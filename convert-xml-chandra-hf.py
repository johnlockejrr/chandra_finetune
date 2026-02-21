#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PAGE-XML / ALTO-XML to Chandra OCR Dataset Converter

Converts PAGE-XML and ALTO-XML ground-truth annotations into a HuggingFace
dataset compatible with train_chandra.py (Chandra OCR fine-tuning).

- Processes XML files in batches (memory-friendly, explicit cleanup + gc)
- Saves all crops (line / paragraph / page) as images on disk
- Writes metadata_train.jsonl with paths + texts
- Builds a HuggingFace dataset at the end, ready for train_chandra.py
- Supports both 'simple' (image + text) and 'finevision' (images + texts) formats
- Optional automatic train/validation split

Output schema (simple format, default):
    image      : PIL Image  (decoded by HF Image feature)
    text       : plain string (ground-truth OCR text)
    filename   : image filename
    source     : XML stem
    type       : "line" / "paragraph" / "page"

Output schema (finevision format):
    images     : [PIL Image]
    texts      : [{"user": "", "assistant": text}]
    text       : plain string
    filename   : image filename
    source     : XML stem
    type       : "line" / "paragraph" / "page"

Usage example:
    python convert-xml-chandra-hf.py \\
        --input_dir /path/to/xml_and_images \\
        --output_dir ./chandra_dataset \\
        --include_full_pages \\
        --include_paragraphs \\
        --max_image_edge 2048 \\
        --aug_copies 1 \\
        --val_ratio 0.1
"""

import argparse
import gc
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import xml.etree.ElementTree as ET
import json
import unicodedata
import random
import math

from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm

from datasets import Dataset, DatasetDict, Features, Value, Sequence, Image as HFImage

try:
    from scipy.spatial import ConvexHull
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Paragraph polygon generation will use simple bounding box.")

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False


# ------------------------------------------------------------
# NEW: High-quality optional resizing
# ------------------------------------------------------------
def resize_if_needed(img: Image.Image, max_edge: Optional[int]) -> Image.Image:
    """
    Resize image so that its longest edge is <= max_edge.
    Keeps aspect ratio. Uses high-quality Lanczos downscaling.
    If max_edge is None or image is already small enough, returns original.
    """
    if not max_edge:
        return img

    w, h = img.size
    longest = max(w, h)

    if longest <= max_edge:
        return img  # no resize needed

    scale = max_edge / float(longest)
    new_w = int(w * scale)
    new_h = int(h * scale)

    return img.resize((new_w, new_h), Image.LANCZOS)


# PAGE-XML namespace schemas
PAGE_NAMESPACES = {
    '2009-03-16': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2009-03-16',
    '2010-01-12': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-01-12',
    '2010-03-19': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19',
    '2013-07-15': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15',
    '2014-08-26': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2014-08-26',
    '2016-07-15': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2016-07-15',
    '2017-07-15': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15',
    '2018-07-15': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2018-07-15',
    '2019-07-15': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15'
}

# ALTO-XML namespace schemas
ALTO_NAMESPACES = {
    'v2': 'http://www.loc.gov/standards/alto/ns-v2#',
    'v3': 'http://www.loc.gov/standards/alto/ns-v3#',
    'v4': 'http://www.loc.gov/standards/alto/ns-v4#'
}


def rotate_expand(pil_img: Image.Image, max_angle: float = 2.0) -> Image.Image:
    """Rotate with expand=True so the canvas grows and nothing gets clipped."""
    angle = random.uniform(-max_angle, max_angle)
    return pil_img.rotate(
        angle,
        expand=True,                  # canvas expands to fit full rotated image
        resample=Image.BICUBIC,       # smooth interpolation
        fillcolor=(255, 255, 255),    # white background for the new corners
    )


def build_augmentation_pipeline() -> Optional[Any]:
    if not ALBUMENTATIONS_AVAILABLE:
        return None
    return A.Compose([
        # Rotation removed — handled by rotate_expand() in apply_augmentation()
        A.Perspective(scale=(0.005, 0.015), p=0.3),
        A.ElasticTransform(alpha=8, sigma=3, p=0.2),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.12, 0.08),
            contrast_limit=(-0.08, 0.08),
            p=0.5,
        ),
        A.HueSaturationValue(
            hue_shift_limit=5,
            sat_shift_limit=(-30, 10),
            val_shift_limit=(-10, 10),
            p=0.3,
        ),
        A.ToGray(p=0.2),
        A.RandomShadow(
            shadow_roi=(0.0, 0.0, 1.0, 1.0),
            num_shadows_limit=(1, 1),
            shadow_dimension=4,
            shadow_intensity_range=(0.05, 0.2),
            p=0.2,
        ),
        A.GaussNoise(std_range=(0.01, 0.04), p=0.35),
        A.OneOf([
            A.Blur(blur_limit=3, p=1.0),
            A.GaussianBlur(blur_limit=3, sigma_limit=(0.1, 0.8), p=1.0),
        ], p=0.2),
        A.Downscale(scale_range=(0.6, 0.9), p=0.2),
        A.Sharpen(alpha=(0.1, 0.3), p=0.2),
        A.ImageCompression(quality_range=(75, 95), p=0.3),
    ])


def pad_image(pil_img: Image.Image, pad: int) -> Image.Image:
    """Add white padding around the image to protect edges from perspective warp."""
    new_w = pil_img.width + pad * 2
    new_h = pil_img.height + pad * 2
    padded = Image.new("RGB", (new_w, new_h), (255, 255, 255))
    padded.paste(pil_img, (pad, pad))
    return padded


def crop_padding(pil_img: Image.Image, pad: int) -> Image.Image:
    """Remove the padding added before perspective warping."""
    w, h = pil_img.size
    left = min(pad, w - 1)
    top = min(pad, h - 1)
    right = max(w - pad, left + 1)
    bottom = max(h - pad, top + 1)
    return pil_img.crop((left, top, right, bottom))


def apply_augmentation(pil_img: Image.Image, pipeline: Any) -> Image.Image:
    if pipeline is None:
        return pil_img

    # Step 1: PIL rotation with canvas expansion — guaranteed no clipping
    if random.random() < 0.5:
        pil_img = rotate_expand(pil_img, max_angle=2.0)

    # Step 2: dynamic padding so A.Perspective hits padding, not text edges
    pad = int(0.015 * max(pil_img.width, pil_img.height)) + 10
    pil_img = pad_image(pil_img, pad=pad)

    # Step 3: albumentations pipeline
    img_np = np.array(pil_img.convert("RGB"))
    result = pipeline(image=img_np)
    pil_img = Image.fromarray(result["image"])

    # Step 4: remove the padding
    pil_img = crop_padding(pil_img, pad=pad)

    return pil_img


def normalize_unicode(text: str, form: str = 'NFC') -> str:
    if not text:
        return ""
    return unicodedata.normalize(form, text.strip())


def detect_xml_format(xml_file: Path) -> Tuple[str, str]:
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        namespace = None
        if root.tag.startswith('{'):
            namespace = root.tag.split('}')[0][1:]
        else:
            xmlns = root.get('{http://www.w3.org/2000/xmlns/}xmlns')
            if xmlns:
                namespace = xmlns

        if namespace:
            for _, alto_ns in ALTO_NAMESPACES.items():
                if namespace == alto_ns:
                    return ('alto', alto_ns)
            for version in ['v4', 'v3', 'v2']:
                alto_ns = ALTO_NAMESPACES[version]
                if alto_ns in namespace or 'alto' in namespace.lower():
                    return ('alto', alto_ns)

        if namespace:
            for _, page_ns in PAGE_NAMESPACES.items():
                if namespace == page_ns:
                    return ('pagexml', page_ns)
            for _, page_ns in PAGE_NAMESPACES.items():
                if page_ns in namespace or 'page' in namespace.lower():
                    return ('pagexml', page_ns)

        root_tag = root.tag.split('}')[-1] if '}' in root.tag else root.tag
        if root_tag.lower() == 'alto':
            return ('alto', ALTO_NAMESPACES['v4'])
        elif 'page' in root_tag.lower():
            return ('pagexml', PAGE_NAMESPACES['2019-07-15'])

        return ('pagexml', PAGE_NAMESPACES['2019-07-15'])

    except Exception as e:
        print(f"Warning: Could not detect format for {xml_file}: {e}")
        return ('pagexml', PAGE_NAMESPACES['2019-07-15'])


def parse_polygon_coords(polygon_str: str, format_type: str = 'pagexml') -> List[Tuple[int, int]]:
    if not polygon_str:
        return []

    points = []

    if format_type == 'alto':
        coords = polygon_str.strip().split()
        if len(coords) % 2 != 0:
            return []
        for i in range(0, len(coords), 2):
            try:
                x = int(float(coords[i]))
                y = int(float(coords[i + 1]))
                points.append((x, y))
            except (ValueError, IndexError):
                continue
    else:
        coords = polygon_str.strip().split()
        for coord in coords:
            try:
                if ',' in coord:
                    x, y = coord.split(',', 1)
                    x = int(float(x))
                    y = int(float(y))
                    points.append((x, y))
            except (ValueError, IndexError):
                continue

    return points


def extract_textlines_from_pagexml(xml_file: Path, namespace: str) -> List[Dict[str, Any]]:
    textlines = []
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        textline_elements = root.findall(f'.//{{{namespace}}}TextLine')

        for textline in textline_elements:
            coords_elem = textline.find(f'{{{namespace}}}Coords')
            if coords_elem is not None:
                polygon_str = coords_elem.get('points')
                if polygon_str:
                    coords = parse_polygon_coords(polygon_str, format_type='pagexml')
                    if len(coords) >= 3:
                        transcription = ""
                        text_equiv = textline.find(f'{{{namespace}}}TextEquiv')
                        if text_equiv is not None:
                            unicode_elem = text_equiv.find(f'{{{namespace}}}Unicode')
                            if unicode_elem is not None and unicode_elem.text:
                                transcription = unicode_elem.text.strip()
                        if transcription:
                            textlines.append({
                                'coords': coords,
                                'transcription': transcription
                            })
    except Exception as e:
        print(f"Error parsing {xml_file}: {e}")
    return textlines


def extract_textlines_from_alto(xml_file: Path, namespace: str) -> List[Dict[str, Any]]:
    textlines = []
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        textline_elements = root.findall(f'.//{{{namespace}}}TextLine')

        for textline in textline_elements:
            coords = []
            transcription = ""

            shape_elem = textline.find(f'{{{namespace}}}Shape')
            if shape_elem is not None:
                polygon_elem = shape_elem.find(f'{{{namespace}}}Polygon')
                if polygon_elem is not None:
                    polygon_str = polygon_elem.get('POINTS')
                    if polygon_str:
                        coords = parse_polygon_coords(polygon_str, format_type='alto')

            if not coords or len(coords) < 3:
                try:
                    hpos = int(textline.get('HPOS', 0))
                    vpos = int(textline.get('VPOS', 0))
                    width = int(textline.get('WIDTH', 0))
                    height = int(textline.get('HEIGHT', 0))
                    if width > 0 and height > 0:
                        coords = [
                            (hpos, vpos),
                            (hpos + width, vpos),
                            (hpos + width, vpos + height),
                            (hpos, vpos + height)
                        ]
                except (ValueError, TypeError):
                    pass

            string_elements = textline.findall(f'{{{namespace}}}String')
            transcription_parts = []
            for string_elem in string_elements:
                content = string_elem.get('CONTENT', '')
                if content:
                    transcription_parts.append(content)
            transcription = ' '.join(transcription_parts).strip()

            if not transcription and textline.text:
                transcription = textline.text.strip()

            if coords and len(coords) >= 3 and transcription:
                textlines.append({
                    'coords': coords,
                    'transcription': transcription
                })
    except Exception as e:
        print(f"Error parsing {xml_file}: {e}")
    return textlines


def crop_image_from_polygon(image: Image.Image,
                            polygon: List[Tuple[int, int]],
                            padding: int = 5,
                            rect_only: bool = False) -> Image.Image:
    """
    Crop image to the bounding box of a polygon.
    If rect_only=True, returns a plain rectangular crop with no polygon masking.
    This is important for paragraphs: the convex hull mask would cut into lines
    with irregular margins, causing character truncation before augmentation runs.
    """
    if not polygon or len(polygon) < 3:
        return image

    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]

    x_min = max(0, min(xs) - padding)
    y_min = max(0, min(ys) - padding)
    x_max = min(image.width, max(xs) + padding)
    y_max = min(image.height, max(ys) + padding)

    cropped_bbox = image.crop((x_min, y_min, x_max, y_max))

    if rect_only:
        # No polygon masking — return clean rectangle, no risk of clipping
        return cropped_bbox

    adjusted_polygon = [(x - x_min, y - y_min) for x, y in polygon]
    mask = Image.new('L', cropped_bbox.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(adjusted_polygon, fill=255)

    background = Image.new('RGB', cropped_bbox.size, (255, 255, 255))
    cropped = Image.composite(cropped_bbox, background, mask)
    return cropped


def merge_polygons_to_surrounding(polygons: List[List[Tuple[int, int]]]) -> List[Tuple[int, int]]:
    if not polygons:
        return []

    all_points = []
    for polygon in polygons:
        if polygon and len(polygon) >= 3:
            all_points.extend(polygon)

    if len(all_points) < 3:
        return []

    if SCIPY_AVAILABLE and len(all_points) >= 3:
        try:
            points_array = np.array(all_points)
            hull = ConvexHull(points_array)
            return [tuple(points_array[vertex]) for vertex in hull.vertices]
        except Exception:
            pass

    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    return [
        (x_min, y_min),
        (x_max, y_min),
        (x_max, y_max),
        (x_min, y_max)
    ]


def group_lines_into_paragraphs(textlines: List[Dict[str, Any]],
                                min_lines: int = 5,
                                max_lines: int = 10) -> List[List[Dict[str, Any]]]:

    if len(textlines) < min_lines:
        return []

    paragraphs = []
    i = 0
    while i < len(textlines):
        paragraph_size = random.randint(min_lines, max_lines)
        paragraph = textlines[i:i + paragraph_size]
        if len(paragraph) >= min_lines:
            paragraphs.append(paragraph)
        i += paragraph_size
    return paragraphs


def find_image_file(xml_file: Path) -> Optional[Path]:
    for ext in ['.jpg', '.jpeg', '.png', '.tiff', '.tif',
                '.JPG', '.JPEG', '.PNG', '.TIFF', '.TIF']:
        potential_image = xml_file.with_suffix(ext)
        if potential_image.exists():
            return potential_image
    return None


def create_sample_metadata(image_rel_path: str,
                           text: str,
                           source: str,
                           sample_type: str) -> Dict[str, Any]:
    """
    Unified metadata:
      - image   : single path (for HF Image viewer)
      - images  : [path] (finevision / LightOnOCR-2 style)
      - text    : plain string
      - texts   : [{"user": "", "assistant": text}]
      - filename, source, type
    """
    filename = os.path.basename(image_rel_path)
    return {
        "image": image_rel_path,
        "images": [image_rel_path],
        "text": text,
        "texts": [{
            "user": "",
            "assistant": text
        }],
        "filename": filename,
        "source": source,
        "type": sample_type,
    }

def convert_xml_to_chandra_train(
    input_dir: str,
    output_dir: str,
    unicode_form: str = 'NFC',
    min_text_length: int = 1,
    min_crop_size: int = 32,
    include_full_pages: bool = False,
    include_paragraphs: bool = False,
    paragraph_min_lines: int = 5,
    paragraph_max_lines: int = 10,
    line_separator: str = '\n',
    batch_size: int = 50,
    aug_copies: int = 0,
    max_image_edge: Optional[int] = 2048,
    output_format: str = "simple",
    val_ratio: float = 0.0,
    seed: int = 3407,
) -> None:

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_path / "metadata_train.jsonl"
    if metadata_path.exists():
        print(f"Overwriting existing metadata file: {metadata_path}")
        metadata_path.unlink()

    aug_pipeline = None
    if aug_copies > 0:
        if not ALBUMENTATIONS_AVAILABLE:
            print("Warning: --aug_copies requested but albumentations is not installed.")
            aug_copies = 0
        else:
            aug_pipeline = build_augmentation_pipeline()
            print(f"Augmentation enabled: {aug_copies} augmented copy/copies per crop.")

    xml_files = sorted(list(input_path.rglob("*.xml")) + list(input_path.rglob("*.XML")))
    if not xml_files:
        print(f"No XML files found in {input_dir}")
        return

    print(f"Found {len(xml_files)} XML files")
    print(f"Include full pages: {include_full_pages}")
    print(f"Include paragraphs: {include_paragraphs}")
    print(f"Max image edge: {max_image_edge}")

    processed_count = 0
    skipped_count = 0
    batch_num = 0
    line_samples = 0
    paragraph_samples = 0
    page_samples = 0
    sample_id = 0

    with metadata_path.open("a", encoding="utf-8") as meta_f:
        for batch_start in range(0, len(xml_files), batch_size):
            batch_end = min(batch_start + batch_size, len(xml_files))
            batch_files = xml_files[batch_start:batch_end]
            batch_num += 1

            print(f"\nProcessing batch {batch_num} ({batch_start+1}-{batch_end})...")

            for xml_file in tqdm(batch_files, desc=f"Batch {batch_num}", leave=False):
                full_image = None
                try:
                    detected_format, namespace = detect_xml_format(xml_file)

                    image_file = find_image_file(xml_file)
                    if not image_file:
                        skipped_count += 1
                        continue

                    try:
                        full_image = Image.open(image_file).convert("RGB")
                    except Exception:
                        skipped_count += 1
                        continue

                    if detected_format == 'alto':
                        textlines = extract_textlines_from_alto(xml_file, namespace)
                    else:
                        textlines = extract_textlines_from_pagexml(xml_file, namespace)

                    if not textlines:
                        skipped_count += 1
                        continue

                    page_transcriptions = []

                    # ------------------------------------------------------------
                    # LINE-LEVEL SAMPLES
                    # ------------------------------------------------------------
                    for line_idx, textline_data in enumerate(textlines):
                        transcription = normalize_unicode(
                            textline_data['transcription'], unicode_form
                        )
                        if len(transcription) < min_text_length:
                            continue

                        if include_full_pages:
                            page_transcriptions.append(transcription)

                        try:
                            cropped_image = crop_image_from_polygon(
                                full_image, textline_data['coords'], padding=5
                            )
                            if cropped_image.width < min_crop_size or cropped_image.height < min_crop_size:
                                continue
                        except Exception:
                            continue

                        # Resize if needed
                        cropped_image = resize_if_needed(cropped_image, max_image_edge)

                        # Save original
                        sample_id += 1
                        img_name = f"{xml_file.stem}_line_{line_idx}_orig_{sample_id}.png"
                        img_path = images_dir / img_name
                        cropped_image.save(img_path)

                        rel_path = os.path.relpath(img_path, output_path)
                        meta = create_sample_metadata(rel_path, transcription, xml_file.stem, "line")
                        meta_f.write(json.dumps(meta, ensure_ascii=False) + "\n")

                        processed_count += 1
                        line_samples += 1

                        # Augmented copies
                        for aug_i in range(aug_copies):
                            aug_img = apply_augmentation(cropped_image, aug_pipeline)
                            aug_img = resize_if_needed(aug_img, max_image_edge)

                            sample_id += 1
                            aug_name = f"{xml_file.stem}_line_{line_idx}_aug{aug_i}_{sample_id}.png"
                            aug_path = images_dir / aug_name
                            aug_img.save(aug_path)

                            rel_aug = os.path.relpath(aug_path, output_path)
                            aug_meta = create_sample_metadata(rel_aug, transcription, xml_file.stem, "line")
                            meta_f.write(json.dumps(aug_meta, ensure_ascii=False) + "\n")

                            processed_count += 1
                            line_samples += 1

                    # ------------------------------------------------------------
                    # PARAGRAPH-LEVEL SAMPLES
                    # ------------------------------------------------------------
                    if include_paragraphs and len(textlines) >= paragraph_min_lines:
                        paragraph_groups = group_lines_into_paragraphs(
                            textlines, paragraph_min_lines, paragraph_max_lines
                        )

                        for para_idx, paragraph_group in enumerate(paragraph_groups):
                            paragraph_transcriptions = []
                            paragraph_polygons = []

                            for line_data in paragraph_group:
                                t = normalize_unicode(line_data['transcription'], unicode_form)
                                if len(t) >= min_text_length:
                                    paragraph_transcriptions.append(t)
                                    paragraph_polygons.append(line_data['coords'])

                            if not paragraph_transcriptions:
                                continue

                            merged_polygon = merge_polygons_to_surrounding(paragraph_polygons)

                            try:
                                para_img = crop_image_from_polygon(full_image, merged_polygon, padding=20, rect_only=True)
                                if para_img.width < min_crop_size or para_img.height < min_crop_size:
                                    continue
                            except Exception:
                                continue

                            para_img = resize_if_needed(para_img, max_image_edge)
                            paragraph_text = line_separator.join(paragraph_transcriptions)

                            sample_id += 1
                            para_name = f"{xml_file.stem}_para_{para_idx}_orig_{sample_id}.png"
                            para_path = images_dir / para_name
                            para_img.save(para_path)

                            rel_para = os.path.relpath(para_path, output_path)
                            para_meta = create_sample_metadata(rel_para, paragraph_text, xml_file.stem, "paragraph")
                            meta_f.write(json.dumps(para_meta, ensure_ascii=False) + "\n")

                            processed_count += 1
                            paragraph_samples += 1

                            # Augmented copies
                            for aug_i in range(aug_copies):
                                aug_para = apply_augmentation(para_img, aug_pipeline)
                                aug_para = resize_if_needed(aug_para, max_image_edge)

                                sample_id += 1
                                aug_para_name = f"{xml_file.stem}_para_{para_idx}_aug{aug_i}_{sample_id}.png"
                                aug_para_path = images_dir / aug_para_name
                                aug_para.save(aug_para_path)

                                rel_para_aug = os.path.relpath(aug_para_path, output_path)
                                aug_para_meta = create_sample_metadata(
                                    rel_para_aug, paragraph_text, xml_file.stem, "paragraph"
                                )
                                meta_f.write(json.dumps(aug_para_meta, ensure_ascii=False) + "\n")

                                processed_count += 1
                                paragraph_samples += 1

                    # ------------------------------------------------------------
                    # FULL-PAGE SAMPLES
                    # ------------------------------------------------------------
                    if include_full_pages and page_transcriptions:
                        full_text = line_separator.join(
                            normalize_unicode(t, unicode_form)
                            for t in page_transcriptions
                            if len(t) >= min_text_length
                        )

                        if full_text:
                            full_resized = resize_if_needed(full_image, max_image_edge)

                            sample_id += 1
                            page_name = f"{xml_file.stem}_page_orig_{sample_id}.png"
                            page_path = images_dir / page_name
                            full_resized.save(page_path)

                            rel_page = os.path.relpath(page_path, output_path)
                            page_meta = create_sample_metadata(rel_page, full_text, xml_file.stem, "page")
                            meta_f.write(json.dumps(page_meta, ensure_ascii=False) + "\n")

                            processed_count += 1
                            page_samples += 1

                            # Augmented copies  ← THIS BLOCK WAS MISSING
                            for aug_i in range(aug_copies):
                                aug_page = apply_augmentation(full_resized, aug_pipeline)
                                aug_page = resize_if_needed(aug_page, max_image_edge)

                                sample_id += 1
                                aug_page_name = f"{xml_file.stem}_page_aug{aug_i}_{sample_id}.png"
                                aug_page_path = images_dir / aug_page_name
                                aug_page.save(aug_page_path)

                                rel_page_aug = os.path.relpath(aug_page_path, output_path)
                                aug_page_meta = create_sample_metadata(
                                    rel_page_aug, full_text, xml_file.stem, "page"
                                )
                                meta_f.write(json.dumps(aug_page_meta, ensure_ascii=False) + "\n")

                                processed_count += 1
                                page_samples += 1

                except Exception as e:
                    print(f"Error processing {xml_file}: {e}")
                    skipped_count += 1
                    continue
                finally:
                    if full_image is not None:
                        full_image.close()
                        del full_image

            gc.collect()

    print("\nFinished conversion.")
    print(f"Total samples: {processed_count}")
    print(f"  Lines:      {line_samples}")
    print(f"  Paragraphs: {paragraph_samples}")
    print(f"  Pages:      {page_samples}")
    print(f"Skipped files: {skipped_count}")
    print(f"Metadata: {metadata_path}")
    print(f"Images:   {images_dir}")

    # ------------------------------------------------------------
    # Build HuggingFace dataset (streaming generator, one image at a time)
    # Compatible with train_chandra.py in both 'simple' and 'finevision' modes.
    # ------------------------------------------------------------
    print(f"\nBuilding HuggingFace dataset (format={output_format}, streaming)...")

    def _hf_generator_simple(meta_path: Path, base_path: Path):
        """Yield {image, text, filename, source, type} -- simple format for Chandra."""
        skipped = 0
        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                ex = json.loads(line)
                img_rel = ex.get("image") or (ex["images"][0] if isinstance(ex.get("images"), list) else "")
                img_abs = base_path / img_rel
                if not img_abs.exists():
                    skipped += 1
                    continue
                try:
                    pil = Image.open(img_abs).convert("RGB")
                except Exception:
                    skipped += 1
                    continue
                yield {
                    "image": pil,
                    "text": ex.get("text", ""),
                    "filename": ex.get("filename", ""),
                    "source": ex.get("source", ""),
                    "type": ex.get("type", ""),
                }
                pil.close()
        if skipped > 0:
            print(f"  Skipped {skipped} samples (missing/invalid images).")

    def _hf_generator_finevision(meta_path: Path, base_path: Path):
        """Yield {images, texts, text, ...} -- finevision format (LightOnOCR compat)."""
        skipped = 0
        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                ex = json.loads(line)
                img_rel = ex["images"][0] if isinstance(ex.get("images"), list) else ex.get("image", "")
                img_abs = base_path / img_rel
                if not img_abs.exists():
                    skipped += 1
                    continue
                try:
                    pil = Image.open(img_abs).convert("RGB")
                except Exception:
                    skipped += 1
                    continue
                texts_list = ex.get("texts", [])
                texts = [
                    {"user": t.get("user", ""), "assistant": t.get("assistant", "")}
                    for t in (texts_list if isinstance(texts_list, list) else [])
                    if isinstance(t, dict)
                ]
                yield {
                    "images": [pil],
                    "texts": texts,
                    "text": ex.get("text", ""),
                    "filename": ex.get("filename", ""),
                    "source": ex.get("source", ""),
                    "type": ex.get("type", ""),
                }
                pil.close()
        if skipped > 0:
            print(f"  Skipped {skipped} samples (missing/invalid images).")

    if output_format == "simple":
        features = Features({
            "image": HFImage(),
            "text": Value("string"),
            "filename": Value("string"),
            "source": Value("string"),
            "type": Value("string"),
        })
        gen_fn = lambda: _hf_generator_simple(metadata_path, output_path)  # noqa: E731
    else:
        features = None
        gen_fn = lambda: _hf_generator_finevision(metadata_path, output_path)  # noqa: E731

    dataset = Dataset.from_generator(gen_fn, features=features)

    if features is None:
        dataset = dataset.cast_column("images", Sequence(HFImage()))

    # Train / validation split
    if val_ratio > 0 and len(dataset) > 10:
        split = dataset.train_test_split(test_size=val_ratio, seed=seed)
        dataset_dict = DatasetDict({"train": split["train"], "validation": split["test"]})
        print(f"  Split: train={len(split['train']):,}  validation={len(split['test']):,}")
    else:
        dataset_dict = DatasetDict({"train": dataset})

    dataset_path = output_path / "hf_dataset_chandra"
    dataset_dict.save_to_disk(str(dataset_path))
    gc.collect()
    print(f"HuggingFace dataset saved to: {dataset_path}")
    for split_name, split_ds in dataset_dict.items():
        print(f"  {split_name}: {len(split_ds):,} samples")


def main():
    parser = argparse.ArgumentParser(
        description="PAGE-XML / ALTO-XML to Chandra OCR HuggingFace dataset converter. "
                    "Output is directly usable with train_chandra.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing XML + image files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for crops, metadata, and HF dataset")

    parser.add_argument("--unicode_form", type=str, default="NFC",
                        choices=["NFC", "NFD", "NFKC", "NFKD"])

    parser.add_argument("--min_text_length", type=int, default=1)
    parser.add_argument("--min_crop_size", type=int, default=32)

    parser.add_argument("--include_full_pages", action="store_true",
                        help="Also create full-page samples")
    parser.add_argument("--include_paragraphs", action="store_true",
                        help="Also create paragraph-level samples")

    parser.add_argument("--paragraph_min_lines", type=int, default=5)
    parser.add_argument("--paragraph_max_lines", type=int, default=10)

    parser.add_argument("--line_separator", type=str, default="\n")
    parser.add_argument("--batch_size", type=int, default=50,
                        help="XML files per processing batch (default: 50)")

    parser.add_argument("--aug_copies", type=int, default=0,
                        help="Number of augmented copies per crop (default: 0)")

    parser.add_argument("--max_image_edge", type=int, default=2048,
                        help="Resize images so longest edge <= this value. "
                             "Default 2048 matches Chandra's optimal resolution.")

    parser.add_argument("--output_format", type=str, default="simple",
                        choices=["simple", "finevision"],
                        help="HF dataset format: 'simple' (image+text, default for Chandra) "
                             "or 'finevision' (images+texts, LightOnOCR-2 compat)")

    parser.add_argument("--val_ratio", type=float, default=0.0,
                        help="Validation split ratio (default: 0.0 = no split). "
                             "Set to e.g. 0.1 to create a 90/10 train/val split.")

    parser.add_argument("--seed", type=int, default=3407,
                        help="Random seed for train/val splitting (default: 3407)")

    args = parser.parse_args()

    convert_xml_to_chandra_train(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        unicode_form=args.unicode_form,
        min_text_length=args.min_text_length,
        min_crop_size=args.min_crop_size,
        include_full_pages=args.include_full_pages,
        include_paragraphs=args.include_paragraphs,
        paragraph_min_lines=args.paragraph_min_lines,
        paragraph_max_lines=args.paragraph_max_lines,
        line_separator=args.line_separator,
        batch_size=args.batch_size,
        aug_copies=args.aug_copies,
        max_image_edge=args.max_image_edge,
        output_format=args.output_format,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

