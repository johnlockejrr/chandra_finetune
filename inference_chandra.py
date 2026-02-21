#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference_chandra.py

Run OCR inference with a fine-tuned Chandra checkpoint (LoRA or merged).
Supports both Unsloth LoRA adapters and merged 16-bit models.

Usage:
    # LoRA checkpoint (loads base model + applies adapter)
    python inference_chandra.py --checkpoint ./chandra_sam_44_mss --image doc.png

    # Merged 16-bit model
    python inference_chandra.py --checkpoint ./chandra_sam_44_mss/merged_16bit --image doc.png --merged

    # Original Chandra (no fine-tuning)
    python inference_chandra.py --checkpoint datalab-to/chandra --image doc.png --merged

    # Custom prompt
    python inference_chandra.py --checkpoint ./ckpt --image doc.png --prompt "Extract the table from this image"

    # Batch: all images in a directory
    python inference_chandra.py --checkpoint ./ckpt --image_dir ./scans/ --output_dir ./results/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run OCR inference with a fine-tuned Chandra checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- input ---
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument("--image", type=str, help="Path to a single image file")
    inp.add_argument("--image_dir", type=str, help="Directory of images to process")

    # --- model ---
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to checkpoint dir (LoRA adapter or merged model)")
    p.add_argument("--base_model", type=str, default="datalab-to/chandra",
                   help="Base model for LoRA loading (default: datalab-to/chandra)")
    p.add_argument("--merged", action="store_true",
                   help="Checkpoint is a merged model (not LoRA adapter)")
    p.add_argument("--load_in_4bit", action="store_true", default=True,
                   help="Load in 4-bit quantisation (default: True, LoRA mode only)")
    p.add_argument("--no_4bit", dest="load_in_4bit", action="store_false")

    # --- generation ---
    p.add_argument("--prompt", type=str, default=None,
                   help="Custom OCR prompt (default: Chandra's built-in OCR prompt)")
    p.add_argument("--prompt_type", type=str, default="ocr",
                   choices=["ocr", "ocr_layout"],
                   help="Prompt type: 'ocr' (plain text) or 'ocr_layout' (with bboxes)")
    p.add_argument("--max_tokens", type=int, default=12384,
                   help="Max output tokens (default: 12384)")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature (default: 0.0 = greedy)")
    p.add_argument("--longest_edge", type=int, default=1540,
                   help="Resize image longest edge (default: 1540)")

    # --- output ---
    p.add_argument("--output_dir", type=str, default=None,
                   help="Save output text files here (one per image)")
    p.add_argument("--output_file", type=str, default=None,
                   help="Save single output to this file")

    return p.parse_args()


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}

OCR_PROMPT = (
    "OCR this image to HTML.\n\n"
    "Only use these tags ['math', 'br', 'i', 'b', 'u', 'del', 'sup', 'sub', "
    "'table', 'tr', 'td', 'p', 'th', 'div', 'pre', 'h1', 'h2', 'h3', 'h4', "
    "'h5', 'ul', 'ol', 'li', 'input', 'a', 'span', 'img', 'hr', 'tbody', "
    "'small', 'caption', 'strong', 'thead', 'big', 'code'], and these attributes "
    "['class', 'colspan', 'rowspan', 'display', 'checked', 'type', 'border', "
    "'value', 'style', 'href', 'alt', 'align'].\n\n"
    "Guidelines:\n"
    "* Inline math: Surround math with <math>...</math> tags. Math expressions "
    "should be rendered in KaTeX-compatible LaTeX. Use display for block math.\n"
    "* Tables: Use colspan and rowspan attributes to match table structure.\n"
    "* Formatting: Maintain consistent formatting with the image, including spacing, "
    "indentation, subscripts/superscripts, and special characters.\n"
    "* Images: Include a description of any images in the alt attribute of an <img> tag. "
    "Do not fill out the src property.\n"
    "* Forms: Mark checkboxes and radio buttons properly.\n"
    "* Text: join lines together properly into paragraphs using <p>...</p> tags. "
    "Use <br> tags for line breaks within paragraphs, but only when absolutely "
    "necessary to maintain meaning.\n"
    "* Use the simplest possible HTML structure that accurately represents the "
    "content of the block.\n"
    "* Make sure the text is accurate and easy for a human to read and interpret. "
    "Reading order should be correct and natural."
)

OCR_LAYOUT_PROMPT = (
    "OCR this image to HTML, arranged as layout blocks. Each layout block should "
    "be a div with the data-bbox attribute representing the bounding box of the "
    "block in [x0, y0, x1, y1] format. Bboxes are normalized 0-1024. The data-label "
    "attribute is the label for the block.\n\n"
    "Use the following labels:\n"
    "- Caption\n- Footnote\n- Equation-Block\n- List-Group\n- Page-Header\n"
    "- Page-Footer\n- Image\n- Section-Header\n- Table\n- Text\n- Complex-Block\n"
    "- Code-Block\n- Form\n- Table-Of-Contents\n- Figure\n\n"
    + OCR_PROMPT.split("Guidelines:")[1]  # reuse guidelines
)

PROMPT_MAP = {"ocr": OCR_PROMPT, "ocr_layout": "Guidelines:" + OCR_LAYOUT_PROMPT}


def load_model_lora(checkpoint: str, base_model: str, load_in_4bit: bool):
    """Load base model via Unsloth and apply LoRA adapter."""
    from unsloth import FastVisionModel
    from transformers import Qwen3VLProcessor
    from peft import PeftModel

    print(f"Loading base model: {base_model} (4-bit={load_in_4bit})")
    model, tokenizer = FastVisionModel.from_pretrained(
        base_model,
        load_in_4bit=load_in_4bit,
        use_gradient_checkpointing=False,
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter: {checkpoint}")
    model = PeftModel.from_pretrained(model, checkpoint)

    FastVisionModel.for_inference(model)

    # Processor: use checkpoint dir if it has one (final save), else base model
    ckpt_path = Path(checkpoint)
    proc_path = ckpt_path if (ckpt_path / "preprocessor_config.json").exists() else base_model
    processor = Qwen3VLProcessor.from_pretrained(str(proc_path))
    return model, tokenizer, processor


def load_model_merged(checkpoint: str):
    """Load a merged (full) model directly via transformers."""
    import torch
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    print(f"Loading merged model: {checkpoint}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    processor = Qwen3VLProcessor.from_pretrained(checkpoint)
    tokenizer = processor.tokenizer
    return model, tokenizer, processor


def resize_image(img, longest_edge: int):
    """Resize so longest edge <= longest_edge, maintaining aspect ratio."""
    from PIL import Image
    w, h = img.size
    if max(w, h) <= longest_edge:
        return img
    scale = longest_edge / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    return img.resize((new_w, new_h), Image.LANCZOS)


def run_inference(
    model, tokenizer, processor, image_path: str, args: argparse.Namespace
) -> str:
    """Run inference on a single image and return the generated text."""
    import torch
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    img = resize_image(img, args.longest_edge)

    if args.prompt:
        prompt_text = args.prompt
    else:
        prompt_text = PROMPT_MAP.get(args.prompt_type, OCR_PROMPT)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    try:
        from qwen_vl_utils import process_vision_info
        image_inputs, _ = process_vision_info(messages)
    except ImportError:
        image_inputs = [img]

    inputs = processor(
        text=text,
        images=image_inputs,
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    gen_kwargs = dict(
        max_new_tokens=args.max_tokens,
        use_cache=True,
    )
    if args.temperature > 0:
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = 0.9
        gen_kwargs["do_sample"] = True
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    trimmed = output_ids[0][inputs["input_ids"].shape[1]:]
    result = processor.tokenizer.decode(trimmed, skip_special_tokens=True)
    return result.strip()


def main():
    args = parse_args()

    if args.merged:
        model, tokenizer, processor = load_model_merged(args.checkpoint)
    else:
        model, tokenizer, processor = load_model_lora(
            args.checkpoint, args.base_model, args.load_in_4bit
        )

    if args.image:
        images = [Path(args.image)]
    else:
        img_dir = Path(args.image_dir)
        images = sorted(
            p for p in img_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            print(f"No images found in {img_dir}")
            sys.exit(1)

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {len(images)} image(s)...\n")

    for img_path in images:
        print(f"--- {img_path.name} ---")
        t0 = time.time()
        result = run_inference(model, tokenizer, processor, str(img_path), args)
        elapsed = time.time() - t0

        print(result)
        print(f"\n[{elapsed:.1f}s]\n")

        if args.output_file and len(images) == 1:
            Path(args.output_file).write_text(result, encoding="utf-8")
            print(f"Saved to {args.output_file}")
        elif args.output_dir:
            out_path = Path(args.output_dir) / f"{img_path.stem}.txt"
            out_path.write_text(result, encoding="utf-8")
            print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
