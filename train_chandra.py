#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_chandra.py

Fine-tuning script for Chandra OCR model (datalab-to/chandra) using Unsloth.
Based on Qwen3-VL-8B architecture with optimized image preprocessing.

Chandra specs (from config.json):
    Architecture  : Qwen3VLForConditionalGeneration
    Base model    : Qwen/Qwen3-VL-8B-Instruct
    Max seq length: 2048
    Image token   : 151655  (<|image_pad|>)
    Vision start  : 151652  (<|vision_start|>)
    Vision end    : 151653  (<|vision_end|>)
    Patch size    : 16, spatial merge 2
    Dtype         : bfloat16

Usage:
    python train_chandra.py --dataset_dir ./data --output_dir ./output
    python train_chandra.py --dataset_name unsloth/LaTeX_OCR --output_dir ./output
    python train_chandra.py --help
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# ---------------------------------------------------------------------------
# 2. Configuration Constants
# ---------------------------------------------------------------------------

MODEL_ID = "datalab-to/chandra"

# Token IDs (from chandra-model/config.json)
IMAGE_TOKEN_ID = 151655
VISION_START_TOKEN_ID = 151652
VISION_END_TOKEN_ID = 151653
VIDEO_TOKEN_ID = 151656
BOS_TOKEN_ID = 151643
EOS_TOKEN_ID = 151645

# Architecture constants
MAX_SEQ_LENGTH = 2048
VOCAB_SIZE = 151936
TEXT_HIDDEN_SIZE = 4096
NUM_HIDDEN_LAYERS = 36
VISION_HIDDEN_SIZE = 1152
PATCH_SIZE = 16
SPATIAL_MERGE_SIZE = 2

# Training defaults
DEFAULT_TARGET_LONGEST_EDGE = 2048
DEFAULT_SHORTEST_EDGE = 28
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUM = 8
DEFAULT_WARMUP_STEPS = 50
DEFAULT_MAX_STEPS = 500
DEFAULT_WEIGHT_DECAY = 0.01

# OCR instruction used in conversation format
OCR_INSTRUCTION = (
    "Extract all text from this document image. "
    "Preserve the layout, structure, and formatting using Markdown. "
    "Include tables, forms, and mathematical equations in proper format."
)

log = logging.getLogger("train_chandra")


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Defined as a standalone function so ``--help`` can be served before
    heavy imports (torch, unsloth, transformers).
    """
    p = argparse.ArgumentParser(
        description="Fine-tune Chandra OCR model (datalab-to/chandra) with Unsloth + LoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- dataset ---
    ds = p.add_mutually_exclusive_group(required=True)
    ds.add_argument("--dataset_dir", type=str, help="Path to local HF dataset directory")
    ds.add_argument("--dataset_name", type=str, help="HuggingFace dataset identifier")
    p.add_argument("--dataset_subset", type=str, default=None, help="HF dataset subset/config")
    p.add_argument("--dataset_split", type=str, default="train", help="Dataset split (default: train)")
    p.add_argument(
        "--image_column", type=str, default="image",
        help="Column name for images when using simple format datasets (default: image)",
    )
    p.add_argument(
        "--text_column", type=str, default="text",
        help="Column name for text when using simple format datasets (default: text)",
    )
    p.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio (default: 0.1)")

    # --- model ---
    p.add_argument("--model_id", type=str, default=MODEL_ID, help=f"Model ID or local path (default: {MODEL_ID})")
    p.add_argument("--load_in_4bit", action="store_true", default=True, help="Load model in 4-bit (default: True)")
    p.add_argument("--no_4bit", dest="load_in_4bit", action="store_false", help="Load model in 16-bit instead of 4-bit")

    # --- image resolution ---
    p.add_argument(
        "--longest_edge", type=int, default=DEFAULT_TARGET_LONGEST_EDGE,
        help=f"Target longest image edge in px (default: {DEFAULT_TARGET_LONGEST_EDGE})",
    )

    # --- LoRA ---
    p.add_argument("--lora_r", type=int, default=DEFAULT_LORA_R, help=f"LoRA rank (default: {DEFAULT_LORA_R})")
    p.add_argument("--lora_alpha", type=int, default=DEFAULT_LORA_ALPHA, help=f"LoRA alpha (default: {DEFAULT_LORA_ALPHA})")
    p.add_argument("--lora_dropout", type=float, default=DEFAULT_LORA_DROPOUT, help=f"LoRA dropout (default: {DEFAULT_LORA_DROPOUT})")
    p.add_argument(
        "--finetune_vision", action="store_true", default=True,
        help="Fine-tune vision encoder layers (default: True)",
    )
    p.add_argument("--no_finetune_vision", dest="finetune_vision", action="store_false")

    # --- training ---
    p.add_argument("--output_dir", type=str, required=True, help="Directory for checkpoints and final model")
    p.add_argument("--num_epochs", type=int, default=None, help="Number of training epochs (overrides --max_steps)")
    p.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS, help=f"Max training steps (default: {DEFAULT_MAX_STEPS})")
    p.add_argument("--per_device_train_batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--gradient_accumulation_steps", type=int, default=DEFAULT_GRAD_ACCUM)
    p.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE)
    p.add_argument("--warmup_steps", type=int, default=DEFAULT_WARMUP_STEPS)
    p.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=["cosine", "linear", "constant"])
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--report_to", type=str, default="none", help="Reporting integration (none, wandb, tensorboard)")
    p.add_argument("--ocr_instruction", type=str, default=OCR_INSTRUCTION, help="System instruction for OCR")

    # --- export ---
    p.add_argument("--save_merged_16bit", action="store_true", help="Also save merged 16-bit model for vLLM")
    p.add_argument("--save_gguf", action="store_true", help="Also save GGUF quantisation (q8_0)")
    p.add_argument("--push_to_hub", type=str, default=None, help="HuggingFace Hub repo to push to (e.g. user/repo)")
    p.add_argument("--hf_token", type=str, default=None, help="HuggingFace token for push_to_hub")

    # --- early stopping ---
    p.add_argument(
        "--eval_steps", type=int, default=50,
        help="Run evaluation every N steps (default: 50). Also controls early stopping granularity.",
    )
    p.add_argument(
        "--early_stopping_patience", type=int, default=3,
        help="Stop after N evaluations without improvement (default: 3). Set 0 to disable.",
    )
    p.add_argument(
        "--early_stopping_threshold", type=float, default=0.0,
        help="Minimum improvement to count as progress (default: 0.0)",
    )
    p.add_argument(
        "--metric_for_best_model", type=str, default="eval_loss",
        help="Metric to monitor for early stopping (default: eval_loss)",
    )

    # --- resume ---
    p.add_argument(
        "--resume_from_checkpoint", type=str, default=None,
        help="Path to checkpoint directory or 'latest' to auto-detect",
    )

    # --- OCR metrics ---
    p.add_argument(
        "--compute_cer_wer", action="store_true", default=True,
        help="Compute CER/WER during evaluation (requires jiwer, default: True)",
    )
    p.add_argument("--no_cer_wer", dest="compute_cer_wer", action="store_false",
                   help="Disable CER/WER computation during evaluation")

    # --- misc ---
    p.add_argument("--skip_verification", action="store_true", help="Skip config verification step")
    p.add_argument("--skip_pre_eval", action="store_true", help="Skip inference test before training")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    return p


# Fast-exit on --help so we never import torch/unsloth just to show usage.
if "--help" in sys.argv or "-h" in sys.argv:
    _build_parser().parse_args()


# ---------------------------------------------------------------------------
# 1. Heavy Imports (deferred past --help)
# ---------------------------------------------------------------------------

from unsloth import FastVisionModel, unsloth_train  # noqa: E402
import torch  # noqa: E402
import numpy as np  # noqa: E402
from transformers import Qwen3VLProcessor, TextStreamer, EarlyStoppingCallback  # noqa: E402
from datasets import load_dataset, load_from_disk, DatasetDict  # noqa: E402
from unsloth.trainer import UnslothVisionDataCollator  # noqa: E402
from trl import SFTTrainer, SFTConfig  # noqa: E402
from PIL import Image  # noqa: E402

try:
    import jiwer  # noqa: E402
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False


# ---------------------------------------------------------------------------
# 3. Model Loading
# ---------------------------------------------------------------------------

def load_model(
    model_id: str = MODEL_ID,
    *,
    load_in_4bit: bool = True,
    longest_edge: int = DEFAULT_TARGET_LONGEST_EDGE,
) -> tuple:
    """Load the Chandra model + tokenizer + processor.

    Uses ``FastVisionModel`` for Unsloth-optimised loading and explicitly
    instantiates ``Qwen3VLProcessor`` to guarantee correct image
    preprocessing (avoids AutoProcessor falling back to wrong class).

    Args:
        model_id: HuggingFace model identifier or local path.
        load_in_4bit: Whether to use 4-bit quantisation.
        longest_edge: Target longest edge for image resizing.

    Returns:
        (model, tokenizer, processor) tuple.
    """
    log.info("Loading model from %s (4-bit=%s)", model_id, load_in_4bit)

    model, tokenizer = FastVisionModel.from_pretrained(
        model_id,
        load_in_4bit=load_in_4bit,
        use_gradient_checkpointing="unsloth",
        trust_remote_code=True,
    )

    # Explicit Qwen3VLProcessor (not AutoProcessor) for correct image preprocessing
    processor = Qwen3VLProcessor.from_pretrained(model_id)

    # Override image resolution for training
    processor.image_processor.size = {
        "longest_edge": longest_edge,
        "shortest_edge": DEFAULT_SHORTEST_EDGE,
    }

    # Video token ID (151656) is left as-is from config -- it simply won't
    # match any input tokens in OCR data.  Setting it to None crashes Qwen3-VL
    # because the model computes `input_ids == video_token_id` which returns a
    # plain bool instead of a tensor mask.

    log.info(
        "Model loaded: type=%s  processor=%s  image_processor=%s",
        model.config.model_type,
        type(processor).__name__,
        type(processor.image_processor).__name__,
    )
    return model, tokenizer, processor


# ---------------------------------------------------------------------------
# 4. Configuration Verification
# ---------------------------------------------------------------------------

def verify_config(model, processor) -> bool:
    """Validate that the loaded model matches Chandra's expected configuration.

    Raises ``ValueError`` if any critical check fails.

    Returns:
        True when all checks pass.
    """
    cfg = model.config
    checks: Dict[str, bool] = {
        "image_token_id == 151655": cfg.image_token_id == IMAGE_TOKEN_ID,
        "vision_start_token_id == 151652": cfg.vision_start_token_id == VISION_START_TOKEN_ID,
        "vision_end_token_id == 151653": cfg.vision_end_token_id == VISION_END_TOKEN_ID,
        "max_seq_length == 2048": cfg.max_seq_length == MAX_SEQ_LENGTH,
        "model_type == qwen3_vl": cfg.model_type == "qwen3_vl",
        "text_config.vocab_size == 151936": cfg.text_config.vocab_size == VOCAB_SIZE,
        "text_config.hidden_size == 4096": cfg.text_config.hidden_size == TEXT_HIDDEN_SIZE,
        "vision_config.patch_size == 16": cfg.vision_config.patch_size == PATCH_SIZE,
        "vision_config.spatial_merge_size == 2": cfg.vision_config.spatial_merge_size == SPATIAL_MERGE_SIZE,
    }

    log.info("Configuration verification:")
    all_pass = True
    for name, ok in checks.items():
        status = "PASS" if ok else "FAIL"
        log.info("  [%s] %s", status, name)
        if not ok:
            all_pass = False

    if not all_pass:
        raise ValueError(
            "Chandra configuration mismatch! The loaded model does not match "
            "the expected token IDs or architecture parameters."
        )
    log.info("All configuration checks passed.")
    return True


# ---------------------------------------------------------------------------
# 5. Dataset Preparation
# ---------------------------------------------------------------------------

def _detect_format(sample: dict, image_col: str, text_col: str) -> str:
    """Detect whether a dataset sample is 'finevision' or 'simple'."""
    keys = set(sample.keys())
    if "images" in keys and "texts" in keys:
        return "finevision"
    if image_col in keys and text_col in keys:
        return "simple"
    return "unknown"


def convert_to_conversation(
    sample: dict,
    instruction: str = OCR_INSTRUCTION,
    image_col: str = "image",
    text_col: str = "text",
    dataset_format: str = "simple",
) -> Dict[str, Any]:
    """Convert a dataset sample into the Qwen3-VL chat format.

    Supports both *simple* (``image`` + ``text`` columns) and
    *finevision* (``images`` + ``texts`` columns) dataset layouts.

    Returns:
        ``{"messages": [user_msg, assistant_msg]}`` dict.
    """
    if dataset_format == "finevision":
        img = sample["images"][0]
        text_entry = sample["texts"][0]
        ground_truth = text_entry.get("assistant", "") if isinstance(text_entry, dict) else str(text_entry)
    else:
        img = sample[image_col]
        ground_truth = sample[text_col]

    if isinstance(img, str) or isinstance(img, Path):
        img = Image.open(img).convert("RGB")

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": img},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": ground_truth}],
        },
    ]
    return {"messages": conversation}


def prepare_dataset(
    args: argparse.Namespace,
) -> tuple:
    """Load and convert the dataset into conversation format.

    Handles both local disk datasets and HuggingFace Hub datasets.
    Automatically detects whether the data uses finevision or simple format.
    Splits into train/eval using ``val_ratio`` when early stopping is enabled.

    Returns:
        ``(train_conversations, eval_conversations)`` tuple.
        ``eval_conversations`` may be ``None`` if early stopping is disabled.
    """
    train_ds = None
    eval_ds = None

    if args.dataset_dir:
        log.info("Loading local dataset from %s", args.dataset_dir)
        ds = load_from_disk(args.dataset_dir)
        if isinstance(ds, DatasetDict):
            split = args.dataset_split if args.dataset_split in ds else list(ds.keys())[0]
            train_ds = ds[split]
            # Use existing validation split if available
            for val_key in ("validation", "val", "test"):
                if val_key in ds:
                    eval_ds = ds[val_key]
                    log.info("Using existing '%s' split for evaluation (%d samples)", val_key, len(eval_ds))
                    break
            log.info("Using split '%s' (%d samples)", split, len(train_ds))
        else:
            train_ds = ds
            log.info("Loaded %d samples", len(train_ds))
    else:
        log.info("Loading HuggingFace dataset: %s", args.dataset_name)
        kwargs: Dict[str, Any] = {}
        if args.dataset_subset:
            kwargs["name"] = args.dataset_subset
        train_ds = load_dataset(args.dataset_name, split=args.dataset_split, **kwargs)
        log.info("Loaded %d samples", len(train_ds))

    # Auto-split for early stopping when no eval set found
    use_early_stopping = args.early_stopping_patience > 0
    if use_early_stopping and eval_ds is None and args.val_ratio > 0:
        split_result = train_ds.train_test_split(test_size=args.val_ratio, seed=args.seed)
        train_ds = split_result["train"]
        eval_ds = split_result["test"]
        log.info(
            "Auto-split: train=%d  eval=%d  (val_ratio=%.2f)",
            len(train_ds), len(eval_ds), args.val_ratio,
        )

    # Detect format
    sample = train_ds[0]
    fmt = _detect_format(sample, args.image_column, args.text_column)
    log.info("Detected dataset format: %s", fmt)

    if fmt == "unknown":
        log.warning(
            "Could not detect dataset format. Available columns: %s. "
            "Assuming 'simple' with image_col='%s', text_col='%s'.",
            list(sample.keys()), args.image_column, args.text_column,
        )
        fmt = "simple"

    def _convert_split(dataset):
        return [
            convert_to_conversation(
                s,
                instruction=args.ocr_instruction,
                image_col=args.image_column,
                text_col=args.text_column,
                dataset_format=fmt,
            )
            for s in dataset
        ]

    log.info("Converting %d train samples to conversation format...", len(train_ds))
    train_conv = _convert_split(train_ds)

    eval_conv = None
    if eval_ds is not None:
        log.info("Converting %d eval samples to conversation format...", len(eval_ds))
        eval_conv = _convert_split(eval_ds)

    log.info("Conversion complete: train=%d  eval=%s", len(train_conv), len(eval_conv) if eval_conv else "None")
    return train_conv, eval_conv


# ---------------------------------------------------------------------------
# 6. Training Setup
# ---------------------------------------------------------------------------

def setup_lora(
    model,
    *,
    r: int = DEFAULT_LORA_R,
    alpha: int = DEFAULT_LORA_ALPHA,
    dropout: float = DEFAULT_LORA_DROPOUT,
    finetune_vision: bool = True,
):
    """Apply LoRA adapters to the model via Unsloth.

    Args:
        model: The base model loaded via ``FastVisionModel``.
        r: LoRA rank.
        alpha: LoRA alpha scaling factor.
        dropout: LoRA dropout rate.
        finetune_vision: Whether to also fine-tune vision encoder layers.

    Returns:
        The PEFT-wrapped model.
    """
    log.info(
        "Applying LoRA: r=%d  alpha=%d  dropout=%.3f  vision=%s",
        r, alpha, dropout, finetune_vision,
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=finetune_vision,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    model.print_trainable_parameters()
    return model


def _preprocess_logits_for_metrics(logits, labels):
    """Reduce logits to argmax token IDs so we don't OOM gathering full vocab tensors."""
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def _make_compute_metrics(tokenizer, max_samples: int = 64):
    """Build a compute_metrics fn that computes CER and WER via jiwer.

    These are *teacher-forced* metrics (the model sees correct previous tokens
    at each step), so they are optimistic compared to actual autoregressive
    generation.  Still very useful for tracking training progress.

    Args:
        tokenizer: The model tokenizer.
        max_samples: Maximum number of eval samples to decode for CER/WER.
                     Decoding the full eval set on long sequences is very slow.
                     64 samples give a reliable metric estimate.
    """
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    vocab_sz = int(getattr(tokenizer, "vocab_size", VOCAB_SIZE))

    def compute_metrics(eval_pred):
        pred_ids, label_ids = eval_pred

        # Cast to int64 and clamp to valid token range
        pred_ids = np.clip(pred_ids, 0, vocab_sz - 1).astype(np.int64)
        label_ids = label_ids.astype(np.int64)

        # --- KEY FIX 1: Cap samples to avoid O(N * seq_len) Python decode ---
        n = pred_ids.shape[0]
        if n > max_samples:
            indices = np.random.choice(n, max_samples, replace=False)
            pred_ids = pred_ids[indices]
            label_ids = label_ids[indices]

        # Only score response tokens: prompt/image positions have label=-100.
        response_mask = label_ids != -100
        pred_ids = np.where(response_mask, pred_ids, pad_id)
        label_ids = np.where(response_mask, label_ids, pad_id)

        # --- KEY FIX 2: Trim trailing padding columns to reduce decode work ---
        # Find the last column that has any real (non-pad) label token
        any_real = response_mask.any(axis=0)  # shape: [seq_len]
        if any_real.any():
            last_real_col = int(np.where(any_real)[0].max()) + 1
            pred_ids = pred_ids[:, :last_real_col]
            label_ids = label_ids[:, :last_real_col]

        try:
            preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        except Exception as exc:
            log.warning("CER/WER decode failed: %s", exc)
            return {"cer": -1.0, "wer": -1.0}

        pairs = [(p.strip(), l.strip()) for p, l in zip(preds, labels) if l.strip()]
        if not pairs:
            return {"cer": 1.0, "wer": 1.0}

        pred_texts, label_texts = zip(*pairs)
        try:
            cer = jiwer.cer(list(label_texts), list(pred_texts))
            wer = jiwer.wer(list(label_texts), list(pred_texts))
        except Exception as exc:
            log.warning("jiwer computation failed: %s", exc)
            return {"cer": -1.0, "wer": -1.0}

        return {"cer": round(cer, 4), "wer": round(wer, 4)}

    return compute_metrics


def build_trainer(
    model,
    tokenizer,
    train_dataset: list,
    args: argparse.Namespace,
    eval_dataset: Optional[list] = None,
) -> SFTTrainer:
    """Construct the ``SFTTrainer`` with Chandra-optimised settings.

    When *eval_dataset* is provided and ``early_stopping_patience > 0``,
    step-based evaluation and ``EarlyStoppingCallback`` are configured
    automatically.

    Returns:
        Configured ``SFTTrainer`` ready for ``.train()``.
    """
    use_early_stopping = eval_dataset is not None and args.early_stopping_patience > 0

    training_kwargs: Dict[str, Any] = dict(
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        optim="adamw_8bit",
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        seed=args.seed,
        output_dir=args.output_dir,
        report_to=args.report_to,
        bf16=True,
        fp16=False,
        # REQUIRED for vision fine-tuning with Unsloth:
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=MAX_SEQ_LENGTH,
    )

    if args.num_epochs is not None:
        training_kwargs["num_train_epochs"] = args.num_epochs
    else:
        training_kwargs["max_steps"] = args.max_steps

    # Step-based evaluation + early stopping
    if use_early_stopping:
        training_kwargs["eval_strategy"] = "steps"
        training_kwargs["eval_steps"] = args.eval_steps
        training_kwargs["load_best_model_at_end"] = True
        training_kwargs["metric_for_best_model"] = args.metric_for_best_model
        training_kwargs["greater_is_better"] = False
        log.info(
            "Early stopping: eval every %d steps, patience=%d, metric=%s",
            args.eval_steps, args.early_stopping_patience, args.metric_for_best_model,
        )
    elif eval_dataset is not None:
        training_kwargs["eval_strategy"] = "steps"
        training_kwargs["eval_steps"] = args.eval_steps

    callbacks = []
    if use_early_stopping:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )

    # CER/WER metrics (teacher-forced, informational alongside eval_loss)
    compute_metrics_fn = None
    preprocess_logits_fn = None
    if args.compute_cer_wer and eval_dataset is not None:
        if JIWER_AVAILABLE:
            compute_metrics_fn = _make_compute_metrics(tokenizer, max_samples=64)
            preprocess_logits_fn = _preprocess_logits_for_metrics
            log.info("CER/WER metrics enabled (teacher-forced, via jiwer)")
        else:
            log.warning(
                "CER/WER requested but jiwer is not installed. "
                "Install with: pip install jiwer"
            )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer, resize=args.longest_edge),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(**training_kwargs),
        callbacks=callbacks if callbacks else None,
        compute_metrics=compute_metrics_fn,
        preprocess_logits_for_metrics=preprocess_logits_fn,
    )
    return trainer


# ---------------------------------------------------------------------------
# 7. Main Training Function
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    """End-to-end training pipeline for Chandra fine-tuning.

    1. Load model + processor
    2. Verify configuration
    3. Apply LoRA
    4. Prepare dataset
    5. (Optional) pre-training inference test
    6. Train
    7. Save
    8. (Optional) export merged / GGUF / push to Hub
    """
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.set_float32_matmul_precision("high")

    # -- 1. Load model --
    log.info("=" * 60)
    log.info("PHASE: Loading Chandra model")
    log.info("=" * 60)

    model, tokenizer, processor = load_model(
        model_id=args.model_id,
        load_in_4bit=args.load_in_4bit,
        longest_edge=args.longest_edge,
    )

    # -- 2. Verify config --
    if not args.skip_verification:
        log.info("=" * 60)
        log.info("PHASE: Verifying configuration")
        log.info("=" * 60)
        verify_config(model, processor)

    # -- 3. Apply LoRA --
    log.info("=" * 60)
    log.info("PHASE: Configuring LoRA adapters")
    log.info("=" * 60)
    model = setup_lora(
        model,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        finetune_vision=args.finetune_vision,
    )

    # -- 4. Prepare dataset --
    log.info("=" * 60)
    log.info("PHASE: Preparing dataset")
    log.info("=" * 60)
    train_data, eval_data = prepare_dataset(args)

    # -- 5. Pre-training inference test --
    if not args.skip_pre_eval:
        log.info("=" * 60)
        log.info("PHASE: Pre-training inference test")
        log.info("=" * 60)
        _run_inference_test(model, tokenizer, processor, train_data, tag="BEFORE training")

    # -- 6. Train --
    log.info("=" * 60)
    log.info("PHASE: Training")
    log.info("=" * 60)
    FastVisionModel.for_training(model)

    gpu_stats = torch.cuda.get_device_properties(0)
    start_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    max_mem = round(gpu_stats.total_mem / 1024**3, 3) if hasattr(gpu_stats, "total_mem") else round(gpu_stats.total_memory / 1024**3, 3)
    log.info("GPU: %s  |  Max VRAM: %.1f GB  |  Reserved: %.1f GB", gpu_stats.name, max_mem, start_mem)

    trainer = build_trainer(model, tokenizer, train_data, args, eval_dataset=eval_data)

    # Resume from checkpoint if requested
    resume_ckpt = args.resume_from_checkpoint
    if resume_ckpt == "latest":
        ckpts = sorted(Path(args.output_dir).glob("checkpoint-*"))
        resume_ckpt = str(ckpts[-1]) if ckpts else None
        if resume_ckpt:
            log.info("Resuming from latest checkpoint: %s", resume_ckpt)
        else:
            log.warning("No checkpoints found in %s, starting fresh", args.output_dir)

    trainer_stats = unsloth_train(trainer, resume_from_checkpoint=resume_ckpt)

    used_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    log.info(
        "Training complete.  Duration: %.1fs  |  Peak VRAM: %.1f GB  |  Training delta: %.1f GB",
        trainer_stats.metrics["train_runtime"],
        used_mem,
        used_mem - start_mem,
    )

    # Report early stopping status
    if eval_data and args.early_stopping_patience > 0:
        global_step = trainer_stats.global_step
        planned = args.max_steps if args.num_epochs is None else None
        if planned and global_step < planned:
            log.info("Early stopping triggered at step %d / %d", global_step, planned)
        else:
            log.info("Training completed all steps (early stopping did not trigger)")

    # -- 7. Save --
    log.info("=" * 60)
    log.info("PHASE: Saving model")
    log.info("=" * 60)
    _save_model(model, tokenizer, processor, args)

    # -- 8. Post-training inference --
    log.info("=" * 60)
    log.info("PHASE: Post-training inference test")
    log.info("=" * 60)
    _run_inference_test(model, tokenizer, processor, train_data, tag="AFTER training")

    log.info("All done.")


# ---------------------------------------------------------------------------
# Helper: inference test
# ---------------------------------------------------------------------------

def _run_inference_test(
    model, tokenizer, processor, converted_dataset: list, *, tag: str = ""
) -> Optional[str]:
    """Run a single inference sample to verify the model works.

    Uses the first sample from the converted dataset.
    """
    FastVisionModel.for_inference(model)

    if not converted_dataset:
        log.warning("No dataset samples available for inference test.")
        return None

    sample_msg = converted_dataset[0]["messages"]
    user_content = sample_msg[0]["content"]
    test_image = None
    for part in user_content:
        if part.get("type") == "image":
            test_image = part.get("image")
            break

    if test_image is None:
        log.warning("First sample has no image; skipping inference test.")
        return None

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_content[0].get("text", OCR_INSTRUCTION)},
            ],
        }
    ]

    inputs = processor(
        text=processor.apply_chat_template(messages, add_generation_prompt=True),
        images=[test_image],
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    log.info("Inference test (%s):", tag)
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    outputs = model.generate(
        **inputs,
        streamer=streamer,
        max_new_tokens=256,
        use_cache=True,
        temperature=0.3,
        top_p=0.9,
        min_p=0.05,
    )

    result = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return result.strip()


# ---------------------------------------------------------------------------
# Helper: save model
# ---------------------------------------------------------------------------

def _save_model(model, tokenizer, processor, args: argparse.Namespace) -> None:
    """Save LoRA adapters, processor, and optionally merged/GGUF exports."""
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # LoRA adapters
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    processor.save_pretrained(str(out))
    log.info("LoRA adapter + processor saved to %s", out)

    # Merged 16-bit (for vLLM deployment)
    if args.save_merged_16bit:
        merged_dir = str(out / "merged_16bit")
        log.info("Saving merged 16-bit model to %s ...", merged_dir)
        model.save_pretrained_merged(merged_dir, tokenizer)
        processor.save_pretrained(merged_dir)
        log.info("Merged 16-bit model saved.")

    # GGUF
    if args.save_gguf:
        gguf_dir = str(out / "gguf")
        log.info("Saving GGUF (q8_0) to %s ...", gguf_dir)
        model.save_pretrained_gguf(gguf_dir, tokenizer)
        log.info("GGUF model saved.")

    # Push to Hub
    if args.push_to_hub:
        log.info("Pushing to HuggingFace Hub: %s", args.push_to_hub)
        token = args.hf_token or os.environ.get("HF_TOKEN")
        model.push_to_hub(args.push_to_hub, token=token)
        tokenizer.push_to_hub(args.push_to_hub, token=token)
        processor.push_to_hub(args.push_to_hub, token=token)
        log.info("Push complete.")

    # Save training config for reproducibility
    config_path = out / "training_config.json"
    config_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)
    log.info("Training config saved to %s", config_path)


# ---------------------------------------------------------------------------
# Helper: memory reporting
# ---------------------------------------------------------------------------

def report_memory() -> Dict[str, float]:
    """Return current CUDA memory statistics in GB."""
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
        "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
        "max_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
    }


# ---------------------------------------------------------------------------
# 8. Inference Helper (post-training convenience)
# ---------------------------------------------------------------------------

def run_ocr(
    image_or_path: Union[str, Path, Image.Image],
    model=None,
    tokenizer=None,
    processor=None,
    *,
    instruction: str = OCR_INSTRUCTION,
    max_new_tokens: int = 1024,
    temperature: float = 0.3,
) -> str:
    """Run OCR inference on a single image with the fine-tuned model.

    This is a convenience wrapper for post-training use.

    Args:
        image_or_path: A PIL Image or file path.
        model: The loaded model (uses global if None).
        tokenizer: The tokenizer.
        processor: The Qwen3VLProcessor.
        instruction: The OCR prompt.
        max_new_tokens: Maximum generation length.
        temperature: Sampling temperature (low = deterministic).

    Returns:
        Extracted text string.
    """
    FastVisionModel.for_inference(model)

    if isinstance(image_or_path, (str, Path)):
        image = Image.open(image_or_path).convert("RGB")
    else:
        image = image_or_path

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction},
            ],
        }
    ]

    inputs = processor(
        text=processor.apply_chat_template(messages, add_generation_prompt=True),
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            temperature=temperature,
            top_p=0.9,
        )

    result = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return result.strip()


# ---------------------------------------------------------------------------
# 9. CLI Argument Parsing  (already defined in _build_parser above)
# 10. __main__
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: parse arguments, configure logging, and launch training."""
    args = _build_parser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log.info("train_chandra.py starting")
    log.info("Arguments: %s", vars(args))

    train(args)


if __name__ == "__main__":
    main()
