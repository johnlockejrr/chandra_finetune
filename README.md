# train_chandra.py -- Complete Documentation

Production-ready fine-tuning script for the **Chandra** OCR model (`datalab-to/chandra`) using Unsloth + LoRA.

---

## Quick Start

```bash
# Minimal: fine-tune on a HuggingFace dataset
python train_chandra.py \
  --dataset_name unsloth/LaTeX_OCR \
  --output_dir ./chandra_output

# Local dataset with early stopping
python train_chandra.py \
  --dataset_dir ./my_ocr_dataset \
  --output_dir ./chandra_output \
  --eval_steps 50 \
  --early_stopping_patience 3

# Resume interrupted training
python train_chandra.py \
  --dataset_name unsloth/LaTeX_OCR \
  --output_dir ./chandra_output \
  --resume_from_checkpoint latest
```

---

## Chandra Model Specifications

| Attribute           | Value                               | Config Key                  |
| ------------------- | ----------------------------------- | --------------------------- |
| Architecture        | `Qwen3VLForConditionalGeneration`   | `architectures[0]`          |
| Base model          | `Qwen/Qwen3-VL-8B-Instruct`        | `base_model_name_or_path`   |
| Model type          | `qwen3_vl`                          | `model_type`                |
| Max sequence length | 2048                                | `max_seq_length`            |
| Image token ID      | 151655 `<\|image_pad\|>`            | `image_token_id`            |
| Vision start ID     | 151652 `<\|vision_start\|>`         | `vision_start_token_id`     |
| Vision end ID       | 151653 `<\|vision_end\|>`           | `vision_end_token_id`       |
| Patch size          | 16                                  | `vision_config.patch_size`  |
| Spatial merge       | 2                                   | `vision_config.spatial_merge_size` |
| Text hidden size    | 4096                                | `text_config.hidden_size`   |
| Vocab size          | 151936                              | `text_config.vocab_size`    |
| Dtype               | bfloat16                            | `dtype`                     |
| Image processor     | `Qwen2VLImageProcessorFast`         | `preprocessor_config.json`  |
| Processor class     | `Qwen3VLProcessor`                  | `preprocessor_config.json`  |

All these are verified at runtime by `verify_config()` before training starts.

---

## Script Structure (10 Sections)

| # | Section                      | Function(s)                              |
|---|------------------------------|------------------------------------------|
| 1 | Imports                      | Deferred past `--help` for fast CLI      |
| 2 | Configuration constants      | Token IDs, architecture params, defaults |
| 3 | Model loading                | `load_model()`                           |
| 4 | Configuration verification   | `verify_config()`                        |
| 5 | Dataset preparation          | `prepare_dataset()`, `convert_to_conversation()` |
| 6 | Training setup               | `setup_lora()`, `build_trainer()`        |
| 7 | Main training function       | `train()`                                |
| 8 | Inference helper             | `run_ocr()`                              |
| 9 | CLI argument parsing         | `_build_parser()`                        |
| 10| `__main__` block             | `main()`                                 |

---

## CLI Arguments Reference

### Dataset

| Argument           | Default   | Description                                      |
|--------------------|-----------|--------------------------------------------------|
| `--dataset_dir`    | --        | Path to local HF dataset (mutually exclusive)    |
| `--dataset_name`   | --        | HuggingFace dataset identifier (mutually exclusive) |
| `--dataset_subset` | `None`    | HF dataset subset/config name                    |
| `--dataset_split`  | `train`   | Which split to use                               |
| `--image_column`   | `image`   | Column name for images (simple format)           |
| `--text_column`    | `text`    | Column name for text (simple format)             |
| `--val_ratio`      | `0.1`     | Auto-split ratio for validation set              |

### Model

| Argument          | Default              | Description                          |
|-------------------|----------------------|--------------------------------------|
| `--model_id`      | `datalab-to/chandra` | Model ID or local path               |
| `--load_in_4bit`  | `True`               | 4-bit quantisation (saves ~8GB VRAM) |
| `--no_4bit`       | --                   | Use 16-bit instead                   |
| `--longest_edge`  | `2048`               | Target image resolution (px)         |

### LoRA

| Argument              | Default | Description                        |
|-----------------------|---------|------------------------------------|
| `--lora_r`            | `16`    | LoRA rank                          |
| `--lora_alpha`        | `16`    | LoRA alpha (recommend alpha == r)  |
| `--lora_dropout`      | `0.05`  | LoRA dropout                       |
| `--finetune_vision`   | `True`  | Also fine-tune vision encoder      |
| `--no_finetune_vision`| --      | Freeze vision encoder              |

### Training

| Argument                       | Default  | Description                           |
|--------------------------------|----------|---------------------------------------|
| `--output_dir`                 | required | Checkpoint and model output directory |
| `--num_epochs`                 | `None`   | Epoch count (overrides `--max_steps`) |
| `--max_steps`                  | `500`    | Max training steps                    |
| `--per_device_train_batch_size`| `1`      | Batch size per GPU                    |
| `--gradient_accumulation_steps`| `8`      | Effective batch = batch_size * accum  |
| `--learning_rate`              | `2e-4`   | Peak learning rate                    |
| `--warmup_steps`               | `50`     | LR warmup steps                       |
| `--weight_decay`               | `0.01`   | AdamW weight decay                    |
| `--lr_scheduler_type`          | `cosine` | `cosine`, `linear`, or `constant`     |
| `--logging_steps`              | `10`     | Log train loss every N steps          |
| `--save_steps`                 | `200`    | Save checkpoint every N steps         |
| `--save_total_limit`           | `3`      | Keep only N most recent checkpoints   |
| `--seed`                       | `3407`   | Random seed                           |
| `--report_to`                  | `none`   | `none`, `wandb`, or `tensorboard`     |
| `--ocr_instruction`            | (long)   | System prompt sent with each image    |

### Early Stopping

| Argument                       | Default      | Description                                       |
|--------------------------------|--------------|---------------------------------------------------|
| `--eval_steps`                 | `50`         | Evaluate every N training steps                   |
| `--early_stopping_patience`    | `3`          | Stop after N evals without improvement (0=disable)|
| `--early_stopping_threshold`   | `0.0`        | Minimum delta to count as improvement             |
| `--metric_for_best_model`      | `eval_loss`  | Metric to monitor                                 |

**How it works:**
1. When `early_stopping_patience > 0`, the dataset is auto-split into train/eval using `val_ratio` (default 10%).
2. If your dataset already has a `validation`, `val`, or `test` split, that is used instead.
3. Every `eval_steps` training steps, the model is evaluated on the eval split.
4. If `eval_loss` does not improve for `patience` consecutive evaluations, training stops.
5. The best checkpoint (lowest eval_loss) is automatically loaded at the end.

**Example:** With `--eval_steps 50 --early_stopping_patience 3`, training stops if eval_loss doesn't improve for 150 steps (3 evals x 50 steps).

### Checkpoint Resume

| Argument                    | Default | Description                                           |
|-----------------------------|---------|-------------------------------------------------------|
| `--resume_from_checkpoint`  | `None`  | Path to checkpoint dir, or `latest` for auto-detect   |

**Usage:**
```bash
# Resume from a specific checkpoint
python train_chandra.py ... --resume_from_checkpoint ./chandra_output/checkpoint-200

# Auto-detect the most recent checkpoint
python train_chandra.py ... --resume_from_checkpoint latest
```

### Export

| Argument              | Default | Description                                 |
|-----------------------|---------|---------------------------------------------|
| `--save_merged_16bit` | off     | Also save merged 16-bit model for vLLM      |
| `--save_gguf`         | off     | Also save GGUF (q8_0) for llama.cpp/Ollama  |
| `--push_to_hub`       | `None`  | HF Hub repo name (e.g. `user/chandra-lora`) |
| `--hf_token`          | `None`  | HF token (or set `HF_TOKEN` env var)        |

### OCR Metrics (CER/WER)

| Argument            | Default | Description                                       |
|---------------------|---------|---------------------------------------------------|
| `--compute_cer_wer` | `True`  | Compute CER/WER during evaluation (requires jiwer)|
| `--no_cer_wer`      | --      | Disable CER/WER computation                       |

**How it works:**
- CER (Character Error Rate) and WER (Word Error Rate) are computed via `jiwer` alongside `eval_loss`.
- These are *teacher-forced* metrics: the model sees correct previous tokens at each step. This makes them optimistic compared to actual generation, but they still track training progress reliably.
- Metrics appear as `eval_cer` and `eval_wer` in the training logs next to `eval_loss`.
- `eval_loss` remains the default early stopping metric (more stable). You can switch with `--metric_for_best_model eval_cer`.
- Requires `pip install jiwer`. Gracefully degrades if not installed.

### Misc

| Argument              | Default | Description                          |
|-----------------------|---------|--------------------------------------|
| `--skip_verification` | off     | Skip config checks on startup        |
| `--skip_pre_eval`     | off     | Skip inference test before training  |
| `-v` / `--verbose`    | off     | Debug-level logging                  |

---

## Dataset Formats

The script auto-detects two formats:

### Simple format
Each sample has an `image` (PIL) and `text` (str) column:
```python
{"image": <PIL.Image>, "text": "ground truth OCR text"}
```

### FineVision format
Each sample has `images` (list of PIL) and `texts` (list of dicts):
```python
{"images": [<PIL.Image>], "texts": [{"user": "", "assistant": "ground truth text"}]}
```

Custom column names are supported via `--image_column` and `--text_column`.

---

## Conversation Format (Qwen3-VL)

Each sample is converted to this structure for the `SFTTrainer`:

```python
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "<OCR instruction>"},
        {"type": "image", "image": <PIL.Image>}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "<ground truth text>"}
      ]
    }
  ]
}
```

---

## Training Pipeline Phases

The `train()` function executes these phases in order:

1. **Load model** -- `FastVisionModel.from_pretrained()` with `trust_remote_code=True`, explicit `Qwen3VLProcessor`, image resolution override, video token disabled
2. **Verify config** -- Checks all 9 critical parameters (token IDs, architecture, sequence length)
3. **Apply LoRA** -- `FastVisionModel.get_peft_model()` targeting vision + language + attention + MLP layers
4. **Prepare dataset** -- Load, auto-detect format, auto-split for eval, convert to conversation format
5. **Pre-training inference** -- Generate on first sample to verify the forward pass works
6. **Train** -- `SFTTrainer` with `UnslothVisionDataCollator`, optional early stopping + resume
7. **Save** -- LoRA adapters + tokenizer + processor + training config; optional merged/GGUF/Hub export
8. **Post-training inference** -- Generate on first sample to compare with pre-training output

---

## Post-Training Inference

Use the `run_ocr()` helper function:

```python
from train_chandra import run_ocr, load_model

model, tokenizer, processor = load_model("./chandra_output")

result = run_ocr(
    "document.png",
    model=model,
    tokenizer=tokenizer,
    processor=processor,
)
print(result)
```

---

## Key Implementation Details

| Requirement                    | Implementation                                         |
|--------------------------------|--------------------------------------------------------|
| Model loading                  | `FastVisionModel.from_pretrained("datalab-to/chandra", trust_remote_code=True)` |
| Processor                      | Explicit `Qwen3VLProcessor.from_pretrained()` (not AutoProcessor) |
| Image resolution               | `processor.image_processor.size = {"longest_edge": 2048, "shortest_edge": 28}` |
| Data collator                  | `UnslothVisionDataCollator(model, tokenizer)`          |
| SFTTrainer flags               | `remove_unused_columns=False`, `dataset_kwargs={"skip_prepare_dataset": True}` |
| Max length                     | `max_length=2048` (Chandra's `max_seq_length`)         |
| Optimizer                      | `adamw_8bit`                                           |
| Learning rate                  | `2e-4`                                                 |
| LoRA                           | `r=16, lora_alpha=16`, vision+language+attention+MLP   |
| Memory                         | 4-bit quantisation + Unsloth gradient checkpointing    |
| Early stopping                 | `EarlyStoppingCallback` with step-based eval           |
| Checkpoint resume              | `trainer.train(resume_from_checkpoint=...)`             |

---

## Verification Commands

```bash
# 1. Syntax check
python -m py_compile train_chandra.py

# 2. Import check (no execution)
python -c "from train_chandra import load_model, verify_config, train; print('OK')"

# 3. CLI help (fast, no heavy imports)
python train_chandra.py --help

# 4. Quick dry-run (1 step, no pre-eval)
python train_chandra.py \
  --dataset_name unsloth/LaTeX_OCR \
  --output_dir ./test_chandra \
  --max_steps 1 \
  --early_stopping_patience 0 \
  --skip_pre_eval \
  --verbose
```

---

## Example Training Runs

### Minimal (no early stopping)
```bash
python train_chandra.py \
  --dataset_name unsloth/LaTeX_OCR \
  --output_dir ./chandra_lora \
  --max_steps 500 \
  --early_stopping_patience 0
```

### Full run with early stopping
```bash
python train_chandra.py \
  --dataset_dir ./my_hf_dataset \
  --output_dir ./chandra_lora \
  --num_epochs 3 \
  --eval_steps 100 \
  --early_stopping_patience 5 \
  --save_steps 100 \
  --learning_rate 1e-4 \
  --longest_edge 2048 \
  --verbose
```

### Production with export
```bash
python train_chandra.py \
  --dataset_dir ./my_hf_dataset \
  --output_dir ./chandra_production \
  --num_epochs 5 \
  --eval_steps 200 \
  --early_stopping_patience 3 \
  --save_merged_16bit \
  --push_to_hub myuser/chandra-ocr-finetuned \
  --hf_token $HF_TOKEN
```

### Resume from crash
```bash
python train_chandra.py \
  --dataset_dir ./my_hf_dataset \
  --output_dir ./chandra_production \
  --resume_from_checkpoint latest
```

---

## End-to-End Workflow: XML to Fine-Tuned Model

### Step 1: Convert PAGE-XML / ALTO-XML to HF dataset

```bash
python convert-xml-chandra-hf.py \
  --input_dir /path/to/xml_and_images \
  --output_dir ./chandra_dataset \
  --include_full_pages \
  --include_paragraphs \
  --max_image_edge 2048 \
  --val_ratio 0.1 \
  --output_format simple
```

This produces `./chandra_dataset/hf_dataset_chandra/` with train + validation splits.

### Step 2: Fine-tune Chandra

```bash
python train_chandra.py \
  --dataset_dir ./chandra_dataset/hf_dataset_chandra \
  --output_dir ./chandra_finetuned \
  --num_epochs 3 \
  --eval_steps 100 \
  --early_stopping_patience 5 \
  --verbose
```

CER/WER metrics appear in logs alongside eval_loss.

### Step 3: Use the fine-tuned model

```python
from train_chandra import run_ocr, load_model
model, tokenizer, processor = load_model("./chandra_finetuned")
print(run_ocr("document.png", model=model, tokenizer=tokenizer, processor=processor))
```

---

## convert-xml-chandra-hf.py Reference

Converts PAGE-XML and ALTO-XML ground-truth annotations to a HuggingFace dataset.

| Argument              | Default   | Description                                              |
|-----------------------|-----------|----------------------------------------------------------|
| `--input_dir`         | required  | Directory with XML + image files                         |
| `--output_dir`        | required  | Output directory for crops, metadata, HF dataset         |
| `--max_image_edge`    | `2048`    | Resize longest edge (matches Chandra's optimal res)      |
| `--output_format`     | `simple`  | `simple` (image+text) or `finevision` (images+texts)     |
| `--val_ratio`         | `0.0`     | Validation split ratio (e.g. 0.1 for 90/10 split)       |
| `--include_full_pages`| off       | Also create full-page samples                            |
| `--include_paragraphs`| off       | Also create paragraph-level samples                      |
| `--aug_copies`        | `0`       | Augmented copies per crop (rotation, noise, blur, etc.)  |
| `--batch_size`        | `50`      | XML files per processing batch                           |
| `--seed`              | `3407`    | Random seed for splitting                                |

**Memory efficiency:**
- Processes one XML file at a time, explicit image cleanup + gc.collect() between batches
- HF dataset built via streaming generator (one image at a time, never holds all in RAM)
- Crops saved to disk immediately, PIL images closed after use

**Output:** `hf_dataset_chandra/` directory with train (+ optional validation) splits.

---

## Training Tips for Chandra OCR

1. **Learning rate**: Start with `2e-4` for LoRA. Reduce to `5e-5` if overfitting (eval_loss rising).
2. **Vision layers**: Keep `--finetune_vision` for domain adaptation (handwriting, specific layouts). Use `--no_finetune_vision` if your data is already in-distribution.
3. **Batch size**: With RTX 3090 (24GB), `batch_size=2` + `grad_accum=8` gives effective batch of 16.
4. **Image resolution**: 2048px is the sweet spot for 24GB VRAM. Use 4096px with 24GB+ GPUs for better accuracy.
5. **Temperature**: 0.3 during inference for deterministic OCR output.
6. **Early stopping**: Recommended to prevent overfitting. Patience=3 with eval_steps=50 is a good starting point.
7. **CER/WER monitoring**: Teacher-forced CER/WER are logged during eval. For actual generation quality, test post-training with `run_ocr()`.
8. **Sequence length**: Chandra's max is 2048 tokens. Documents with very long text may be truncated.
