# Qwen3-VL-8B × SPIQA Fine-Tuning

QLoRA fine-tuning of **Qwen3-VL-8B-Instruct** on the
[SPIQA](https://huggingface.co/datasets/google/spiqa) scientific figure QA dataset
(NeurIPS 2024 D&B track).

---

## File overview

```
download_spiqa.py   – download SPIQA from HuggingFace (~30 GB)
train.py            – QLoRA fine-tuning with PEFT + TRL SFTTrainer
evaluate.py         – inference + ROUGE-L / METEOR / BERTScore evaluation
requirements.txt    – Python dependencies
```

---

## Quick-start

### 1. Install dependencies

```bash
# Qwen3-VL requires transformers built from source (v4.57 not yet on PyPI)
pip install git+https://github.com/huggingface/transformers

pip install -r requirements.txt

# Recommended for A100/H100 — skip on T4
pip install flash-attn --no-build-isolation
```

### 2. Download SPIQA

```bash
python download_spiqa.py
# Downloads to ./spiqa_data/  (~30 GB total including images)
```

### 3. Run zero-shot baseline

```bash
# Instruct mode — direct baseline
python evaluate.py \
    --model_id Qwen/Qwen3-VL-8B-Instruct \
    --output_dir ./eval_results

# Thinking mode — free ablation (no fine-tuning)
python evaluate.py \
    --model_id Qwen/Qwen3-VL-8B-Instruct \
    --thinking \
    --output_dir ./eval_results

# Quick check on 200 examples first
python evaluate.py \
    --model_id Qwen/Qwen3-VL-8B-Instruct \
    --max_samples 200 \
    --output_dir ./eval_results
```

### 4. Fine-tun

**Single GPU (A100 40 GB)**
```bash
python train.py \
    --model_id Qwen/Qwen3-VL-8B-Instruct \
    --spiqa_train_json ./spiqa_data/train/SPIQA_train.json \
    --spiqa_image_dir  ./spiqa_data/train/images \
    --output_dir       ./outputs/qwen3vl-spiqa-lora \
    --max_samples      15000 \
    --num_train_epochs 2 \
    --lora_r           32 \
    --lora_alpha       64 \
    --use_qlora
```

### 5. Evaluate fine-tuned mode

```bash
python evaluate.py \
    --model_id  Qwen/Qwen3-VL-8B-Instruct \
    --lora_path ./outputs/qwen3vl-spiqa-lora/final_lora_adapter \
    --output_dir ./eval_results
```

---

## GPU memory reference

| Config | VRAM |
|--------|------|
| QLoRA 4-bit, vision frozen, batch=1 | ~18 GB |
| QLoRA 4-bit, vision trainable, batch=1 | ~24 GB |
| BF16 LoRA, vision frozen, batch=1 | ~32 GB |

---

## Notes on Qwen3-VL specifics

- **Image tokens**: Qwen3-VL uses `token × 32 × 32` patches (vs 28×28 in Qwen2.5-VL).
  `max_pixels=1280*28*28` in the processor config is kept compatible.
- **Transformers version**: Must install from GitHub source until v4.57 is released on PyPI.
- **FP8 checkpoints**: Not yet loadable via transformers — use BF16 or QLoRA (NF4) for training.
- **ZeRO strategy**: For the dense 8B model, ZeRO-2 or ZeRO-3 both work. ZeRO-3 is
  slower but saves more memory on multi-GPU setups.
