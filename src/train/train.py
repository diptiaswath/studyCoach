"""
Fine-tuning Qwen3-VL-8B-Instruct on SPIQA (Test-A / Train split)
with LoRA using HuggingFace PEFT + Transformers Trainer.

Logging: Weights & Biases (wandb) — logs training loss, eval loss,
         learning rate, gradient norm, GPU memory, and periodic
         qualitative sample predictions.

Hardware target : 1x A100 40GB (or 2x T4 with gradient checkpointing)
Expected runtime: ~4-6 h for 10 k samples / 1 epoch on A100

Usage
-----
# Single GPU
python train.py --wandb_project spiqa-qwen3vl

# Multi-GPU (e.g. 2x T4)
torchrun --nproc_per_node 2 train.py --wandb_project spiqa-qwen3vl

# Disable W&B (offline / CI)
python train.py --wandb_mode disabled
"""

import os
import json
import math
import random
import argparse
import logging
from pathlib import Path

import torch
from datasets import Dataset
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration as AutoModelForVision2Seq,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
DEFAULT_CONFIG = dict(
    # Model
    model_id           = "Qwen/Qwen3-VL-8B-Instruct",
    # SPIQA paths (produced by download_spiqa.py)
    spiqa_train_json      = "./spiqa/train_val/SPIQA_train.json",
    spiqa_image_dir       = "./spiqa/train_val/SPIQA_train_val_Images",
    # Dedicated val split — paper-disjoint from train
    spiqa_val_json        = "./spiqa/train_val/SPIQA_val.json",
    spiqa_val_image_dir   = "./spiqa/train_val/SPIQA_train_val_Images",
    # Training
    output_dir         = "./outputs/qwen3vl-8b-spiqa-lora",
    num_train_epochs   = 2,
    per_device_batch   = 1,
    grad_accum_steps   = 8,
    learning_rate      = 2e-4,
    max_seq_length     = 2048,
    warmup_ratio       = 0.05,
    lr_scheduler       = "cosine",
    save_steps         = 200,
    logging_steps      = 10,
    eval_steps         = 200,
    val_split_ratio    = 0.02,
    max_samples        = 15_000,
    seed               = 42,
    # Image resolution
    min_pixels         = 256 * 28 * 28,
    max_pixels         = 1280 * 28 * 28,
    # LoRA
    lora_r             = 32,
    lora_alpha         = 64,
    lora_dropout       = 0.05,
    # QLoRA
    use_qlora          = False,   # A40 46GB — bf16 fits, QLoRA not needed
    freeze_vision      = True,
    # ── W&B ────────────────────────────────────
    wandb_project      = "spiqa-qwen3vl",
    wandb_run_name     = None,       # auto-generated if None
    wandb_mode         = "online",   # "online" | "offline" | "disabled"
    # How many qualitative sample predictions to log per eval
    wandb_sample_preds = 8,
)


# ─────────────────────────────────────────────
# 1. W&B initialisation
# ─────────────────────────────────────────────

def init_wandb(cfg: dict, trainable_params: int, all_params: int):
    """
    Initialise a W&B run and log the full experiment config +
    a model-architecture summary as run metadata.
    Only runs on rank 0 in DDP setups.
    """
    if int(os.environ.get("LOCAL_RANK", 0)) != 0:
        return  # Only log from rank-0 process

    import wandb

    os.environ["WANDB_MODE"] = cfg["wandb_mode"]

    run_name = cfg["wandb_run_name"] or (
        f"r{cfg['lora_r']}-lr{cfg['learning_rate']:.0e}"
        f"-{'qlora' if cfg['use_qlora'] else 'lora'}"
        f"-{'visfrozen' if cfg['freeze_vision'] else 'vistuned'}"
    )

    wandb.init(
        project = cfg["wandb_project"],
        name    = run_name,
        config  = {
            # ── Experiment ─────────────────────
            **{k: v for k, v in cfg.items() if not k.startswith("wandb_")},
            # ── Model summary ──────────────────
            "trainable_params":     trainable_params,
            "total_params":         all_params,
            "trainable_pct":        100 * trainable_params / all_params,
            # ── Derived ────────────────────────
            "effective_batch_size": cfg["per_device_batch"] * cfg["grad_accum_steps"],
        },
        tags = [
            "qwen3-vl",
            "spiqa",
            "qlora" if cfg["use_qlora"] else "lora",
            f"r{cfg['lora_r']}",
        ],
    )

    log.info(f"W&B run initialised: {wandb.run.url}")


# ─────────────────────────────────────────────
# 2. Custom W&B callback
# ─────────────────────────────────────────────

class WandbSPIQACallback(TrainerCallback):
    """
    Extends HuggingFace's built-in W&B logging with:

    • GPU memory usage (allocated / reserved) at every logging step
    • Gradient norm (retrieved from trainer state)
    • Learning rate (already in trainer logs, re-surfaced explicitly)
    • Tokens-per-second throughput estimate
    • Qualitative sample predictions table at each eval checkpoint
    """

    def __init__(
        self,
        processor,
        val_examples: list[dict],
        n_samples: int = 8,
        max_new_tokens: int = 128,
    ):
        self.processor       = processor
        # Keep a fixed random subset of val examples for qualitative logging
        rng = random.Random(0)
        self.qual_examples   = rng.sample(val_examples, min(n_samples, len(val_examples)))
        self.max_new_tokens  = max_new_tokens
        self._step_start_time: float | None = None

    # ── Training step hooks ───────────────────

    def on_step_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        import time
        self._step_start_time = time.perf_counter()

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Called every `logging_steps`. Adds GPU + throughput metrics."""
        if int(os.environ.get("LOCAL_RANK", 0)) != 0:
            return
        if logs is None:
            return

        import time
        import wandb

        extra = {}

        # ── GPU memory ───────────────────────
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                extra[f"gpu/{i}/mem_allocated_gb"] = (
                    torch.cuda.memory_allocated(i) / 1e9
                )
                extra[f"gpu/{i}/mem_reserved_gb"] = (
                    torch.cuda.memory_reserved(i) / 1e9
                )
                extra[f"gpu/{i}/mem_utilisation_pct"] = (
                    100 * torch.cuda.memory_allocated(i)
                    / max(torch.cuda.memory_reserved(i), 1)
                )

        # ── Throughput (samples / sec) ────────
        if self._step_start_time is not None:
            elapsed = time.perf_counter() - self._step_start_time
            samples_per_step = args.per_device_train_batch_size * args.gradient_accumulation_steps
            extra["throughput/samples_per_sec"] = samples_per_step / max(elapsed, 1e-6)

        # ── Perplexity from training loss ──────
        if "loss" in logs:
            try:
                extra["train/perplexity"] = math.exp(logs["loss"])
            except OverflowError:
                extra["train/perplexity"] = float("inf")

        # ── Rename HF keys → cleaner W&B names ─
        key_map = {
            "loss":               "train/loss",
            "learning_rate":      "train/learning_rate",
            "grad_norm":          "train/grad_norm",
            "eval_loss":          "eval/loss",
            "eval_runtime":       "eval/runtime_sec",
            "eval_samples_per_second": "eval/samples_per_sec",
        }
        renamed = {key_map.get(k, k): v for k, v in logs.items()}

        # Eval perplexity
        if "eval/loss" in renamed:
            try:
                renamed["eval/perplexity"] = math.exp(renamed["eval/loss"])
            except OverflowError:
                renamed["eval/perplexity"] = float("inf")

        wandb.log({**renamed, **extra})

    # ── Eval hook — qualitative predictions ───

    def on_evaluate(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        """Log a W&B Table of sample predictions after each eval checkpoint."""
        if int(os.environ.get("LOCAL_RANK", 0)) != 0:
            return
        if model is None:
            return

        import wandb

        log.info(
            f"[W&B] Generating {len(self.qual_examples)} qualitative "
            f"sample predictions for step {state.global_step} ..."
        )

        table = wandb.Table(
            columns=["step", "paper_id", "question", "student", "ground_truth", "prediction", "figure"]
        )

        model.eval()
        with torch.inference_mode():
            for ex in self.qual_examples:
                caption_text = (
                    f"\nfigure_caption: {ex['caption']}" if ex.get("caption") else ""
                )
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": ex["image_path"]},
                            {
                                "type": "text",
                                "text": f"{caption_text}\n\nquestion: {ex['question']}\n\nstudent: {ex['student']}",
                            },
                        ],
                    },
                ]
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                try:
                    img = Image.open(ex["image_path"]).convert("RGB")
                except Exception:
                    img = Image.new("RGB", (224, 224))

                inputs = self.processor(
                    text=text,
                    images=[img],
                    return_tensors="pt",
                ).to(model.device)

                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
                new_tokens = gen_ids[:, inputs["input_ids"].shape[1]:]
                prediction = self.processor.batch_decode(
                    new_tokens, skip_special_tokens=True
                )[0].strip()

                ground_truth = (
                    f"verdict = {ex['verdict']}\n"
                    f"error_category = {ex['error_category']}\n"
                    f"feedback = {ex['feedback']}"
                )
                table.add_data(
                    state.global_step,
                    ex.get("paper_id", ""),
                    ex["question"],
                    ex.get("student", ""),
                    ground_truth,
                    prediction,
                    wandb.Image(img),
                )

        wandb.log({"eval/qualitative_samples": table})
        model.train()

    # ── End-of-training summary ───────────────

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if int(os.environ.get("LOCAL_RANK", 0)) != 0:
            return

        import wandb

        # Log best checkpoint metadata as summary
        wandb.run.summary["best_eval_loss"]  = state.best_metric
        wandb.run.summary["best_checkpoint"] = state.best_model_checkpoint
        wandb.run.summary["total_steps"]     = state.global_step
        wandb.run.summary["total_epochs"]    = state.epoch

        log.info(
            f"[W&B] Training complete. Best eval loss: {state.best_metric:.4f} "
            f"at checkpoint {state.best_model_checkpoint}"
        )
        wandb.finish()


# ─────────────────────────────────────────────
# 3. SPIQA data loader
# ─────────────────────────────────────────────

def load_spiqa_train(
    json_path: str,
    image_dir: str,
    max_samples: int | None = None,
    seed: int = 42,
) -> list[dict]:
    """
    Parse SPIQA train JSON into a flat list of QA examples.

    Actual SPIQA schema:
      paper_id → all_figures → figure_filename → {caption, content_type, figure_type}
      paper_id → qa → [{question, answer, reference, ...}]

    Each QA item's "reference" field is the figure filename; we join it to
    all_figures to get the image path and caption.
    """
    log.info(f"Loading SPIQA train data from {json_path}")
    with open(json_path) as f:
        raw = json.load(f)

    examples  = []
    image_dir = Path(image_dir)
    skipped   = 0

    for paper_id, paper_data in raw.items():
        all_figures = paper_data.get("all_figures", {})
        for qa in paper_data.get("qa", []):
            ref = qa.get("reference", "")
            if not ref:
                skipped += 1
                continue
            img_path = image_dir / paper_id / ref
            if not img_path.exists():
                skipped += 1
                continue
            fig_meta = all_figures.get(ref, {})
            examples.append({
                "paper_id":       paper_id,
                "fig_id":         ref,
                "image_path":     str(img_path),
                "caption":        fig_meta.get("caption", "").strip(),
                "fig_type":       fig_meta.get("figure_type", "unknown"),
                "question":       qa["question"],
                "answer":         qa["answer"],
                "student":        qa.get("student", ""),
                "verdict":        qa.get("verdict", ""),
                "error_category": qa.get("error_category", ""),
                "feedback":       qa.get("feedback", ""),
            })

    if skipped:
        log.debug(f"Skipped {skipped} QA items (missing reference or image)")
    log.info(f"Loaded {len(examples)} QA pairs from {len(raw)} papers")

    random.seed(seed)
    random.shuffle(examples)
    if max_samples is not None:
        examples = examples[:max_samples]
        log.info(f"Subsampled to {len(examples)} examples")

    return examples


# ─────────────────────────────────────────────
# 4. Chat-format conversion
# ─────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a study coach assessing a student's answer to a question about an image from a scientific paper.\n"
    "You have provide a verdict, an error_category and a feedback after assessing the student's answer.\n\n"
    "Possible verdicts:\n"
    " - correct: the answer is accurate and complete\n"
    " - partially correct : the answer gets some aspects right but misses or misidentifies others\n"
    " - incorrect : the answer is completely wrong or contradicts the figure\n\n"
    "An incorrect or partially correct student answer may have one of these error types:\n"
    " - omission: Missing one or more key points in the answer\n"
    " - factual: Giving factual data that contradicts the figure, or misreading it (e.g. axes, legends, trends)\n"
    " - conceptual: Misunderstanding a concept or drawing a wrong conclusion from the data\n\n"
    "Assess the student's response and format your output exactly as:\n"
    "verdict = <incorrect | partially correct | correct>\n"
    "error_category = <omission | factual | conceptual | N/A>\n"
    "feedback = <concise study coach explanation, second-person tone>\n\n"
    "If verdict is correct, set error_category = N/A and feedback = N/A. Do not use bullet points or bold text."
)


def make_conversation(example: dict) -> dict:
    return {
        "paper_id":       example.get("paper_id", ""),
        "image_path":     example["image_path"],
        "caption":        example.get("caption", ""),
        "question":       example["question"],
        "student":        example["student"],
        "verdict":        example["verdict"],
        "error_category": example["error_category"],
        "feedback":       example["feedback"],
    }


# ─────────────────────────────────────────────
# 5. Collator
# ─────────────────────────────────────────────

class SPIQACollator:
    """
    Batch collator:
    1. Applies the Qwen3-VL chat template
    2. Loads images from disk
    3. Runs the processor → model inputs
    4. Masks prompt tokens in labels (train only on assistant answer)
    """

    def __init__(self, processor, max_length: int = 2048):
        self.processor  = processor
        self.max_length = max_length

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        texts, all_images, valid = [], [], []

        for ex in examples:
            caption_text = (
                f"\nfigure_caption: {ex['caption']}" if ex.get("caption") else ""
            )
            conv = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": ex["image_path"]},
                        {"type": "text",  "text": f"{caption_text}\n\nquestion: {ex['question']}\n\nstudent: {ex['student']}"},
                    ],
                },
                {"role": "assistant", "content": f"verdict = {ex['verdict']}\nerror_category = {ex['error_category']}\nfeedback = {ex['feedback']}"},
            ]
            text = self.processor.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)

            try:
                img = Image.open(ex["image_path"]).convert("RGB")
            except Exception as e:
                log.warning(f"Could not open image {ex['image_path']}: {e}")
                img = Image.new("RGB", (224, 224))
            all_images.append([img])
            valid.append(ex)

        # Skip examples whose token length exceeds max_length rather than
        # truncating — truncation desyncs <image> placeholder counts and
        # causes a hard crash in the Qwen3-VL processor.
        safe_texts, safe_images = [], []
        for i, (text, imgs) in enumerate(zip(texts, all_images)):
            tok_len = len(self.processor.tokenizer.encode(text))
            if tok_len > self.max_length:
                log.warning(
                    f"Skipping example {valid[i].get('paper_id', '?')} — "
                    f"sequence length {tok_len} > max_length {self.max_length}. "
                    f"Increase max_seq_length to include it."
                )
                continue
            safe_texts.append(text)
            safe_images.append(imgs)

        if not safe_texts:
            safe_texts  = [texts[0]]
            safe_images = [all_images[0]]

        inputs = self.processor(
            text=safe_texts,
            images=safe_images,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        # Mask prompt tokens — only supervise on the assistant answer
        labels = inputs["input_ids"].clone()
        assistant_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        for i, label_row in enumerate(labels):
            im_start_positions = (label_row == assistant_token_id).nonzero(as_tuple=True)[0]
            if len(im_start_positions) >= 1:
                last_pos = im_start_positions[-1].item()
                labels[i, : last_pos + 2] = -100
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        inputs["labels"] = labels
        return inputs


# ─────────────────────────────────────────────
# 6. Model + LoRA setup
# ─────────────────────────────────────────────

def build_model_and_processor(cfg: dict):
    log.info(f"Loading processor from {cfg['model_id']}")
    processor = AutoProcessor.from_pretrained(
        cfg["model_id"],
        min_pixels=cfg["min_pixels"],
        max_pixels=cfg["max_pixels"],
    )
    processor.tokenizer.padding_side = "right"

    bnb_config = None
    if cfg["use_qlora"]:
        log.info("Using QLoRA (4-bit NF4 quantisation)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    log.info(f"Loading model {cfg['model_id']}")
    model = AutoModelForVision2Seq.from_pretrained(
        cfg["model_id"],
        quantization_config=bnb_config,
        dtype=torch.bfloat16 if not cfg["use_qlora"] else None,
        device_map="auto",
        trust_remote_code=True,
    )

    # Always freeze the full vision encoder first
    log.info("Freezing vision encoder weights")
    for name, param in model.named_parameters():
        if "visual" in name:
            param.requires_grad = False

    # Optionally unfreeze just the merger (cross-modal projector)
    # so it can co-adapt with the LoRA adapters during training.
    if not cfg["freeze_vision"]:
        log.info("Unfreezing merger (visual projector) weights")
        for name, param in model.named_parameters():
            if "merger" in name:
                param.requires_grad = True
                log.info(f"  Unfreezing: {name}")

    lora_cfg = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=[
            "q_proj", "v_proj", "o_proj", "gate_proj",
        ],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    if cfg["use_qlora"]:
        model.enable_input_require_grads()

    # Return param counts for W&B config
    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total      = sum(p.numel() for p in model.parameters())
    return model, processor, trainable, total


# ─────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Qwen3-VL-8B on SPIQA")
    for k, v in DEFAULT_CONFIG.items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", default=v, action="store_true")
            p.add_argument(f"--no_{k}", dest=k, action="store_false")
        else:
            p.add_argument(f"--{k}", default=v, type=type(v) if v is not None else str)
    return vars(p.parse_args())


def main():
    cfg = parse_args()
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    # ── Load data ──────────────────────────────
    raw_examples = load_spiqa_train(
        json_path   = cfg["spiqa_train_json"],
        image_dir   = cfg["spiqa_image_dir"],
        max_samples = cfg["max_samples"],
        seed        = cfg["seed"],
    )
    train_convs = [make_conversation(ex) for ex in raw_examples]

    # Load val from dedicated SPIQA_val.json (paper-disjoint from train)
    n_val = max(1, int(len(train_convs) * cfg["val_split_ratio"]))
    val_examples = load_spiqa_train(
        json_path   = cfg["spiqa_val_json"],
        image_dir   = cfg["spiqa_val_image_dir"],
        max_samples = n_val,
        seed        = cfg["seed"],
    )
    val_convs = [make_conversation(ex) for ex in val_examples]

    # Fail fast if val split doesn't have enough examples
    if len(val_convs) < n_val:
        log.warning(
            f"Requested {n_val} val examples but SPIQA_val.json only has "
            f"{len(val_convs)} after filtering. Consider reducing val_split_ratio."
        )

    train_ds = Dataset.from_list(train_convs)
    val_ds   = Dataset.from_list(val_convs)
    log.info(f"Train: {len(train_ds)}  |  Val: {len(val_ds)} (from SPIQA_val.json)")

    # ── Model ──────────────────────────────────
    model, processor, trainable_params, all_params = build_model_and_processor(cfg)

    # ── W&B init ───────────────────────────────
    if cfg["wandb_mode"] != "disabled":
        init_wandb(cfg, trainable_params, all_params)

    # ── Collator ───────────────────────────────
    collator = SPIQACollator(processor, max_length=cfg["max_seq_length"])

    # ── TrainingArguments ──────────────────────
    training_args = TrainingArguments(
        output_dir                   = cfg["output_dir"],
        num_train_epochs             = cfg["num_train_epochs"],
        per_device_train_batch_size  = cfg["per_device_batch"],
        per_device_eval_batch_size   = 1,
        gradient_accumulation_steps  = cfg["grad_accum_steps"],
        learning_rate                = cfg["learning_rate"],
        lr_scheduler_type            = cfg["lr_scheduler"],
        warmup_ratio                 = cfg["warmup_ratio"],
        bf16                         = True,
        gradient_checkpointing       = True,
        gradient_checkpointing_kwargs= {"use_reentrant": False},
        logging_steps                = cfg["logging_steps"],
        save_steps                   = cfg["save_steps"],
        eval_steps                   = cfg["eval_steps"],
        eval_strategy          = "steps",
        save_total_limit             = 3,
        load_best_model_at_end       = True,
        metric_for_best_model        = "eval_loss",
        greater_is_better            = False,
        remove_unused_columns        = False,
        dataloader_num_workers       = 4,
        # Hand off to W&B; disable tensorboard
        report_to                    = "wandb" if cfg["wandb_mode"] != "disabled" else "none",
        run_name                     = cfg.get("wandb_run_name"),
        seed                         = cfg["seed"],
    )

    # ── W&B custom callback ────────────────────
    # Build raw val example dicts for the qualitative prediction table
    val_raw = list(val_convs)   # already flat dicts
    wandb_callback = WandbSPIQACallback(
        processor    = processor,
        val_examples = val_raw,
        n_samples    = cfg["wandb_sample_preds"],
        max_new_tokens = 128,
    )

    # ── Trainer ────────────────────────────────
    trainer = Trainer(
        model              = model,
        args               = training_args,
        train_dataset      = train_ds,
        eval_dataset       = val_ds,
        data_collator      = collator,
        callbacks          = [wandb_callback] if cfg["wandb_mode"] != "disabled" else [],
    )

    log.info("Starting training ...")
    trainer.train()

    # ── Save ───────────────────────────────────
    save_path = Path(cfg["output_dir"]) / "final_lora_adapter"
    log.info(f"Saving LoRA adapter to {save_path}")
    trainer.model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    log.info("Done.")


if __name__ == "__main__":
    main()
