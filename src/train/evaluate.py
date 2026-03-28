"""
evaluate.py
-----------
Run a trained (or baseline) Qwen3-VL model on SPIQA Test-A and
compute METEOR, ROUGE-L, BERTScore, and optionally CIDEr.

Outputs a results JSON and a per-metric summary.

Usage
-----
# Zero-shot baseline
python evaluate.py --model_id Qwen/Qwen3-VL-8B-Instruct

# Fine-tuned LoRA adapter
python evaluate.py \
    --model_id Qwen/Qwen3-VL-8B-Instruct \
    --lora_path ./outputs/qwen3vl-8b-spiqa-lora/final_lora_adapter

# Thinking mode (zero-shot ablation)
python evaluate.py --model_id Qwen/Qwen3-VL-8B-Thinking --thinking
"""

import argparse
import json
import logging
import re
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration as AutoModelForVision2Seq

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

def load_spiqa_testA(json_path: str, image_dir: str) -> list[dict]:
    """
    Actual SPIQA Test-A schema:
      paper_id → all_figures → figure_filename → {caption, content_type, figure_type}
      paper_id → qa → [{question, answer, reference, ...}]
    """
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

    log.info(f"Loaded {len(examples)} Test-A examples (skipped {skipped})")
    return examples


# ─────────────────────────────────────────────
# Inference
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


def build_prompt(processor, example: dict, thinking: bool = False) -> tuple:
    caption_text = (
        f"\nfigure_caption: {example['caption']}" if example.get("caption") else ""
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": example["image_path"]},
                {"type": "text",  "text": f"{caption_text}\n\nquestion: {example['question']}\n\nstudent: {example['student']}"},
            ],
        },
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if thinking:
        text += "<think>\n"
    return text, messages


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from Thinking-mode output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


@torch.inference_mode()
def run_inference(
    model,
    processor,
    examples: list[dict],
    thinking: bool = False,
    max_new_tokens: int = 256,
) -> list[dict]:
    # With device_map="auto" the model may be split across GPUs;
    # model.device raises AttributeError in that case — use parameter device instead.
    device = next(model.parameters()).device
    results = []
    for ex in tqdm(examples, desc="Inference"):
        text, _ = build_prompt(processor, ex, thinking=thinking)
        try:
            image = Image.open(ex["image_path"]).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224))

        inputs = processor(
            text=text,
            images=[image],
            return_tensors="pt",
        ).to(device)

        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
        generated = gen_ids[:, inputs["input_ids"].shape[1]:]
        prediction = processor.batch_decode(generated, skip_special_tokens=True)[0]

        if thinking:
            prediction = strip_thinking(prediction)

        results.append({
            "paper_id":     ex["paper_id"],
            "fig_id":       ex.get("fig_id", ""),
            "fig_type":     ex.get("fig_type", "unknown"),
            "question":     ex["question"],
            "answer":       ex.get("answer", ""),
            "student":      ex.get("student", ""),
            "ground_truth": f"verdict = {ex['verdict']}\nerror_category = {ex['error_category']}\nfeedback = {ex['feedback']}",
            "prediction":   prediction.strip(),
        })
    return results


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    predictions  = [r["prediction"]   for r in results]
    ground_truths = [r["ground_truth"] for r in results]

    scores = {}

    # ROUGE-L
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_l = [
            scorer.score(gt, pred)["rougeL"].fmeasure
            for gt, pred in zip(ground_truths, predictions)
        ]
        scores["ROUGE-L"] = sum(rouge_l) / len(rouge_l)
    except ImportError:
        log.warning("rouge_score not installed — skipping ROUGE-L")

    # METEOR
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
            nltk.download("punkt", quiet=True)
            nltk.download("wordnet", quiet=True)
        from nltk.translate.meteor_score import meteor_score
        from nltk.tokenize import word_tokenize
        meteors = [
            meteor_score([word_tokenize(gt)], word_tokenize(pred))
            for gt, pred in zip(ground_truths, predictions)
        ]
        scores["METEOR"] = sum(meteors) / len(meteors)
    except ImportError:
        log.warning("nltk not installed — skipping METEOR")

    # BERTScore
    try:
        from bert_score import score as bert_score
        P, R, F1 = bert_score(predictions, ground_truths, model_type="distilbert-base-uncased", lang="en", verbose=False)
        scores["BERTScore-F1"] = F1.mean().item()
    except ImportError:
        log.warning("bert_score not installed — skipping BERTScore")

    return scores


# ─────────────────────────────────────────────
# Per-figure-type metric breakdown
# ─────────────────────────────────────────────

def analyse_by_fig_type(results: list[dict]) -> dict:
    """
    Compute per-metric breakdown by figure type.
    Mirrors evaluate_baseline.py for consistent cross-run comparison.
    """
    from collections import defaultdict
    by_type = defaultdict(list)
    for r in results:
        by_type[r.get("fig_type", "unknown")].append(r)

    breakdown = {}
    for fig_type, subset in sorted(by_type.items()):
        breakdown[fig_type] = {
            "n_examples": len(subset),
            **compute_metrics(subset),
        }
    return breakdown


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id",  default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--lora_path", default=None, help="Path to saved LoRA adapter")
    p.add_argument("--thinking",  action="store_true", help="Use Thinking mode")
    p.add_argument("--test_json", default="./spiqa/test-A/SPIQA_testA.json")
    p.add_argument("--image_dir", default="./spiqa/test-A/SPIQA_testA_Images")
    p.add_argument("--output_dir", default="./eval_results")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--ablation",    default=None,
                   help="Ablation name used as output filename and top-level JSON field")
    p.add_argument("--max_samples",    type=int, default=None,
                   help="Limit eval to N examples (useful for quick checks)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load model ────────────────────────────
    log.info(f"Loading processor from {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)

    log.info(f"Loading model from {args.model_id}")
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if args.lora_path:
        from peft import PeftModel
        log.info(f"Loading LoRA adapter from {args.lora_path}")
        model = PeftModel.from_pretrained(model, args.lora_path)
        model = model.merge_and_unload()
        log.info("LoRA adapter merged into base model")

    model.eval()

    # ── Load data ─────────────────────────────
    examples = load_spiqa_testA(args.test_json, args.image_dir)
    if args.max_samples:
        examples = examples[: args.max_samples]
        log.info(f"Evaluating on {len(examples)} examples (limited)")

    # ── Inference ─────────────────────────────
    results = run_inference(
        model, processor, examples,
        thinking=args.thinking,
        max_new_tokens=args.max_new_tokens,
    )

    # ── Metrics ───────────────────────────────
    scores      = compute_metrics(results)
    by_fig_type = analyse_by_fig_type(results)

    log.info("=" * 50)
    log.info("Evaluation Results")
    log.info("=" * 50)
    for metric, value in scores.items():
        log.info(f"  {metric:20s}: {value:.4f}")
    log.info("-" * 50)
    log.info("Per figure-type breakdown:")
    for fig_type, type_scores in sorted(by_fig_type.items()):
        n = type_scores.get("n_examples", 0)
        row = "  ".join(f"{k}={v:.3f}" for k, v in type_scores.items() if k != "n_examples")
        log.info(f"  {fig_type:15s} (n={n:4d})  {row}")
    log.info("=" * 50)

    # ── Save ──────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.ablation:
        ablation = args.ablation
    else:
        tag = "thinking" if args.thinking else "instruct"
        if args.lora_path:
            adapter_name = Path(args.lora_path).name
            tag += f"-lora-{adapter_name}"
        else:
            tag += "-zeroshot"
        ablation = tag

    results_path = out_dir / f"{ablation}.json"
    with open(results_path, "w") as f:
        json.dump({
            "ablation":    ablation,
            "n_examples":  len(results),
            "scores":      scores,
            "by_fig_type": by_fig_type,
            "predictions": results,
        }, f, indent=2)
    log.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
