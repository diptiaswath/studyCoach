"""
evaluate_molora.py
──────────────────
Evaluates a trained MoLoRA model on SPIQA Test-A.

Beyond the standard metrics (ROUGE-L, METEOR, BERTScore) it also:
  • Runs routing analysis — which experts activate for which figure types
  • Produces a per-expert accuracy breakdown
  • Saves a detailed results JSON + routing stats

Usage
─────
# Evaluate trained MoLoRA checkpoint
python evaluate_molora.py \\
    --model_id   Qwen/Qwen3-VL-8B-Instruct \\
    --molora_path ./outputs/qwen3vl-8b-spiqa-molora/final_molora

# Compare against vanilla LoRA (uses evaluate.py from original package)
python evaluate.py \\
    --model_id  Qwen/Qwen3-VL-8B-Instruct \\
    --lora_path ./outputs/qwen3vl-8b-spiqa-lora/final_lora_adapter
"""

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration as AutoModelForVision2Seq

from train_molora import MoLoRAModel, MoLoRARouter, MoLoRALinear, LoRAAdapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

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
"If verdict is correct, set error_category = N/A and feedback = N/A. Do not use bullet points of bold text."
)


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

def load_spiqa_testA(json_path: str, image_dir: str) -> list[dict]:
    """
    Actual SPIQA Test-A schema:
      paper_id → all_figures → figure_filename → {caption, content_type, figure_type}
      paper_id → qa → [{question, answer, reference, ...}]
    Each QA item's "reference" field is the figure filename used to look up
    the image path and caption from all_figures.
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
                "paper_id":   paper_id,
                "fig_id":     ref,
                "image_path": str(img_path),
                "caption":    fig_meta.get("caption", "").strip(),
                "fig_type":   fig_meta.get("figure_type", "unknown"),
                "question":   qa["question"],
                "answer":     qa["answer"],
                "student" :   qa["student"],
                "verdict":     qa["verdict"],
                "error_category": qa["error_category"],
                "feedback" : qa["feedback"]
            })

    log.info(f"Loaded {len(examples)} Test-A examples (skipped {skipped})")
    return examples


# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────

def load_model(model_id: str, molora_path: str, cfg_overrides: dict):
    """Load base model and reinject MoLoRA weights from checkpoint."""
    log.info(f"Loading processor from {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)

    log.info(f"Loading base model {model_id}")
    base_model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        dtype             = torch.bfloat16,
        device_map        = "auto",
        trust_remote_code = True,
    )
    for param in base_model.parameters():
        param.requires_grad = False

    # Load MoLoRA config
    molora_path = Path(molora_path)
    with open(molora_path / "molora_config.json") as f:
        saved_cfg = json.load(f)
    cfg = {**saved_cfg, **cfg_overrides}

    log.info(f"Reconstructing MoLoRA model (K={cfg['num_experts']})")
    model = MoLoRAModel(
        base_model        = base_model,
        num_experts       = cfg["num_experts"],
        r                 = cfg.get("molora_r", 16),
        alpha             = cfg.get("molora_alpha", 32),
        dropout           = 0.0,   # no dropout at eval
        target_modules    = cfg.get("target_modules", ["q_proj", "v_proj", "o_proj", "gate_proj"]),
        router_hidden_dim = cfg.get("router_hidden_dim", 128),
        router_noise      = 0.0,   # no noise at eval
        aux_loss_coef     = 0.0,
    )

    # Detect which checkpoint file was saved.
    # _save_checkpoint writes "molora_adapter.pt" (de-duplicated trainable params).
    # save_pretrained writes "molora_weights.pt" (also trainable params).
    # Try both.
    adapter_file = None
    for candidate in ["molora_adapter.pt", "molora_weights.pt"]:
        if (molora_path / candidate).exists():
            adapter_file = molora_path / candidate
            log.info(f"Loading adapter weights from {adapter_file}")
            break
    if adapter_file is None:
        raise FileNotFoundError(
            f"No adapter file found in {molora_path}. "
            "Expected 'molora_adapter.pt' or 'molora_weights.pt'."
        )

    saved_state = torch.load(adapter_file, map_location="cpu", weights_only=False)

    # The de-duplicated checkpoint stores router params under a short key
    # like "router.net.0.weight" rather than the full nested path used by
    # load_state_dict. We need to inject them directly by name.
    loaded_router = False
    loaded_experts = 0

    for key, tensor in saved_state.items():
        tensor_bf16 = tensor.to(torch.bfloat16)

        # ── Router params ──────────────────────────────────────────────
        # Keys saved by _save_checkpoint look like "router.net.0.weight"
        if key.startswith("router."):
            # e.g. "router.net.0.weight" → attr path on model.router
            attr_path = key[len("router."):]   # "net.0.weight"
            obj = model.router
            parts = attr_path.split(".")
            for part in parts[:-1]:
                obj = getattr(obj, part)
            param = getattr(obj, parts[-1])
            device = param.device if hasattr(param, "device") else next(model.router.parameters()).device
            param.data.copy_(tensor_bf16.to(device))
            loaded_router = True
            continue

        # ── LoRA expert params ─────────────────────────────────────────
        # Keys look like "base_model.model...q_proj.experts.0.lora_A.weight"
        # or short form "experts.0.lora_A.weight" if saved differently.
        # Walk the model's named parameters to find a match by suffix.
        matched = False
        for name, param in model.named_parameters():
            if name == key or name.endswith("." + key) or key.endswith("." + name):
                device = param.device
                param.data.copy_(tensor_bf16.to(device))
                loaded_experts += 1
                matched = True
                break
        if not matched and not key.startswith("router."):
            log.debug(f"  No match for key: {key}")

    # Cast all MoLoRA adapters and router to bfloat16 on the correct device
    # (catches any that weren't explicitly loaded above)
    base_device = next(base_model.parameters()).device
    for layer in model.molora_layers:
        layer.to(dtype=torch.bfloat16, device=base_device)
    model.router.to(dtype=torch.bfloat16, device=base_device)

    log.info(
        f"Loaded adapter weights — router: {'yes' if loaded_router else 'NO'}, "
        f"expert params matched: {loaded_experts}"
    )

    model.eval()
    return model, processor


# ─────────────────────────────────────────────
# Inference with routing capture
# ─────────────────────────────────────────────

@torch.inference_mode()
def run_inference(
    model: MoLoRAModel,
    processor,
    examples: list[dict],
    max_new_tokens: int = 256,
) -> list[dict]:
    results = []
    routing_by_fig_type = defaultdict(list)  # fig_type → list of routing weight vectors

    for ex in tqdm(examples, desc="Inference"):
        caption_text = f"\nfigure_caption: {ex['caption']}" if ex.get("caption") else ""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": ex["image_path"]},
                    {"type": "text",  "text": f"{caption_text}\n\nquestion: {ex['question']}\n\nstudent: {ex['student']}"},
                ],
            },
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        try:    img = Image.open(ex["image_path"]).convert("RGB")
        except: img = Image.new("RGB", (224, 224))

        inputs = processor(
            text=text, images=[img], return_tensors="pt"
        ).to(model.base_model.device)

        # Register a one-shot hook on the visual merger output.
        # During generate()'s prefill the vision encoder runs exactly once,
        # firing the merger hook and setting model._visual_summary.
        # Our hook intercepts that moment to:
        #   (a) broadcast the visual summary to all MoLoRALinear layers, and
        #   (b) capture the resulting routing weights.
        # This avoids a separate full forward pass and the deadlock it caused
        # with device_map="auto".
        captured_weights = []
        _broadcast_done  = [False]

        def _merger_broadcast_hook(module, input, output):
            if _broadcast_done[0]:
                return  # only act on the first (prefill) call
            _broadcast_done[0] = True
            # Mean-pool merger output → visual summary, same as training hook
            vs = output.detach()
            if vs.dim() == 2:
                vs = vs.mean(0, keepdim=True)   # (1, D)
            else:
                vs = vs.mean(1)                  # (B, D)
            model._visual_summary = vs
            model._broadcast_visual_summary()
            # Capture routing weights from the first MoLoRALinear layer's router
            weights, _ = model.router(vs, training=False)
            captured_weights.append(weights.detach().mean(0).cpu())

        # Resolve merger module the same way training does
        _merger_module = None
        for name, module in model.base_model.named_modules():
            if name in ("visual.merger", "model.visual.merger"):
                if _merger_module is None or len(name) > len(_merger_module[0]):
                    _merger_module = (name, module)
        if _merger_module is None:
            raise RuntimeError("Could not find visual merger module for eval hook")
        _hook_handle = _merger_module[1].register_forward_hook(
            _merger_broadcast_hook
        )

        gen_ids = model.base_model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            do_sample      = False,
        )

        _hook_handle.remove()

        # Routing weights captured during prefill (image-conditioned)
        routing_weights = None
        if captured_weights:
            routing_weights = captured_weights[0].tolist()
            routing_by_fig_type[ex.get("fig_type", "unknown")].append(routing_weights)
        new_toks = gen_ids[:, inputs["input_ids"].shape[1]:]
        prediction = processor.batch_decode(new_toks, skip_special_tokens=True)[0].strip()
        ground_truth = f"verdict = {ex['verdict']}\nerror_category = {ex['error_category']}\nfeedback = {ex['feedback']}"

        results.append({
            "paper_id":        ex["paper_id"],
            "fig_id":          ex.get("fig_id", ""),
            "fig_type":        ex.get("fig_type", "unknown"),
            "question":        ex["question"],
            "answer":          ex.get("answer", ""),
            "student":         ex.get("student", ""),
            "ground_truth":    ground_truth,
            "prediction":      prediction,
            "routing_weights": routing_weights,
        })

    return results, dict(routing_by_fig_type)


# ─────────────────────────────────────────────
# Routing analysis
# ─────────────────────────────────────────────

def analyse_routing(
    results: list[dict],
    routing_by_fig_type: dict,
    num_experts: int = None,
) -> dict:
    """
    Compute per-expert statistics and figure-type routing affinity.
    num_experts is derived from the routing weights if not provided.
    """
    analysis = {}

    # Global expert load
    all_weights = [r["routing_weights"] for r in results if r["routing_weights"]]
    if all_weights:
        import torch
        w = torch.tensor(all_weights)  # (N, K)
        K = w.shape[1]  # derive K from data — never rely on arg being non-None
        if num_experts is None:
            num_experts = K
        mean_load = w.mean(0).tolist()
        analysis["global_expert_load"] = {
            f"expert_{i}": mean_load[i] for i in range(K)
        }
        # Router entropy
        eps = 1e-8
        mean_w = w.mean(0)
        entropy = -(mean_w * (mean_w + eps).log()).sum().item()
        analysis["router_entropy"] = entropy
        analysis["top_expert_dominance"] = mean_w.max().item()

    # Per figure-type routing affinity
    fig_type_routing = {}
    for fig_type, weights_list in routing_by_fig_type.items():
        if not weights_list:
            continue
        w = torch.tensor(weights_list).mean(0)  # (K,)
        fig_type_routing[fig_type] = {
            f"expert_{i}": w[i].item() for i in range(len(w))
        }
        # Dominant expert for this figure type
        fig_type_routing[fig_type]["dominant_expert"] = int(w.argmax().item())

    analysis["routing_by_fig_type"] = fig_type_routing
    return analysis


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    predictions   = [r["prediction"]   for r in results]
    ground_truths = [r["ground_truth"] for r in results]
    scores = {}

    try:
        from rouge_score import rouge_scorer
        scorer  = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_l = [scorer.score(gt, p)["rougeL"].fmeasure
                   for gt, p in zip(ground_truths, predictions)]
        scores["ROUGE-L"] = sum(rouge_l) / len(rouge_l)
    except ImportError:
        log.warning("rouge_score not installed — skipping ROUGE-L")

    try:
        import nltk
        for resource in ["punkt", "punkt_tab", "wordnet"]:
            try: nltk.data.find(f"tokenizers/{resource}")
            except LookupError: nltk.download(resource, quiet=True)
        from nltk.translate.meteor_score import meteor_score
        from nltk.tokenize import word_tokenize
        meteors = [meteor_score([word_tokenize(gt)], word_tokenize(p))
                   for gt, p in zip(ground_truths, predictions)]
        scores["METEOR"] = sum(meteors) / len(meteors)
    except ImportError:
        log.warning("nltk not installed — skipping METEOR")

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
    p.add_argument("--model_id",    default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--molora_path", required=True, help="Path to saved MoLoRA checkpoint")
    p.add_argument("--test_json",   default="./spiqa_data/test-A/SPIQA_testA.json")
    p.add_argument("--image_dir",   default="./spiqa_data/test-A/images")
    p.add_argument("--output_dir",  default="./eval_results")
    p.add_argument("--ablation",    default=None,
                   help="Ablation name used as output filename and top-level JSON field")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--max_samples",    type=int, default=None)
    # Config overrides — only values explicitly passed on the CLI override
    # the saved molora_config.json. Defaults are set to None so that the
    # saved config value wins when the user doesn't pass the flag.
    p.add_argument("--num_experts",      type=int,   default=None)
    p.add_argument("--molora_r",         type=int,   default=None)
    p.add_argument("--molora_alpha",     type=float, default=None)
    p.add_argument("--router_hidden_dim",type=int,   default=None)
    p.add_argument("--target_modules",  default=None)
    return p.parse_args()


def main():
    args    = parse_args()
    # Build cfg_ovr only from args that were explicitly provided
    cfg_ovr = {}
    if args.num_experts      is not None: cfg_ovr["num_experts"]       = args.num_experts
    if args.molora_r         is not None: cfg_ovr["molora_r"]          = args.molora_r
    if args.molora_alpha     is not None: cfg_ovr["molora_alpha"]      = args.molora_alpha
    if args.router_hidden_dim is not None: cfg_ovr["router_hidden_dim"] = args.router_hidden_dim
    if args.target_modules   is not None:
        cfg_ovr["target_modules"] = [m.strip() for m in args.target_modules.split(",")]

    model, processor = load_model(args.model_id, args.molora_path, cfg_ovr)

    examples = load_spiqa_testA(args.test_json, args.image_dir)
    if args.max_samples:
        examples = examples[: args.max_samples]
        log.info(f"Evaluating on {len(examples)} examples (limited)")

    results, routing_by_fig_type = run_inference(
        model, processor, examples, max_new_tokens=args.max_new_tokens
    )

    scores     = compute_metrics(results)
    routing    = analyse_routing(results, routing_by_fig_type, model.num_experts)
    by_fig_type = analyse_by_fig_type(results)

    log.info("=" * 55)
    log.info("Evaluation Results")
    log.info("=" * 55)
    for m, v in scores.items():
        log.info(f"  {m:20s}: {v:.4f}")
    log.info("-" * 55)
    log.info("Routing Analysis")
    log.info(f"  Router entropy     : {routing.get('router_entropy', 'N/A'):.4f}")
    log.info(f"  Top expert dominance: {routing.get('top_expert_dominance', 'N/A'):.4f}")
    log.info("  Global expert load :")
    for k, v in routing.get("global_expert_load", {}).items():
        log.info(f"    {k}: {v:.3f}")
    log.info("-" * 55)
    log.info("Per figure-type breakdown:")
    for fig_type, type_scores in sorted(by_fig_type.items()):
        n = type_scores.get("n_examples", 0)
        row = "  ".join(f"{k}={v:.3f}" for k, v in type_scores.items() if k != "n_examples")
        log.info(f"  {fig_type:15s} (n={n:4d})  {row}")
    log.info("=" * 55)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ablation = args.ablation or Path(args.molora_path).parent.name
    out_path = out_dir / f"{ablation}.json"
    with open(out_path, "w") as f:
        json.dump({
            "ablation":    ablation,
            "scores":      scores,
            "routing":     routing,
            "by_fig_type": by_fig_type,
            "n_examples":  len(results),
            "predictions": results,
        }, f, indent=2)
    log.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
