#!/usr/bin/env python3
"""Evaluation Script for Qwen3.5-9B (Modification 2: Architectural Ablation)

This script runs the Modification 2 architectural ablation from Report 3:
Testing Qwen3.5-9B (early fusion architecture) vs Qwen3-VL-8B (bolt-on ViT).

Hypothesis: If the bolt-on ViT architecture of Qwen3-VL causes the PC Recall
collapse observed under visual input, Qwen3.5-9B should show a smaller or
reversed verdict gap (C4 accuracy >= C1 accuracy).

================================================================================
IMPLEMENTATION STEPS (from Report 3)
================================================================================

1. Confirm Together.ai access.
   Verify Qwen/Qwen3.5-9B is available under the existing Together.ai API key
   used for Report 2. Model is live as of March 2, 2026 ($0.10/1M input,
   $0.15/1M output tokens).

2. Disable thinking mode.
   Qwen3.5-9B runs in thinking mode by default, which prepends a chain-of-thought
   reasoning block before the final response. This will break the structured
   output parser (which expects Verdict = ..., Error Category = ...,
   Feedback = ...). Disable it explicitly:

       extra_body={"chat_template_kwargs": {"enable_thinking": False}}

   Use sampling parameters: temperature=0.0 (greedy, for reproducibility),
   top_p=1.0, presence_penalty=0.0.

3. Use identical prompts.
   Use the exact same prompt templates as Report 2 for all four conditions
   (C1-C4). No prompt changes. The only change is the model ID string. This
   ensures any performance difference is attributable to the model architecture,
   not prompt variation.

4. Run C1 and C4 on the H3/H4 evaluation set (N=108).
   These are the two conditions that define the verdict-feedback gap. C1
   (text-only) establishes the same-architecture text baseline. C4 (multimodal)
   is the hypothesis-test condition: does early fusion eliminate the verdict
   drop? Running all four conditions (C1-C4) is preferred if API budget allows.

5. Parse outputs with the same extraction logic.
   Qwen3.5-9B output formatting may differ slightly from Qwen3-VL (e.g., extra
   whitespace, different capitalization of "Partially Correct" vs "partially
   correct"). Before computing metrics, verify the parser correctly extracts
   Verdict, Error Category, and Feedback fields from 5-10 sample outputs.
   Normalize verdict strings to lowercase before comparison.

6. Run the same three evaluation scripts:
   - Verdict accuracy: compare predicted vs. ground-truth verdict labels
   - Intrinsic metrics (PC Recall, FSR, VGS): same computation from raw
     per-example outputs
   - Feedback quality (LLM-as-Judge): run analyze_feedback_by_error_type.py
     with model=claude-sonnet-4-6, temperature=0, N=108

7. Interpret the result against two hypotheses:
   - If C4 verdict accuracy >= C1 for Qwen3.5-9B: early fusion resolves the
     architectural mismatch. Report 4 should use Qwen3.5-9B as the backbone
     for fine-tuning on SPIQA+, not Qwen3-VL.
   - If C4 verdict accuracy < C1 for Qwen3.5-9B: the bolt-on ViT is not the
     cause. The failure is inherent to how visual input interacts with
     three-class verdict decisions at this parameter scale.

================================================================================
KEY IMPLEMENTATION DETAILS
================================================================================

Model: Qwen/Qwen3.5-9B (early fusion, no separate vision encoder)
Thinking mode: DISABLED via extra_body={"chat_template_kwargs": {"enable_thinking": False}}
Sampling: temperature=0.0, top_p=1.0, presence_penalty=0.0

================================================================================
USAGE
================================================================================

    python src/eval_qwen35_9b.py \\
        --data data/test-A/SPIQA_testA_part1_output_latest.json \\
        --images data/test-A/SPIQA_testA_Images \\
        --output data/eval/qwen35_9b

    # Run specific conditions only:
    python src/eval_qwen35_9b.py \\
        --data data/test-A/SPIQA_testA_part1_output_latest.json \\
        --images data/test-A/SPIQA_testA_Images \\
        --output data/eval/qwen35_9b \\
        --conditions text_only multimodal

    # Smoke test with 3 examples per error type:
    python src/eval_qwen35_9b.py \\
        --data data/test-A/SPIQA_testA_part1_output_latest.json \\
        --images data/test-A/SPIQA_testA_Images \\
        --output data/eval/qwen35_9b \\
        --max-per-type 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import openai

sys.path.insert(0, str(Path(__file__).parent))
from eval_utils import parse_eval_output, to_data_url, SYSTEM_PROMPT

# Qwen3.5-9B: Early fusion architecture (no separate vision encoder)
MODEL = "Qwen/Qwen3.5-9B"

# Together.ai pricing: $0.10/1M input, $0.15/1M output tokens


def build_client() -> openai.OpenAI:
    """Build Together.ai client."""
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "TOGETHER_API_KEY environment variable is not set. "
            "Get a key at https://api.together.ai"
        )
    return openai.OpenAI(
        base_url="https://api.together.xyz/v1",
        api_key=api_key,
    )


def load_all_examples(
    output_json_paths: List[str | Path],
    images_root: str | Path,
) -> List[Dict[str, Any]]:
    """Load ALL examples from augmented SPIQA output JSONs (no sampling).

    Returns all examples with valid student answers.
    """
    images_root = Path(images_root)
    examples = []

    for path in output_json_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for paper_key, paper in data.items():
            all_figures = paper.get("all_figures", {})
            for qa in paper.get("qa", []):
                student = (qa.get("student") or "").strip()
                if not student:
                    continue

                verdict = (qa.get("verdict") or "").strip()
                error_category = (qa.get("error_category") or "").strip()
                feedback = (qa.get("feedback") or "").strip()
                question = (qa.get("question") or "").strip()
                answer = (qa.get("answer") or "").strip()
                reference = (qa.get("reference") or "").strip()
                figure_type = (qa.get("figure_type") or "").strip()

                fig_details = all_figures.get(reference, {})
                caption = (fig_details.get("caption") or "").strip()
                image_path = images_root / paper_key / reference

                examples.append({
                    "paper_id": paper_key,
                    "question": question,
                    "answer": answer,
                    "caption": caption,
                    "image_path": str(image_path),
                    "student": student,
                    "figure_type": figure_type,
                    "ground_truth": {
                        "verdict": verdict,
                        "error_category": error_category,
                        "feedback": feedback,
                    },
                })

    return examples


def filter_error_examples(examples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Filter to error examples only and group by error category.

    Excludes examples where verdict is "Correct" or "correct".
    Groups remaining by error_category (factual, conceptual, omission).
    """
    by_error_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for ex in examples:
        verdict = ex["ground_truth"]["verdict"].lower()
        if verdict == "correct":
            continue

        error_cat = ex["ground_truth"]["error_category"].lower()
        if error_cat in ("factual", "conceptual", "omission"):
            by_error_type[error_cat].append(ex)

    return dict(by_error_type)


def call_model(
    client: openai.OpenAI,
    messages: List[Dict[str, Any]],
) -> str:
    """Call Qwen3.5-9B with thinking mode disabled.

    Critical settings per Report 3 Modification 2:
    - extra_body with enable_thinking=False to disable chain-of-thought
    - temperature=0.0 for reproducibility
    - top_p=1.0, presence_penalty=0.0
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.0,
        top_p=1.0,
        presence_penalty=0.0,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        },
    )
    return response.choices[0].message.content or ""


def evaluate_text_only(client: openai.OpenAI, example: dict) -> Tuple[dict, str]:
    """Scenario C1: text_only (Q + Student Answer)."""
    question = example["question"]
    student = example["student"]

    user_text = f"Question:\n{question}\nStudent Answer:\n{student}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": user_text}]},
    ]

    output_text = call_model(client, messages)
    return parse_eval_output(output_text), output_text


def evaluate_caption_only(client: openai.OpenAI, example: dict) -> Tuple[dict, str]:
    """Scenario C2: caption_only (Q + Caption + Student Answer)."""
    caption = example["caption"]
    question = example["question"]
    student = example["student"]

    user_text = f"Caption:\n{caption}\nQuestion:\n{question}\nStudent Answer:\n{student}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": user_text}]},
    ]

    output_text = call_model(client, messages)
    return parse_eval_output(output_text), output_text


def evaluate_vision_only(client: openai.OpenAI, example: dict) -> Tuple[dict, str]:
    """Scenario C3: vision_only (Q + Image + Student Answer)."""
    question = example["question"]
    student = example["student"]
    image_path = example["image_path"]

    data_url = to_data_url(image_path)
    text = f"Question:\n{question}\nStudent Answer:\n{student}"

    user_content = [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    output_text = call_model(client, messages)
    return parse_eval_output(output_text), output_text


def evaluate_multimodal(client: openai.OpenAI, example: dict) -> Tuple[dict, str]:
    """Scenario C4: multimodal (Q + Caption + Image + Student Answer)."""
    caption = example["caption"]
    question = example["question"]
    student = example["student"]
    image_path = example["image_path"]

    data_url = to_data_url(image_path)
    text = f"Caption:\n{caption}\nQuestion:\n{question}\nStudent Answer:\n{student}"

    user_content = [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    output_text = call_model(client, messages)
    return parse_eval_output(output_text), output_text


SCENARIOS = {
    "text_only": evaluate_text_only,
    "caption_only": evaluate_caption_only,
    "vision_only": evaluate_vision_only,
    "multimodal": evaluate_multimodal,
}


def run_evaluation(
    client: openai.OpenAI,
    examples_by_type: Dict[str, List[Dict[str, Any]]],
    conditions: List[str],
    max_per_type: int | None = None,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Run specified scenarios on each error type subset.

    Returns: {error_type: {scenario: [results]}}
    """
    results: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for error_type, examples in examples_by_type.items():
        if max_per_type and len(examples) > max_per_type:
            examples = examples[:max_per_type]

        print(f"\n{'='*60}", flush=True)
        print(f"ERROR TYPE: {error_type.upper()} ({len(examples)} examples)", flush=True)
        print(f"{'='*60}", flush=True)

        results[error_type] = {}

        for scenario_name in conditions:
            if scenario_name not in SCENARIOS:
                print(f"  Skipping unknown scenario: {scenario_name}", flush=True)
                continue

            eval_func = SCENARIOS[scenario_name]
            print(f"\n  Scenario: {scenario_name}", flush=True)
            scenario_results = []

            for i, example in enumerate(examples, 1):
                print(f"    [{i}/{len(examples)}] paper={example['paper_id']} "
                      f"gt={example['ground_truth']['verdict']}", end="", flush=True)
                try:
                    predicted, raw_output = eval_func(client, example)
                    result = {
                        **example,
                        "predicted": predicted,
                        "raw_output": raw_output,
                    }
                    scenario_results.append(result)
                    print(f" -> {predicted['verdict']}", flush=True)
                except Exception as e:
                    print(f" ERROR: {e}", flush=True)
                    scenario_results.append({
                        **example,
                        "predicted": {"verdict": "", "error_category": "N/A", "feedback": "N/A"},
                        "raw_output": f"ERROR: {e}",
                    })

            results[error_type][scenario_name] = scenario_results

    return results


def calculate_accuracy(results: List[Dict[str, Any]]) -> Tuple[int, int, float]:
    """Calculate verdict accuracy for a list of results.

    Returns: (correct_count, total_count, accuracy_percentage)
    """
    correct = sum(
        1 for r in results
        if r["predicted"]["verdict"].lower() == r["ground_truth"]["verdict"].lower()
    )
    total = len(results)
    accuracy = (correct / total * 100) if total > 0 else 0.0
    return correct, total, accuracy


def calculate_pc_recall(results: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> Dict[str, float]:
    """Calculate PC Recall for each scenario.

    PC Recall = TP_PC / (TP_PC + FN_PC)
    where TP_PC = correctly classified Partially Correct examples
    """
    pc_recall = {}

    for scenario_name in ["text_only", "caption_only", "vision_only", "multimodal"]:
        tp_pc = 0
        fn_pc = 0

        for error_type, scenarios in results.items():
            if scenario_name not in scenarios:
                continue
            for r in scenarios[scenario_name]:
                gt_verdict = r["ground_truth"]["verdict"].lower()
                pred_verdict = r["predicted"]["verdict"].lower()

                if gt_verdict == "partially correct":
                    if pred_verdict == "partially correct":
                        tp_pc += 1
                    else:
                        fn_pc += 1

        total_pc = tp_pc + fn_pc
        pc_recall[scenario_name] = (tp_pc / total_pc * 100) if total_pc > 0 else 0.0

    return pc_recall


def calculate_fsr(results: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> Dict[str, float]:
    """Calculate Feedback Suppression Rate for each scenario.

    FSR = # Correct predictions on non-correct GT / # non-correct GT examples
    """
    fsr = {}

    for scenario_name in ["text_only", "caption_only", "vision_only", "multimodal"]:
        correct_on_non_correct = 0
        total_non_correct = 0

        for error_type, scenarios in results.items():
            if scenario_name not in scenarios:
                continue
            for r in scenarios[scenario_name]:
                gt_verdict = r["ground_truth"]["verdict"].lower()
                pred_verdict = r["predicted"]["verdict"].lower()

                if gt_verdict != "correct":
                    total_non_correct += 1
                    if pred_verdict == "correct":
                        correct_on_non_correct += 1

        fsr[scenario_name] = (correct_on_non_correct / total_non_correct * 100) if total_non_correct > 0 else 0.0

    return fsr


def analyze_results(
    results: Dict[str, Dict[str, List[Dict[str, Any]]]]
) -> Dict[str, Any]:
    """Analyze results including intrinsic metrics.

    Returns analysis dict with:
    - accuracy_matrix: {error_type: {scenario: accuracy}}
    - pc_recall: {scenario: pc_recall}
    - fsr: {scenario: fsr}
    - context_benefit: {error_type: delta}
    """
    accuracy_matrix: Dict[str, Dict[str, float]] = {}
    counts_matrix: Dict[str, Dict[str, Tuple[int, int]]] = {}

    for error_type, scenarios in results.items():
        accuracy_matrix[error_type] = {}
        counts_matrix[error_type] = {}
        for scenario_name, scenario_results in scenarios.items():
            correct, total, accuracy = calculate_accuracy(scenario_results)
            accuracy_matrix[error_type][scenario_name] = accuracy
            counts_matrix[error_type][scenario_name] = (correct, total)

    # Calculate context benefit (delta = multimodal - text_only)
    context_benefit: Dict[str, float] = {}
    for error_type in accuracy_matrix:
        if "multimodal" in accuracy_matrix[error_type] and "text_only" in accuracy_matrix[error_type]:
            delta = accuracy_matrix[error_type]["multimodal"] - accuracy_matrix[error_type]["text_only"]
            context_benefit[error_type] = delta

    # Calculate intrinsic metrics
    pc_recall = calculate_pc_recall(results)
    fsr = calculate_fsr(results)

    # Calculate overall verdict accuracy per scenario
    verdict_accuracy: Dict[str, float] = {}
    for scenario_name in ["text_only", "caption_only", "vision_only", "multimodal"]:
        correct_total = 0
        total_total = 0
        for error_type, scenarios in results.items():
            if scenario_name in scenarios:
                c, t, _ = calculate_accuracy(scenarios[scenario_name])
                correct_total += c
                total_total += t
        verdict_accuracy[scenario_name] = (correct_total / total_total * 100) if total_total > 0 else 0.0

    return {
        "accuracy_matrix": accuracy_matrix,
        "counts_matrix": counts_matrix,
        "context_benefit": context_benefit,
        "pc_recall": pc_recall,
        "fsr": fsr,
        "verdict_accuracy": verdict_accuracy,
    }


def print_summary(analysis: Dict[str, Any], conditions: List[str]) -> None:
    """Print summary tables to console."""
    print("\n" + "=" * 70)
    print(f"QWEN3.5-9B EVALUATION RESULTS (Modification 2)")
    print("=" * 70)

    # Intrinsic metrics
    print("\n" + "-" * 70)
    print("INTRINSIC METRICS")
    print("-" * 70)

    print(f"\n{'Condition':<15} {'PC Recall':>12} {'FSR':>12} {'Verdict':>12}")
    print("-" * 55)
    for cond in conditions:
        if cond in analysis.get("pc_recall", {}):
            pc = analysis["pc_recall"].get(cond, 0)
            fsr_val = analysis["fsr"].get(cond, 0)
            verd = analysis["verdict_accuracy"].get(cond, 0)
            print(f"{cond:<15} {pc:>11.1f}% {fsr_val:>11.1f}% {verd:>11.1f}%")

    # Accuracy by error type
    print("\n" + "-" * 70)
    print("ACCURACY BY ERROR TYPE")
    print("-" * 70)

    error_types = ["factual", "conceptual", "omission"]

    print(f"\n{'Error Type':<15}", end="")
    for s in conditions:
        print(f"{s:>15}", end="")
    print()

    for et in error_types:
        if et not in analysis["accuracy_matrix"]:
            continue
        print(f"{et:<15}", end="")
        for s in conditions:
            if s in analysis["accuracy_matrix"][et]:
                acc = analysis["accuracy_matrix"][et][s]
                print(f"{acc:>14.1f}%", end="")
            else:
                print(f"{'N/A':>15}", end="")
        print()

    # Context benefit
    if "text_only" in conditions and "multimodal" in conditions:
        print("\n" + "-" * 70)
        print("CONTEXT BENEFIT (delta = multimodal - text_only)")
        print("-" * 70)
        for et in error_types:
            if et in analysis["context_benefit"]:
                delta = analysis["context_benefit"][et]
                sign = "+" if delta >= 0 else ""
                print(f"  {et:<15}: {sign}{delta:.1f}pp")

    print("\n" + "=" * 70)


def save_results_json(
    results: Dict[str, Dict[str, List[Dict[str, Any]]]],
    analysis: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Save full results to JSON."""
    results_path = output_dir / "error_type_results.json"

    # Convert results to serializable format
    serializable = {}
    for et, scenarios in results.items():
        serializable[et] = {}
        for s, res_list in scenarios.items():
            serializable[et][s] = res_list

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": MODEL,
            "results": serializable,
            "analysis": analysis,
        }, f, indent=2, ensure_ascii=False)

    print(f"Saved full results to {results_path}")


def save_markdown_report(
    analysis: Dict[str, Any],
    conditions: List[str],
    output_dir: Path,
) -> None:
    """Save markdown report."""
    report_path = output_dir / "qwen35_9b_analysis.md"

    error_types = ["factual", "conceptual", "omission"]

    lines = [
        "# Qwen3.5-9B Evaluation: Modification 2 (Architectural Ablation)",
        "",
        f"**Model:** `{MODEL}`",
        "",
        "## Hypothesis",
        "",
        "If the bolt-on ViT architecture of Qwen3-VL causes the PC Recall collapse",
        "observed under visual input, Qwen3.5-9B (early fusion) should show a smaller",
        "or reversed verdict gap (C4 accuracy >= C1 accuracy).",
        "",
        "## Intrinsic Metrics",
        "",
        "| Condition | PC Recall | FSR | Verdict |",
        "|-----------|-----------|-----|---------|",
    ]

    for cond in conditions:
        if cond in analysis.get("pc_recall", {}):
            pc = analysis["pc_recall"].get(cond, 0)
            fsr_val = analysis["fsr"].get(cond, 0)
            verd = analysis["verdict_accuracy"].get(cond, 0)
            lines.append(f"| {cond} | {pc:.1f}% | {fsr_val:.1f}% | {verd:.1f}% |")

    lines.extend([
        "",
        "## Accuracy Matrix (Error Type x Condition)",
        "",
    ])

    # Build header
    header = "| Error Type |"
    sep = "|------------|"
    for cond in conditions:
        header += f" {cond} |"
        sep += "------------|"
    lines.append(header)
    lines.append(sep)

    for et in error_types:
        if et not in analysis["accuracy_matrix"]:
            continue
        row = f"| {et} |"
        for cond in conditions:
            if cond in analysis["accuracy_matrix"][et]:
                acc = analysis["accuracy_matrix"][et][cond]
                counts = analysis["counts_matrix"][et].get(cond, (0, 0))
                row += f" {acc:.1f}% ({counts[0]}/{counts[1]}) |"
            else:
                row += " N/A |"
        lines.append(row)

    if "text_only" in conditions and "multimodal" in conditions:
        lines.extend([
            "",
            "## Context Benefit (delta = multimodal - text_only)",
            "",
            "| Error Type | delta (pp) |",
            "|------------|-----------|",
        ])
        for et in error_types:
            if et in analysis["context_benefit"]:
                delta = analysis["context_benefit"][et]
                sign = "+" if delta >= 0 else ""
                lines.append(f"| {et} | {sign}{delta:.1f} |")

    # Interpretation
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Compare C1 vs C4 verdict accuracy:",
        "",
    ])

    if "text_only" in analysis["verdict_accuracy"] and "multimodal" in analysis["verdict_accuracy"]:
        c1_acc = analysis["verdict_accuracy"]["text_only"]
        c4_acc = analysis["verdict_accuracy"]["multimodal"]
        gap = c4_acc - c1_acc

        lines.append(f"- C1 (text_only) verdict: {c1_acc:.1f}%")
        lines.append(f"- C4 (multimodal) verdict: {c4_acc:.1f}%")
        lines.append(f"- Gap (C4 - C1): {gap:+.1f}pp")
        lines.append("")

        if gap >= 0:
            lines.append("**Result: Gap resolved or reversed.** Early fusion at 9B may be sufficient.")
            lines.append("Implication: Fine-tune Qwen3.5-9B on SPIQA+.")
        else:
            lines.append("**Result: Gap persists.** The bolt-on ViT is not the sole cause.")
            lines.append("Implication: The failure may be scale-dependent. Test Modification 3 (Qwen3.5-397B).")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved markdown report to {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3.5-9B for Modification 2 (Architectural Ablation)"
    )
    parser.add_argument(
        "--data",
        required=True,
        nargs="+",
        help="Path(s) to *_output.json augmented dataset files",
    )
    parser.add_argument(
        "--images",
        required=True,
        help="Root directory containing per-paper image subdirectories",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["text_only", "multimodal"],
        choices=["text_only", "caption_only", "vision_only", "multimodal"],
        help="Conditions to run (default: text_only multimodal)",
    )
    parser.add_argument(
        "--max-per-type",
        type=int,
        default=None,
        help="Maximum examples per error type (for smoke testing)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {MODEL}")
    print(f"Conditions: {args.conditions}")
    print(f"Output: {output_dir}")

    client = build_client()

    # Step 1: Load all examples
    print("\nStep 1: Loading all examples...", flush=True)
    all_examples = load_all_examples(args.data, args.images)
    print(f"  Loaded {len(all_examples)} total examples", flush=True)

    # Step 2: Filter to error examples and group
    print("\nStep 2: Filtering to error examples (N=108)...", flush=True)
    examples_by_type = filter_error_examples(all_examples)
    total_errors = sum(len(v) for v in examples_by_type.values())
    print(f"  Error examples: {total_errors}", flush=True)
    for et, exs in sorted(examples_by_type.items()):
        pct = len(exs) / total_errors * 100 if total_errors > 0 else 0
        print(f"    {et}: {len(exs)} ({pct:.0f}%)", flush=True)

    # Step 3: Run evaluation
    print(f"\nStep 3: Running evaluation on {args.conditions}...", flush=True)
    results = run_evaluation(
        client,
        examples_by_type,
        conditions=args.conditions,
        max_per_type=args.max_per_type
    )

    # Step 4: Analyze results
    print("\nStep 4: Analyzing results...", flush=True)
    analysis = analyze_results(results)

    # Step 5: Print summary
    print_summary(analysis, args.conditions)

    # Step 6: Save results
    print("\nStep 6: Saving results...", flush=True)
    save_results_json(results, analysis, output_dir)
    save_markdown_report(analysis, args.conditions, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
