#!/usr/bin/env python3
"""Evaluation by Error Type using HuggingFace Inference Endpoint.

Same evaluation as eval_by_error_type.py (H3 hypothesis test) but uses
a HuggingFace endpoint via requests instead of the Together.ai/OpenAI client.

Usage:
    python src/llava.py \\
        --data data/test-A/SPIQA_testA_part1_output_latest.json \\
        --images data/test-A/SPIQA_testA_Images \\
        --output data/eval/llava_error_type_analysis

    # Smoke test with 3 examples per error type:
    python src/llava.py \\
        --data data/test-A/SPIQA_testA_part1_output_latest.json \\
        --images data/test-A/SPIQA_testA_Images \\
        --output data/eval/llava_error_type_analysis \\
        --max-per-type 3
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

sys.path.insert(0, str(Path(__file__).parent))
from eval_utils import parse_eval_output, to_data_url, SYSTEM_PROMPT

# ENDPOINT = "https://fq17s6jmw5bp3w95.us-east-1.aws.endpoints.huggingface.cloud"  # llama3
ENDPOINT = "https://lwoz4ljbkw9bu0zk.us-east-1.aws.endpoints.huggingface.cloud"   # mistral


def run_inference(messages: List[Dict[str, Any]]) -> str:
    """Post a chat-completion request to the HuggingFace endpoint."""
    response = requests.post(
        f"{ENDPOINT}/v1/chat/completions",
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
        json={"messages": messages},
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"] or ""


def load_all_examples(
    output_json_paths: List[str | Path],
    images_root: str | Path,
) -> List[Dict[str, Any]]:
    """Load ALL examples from augmented SPIQA output JSONs (no sampling)."""
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
                    "ground_truth": {
                        "verdict": verdict,
                        "error_category": error_category,
                        "feedback": feedback,
                    },
                })

    return examples


def filter_error_examples(examples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Filter to error examples only and group by error category."""
    by_error_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for ex in examples:
        verdict = ex["ground_truth"]["verdict"].lower()
        if verdict == "correct":
            continue

        error_cat = ex["ground_truth"]["error_category"].lower()
        if error_cat in ("factual", "conceptual", "omission"):
            by_error_type[error_cat].append(ex)

    return dict(by_error_type)


def evaluate_text_only(example: dict) -> Tuple[dict, str]:
    """Scenario C1: text_only (Q + Student Answer)."""
    user_text = f"Question:\n{example['question']}\nStudent Answer:\n{example['student']}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": user_text}]},
    ]
    output_text = run_inference(messages)
    return parse_eval_output(output_text), output_text


def evaluate_caption_only(example: dict) -> Tuple[dict, str]:
    """Scenario C2: caption_only (Q + Caption + Student Answer)."""
    user_text = (
        f"Caption:\n{example['caption']}\n"
        f"Question:\n{example['question']}\n"
        f"Student Answer:\n{example['student']}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": user_text}]},
    ]
    output_text = run_inference(messages)
    return parse_eval_output(output_text), output_text


def evaluate_vision_only(example: dict) -> Tuple[dict, str]:
    """Scenario C3: vision_only (Q + Image + Student Answer)."""
    user_text = f"Question:\n{example['question']}\nStudent Answer:\n{example['student']}"
    data_url = to_data_url(example["image_path"])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]
    output_text = run_inference(messages)
    return parse_eval_output(output_text), output_text


def evaluate_multimodal(example: dict) -> Tuple[dict, str]:
    """Scenario C4: multimodal (Q + Caption + Image + Student Answer)."""
    user_text = (
        f"Caption:\n{example['caption']}\n"
        f"Question:\n{example['question']}\n"
        f"Student Answer:\n{example['student']}"
    )
    data_url = to_data_url(example["image_path"])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]
    output_text = run_inference(messages)
    return parse_eval_output(output_text), output_text


SCENARIOS = {
    "text_only": evaluate_text_only,
    "caption_only": evaluate_caption_only,
    "vision_only": evaluate_vision_only,
    "multimodal": evaluate_multimodal,
}


def run_evaluation(
    examples_by_type: Dict[str, List[Dict[str, Any]]],
    max_per_type: int | None = None,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Run all 4 scenarios on each error type subset.

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

        for scenario_name, eval_func in SCENARIOS.items():
            print(f"\n  Scenario: {scenario_name}", flush=True)
            scenario_results = []

            for i, example in enumerate(examples, 1):
                print(f"    [{i}/{len(examples)}] paper={example['paper_id']} "
                      f"gt={example['ground_truth']['verdict']}", end="", flush=True)
                try:
                    predicted, raw_output = eval_func(example)
                    scenario_results.append({
                        **example,
                        "predicted": predicted,
                        "raw_output": raw_output,
                    })
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
    """Returns: (correct_count, total_count, accuracy_percentage)."""
    correct = sum(
        1 for r in results
        if r["predicted"]["verdict"].lower() == r["ground_truth"]["verdict"].lower()
    )
    total = len(results)
    accuracy = (correct / total * 100) if total > 0 else 0.0
    return correct, total, accuracy


def analyze_results(
    results: Dict[str, Dict[str, List[Dict[str, Any]]]]
) -> Dict[str, Any]:
    """Analyze results and test H3 hypothesis."""
    accuracy_matrix: Dict[str, Dict[str, float]] = {}
    counts_matrix: Dict[str, Dict[str, Tuple[int, int]]] = {}

    for error_type, scenarios in results.items():
        accuracy_matrix[error_type] = {}
        counts_matrix[error_type] = {}
        for scenario_name, scenario_results in scenarios.items():
            correct, total, accuracy = calculate_accuracy(scenario_results)
            accuracy_matrix[error_type][scenario_name] = accuracy
            counts_matrix[error_type][scenario_name] = (correct, total)

    context_benefit: Dict[str, float] = {}
    for error_type in accuracy_matrix:
        if "multimodal" in accuracy_matrix[error_type] and "text_only" in accuracy_matrix[error_type]:
            context_benefit[error_type] = (
                accuracy_matrix[error_type]["multimodal"] - accuracy_matrix[error_type]["text_only"]
            )

    h3_part1_passed = False
    factual_avg = conceptual_avg = None
    if "factual" in accuracy_matrix and "conceptual" in accuracy_matrix:
        factual_avg = sum(accuracy_matrix["factual"].values()) / len(accuracy_matrix["factual"])
        conceptual_avg = sum(accuracy_matrix["conceptual"].values()) / len(accuracy_matrix["conceptual"])
        h3_part1_passed = factual_avg > conceptual_avg

    h3_part2_passed = False
    if "factual" in context_benefit and "conceptual" in context_benefit:
        h3_part2_passed = context_benefit["conceptual"] > context_benefit["factual"]

    return {
        "accuracy_matrix": accuracy_matrix,
        "counts_matrix": counts_matrix,
        "context_benefit": context_benefit,
        "h3_part1": {
            "passed": h3_part1_passed,
            "factual_avg": factual_avg,
            "conceptual_avg": conceptual_avg,
        },
        "h3_part2": {
            "passed": h3_part2_passed,
            "delta_factual": context_benefit.get("factual"),
            "delta_conceptual": context_benefit.get("conceptual"),
        },
    }


def print_summary(analysis: Dict[str, Any]) -> None:
    """Print summary tables to console."""
    print("\n" + "=" * 70)
    print("ACCURACY MATRIX (Error Type × Scenario)")
    print("=" * 70)

    scenarios = ["text_only", "caption_only", "vision_only", "multimodal"]
    error_types = ["factual", "conceptual", "omission"]

    print(f"{'Error Type':<15}", end="")
    for s in scenarios:
        print(f"{s:>15}", end="")
    print(f"{'Avg':>10}")

    for et in error_types:
        if et not in analysis["accuracy_matrix"]:
            continue
        print(f"{et:<15}", end="")
        row_values = []
        for s in scenarios:
            acc = analysis["accuracy_matrix"][et].get(s, 0)
            print(f"{acc:>12.1f}%", end="  ")
            row_values.append(acc)
        avg = sum(row_values) / len(row_values) if row_values else 0
        print(f"{avg:>8.1f}%")

    print("\n" + "=" * 70)
    print("CONTEXT BENEFIT (Δ = multimodal - text_only)")
    print("=" * 70)

    for et in error_types:
        if et in analysis["context_benefit"]:
            delta = analysis["context_benefit"][et]
            sign = "+" if delta >= 0 else ""
            print(f"  {et:<15}: {sign}{delta:.1f}pp")

    print("\n" + "=" * 70)
    print("H3 HYPOTHESIS TEST")
    print("=" * 70)

    h3_p1 = analysis["h3_part1"]
    h3_p2 = analysis["h3_part2"]

    print(f"\nPart 1: Factual errors detected more reliably than conceptual?")
    print(f"  Factual avg:     {h3_p1['factual_avg']:.1f}%")
    print(f"  Conceptual avg:  {h3_p1['conceptual_avg']:.1f}%")
    print(f"  Result:          {'PASS' if h3_p1['passed'] else 'FAIL'}")

    print(f"\nPart 2: Context helps conceptual more than factual?")
    print(f"  Δ_factual:     {h3_p2['delta_factual']:+.1f}pp")
    print(f"  Δ_conceptual:  {h3_p2['delta_conceptual']:+.1f}pp")
    print(f"  Result:          {'PASS' if h3_p2['passed'] else 'FAIL'}")

    print(f"\n{'='*70}")
    overall = (
        "PASS" if (h3_p1["passed"] and h3_p2["passed"])
        else "PARTIAL" if (h3_p1["passed"] or h3_p2["passed"])
        else "FAIL"
    )
    print(f"H3 OVERALL: {overall}")
    print(f"{'='*70}")


def save_markdown_report(analysis: Dict[str, Any], output_dir: Path) -> None:
    """Save markdown report."""
    report_path = output_dir / "error_type_analysis.md"

    scenarios = ["text_only", "caption_only", "vision_only", "multimodal"]
    error_types = ["factual", "conceptual", "omission"]

    lines = [
        "# Error Type Analysis: H3 Hypothesis Test",
        "",
        "## Hypothesis",
        "",
        "**H3:** Factual errors detected more reliably than conceptual errors,",
        "and visual context helps conceptual errors more than factual errors.",
        "",
        "## Accuracy Matrix",
        "",
        "| Error Type | text_only | caption_only | vision_only | multimodal | Avg |",
        "|------------|-----------|--------------|-------------|------------|-----|",
    ]

    for et in error_types:
        if et not in analysis["accuracy_matrix"]:
            continue
        row = [f"| {et}"]
        row_values = []
        for s in scenarios:
            acc = analysis["accuracy_matrix"][et].get(s, 0)
            counts = analysis["counts_matrix"][et].get(s, (0, 0))
            row.append(f"{acc:.1f}% ({counts[0]}/{counts[1]})")
            row_values.append(acc)
        avg = sum(row_values) / len(row_values) if row_values else 0
        row.append(f"{avg:.1f}%")
        row.append("|")
        lines.append(" | ".join(row))

    lines.extend([
        "",
        "## Context Benefit (Δ = multimodal - text_only)",
        "",
        "| Error Type | Δ (pp) |",
        "|------------|--------|",
    ])

    for et in error_types:
        if et in analysis["context_benefit"]:
            delta = analysis["context_benefit"][et]
            sign = "+" if delta >= 0 else ""
            lines.append(f"| {et} | {sign}{delta:.1f} |")

    h3_p1 = analysis["h3_part1"]
    h3_p2 = analysis["h3_part2"]

    lines.extend([
        "",
        "## H3 Hypothesis Test",
        "",
        "### Part 1: Factual > Conceptual?",
        "",
        f"- Factual avg: {h3_p1['factual_avg']:.1f}%",
        f"- Conceptual avg: {h3_p1['conceptual_avg']:.1f}%",
        f"- **Result: {'PASS' if h3_p1['passed'] else 'FAIL'}**",
        "",
        "### Part 2: Δ_conceptual > Δ_factual?",
        "",
        f"- Δ_factual: {h3_p2['delta_factual']:+.1f}pp",
        f"- Δ_conceptual: {h3_p2['delta_conceptual']:+.1f}pp",
        f"- **Result: {'PASS' if h3_p2['passed'] else 'FAIL'}**",
        "",
        "---",
        "",
    ])

    overall = (
        "PASS" if (h3_p1["passed"] and h3_p2["passed"])
        else "PARTIAL" if (h3_p1["passed"] or h3_p2["passed"])
        else "FAIL"
    )
    lines.append(f"**H3 Overall: {overall}**")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved report to {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate by error type (H3) using HuggingFace endpoint"
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
        "--max-per-type",
        type=int,
        default=None,
        help="Maximum examples per error type (for smoke testing)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1: Loading all examples...", flush=True)
    all_examples = load_all_examples(args.data, args.images)
    print(f"  Loaded {len(all_examples)} total examples", flush=True)

    print("\nStep 2: Filtering to error examples...", flush=True)
    examples_by_type = filter_error_examples(all_examples)
    total_errors = sum(len(v) for v in examples_by_type.values())
    print(f"  Error examples: {total_errors}", flush=True)
    for et, exs in sorted(examples_by_type.items()):
        pct = len(exs) / total_errors * 100 if total_errors > 0 else 0
        print(f"    {et}: {len(exs)} ({pct:.0f}%)", flush=True)

    print("\nSteps 3-4: Running 4 scenarios per error type...", flush=True)
    results = run_evaluation(examples_by_type, max_per_type=args.max_per_type)

    print("\nSteps 5-6: Calculating accuracy and context benefit...")
    analysis = analyze_results(results)

    print("\nStep 7: Testing H3 hypothesis...")
    print_summary(analysis)

    print("\nStep 8: Saving results...")

    results_path = output_dir / "error_type_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "analysis": analysis}, f, indent=2, ensure_ascii=False)
    print(f"  Saved full results to {results_path}")

    save_markdown_report(analysis, output_dir)


if __name__ == "__main__":
    main()
