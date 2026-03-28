#!/usr/bin/env python3
"""Modification 4: Full N=108 C4-CoT Evaluation by Error Type

Runs C4-CoT (multimodal + visual chain-of-thought) across all 108 non-correct
examples grouped by error type (factual, conceptual, omission).

Unlike eval_by_error_type.py which runs all 4 scenarios (C1-C4), this script
runs only the C4-CoT condition (multimodal + CoT prompt) since we are comparing
against the existing C4 baseline from the 32B error_type_results.json.

Output format matches the existing error_type_results.json structure so that
compute_pc_recall_fsr.py and compute_vgs.py can consume it with --cot flag.

Usage:
    python src/eval_cot.py \\
        --data data/test-A/SPIQA_testA_part1_output_latest.json \\
        --images data/test-A/SPIQA_testA_Images \\
        --output data/eval/cot_analysis

    # Smoke test with 3 examples per error type:
    python src/eval_cot.py \\
        --data data/test-A/SPIQA_testA_part1_output_latest.json \\
        --images data/test-A/SPIQA_testA_Images \\
        --output data/eval/cot_analysis \\
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
from eval_utils import parse_eval_output_cot, to_data_url, SYSTEM_PROMPT_COT

MODEL = "shahkhubi_e8d4/Qwen/Qwen3-VL-32B-Instruct-6a8179a2"


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


def evaluate_cot(client: openai.OpenAI, example: dict) -> Tuple[dict, str]:
    """C4-CoT: multimodal + visual CoT (Q + Caption + Image + Student Answer + CoT prompt)."""
    caption = example["caption"]
    question = example["question"]
    student = example["student"]
    image_path = example["image_path"]

    data_url = to_data_url(image_path)
    text = f"/no_think\nCaption:\n{caption}\nQuestion:\n{question}\nStudent Answer:\n{student}"

    user_content = [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_COT},
            {"role": "user", "content": user_content},
        ],
    )

    output_text = response.choices[0].message.content or ""
    return parse_eval_output_cot(output_text), output_text


def run_evaluation(
    client: openai.OpenAI,
    examples_by_type: Dict[str, List[Dict[str, Any]]],
    max_per_type: int | None = None,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Run C4-CoT on each error type subset.

    Returns: {error_type: {"multimodal": [results]}}
    Output structure matches eval_by_error_type.py so metrics scripts work.
    """
    results: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for error_type, examples in examples_by_type.items():
        if max_per_type and len(examples) > max_per_type:
            examples = examples[:max_per_type]

        print(f"\n{'='*60}", flush=True)
        print(f"ERROR TYPE: {error_type.upper()} ({len(examples)} examples)", flush=True)
        print(f"{'='*60}", flush=True)

        scenario_results = []

        for i, example in enumerate(examples, 1):
            print(
                f"  [{i}/{len(examples)}] paper={example['paper_id']} "
                f"gt={example['ground_truth']['verdict']}",
                end="",
                flush=True,
            )
            try:
                predicted, raw_output = evaluate_cot(client, example)
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
                    "predicted": {
                        "visual_analysis": "",
                        "verdict": "",
                        "error_category": "N/A",
                        "feedback": "N/A",
                    },
                    "raw_output": f"ERROR: {e}",
                })

        # Store under "multimodal" key so metrics scripts can access it
        results[error_type] = {"multimodal": scenario_results}

    return results


def calculate_accuracy(results: List[Dict[str, Any]]) -> Tuple[int, int, float]:
    """Calculate verdict accuracy for a list of results."""
    correct = sum(
        1 for r in results
        if r["predicted"]["verdict"].lower() == r["ground_truth"]["verdict"].lower()
    )
    total = len(results)
    accuracy = (correct / total * 100) if total > 0 else 0.0
    return correct, total, accuracy


def print_summary(results: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> None:
    """Print summary table to console."""
    print("\n" + "=" * 60)
    print("C4-CoT VERDICT ACCURACY BY ERROR TYPE")
    print("=" * 60)

    error_types = ["factual", "conceptual", "omission"]

    print(f"{'Error Type':<15} {'Correct':>10} {'Total':>8} {'Accuracy':>10}")
    print("-" * 45)

    total_correct = 0
    total_count = 0

    for et in error_types:
        if et not in results:
            continue
        correct, total, accuracy = calculate_accuracy(results[et]["multimodal"])
        total_correct += correct
        total_count += total
        print(f"{et:<15} {correct:>10} {total:>8} {accuracy:>9.1f}%")

    if total_count > 0:
        overall = total_correct / total_count * 100
        print("-" * 45)
        print(f"{'OVERALL':<15} {total_correct:>10} {total_count:>8} {overall:>9.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mod 4: Full N=108 C4-CoT evaluation by error type"
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

    client = build_client()

    # Step 1: Load all examples
    print("Loading all examples...", flush=True)
    all_examples = load_all_examples(args.data, args.images)
    print(f"  Loaded {len(all_examples)} total examples", flush=True)

    # Step 2: Filter to error examples and group
    print("\nFiltering to error examples...", flush=True)
    examples_by_type = filter_error_examples(all_examples)
    total_errors = sum(len(v) for v in examples_by_type.values())
    print(f"  Error examples: {total_errors}", flush=True)
    for et, exs in sorted(examples_by_type.items()):
        pct = len(exs) / total_errors * 100 if total_errors > 0 else 0
        print(f"    {et}: {len(exs)} ({pct:.0f}%)", flush=True)

    # Step 3: Run C4-CoT evaluation
    print("\nRunning C4-CoT evaluation...", flush=True)
    results = run_evaluation(client, examples_by_type, max_per_type=args.max_per_type)

    # Print summary
    print_summary(results)

    # Save full results JSON (same structure as error_type_results.json)
    results_path = output_dir / "error_type_results.json"

    analysis = {}
    for et in results:
        correct, total, accuracy = calculate_accuracy(results[et]["multimodal"])
        analysis[et] = {"multimodal": accuracy}

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "results": results,
            "analysis": {"accuracy_matrix": analysis},
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()
