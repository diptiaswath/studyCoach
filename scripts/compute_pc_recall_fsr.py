#!/usr/bin/env python3
"""
Compute PC Recall and FSR (Feedback Suppression Rate) for Qwen3-VL models.

Source Files:
- 8B: data/eval/error_type_analysis/error_type_results.json
- 32B: data/eval/qwen3_32b/error_type_analysis/error_type_results.json

These files contain H3 evaluation results (N=108 non-correct examples):
- 52 factual errors
- 41 conceptual errors
- 15 omission errors

Ground truth verdicts in these files are either "partially correct" or "incorrect"
(correct examples are excluded since they don't need coaching).

Metrics:
- PC Recall: TP_PC / (TP_PC + FN_PC) on 61 ground-truth PC examples
  Measures: Can the model recognize incomplete (vs fully wrong/right) answers?

- FSR: # Correct predictions on non-correct GT / 108
  Measures: Does the model suppress feedback when coaching is needed?
  Lower is better - high FSR means model incorrectly accepts wrong answers.

Usage:
    python scripts/compute_pc_recall_fsr.py [--model 8b|32b|both]
"""

import json
import argparse
import os

def compute_metrics(data_path: str, model_name: str) -> dict:
    """Compute PC Recall and FSR from error_type_results.json."""

    with open(data_path, 'r') as f:
        data = json.load(f)

    conditions = ['text_only', 'caption_only', 'vision_only', 'multimodal']
    error_types = ['factual', 'conceptual', 'omission']

    results = {}

    for cond in conditions:
        # Collect all examples for this condition
        all_examples = []
        for error_type in error_types:
            all_examples.extend(data['results'][error_type][cond])

        # Count for PC Recall (examples where GT is "partially correct")
        pc_examples = [ex for ex in all_examples
                       if ex['ground_truth']['verdict'].lower() == 'partially correct']
        tp_pc = sum(1 for ex in pc_examples
                    if ex['predicted']['verdict'].lower() == 'partially correct')
        fn_pc = len(pc_examples) - tp_pc
        pc_recall = tp_pc / len(pc_examples) * 100 if pc_examples else 0

        # Count for FSR (model predicts "Correct" on non-correct GT)
        correct_predictions = sum(1 for ex in all_examples
                                  if ex['predicted']['verdict'].lower() == 'correct')
        fsr = correct_predictions / len(all_examples) * 100

        results[cond] = {
            'total_examples': len(all_examples),
            'pc_examples': len(pc_examples),
            'tp_pc': tp_pc,
            'fn_pc': fn_pc,
            'pc_recall': pc_recall,
            'correct_predictions': correct_predictions,
            'fsr': fsr
        }

    return results


def print_results(results: dict, model_name: str, data_path: str):
    """Print formatted results."""

    print("=" * 70)
    print(f"{model_name}: PC Recall and FSR Computation")
    print(f"Source: {data_path}")
    print("=" * 70)
    print()

    for cond, metrics in results.items():
        print(f"{cond}:")
        print(f"  Total examples: {metrics['total_examples']}")
        print(f"  PC examples (GT): {metrics['pc_examples']}")
        print(f"  TP_PC (predicted PC when GT=PC): {metrics['tp_pc']}")
        print(f"  FN_PC (predicted other when GT=PC): {metrics['fn_pc']}")
        print(f"  PC Recall: {metrics['pc_recall']:.1f}%")
        print(f"  Correct predictions (suppressed): {metrics['correct_predictions']}")
        print(f"  FSR: {metrics['fsr']:.1f}%")
        print()

    # Summary table
    print("=" * 70)
    print("Summary Table")
    print("=" * 70)
    print(f"{'Condition':<15} {'PC Recall':>12} {'FSR':>12}")
    print("-" * 40)

    for cond, metrics in results.items():
        print(f"{cond:<15} {metrics['pc_recall']:>11.1f}% {metrics['fsr']:>11.1f}%")

    print()


def main():
    parser = argparse.ArgumentParser(description='Compute PC Recall and FSR metrics')
    parser.add_argument('--model', choices=['8b', '32b', 'both'], default='both',
                        help='Which model to compute metrics for (default: both)')
    args = parser.parse_args()

    # Define data paths
    paths = {
        '8b': ('data/eval/error_type_analysis/error_type_results.json', 'Qwen3-VL-8B'),
        '32b': ('data/eval/qwen3_32b/error_type_analysis/error_type_results.json', 'Qwen3-VL-32B')
    }

    models_to_run = ['8b', '32b'] if args.model == 'both' else [args.model]

    for model in models_to_run:
        data_path, model_name = paths[model]

        if not os.path.exists(data_path):
            print(f"Warning: {data_path} not found, skipping {model_name}")
            continue

        results = compute_metrics(data_path, model_name)
        print_results(results, model_name, data_path)


if __name__ == '__main__':
    main()
