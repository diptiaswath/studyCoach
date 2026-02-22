"""Scenario 1: Text-only incongruence detection using Qwen3-VL-8B-Instruct (Together.ai).

No caption, no image — the model sees only the question, reference answer, and student answer.
This is the baseline that isolates how much the model can infer from text alone,
without any visual or descriptive information about the figure.

Usage:
    python src/eval_text_only.py \\
        --data data/test-A/SPIQA_testA_part1_output_latest.json \\
        --images data/test-A/SPIQA_testA_Images \\
        --output data/eval/text_only_results.json

    # Smoke test with 3 examples:
    python src/eval_text_only.py \\
        --data data/test-A/SPIQA_testA_part1_output_latest.json \\
        --images data/test-A/SPIQA_testA_Images \\
        --output data/eval/text_only_results.json \\
        --max 3
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import openai

sys.path.insert(0, str(Path(__file__).parent))
from eval_utils import load_examples, parse_eval_output, save_results, SYSTEM_PROMPT

MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def build_client() -> openai.OpenAI:
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


def evaluate_example(client: openai.OpenAI, example: dict, include_answer: bool = True) -> dict:
    question = example["question"]
    answer = example["answer"]
    student = example["student"]

    user_text = "/no_think\n"
    if include_answer:
        user_text += f"Reference Answer:\n{answer}\n"
    user_text += f"Question:\n{question}\nStudent Answer:\n{student}"

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [{"type": "text", "text": user_text}],
            },
        ],
    )

    output_text = response.choices[0].message.content or ""
    predicted = parse_eval_output(output_text)

    return {
        "paper_id": example["paper_id"],
        "question": example["question"],
        "answer": example["answer"],
        "caption": example["caption"],
        "image_path": example["image_path"],
        "student": example["student"],
        "ground_truth": example["ground_truth"],
        "predicted": predicted,
        "raw_output": output_text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scenario 1: Text-only incongruence detection via Together.ai"
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
        help="Path to write results JSON",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=50,
        help="Maximum number of examples to evaluate (default: 50)",
    )
    parser.add_argument(
        "--no-answer",
        action="store_true",
        help="Omit the reference answer from the prompt",
    )
    args = parser.parse_args()

    client = build_client()

    print(f"Loading examples from: {args.data}")
    examples = load_examples(args.data, args.images, max_examples=args.max)
    print(f"Loaded {len(examples)} examples\n")

    results = []
    for i, example in enumerate(examples, 1):
        print(f"[{i}/{len(examples)}] paper={example['paper_id']} "
              f"gt={example['ground_truth']['verdict']}")
        try:
            result = evaluate_example(client, example, include_answer=not args.no_answer)
            results.append(result)
            print(f"  -> predicted: {result['predicted']['verdict']} / "
                  f"{result['predicted']['error_category']}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                **{k: example[k] for k in
                   ("paper_id", "question", "caption", "image_path", "student", "ground_truth")},
                "predicted": {"verdict": "", "error_category": "N/A", "feedback": "N/A"},
                "raw_output": f"ERROR: {e}",
            })

    save_results(results, args.output)

    correct = sum(
        1 for r in results
        if r["predicted"]["verdict"].lower() == r["ground_truth"]["verdict"].lower()
    )
    print(f"\nVerdict accuracy: {correct}/{len(results)} "
          f"({100*correct/len(results):.1f}%)")


if __name__ == "__main__":
    main()
