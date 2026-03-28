"""Modification 4: Multimodal + Visual CoT incongruence detection via Together.ai.

Model: Qwen3-VL-32B with visual chain-of-thought prompt.
Input to model: question + image + caption + student answer + CoT system prompt.
The CoT prompt forces explicit visual analysis before verdict.

Usage:
    python src/eval_multimodal_cot.py \
        --data data/test-A/SPIQA_testA_part1_output_latest.json \
        --images data/test-A/SPIQA_testA_Images \
        --output data/eval/cot_smoke_test.json

    # Smoke test with 3 examples:
    python src/eval_multimodal_cot.py \
        --data data/test-A/SPIQA_testA_part1_output_latest.json \
        --images data/test-A/SPIQA_testA_Images \
        --output data/eval/cot_smoke_test.json \
        --max 3
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import openai

sys.path.insert(0, str(Path(__file__).parent))
from eval_utils import (
    load_examples,
    parse_eval_output_cot,
    save_results,
    to_data_url,
    SYSTEM_PROMPT_COT,
)

MODEL = "shahkhubi_e8d4/Qwen/Qwen3-VL-32B-Instruct-6a8179a2"


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


def evaluate_example(client: openai.OpenAI, example: dict) -> dict:
    """Run C4-CoT: multimodal + visual CoT prompt."""
    caption = example["caption"]
    question = example["question"]
    answer = example["answer"]
    student = example["student"]
    image_path = example["image_path"]

    data_url = to_data_url(image_path)

    text = (
        f"/no_think\n"
        f"Caption:\n{caption}\n"
        f"Reference Answer:\n{answer}\n"
        f"Question:\n{question}\n"
        f"Student Answer:\n{student}"
    )

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
    predicted = parse_eval_output_cot(output_text)

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
        description="Mod 4: Multimodal + Visual CoT incongruence detection"
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
    args = parser.parse_args()

    client = build_client()

    print(f"Loading examples from: {args.data}")
    examples = load_examples(args.data, args.images, max_examples=args.max)
    print(f"Loaded {len(examples)} examples\n")

    results = []
    for i, example in enumerate(examples, 1):
        print(
            f"[{i}/{len(examples)}] paper={example['paper_id']} "
            f"gt={example['ground_truth']['verdict']}"
        )
        try:
            result = evaluate_example(client, example)
            results.append(result)
            va_preview = result["predicted"].get("visual_analysis", "")[:60]
            print(
                f"  -> predicted: {result['predicted']['verdict']} / "
                f"{result['predicted']['error_category']}"
            )
            print(f"  -> VA: {va_preview}...")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append(
                {
                    **{
                        k: example[k]
                        for k in (
                            "paper_id",
                            "question",
                            "caption",
                            "image_path",
                            "student",
                            "ground_truth",
                        )
                    },
                    "predicted": {
                        "visual_analysis": "",
                        "verdict": "",
                        "error_category": "N/A",
                        "feedback": "N/A",
                    },
                    "raw_output": f"ERROR: {e}",
                }
            )

    save_results(results, args.output)

    correct = sum(
        1
        for r in results
        if r["predicted"]["verdict"].lower()
        == r["ground_truth"]["verdict"].lower()
    )
    print(
        f"\nVerdict accuracy: {correct}/{len(results)} "
        f"({100 * correct / len(results):.1f}%)"
    )


if __name__ == "__main__":
    main()
