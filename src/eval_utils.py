"""Shared utilities for multimodal incongruence inference evaluation.

Provides:
- load_examples(): equal-stratum sampling from augmented SPIQA dataset
- parse_eval_output(): extract Verdict/Error Category/Feedback from model output
- to_data_url(): base64-encode image as data URL
- save_results(): write results JSON
"""
from __future__ import annotations

import base64
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Shared system prompt (all 3 scenarios) — loaded from prompts/ at import time
# ---------------------------------------------------------------------------
_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
SYSTEM_PROMPT = (_PROMPTS_DIR / "incongruence_eval_v3.txt").read_text(encoding="utf-8").strip()


def load_examples(
    output_json_paths: List[str | Path],
    images_root: str | Path,
    max_examples: int = 50,
) -> List[Dict[str, Any]]:
    """Load and equal-stratum-sample examples from augmented SPIQA output JSONs.

    Samples as evenly as possible across (verdict × error_category) strata,
    allocating budget greedily from smallest-pool strata first so every stratum
    gets as close to an equal share as the data allows.

    Args:
        output_json_paths: One or more paths to *_output.json files.
        images_root: Root directory containing per-paper image subdirectories.
        max_examples: Maximum number of examples to return (default 50).

    Returns:
        List of example dicts with keys: paper_id, question, answer, caption,
        image_path, student, ground_truth (verdict/error_category/feedback).
    """
    images_root = Path(images_root)

    # Collect all valid examples grouped by stratum
    strata: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)

    for path in output_json_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for paper_key, paper in data.items():
            all_figures = paper.get("all_figures", {})
            for qa in paper.get("qa", []):
                student = (qa.get("student") or "").strip()
                if not student:
                    continue

                verdict = (qa.get("verdict") or "").strip().lower()
                error_category = (qa.get("error_category") or "").strip().lower()
                feedback = (qa.get("feedback") or "").strip()
                question = (qa.get("question") or "").strip()
                answer = (qa.get("answer") or "").strip()
                reference = (qa.get("reference") or "").strip()

                fig_details = all_figures.get(reference, {})
                caption = (fig_details.get("caption") or "").strip()
                image_path = images_root / paper_key / reference

                strata[(verdict, error_category)].append(
                    {
                        "paper_id": paper_key,
                        "question": question,
                        "answer": answer,
                        "caption": caption,
                        "image_path": str(image_path),
                        "student": student,
                        "ground_truth": {
                            "verdict": qa.get("verdict", "").strip(),
                            "error_category": qa.get("error_category", "").strip(),
                            "feedback": feedback,
                        },
                    }
                )

    # Equal-stratum sampling: allocate budget greedily from smallest pools first
    # so no stratum is over-represented and every stratum gets as close to an
    # equal share as the data allows.
    rng = random.Random(RANDOM_SEED)
    sampled: List[Dict[str, Any]] = []

    budget = max_examples
    # Sort strata smallest-pool first so constrained strata are satisfied first
    sorted_strata = sorted(strata.items(), key=lambda kv: len(kv[1]))
    for i, (stratum, pool) in enumerate(sorted_strata):
        remaining_strata = len(sorted_strata) - i
        target = min(len(pool), budget // remaining_strata)
        chosen = rng.sample(pool, target)
        sampled.extend(chosen)
        budget -= target

    rng.shuffle(sampled)
    return sampled[:max_examples]


def parse_eval_output(output_text: str) -> Dict[str, str]:
    """Extract Verdict, Error Category, Feedback from model output.

    Mirrors the logic in icl.py:parse_inference_output() but adapted for
    eval responses (no Student: / Agent: framing expected).
    """
    text = output_text.replace("\r\n", "\n").replace("\r", "\n")

    # The eval scripts don't wrap in Student:/Agent: blocks, but handle both.
    lower = text.lower()
    agent_idx = lower.find("agent:")
    agent_block = text[agent_idx + len("agent:"):].strip() if agent_idx != -1 else text

    verdict = ""
    error_category = ""
    feedback = ""

    verdict_match = re.search(r"Verdict\s*=\s*([^\n]+)", agent_block)
    error_match = re.search(r"Error Category\s*=\s*([^\n]+)", agent_block)
    feedback_match = re.search(
        r"Feedback\s*=\s*(.*?)(?:\n[A-Za-z ]+\s*=|\Z)", agent_block, re.DOTALL
    )

    if verdict_match:
        verdict = re.sub(r"\s+", " ", verdict_match.group(1).strip())
    if error_match:
        error_category = re.sub(r"\s+", " ", error_match.group(1).strip())
    if feedback_match:
        feedback = feedback_match.group(1).strip()

    if verdict.lower() == "correct":
        feedback = "N/A"
        error_category = "N/A"

    return {
        "verdict": verdict,
        "error_category": error_category if error_category else "N/A",
        "feedback": feedback if feedback else "N/A",
    }


def to_data_url(image_path: str | Path) -> str:
    """Base64-encode an image file and return as a data URL.

    Uses image/png MIME type — SPIQA images are .png files.
    """
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def save_results(results: List[Dict[str, Any]], output_path: str | Path) -> None:
    """Write results to a JSON file, creating parent directories as needed."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} results to {output_path}")
