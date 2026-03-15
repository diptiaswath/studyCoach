import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.api.claude_client import ask_claude, DEFAULT_MODEL

def read_json(path: Path) -> List[Dict[str, Any]]:
    """Your eval files are JSON arrays: [ {...}, {...} ]"""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path.name} is not a JSON list.")
    return data


def write_json(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def safe_get(rec: Dict[str, Any], keys: List[str], default: str = "") -> str:
    cur: Any = rec
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur if isinstance(cur, str) else default


def build_prompt(question: str, student: str, feedback_a: str, feedback_b: str) -> str:
    return f"""
You are evaluating AI-generated student feedback against educator-written reference feedback.

Background:
- A student answered a question.
- Feedback A is ground-truth feedback written by an educator.
- Feedback B is AI-generated feedback.
- Note: The original question may reference figures or tables that are not available here. 
  If feedback references visual content you cannot see, factor that uncertainty into your confidence score.

Student context:
Question: {question}
Student answer: {student}

Feedback A (ground truth):
{feedback_a}

Feedback B (generated):
{feedback_b}

Task:
Judge whether Feedback B provides equivalent guidance to Feedback A, considering:
1) Semantic meaning (same diagnosis of what’s right/wrong)
2) Key points (covers the main correction(s))
3) Helpfulness (would the student learn the same thing)

Labels:
- match: equivalent meaning and key points
- partial: overlaps but misses some key points or is incomplete
- unmatched: different meaning or misses the main point

Return JSON only (no extra text):
{{
  "label": "match" | "partial" | "unmatched",
  "confidence": "high" | "medium" | "low",
  "rationale": "short explanation"
}}
""".strip()


def parse_json_safely(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    # Best-effort extraction if Claude adds extra whitespace/text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", default="data/eval", help="Directory with eval JSON files")
    parser.add_argument("--out_dir", default="data/llm_judged", help="Directory to save judged outputs")
    parser.add_argument(
        "--pattern",
        default="*_no_answer_results.json",
        help="Only process files matching this glob (default: *_no_answer_results.json)",
    )
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep between API calls (seconds)")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip files that already have outputs in out_dir",
    )
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files found in {in_dir} matching {args.pattern}")

    for fp in files:
        out_path = out_dir / fp.name.replace(".json", ".llmjudged.json")

        if args.resume and out_path.exists():
            print(f"Skipping (already exists): {out_path.name}")
            continue

        records = read_json(fp)
        if not records:
            print(f"Skipping empty file: {fp.name}")
            continue

        print(f"Judging: {fp.name} -> {out_path.name} (n={len(records)})")

        for i, rec in enumerate(records, start=1):
            question = rec.get("question", "") or ""
            ref_answer = rec.get("answer", "") or ""

            gt_feedback = safe_get(rec, ["ground_truth", "feedback"])
            pred_feedback = safe_get(rec, ["predicted", "feedback"])

            prompt = build_prompt(question, ref_answer, gt_feedback, pred_feedback)

            raw = ask_claude(prompt)  # model is defaulted in your ask_claude()
            parsed = parse_json_safely(raw)

            rec["llm_judge"] = {
                "provider": "anthropic",
                "model": DEFAULT_MODEL,
                "label": parsed.get("label") if parsed else None,
                "confidence": parsed.get("confidence") if parsed else None,
                "rationale": parsed.get("rationale") if parsed else None,
                "raw_output": raw,
            }

            if i % 10 == 0:
                print(f"  judged {i}/{len(records)}")

            time.sleep(args.sleep)

        write_json(out_path, records)
        print(f"Saved: {out_path}\n")


if __name__ == "__main__":
    main()