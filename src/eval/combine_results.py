"""
combine_results.py

Merges human annotation, LLM judge, and automated metrics (F1/ROUGE-L/BLEU)
into a single summary table per scenario.

Usage:
    python -m src.eval.combine_results \
        --human_dir data/human_annotated/qwen3_32b \
        --llm_dir   data/llm_judged/qwen3_32b \
        --scored_dir data/eval_scored/qwen3_32b \
        --out_dir   data/eval_summary/qwen3_32b \
        --run_name  qwen3_32b
"""

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def read_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def avg(values: List[Optional[float]]) -> Optional[float]:
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 4) if values else None


def human_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    match = sum(1 for r in records if r.get("human_match") == 1)
    partial = sum(1 for r in records if r.get("human_match") == 2)
    unmatched = sum(1 for r in records if r.get("human_match") == 0)
    return {
        "human_N": total,
        "human_match": match,
        "human_partial": partial,
        "human_unmatched": unmatched,
        "human_match_pct": round(match / total * 100, 1) if total else 0.0,
        "human_soft_match_pct": round((match + partial) / total * 100, 1) if total else 0.0,
    }


def llm_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    match = partial = unmatched = 0
    for rec in records:
        label = rec.get("llm_judge", {}).get("label", "").lower().strip()
        if label == "match":
            match += 1
        elif label == "partial":
            partial += 1
        elif label == "unmatched":
            unmatched += 1
    return {
        "llm_N": total,
        "llm_match": match,
        "llm_partial": partial,
        "llm_unmatched": unmatched,
        "llm_match_pct": round(match / total * 100, 1) if total else 0.0,
        "llm_soft_match_pct": round((match + partial) / total * 100, 1) if total else 0.0,
    }


def auto_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    f1s = [r.get("metrics", {}).get("feedback", {}).get("f1") for r in records]
    rouges = [r.get("metrics", {}).get("feedback", {}).get("rougeL") for r in records]
    bleus = [r.get("metrics", {}).get("feedback", {}).get("bleu") for r in records]
    return {
        "avg_f1": avg(f1s),
        "avg_rougeL": avg(rouges),
        "avg_bleu": avg(bleus),
    }


def normalize(fname: str) -> str:
    """Strip suffixes to get a bare scenario name, e.g. 'text_only'."""
    for suffix in [".annotated.jsonl", ".llmjudged.json", ".scored.jsonl", ".json"]:
        if fname.endswith(suffix):
            fname = fname[: -len(suffix)]
    for suffix in ["_results", "_no_answer_results"]:
        if fname.endswith(suffix):
            fname = fname[: -len(suffix)]
    return fname


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human_dir", default="data/human_annotated/qwen3_32b")
    parser.add_argument("--llm_dir", default="data/llm_judged/qwen3_32b")
    parser.add_argument("--scored_dir", default="data/eval_scored/qwen3_32b")
    parser.add_argument("--out_dir", default="data/eval_summary/qwen3_32b")
    parser.add_argument("--run_name", default="qwen3_32b", help="Prefix for output filenames")
    args = parser.parse_args()

    human_dir = Path(args.human_dir)
    llm_dir = Path(args.llm_dir)
    scored_dir = Path(args.scored_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Index all files by normalized scenario name
    human_files = {normalize(f.name): f for f in sorted(human_dir.glob("*.annotated.jsonl"))}
    llm_files = {normalize(f.name): f for f in sorted(llm_dir.glob("*.llmjudged.json"))}
    scored_files = {normalize(f.name): f for f in sorted(scored_dir.glob("*.scored.jsonl"))}

    scenarios = sorted(set(human_files) | set(llm_files) | set(scored_files))
    if not scenarios:
        raise SystemExit("No files found. Check --human_dir, --llm_dir, --scored_dir.")

    rows = []
    for scenario in scenarios:
        row: Dict[str, Any] = {"scenario": scenario}

        if scenario in human_files:
            row.update(human_metrics(read_jsonl(human_files[scenario])))
        else:
            print(f"[WARN] No human annotation file for: {scenario}")

        if scenario in llm_files:
            row.update(llm_metrics(read_json(llm_files[scenario])))
        else:
            print(f"[WARN] No LLM judge file for: {scenario}")

        if scenario in scored_files:
            row.update(auto_metrics(read_jsonl(scored_files[scenario])))
        else:
            print(f"[WARN] No scored file for: {scenario}")

        rows.append(row)

    prefix = f"{args.run_name}_" if args.run_name else ""
    out_json = out_dir / f"{prefix}combined_summary.json"
    out_csv = out_dir / f"{prefix}combined_summary.csv"

    out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(out_csv, rows)

    print(f"\nCombined Summary ({args.run_name})\n")
    header = f"{'Scenario':<20} {'Human Match%':>13} {'Human Soft%':>12} {'LLM Match%':>11} {'LLM Soft%':>10} {'Avg F1':>8} {'ROUGE-L':>8} {'BLEU':>7}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['scenario']:<20} "
            f"{str(row.get('human_match_pct', '—')) + '%':>13} "
            f"{str(row.get('human_soft_match_pct', '—')) + '%':>12} "
            f"{str(row.get('llm_match_pct', '—')) + '%':>11} "
            f"{str(row.get('llm_soft_match_pct', '—')) + '%':>10} "
            f"{str(row.get('avg_f1', '—')):>8} "
            f"{str(row.get('avg_rougeL', '—')):>8} "
            f"{str(row.get('avg_bleu', '—')):>7}"
        )

    print(f"\nSaved:\n- {out_json}\n- {out_csv}")


if __name__ == "__main__":
    main()
