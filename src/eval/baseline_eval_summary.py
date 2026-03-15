"""
compute_eval_summary.py

Reads all JSON files under data/eval/llm_judged/, computes LLM-judge metrics,
merges with the eval_summary JSON, and writes a combined summary to
data/eval/eval_summary/combined_summary.json (and a CSV for easy viewing).

Filenames like 'caption_only_results.llmjudged.json' are normalized to
'caption_only_results.json' for joining with eval_summary rows.
"""

import json
import os
import glob
import csv
from pathlib import Path

# ── Paths (relative to project root) ──────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]  # adjust if script depth differs

LLM_JUDGED_DIR    = BASE_DIR / "data/llm_judged"
EVAL_SUMMARY_FILE = BASE_DIR / "data/eval_summary/eval_metrics_summary.json"
OUTPUT_JSON       = BASE_DIR / "data/eval_summary/baseline_summary.json"
OUTPUT_CSV        = BASE_DIR / "data/eval_summary/baselin_summary.csv"


def normalize_filename(fname: str) -> str:
    """
    Strip '.llmjudged' from filename so it matches the eval_summary source_file key.
    e.g. 'caption_only_results.llmjudged.json' -> 'caption_only_results.json'
    """
    return fname.replace(".llmjudged.json", ".json")


# ── Step 1: compute LLM-judge metrics per file ────────────────────────────────
def compute_llm_metrics(judged_dir: Path) -> dict:
    results = {}
    files = glob.glob(str(judged_dir / "*.json"))

    if not files:
        print(f"[WARN] No JSON files found in {judged_dir}")

    for fpath in files:
        fname = os.path.basename(fpath)
        normalized = normalize_filename(fname)

        with open(fpath, "r") as f:
            records = json.load(f)

        total     = len(records)
        match     = 0
        partial   = 0
        unmatched = 0

        for rec in records:
            label = rec.get("llm_judge", {}).get("label", "").lower().strip()
            if label == "match":
                match += 1
            elif label == "partial":
                partial += 1
            elif label == "unmatched":
                unmatched += 1
            else:
                print(f"[WARN] Unknown label '{label}' in {fname}")

        match_rate      = match / total if total else 0.0
        soft_match_rate = (match + partial) / total if total else 0.0

        results[normalized] = {
            "N_total":             total,
            "N_match":             match,
            "N_partial":           partial,
            "N_unmatched":         unmatched,
            "match_rate":          round(match_rate, 4),
            "match_rate_pct":      round(match_rate * 100, 2),
            "soft_match_rate":     round(soft_match_rate, 4),
            "soft_match_rate_pct": round(soft_match_rate * 100, 2),
        }

    return results


# ── Step 2: load the existing eval_summary ────────────────────────────────────
def load_eval_summary(fpath: Path) -> dict:
    if not fpath.exists():
        print(f"[WARN] eval_summary file not found: {fpath}. Skipping merge.")
        return {}
    with open(fpath, "r") as f:
        records = json.load(f)
    return {rec["source_file"]: rec for rec in records}


# ── Step 3: merge and write output ────────────────────────────────────────────
def merge_and_save(llm_metrics: dict, eval_summary: dict, out_json: Path, out_csv: Path):
    out_json.parent.mkdir(parents=True, exist_ok=True)

    all_keys = sorted(set(llm_metrics.keys()) | set(eval_summary.keys()))

    combined = []
    for key in all_keys:
        row = {"source_file": key}
        if key in eval_summary:
            row.update(eval_summary[key])
        if key in llm_metrics:
            row.update(llm_metrics[key])
        combined.append(row)

    # Write JSON
    with open(out_json, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"[OK] Combined JSON written to {out_json}")

    # Write CSV with explicit column order
    priority_cols = [
        "source_file", "num_records",
        "N_total", "N_match", "N_partial", "N_unmatched",
        "match_rate", "match_rate_pct",
        "soft_match_rate", "soft_match_rate_pct",
        "avg_f1", "avg_rougeL", "avg_bleu",
        "scored_output",
    ]
    extra_cols = [c for c in (combined[0].keys() if combined else []) if c not in priority_cols]
    all_cols = priority_cols + extra_cols

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(combined)
    print(f"[OK] Combined CSV written to {out_csv}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Computing LLM-judge metrics...")
    llm_metrics = compute_llm_metrics(LLM_JUDGED_DIR)

    for fname, m in llm_metrics.items():
        print(
            f"  {fname}: total={m['N_total']} match={m['N_match']} "
            f"partial={m['N_partial']} unmatched={m['N_unmatched']} "
            f"match%={m['match_rate_pct']} soft%={m['soft_match_rate_pct']}"
        )

    print("\nLoading eval summary...")
    eval_summary = load_eval_summary(EVAL_SUMMARY_FILE)

    print("\nMerging and saving...")
    merge_and_save(llm_metrics, eval_summary, OUTPUT_JSON, OUTPUT_CSV)