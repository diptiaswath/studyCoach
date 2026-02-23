import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
import statistics
import csv


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def safe_metric(rec, metric_name):
    return (
        rec.get("metrics", {})
        .get("feedback", {})
        .get(metric_name, None)
    )


def summarize_file(path: Path) -> Dict[str, Any]:
    records = read_jsonl(path)
    total = len(records)

    match = sum(1 for r in records if r.get("human_match") == 1)
    unmatched = sum(1 for r in records if r.get("human_match") == 0)
    partial = sum(1 for r in records if r.get("human_match") == 2)

    def avg(values):
        values = [v for v in values if v is not None]
        return round(statistics.mean(values), 4) if values else None

    f1_all = [safe_metric(r, "f1") for r in records]
    rouge_all = [safe_metric(r, "rougeL") for r in records]
    bleu_all = [safe_metric(r, "bleu") for r in records]

    match_rate = match / total if total else 0.0
    soft_match_rate = (match + partial) / total if total else 0.0

    summary = {
        "model_file": path.name,
        "N_total": total,
        "N_match": match,
        "N_unmatched": unmatched,
        "N_partial": partial,
        "match_rate": round(match_rate, 4),
        "match_rate_pct": round(match_rate * 100, 2),
        "soft_match_rate": round(soft_match_rate, 4),
        "soft_match_rate_pct": round(soft_match_rate * 100, 2),
        "avg_f1": avg(f1_all),
        "avg_rougeL": avg(rouge_all),
        "avg_bleu": avg(bleu_all),
    }

    return summary


def write_csv(path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        return
    headers = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", default="data/human_annotated")
    parser.add_argument("--out_dir", default="data/eval_summary")
    parser.add_argument("--pattern", default="*.annotated.jsonl")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)



    files = sorted(in_dir.glob(args.pattern))
    if not files:
        raise SystemExit("No annotated files found.")

    summaries = []
    for fp in files:
        summaries.append(summarize_file(fp))

    # Write outputs
    csv_path = out_dir / "human_vs_metrics_summary.csv"
    json_path = out_dir / "human_vs_metrics_summary.json"

    write_csv(csv_path, summaries)
    json_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    # Pretty print
    print("\nModel Comparison Summary\n")
    for row in summaries:
        print(
            f"{row['model_file']} | "
            f"N={row['N_total']} | "
            f"Match={row['match_rate_pct']}% | "
            f"SoftMatch={row['soft_match_rate_pct']}% | "
            f"F1={row['avg_f1']} | "
            f"ROUGE-L={row['avg_rougeL']} | "
            f"BLEU={row['avg_bleu']}"
        )

    print(f"\nSaved:")
    print(f"- {csv_path}")
    print(f"- {json_path}")


if __name__ == "__main__":
    main()