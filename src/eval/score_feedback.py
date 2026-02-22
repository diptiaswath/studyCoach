import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import sacrebleu
from rouge_score import rouge_scorer


# ----------------- text utils -----------------
def normalize_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def tokenize(s: str) -> List[str]:
    s = normalize_text(s)
    return re.findall(r"[a-z0-9]+", s)

def token_f1(ref: str, cand: str) -> float:
    ref_toks = tokenize(ref)
    cand_toks = tokenize(cand)

    if not ref_toks and not cand_toks:
        return 1.0
    if not ref_toks or not cand_toks:
        return 0.0

    ref_counts = Counter(ref_toks)
    cand_counts = Counter(cand_toks)
    overlap = sum((ref_counts & cand_counts).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(cand_toks)
    recall = overlap / len(ref_toks)
    return 2 * precision * recall / (precision + recall)

_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def rouge_l(ref: str, cand: str) -> float:
    return float(_rouge.score(ref or "", cand or "")["rougeL"].fmeasure)

def bleu(ref: str, cand: str) -> float:
    # sacrebleu returns 0..100
    return float(sacrebleu.sentence_bleu(cand or "", [ref or ""]).score)

def score_pair(ref: str, cand: str) -> Dict[str, float]:
    return {"f1": token_f1(ref, cand), "rougeL": rouge_l(ref, cand), "bleu": bleu(ref, cand)}


# ----------------- IO -----------------
def load_json_list(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON list.")
    return data

def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def write_csv(path: Path, rows: List[Dict[str, Any]], header: List[str]) -> None:
    # simple CSV writer to avoid extra deps
    import csv
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", default="data/eval", help="Directory containing input *.json files")
    parser.add_argument("--out_scored_dir", default="data/eval_scored", help="Directory to write per-file scored JSONL")
    parser.add_argument("--out_summary_dir", default="data/eval_summary", help="Directory to write summary CSV/JSON")
    parser.add_argument("--pattern", default="*.json", help="Glob for input files")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_scored_dir = Path(args.out_scored_dir)
    out_summary_dir = Path(args.out_summary_dir)

    out_scored_dir.mkdir(parents=True, exist_ok=True)
    out_summary_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files found in {in_dir} matching {args.pattern}")

    summary_rows: List[Dict[str, Any]] = []

    print(f"Found {len(files)} files.")

    for fp in files:
        records = load_json_list(fp)

        scored_records: List[Dict[str, Any]] = []
        f1s: List[float] = []
        rouges: List[float] = []
        bleus: List[float] = []

        for rec in records:
            gt_fb = (rec.get("ground_truth") or {}).get("feedback", "")
            pr_fb = (rec.get("predicted") or {}).get("feedback", "")

            metrics = score_pair(gt_fb, pr_fb)

            # attach provenance + metrics + annotation placeholders
            rec["source_file"] = fp.name

            rec["metrics"] = rec.get("metrics", {})
            rec["metrics"]["feedback"] = metrics

            scored_records.append(rec)

            f1s.append(metrics["f1"])
            rouges.append(metrics["rougeL"])
            bleus.append(metrics["bleu"])

        # write per-file output (keeps identity by filename)
        out_path = out_scored_dir / f"{fp.stem}.scored.jsonl"
        write_jsonl(out_path, scored_records)

        avg_f1 = sum(f1s) / len(f1s) if f1s else 0.0
        avg_rouge = sum(rouges) / len(rouges) if rouges else 0.0
        avg_bleu = sum(bleus) / len(bleus) if bleus else 0.0

        summary_rows.append({
            "source_file": fp.name,
            "num_records": len(records),
            "avg_f1": round(avg_f1, 6),
            "avg_rougeL": round(avg_rouge, 6),
            "avg_bleu": round(avg_bleu, 6),
            "scored_output": str(out_path),
        })

        print(
            f"{fp.name}: n={len(records)} "
            f"avg_f1={avg_f1:.3f} avg_rougeL={avg_rouge:.3f} avg_bleu={avg_bleu:.2f} "
            f"-> {out_path}"
        )

    # write summary outputs
    summary_csv = out_summary_dir / "eval_metrics_summary.csv"
    summary_json = out_summary_dir / "eval_metrics_summary.json"

    write_csv(
        summary_csv,
        summary_rows,
        header=["source_file", "num_records", "avg_f1", "avg_rougeL", "avg_bleu", "scored_output"],
    )
    summary_json.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nWrote summary:\n- {summary_csv}\n- {summary_json}")


if __name__ == "__main__":
    main()