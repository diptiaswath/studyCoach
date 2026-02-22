import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple


LABEL_HELP = {
    "m": ("match", 1),
    "u": ("unmatched", 0),
    "p": ("partial/unclear", 2),
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def safe_get(rec: Dict[str, Any], keys: List[str], default: str = "") -> str:
    cur: Any = rec
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur if isinstance(cur, str) else default


def print_record(rec: Dict[str, Any], idx: int, total: int) -> None:
    paper_id = rec.get("paper_id", "")
    source_file = rec.get("source_file", "")

    question = rec.get("question", "")
    student = rec.get("student", "")

    gt_feedback = safe_get(rec, ["ground_truth", "feedback"])
    pr_feedback = safe_get(rec, ["predicted", "feedback"])

    gt_verdict = safe_get(rec, ["ground_truth", "verdict"])
    pr_verdict = safe_get(rec, ["predicted", "verdict"])

    gt_cat = safe_get(rec, ["ground_truth", "error_category"])
    pr_cat = safe_get(rec, ["predicted", "error_category"])

    metrics = (rec.get("metrics") or {}).get("feedback", {})

    print("\n" + "=" * 100)
    print(f"[{idx}/{total}] source_file={source_file} paper_id={paper_id}")
    print("-" * 100)
    if question:
        print("Question:\n", question)
    if student:
        print("\nStudent answer:\n", student)

    print("\nGround truth:")
    if gt_verdict or gt_cat:
        print(f"  verdict={gt_verdict}  category={gt_cat}")
    print("  feedback:\n", gt_feedback)

    print("\nPredicted:")
    if pr_verdict or pr_cat:
        print(f"  verdict={pr_verdict}  category={pr_cat}")
    print("  feedback:\n", pr_feedback)

    if metrics:
        print("\nAuto metrics (feedback):", metrics)
    print()


def prompt_label() -> Tuple[str, int]:
    while True:
        resp = input("Label [m=match, u=unmatched, p=partial/unclear, q=quit]: ").strip().lower()
        if resp == "q":
            raise KeyboardInterrupt()
        if resp in LABEL_HELP:
            return LABEL_HELP[resp]
        print("Please enter m, u, p, or q.")


def load_or_create_shared_indices(
    files: List[Path],
    k: int,
    seed: int,
    out_dir: Path,
    index_filename: str = "annotation_index.json",
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Shared sampling:
    - Determine N from the first file
    - Sample indices once with seed
    - Persist to annotation_index.json so reruns keep the same indices
    """
    index_path = out_dir / index_filename
    if index_path.exists():
        existing = json.loads(index_path.read_text(encoding="utf-8"))
        indices = existing.get("shared_sample_indices")
        if isinstance(indices, list) and all(isinstance(x, int) for x in indices):
            return indices, existing

    # Otherwise create fresh shared indices
    first = read_jsonl(files[0])
    n = len(first)
    if n == 0:
        raise ValueError(f"First file {files[0].name} is empty; cannot sample indices.")

    k_eff = min(k, n)
    rng = random.Random(seed)
    indices = rng.sample(range(n), k_eff)
    indices.sort()

    index_obj: Dict[str, Any] = {
        "seed": seed,
        "k_per_file": k,
        "k_effective": k_eff,
        "n_records_in_first_file": n,
        "shared_sample_indices": indices,
        "files": [],  # we'll append per-file completion info
    }

    index_path.write_text(json.dumps(index_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return indices, index_obj


def update_index_file(out_dir: Path, index_obj: Dict[str, Any]) -> None:
    index_path = out_dir / "annotation_index.json"
    index_path.write_text(json.dumps(index_obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", default="data/eval_scored", help="Directory with *.scored.jsonl files")
    parser.add_argument("--out_dir", default="data/human_annotated", help="Output directory for annotated subsets")
    parser.add_argument("--pattern", default="*.scored.jsonl", help="Glob pattern for input files")
    parser.add_argument("--k", type=int, default=10, help="Number of records to sample per file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (only used when creating indices first time)")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If set, skip files that already have an annotated output in out_dir",
    )
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files found in {in_dir} matching {args.pattern}")

    # Shared sampling indices across all files
    shared_indices, index_obj = load_or_create_shared_indices(files, args.k, args.seed, out_dir)

    print(f"Found {len(files)} scored files in {in_dir}.")
    print(f"Shared indices (k={len(shared_indices)}): {shared_indices}")
    print("You will annotate file-by-file.\n")

    try:
        for fp in files:
            out_path = out_dir / fp.name.replace(".scored.jsonl", ".annotated.jsonl")
            if args.resume and out_path.exists():
                print(f"Skipping (already exists): {out_path.name}")
                continue

            records = read_jsonl(fp)
            n = len(records)
            if n == 0:
                print(f"Skipping empty file: {fp.name}")
                continue

            # Sanity check: shared indices must be within range
            if max(shared_indices) >= n:
                raise ValueError(
                    f"{fp.name} has only {n} records but shared index {max(shared_indices)} is out of range."
                )

            print("=" * 100)
            print(f"ANNOTATING FILE: {fp.name} (n={n}) -> will save to {out_path.name}")
            print("=" * 100)

            sampled: List[Dict[str, Any]] = []
            for j, rec_idx in enumerate(shared_indices, start=1):
                rec = records[rec_idx]

                # Ensure provenance fields exist
                rec["source_file"] = rec.get("source_file", fp.name)
                rec["sample_index"] = rec_idx  # index within the scored file (shared across files)

                print_record(rec, j, len(shared_indices))

                label_name, label_value = prompt_label()
                notes = input("Notes (optional): ").strip() or None

                rec["human_label"] = label_name
                rec["human_match"] = label_value
                rec["human_notes"] = notes

                sampled.append(rec)

            write_jsonl(out_path, sampled)

            # Update index tracking
            index_obj.setdefault("files", [])
            index_obj["files"].append(
                {
                    "input_file": fp.name,
                    "output_file": out_path.name,
                    "num_records_in_file": n,
                    "shared_sample_indices": shared_indices,
                }
            )
            update_index_file(out_dir, index_obj)

            print(f"\nSaved annotated subset: {out_path}\n")

    except KeyboardInterrupt:
        print("\nStopped early (quit). Progress saved for completed files.")
        update_index_file(out_dir, index_obj)
        return

    print("All done.")


if __name__ == "__main__":
    main()