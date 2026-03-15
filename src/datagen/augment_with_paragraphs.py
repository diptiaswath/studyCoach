"""Augment QA examples with paragraphs from the corresponding paper text that
mention the referenced Figure or Table.

For each QA, the script:
  1. Parses the figure/table identifier from the 'reference' field
     (e.g. '1705.09966v2-Table1-1.png'  -> keyword='Table', number='1')
  2. Reads the matching per-paper .txt file from the paragraphs directory
  3. Finds paragraphs that mention the identifier (case-insensitive, optional
     space between keyword and number, e.g. 'Figure 3', 'figure3', 'Table 1')
  4. Concatenates matching paragraphs and stores them in a new 'paragraphs'
     field; uses 'N/A' when nothing is found

Usage:
    python src/datagen/augment_with_paragraphs.py \\
        data/test-A/SPIQA_testA_part1_output_latest.json \\
        data/test-A/SPIQA_testA_part1_output_augmented.json

    python src/datagen/augment_with_paragraphs.py \\
        data/test-A/SPIQA_testA_part1_output_latest.json \\
        data/test-A/SPIQA_testA_part1_output_augmented.json \\
        --paragraphs /custom/paragraphs/dir
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_PARAGRAPHS_DIR = (
    _REPO_ROOT / "data" / "test-A" / "SPIQA_train_val_test-A_extracted_paragraphs"
)

# Matches the keyword and number in a reference filename, e.g.
# '1705.09966v2-Table1-1.png' -> ('Table', '1')
# '1603.00286v5-Figure3-1.png' -> ('Figure', '3')
_REF_RE = re.compile(r"-(Figure|Table)(\d+)-", re.IGNORECASE)

PARAGRAPH_SEP = "\n\n"


def parse_reference(reference: str) -> tuple[str, str] | None:
    """Return (keyword, number) from a reference filename, or None."""
    m = _REF_RE.search(reference)
    if not m:
        return None
    return m.group(1), m.group(2)


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split(PARAGRAPH_SEP) if p.strip()]


def find_paragraphs(paragraphs: list[str], keyword: str, number: str) -> list[str]:
    """Return paragraphs that mention '<keyword> <number>' (optional space, case-insensitive)."""
    pattern = re.compile(
        rf"(?i)\b{re.escape(keyword)}\s*{re.escape(number)}\b"
    )
    return [p for p in paragraphs if pattern.search(p)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment SPIQA QA examples with relevant paragraphs from paper text."
    )
    parser.add_argument("input", type=Path, help="Path to input JSON file")
    parser.add_argument("output", type=Path, help="Path for augmented output JSON file")
    parser.add_argument(
        "--paragraphs",
        type=Path,
        default=_DEFAULT_PARAGRAPHS_DIR,
        help=f"Directory of per-paper .txt files (default: {_DEFAULT_PARAGRAPHS_DIR})",
    )
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"Input file not found: {args.input}")
    if not args.paragraphs.is_dir():
        sys.exit(f"Paragraphs directory not found: {args.paragraphs}")

    with open(args.input, encoding="utf-8") as f:
        data: dict = json.load(f)

    # ── counters ──────────────────────────────────────────────────────────────
    total_qa = 0
    ref_parse_failures = 0   # reference didn't match expected pattern
    missing_txt = 0          # no .txt file for the paper
    found_paragraphs = 0     # at least one paragraph matched
    not_found = 0            # no paragraphs matched → 'N/A'
    total_paras_added = 0    # cumulative paragraph count across all hits
    papers_missing_txt: set[str] = set()

    for paper_id, paper_data in data.items():
        txt_path = args.paragraphs / f"{paper_id}.txt"
        paragraphs_cache: list[str] | None = None  # lazy-load once per paper

        for qa in paper_data.get("qa", []):
            total_qa += 1
            reference: str = qa.get("reference", "")

            parsed = parse_reference(reference)
            if parsed is None:
                qa["paragraphs"] = "N/A"
                ref_parse_failures += 1
                continue

            keyword, number = parsed

            if not txt_path.exists():
                qa["paragraphs"] = "N/A"
                missing_txt += 1
                papers_missing_txt.add(paper_id)
                continue

            if paragraphs_cache is None:
                paragraphs_cache = split_paragraphs(
                    txt_path.read_text(encoding="utf-8")
                )

            matches = find_paragraphs(paragraphs_cache, keyword, number)
            if matches:
                qa["paragraphs"] = PARAGRAPH_SEP.join(matches)
                found_paragraphs += 1
                total_paras_added += len(matches)
            else:
                qa["paragraphs"] = "N/A"
                not_found += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # ── statistics ────────────────────────────────────────────────────────────
    print(f"Output written to : {args.output}")
    print()
    print("=== Augmentation statistics ===")
    print(f"  Total QA examples      : {total_qa}")
    print(f"  Paragraphs found       : {found_paragraphs}  ({found_paragraphs/total_qa*100:.1f}%)")
    print(f"  No paragraphs (N/A)    : {not_found}  ({not_found/total_qa*100:.1f}%)")
    print(f"  Ref parse failures     : {ref_parse_failures}")
    print(f"  Missing .txt files     : {missing_txt} QA across {len(papers_missing_txt)} paper(s)")
    if total_paras_added and found_paragraphs:
        print(f"  Avg paragraphs/hit     : {total_paras_added/found_paragraphs:.2f}")
    if papers_missing_txt:
        print()
        print("  Papers with no .txt:")
        for pid in sorted(papers_missing_txt):
            print(f"    {pid}")


if __name__ == "__main__":
    main()
