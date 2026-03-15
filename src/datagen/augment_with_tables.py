"""Augment QA examples with OCR-extracted table data.

For each QA example referencing a table figure, runs OCR via img2table + EasyOCR
and adds a 'table' field with the stringified table JSON. Non-table references
get 'table': 'N/A'.

Usage:
    python src/datagen/augment_with_tables.py \\
        data/test-A/SPIQA_testA_part1_output_latest.json \\
        data/test-A/SPIQA_testA_part1_output_augmented.json \\
        --images data/test-A/SPIQA_testA_Images
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def extract_table_json(image_path: Path, ocr) -> str:
    """Run img2table OCR on image_path, return stringified JSON or 'N/A'."""
    from img2table.document import Image as Img2TableImage

    doc = Img2TableImage(src=str(image_path))
    tables = doc.extract_tables(
        ocr=ocr,
        implicit_rows=True,
        implicit_columns=False,
        borderless_tables=False,
        min_confidence=50,
    )
    if not tables:
        return "N/A"

    result = []
    for table in tables:
        df = table.df
        # Use first row as header if column names are generic integers
        if all(isinstance(c, int) for c in df.columns):
            df.columns = df.iloc[0]
            df = df[1:].reset_index(drop=True)
        result.append(json.loads(df.to_json(orient="records", force_ascii=False)))

    return json.dumps(result, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment SPIQA QA examples with OCR table data."
    )
    parser.add_argument("input", help="Path to input JSON file")
    parser.add_argument("output", help="Path for augmented output JSON file")
    parser.add_argument(
        "--images",
        default="data/test-A/SPIQA_testA_Images",
        help="Root directory containing per-paper image subdirectories",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    images_root = Path(args.images)

    if not input_path.exists():
        sys.exit(f"Input file not found: {input_path}")
    if not images_root.exists():
        sys.exit(f"Images directory not found: {images_root}")

    with open(input_path, encoding="utf-8") as f:
        data: dict = json.load(f)

    # Lazy-init OCR only if there are tables to process
    ocr = None

    total_qa = 0
    table_hits = 0
    ocr_failures = 0

    for paper_id, paper_data in data.items():
        all_figures: dict = paper_data.get("all_figures", {})
        qa_list: list = paper_data.get("qa", [])

        for qa in qa_list:
            total_qa += 1
            reference: str = qa.get("reference", "")
            figure_info: dict = all_figures.get(reference, {})
            content_type: str = figure_info.get("content_type", "")

            if content_type == "table":
                table_hits += 1
                image_path = images_root / paper_id / reference

                if not image_path.exists():
                    print(f"  [WARN] Image not found: {image_path}", flush=True)
                    qa["table"] = "N/A"
                    continue

                if ocr is None:
                    print("Initialising EasyOCR (first table found)...", flush=True)
                    from img2table.ocr import EasyOCR
                    ocr = EasyOCR(lang=["en"])

                print(f"  OCR: {image_path.name}", flush=True)
                try:
                    qa["table"] = extract_table_json(image_path, ocr)
                except Exception as exc:
                    print(f"  [ERROR] OCR failed for {image_path.name}: {exc}", flush=True)
                    qa["table"] = "N/A"
                    ocr_failures += 1
            else:
                qa["table"] = "N/A"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(
        f"\nDone. {total_qa} QA examples processed, "
        f"{table_hits} tables found ({ocr_failures} OCR failures). "
        f"Output written to {output_path}"
    )


if __name__ == "__main__":
    main()
