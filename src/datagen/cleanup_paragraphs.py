#!/usr/bin/env python3
"""Batch-sanitize paragraph text files in a directory.

Replaces NO-BREAK SPACE (U+00A0) after 'Figure' / 'Table' with a normal space
in every .txt file found in the target directory, overwriting each file in-place.

Usage:
    python src/datagen/cleanup_paragraphs.py
    python src/datagen/cleanup_paragraphs.py --dir /path/to/paragraphs
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_DIR = _REPO_ROOT / "data" / "test-A" / "SPIQA_train_val_test-A_extracted_paragraphs"


def sanitize_text(text: str) -> tuple[str, int]:
    pattern = re.compile(r"(?i)\b(Figure|Table)\u00A0")
    sanitized_text, replacements = pattern.subn(lambda m: f"{m.group(1)} ", text)
    return sanitized_text, replacements


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sanitize all .txt paragraph files in a directory, replacing "
            "NO-BREAK SPACE after 'Figure'/'Table' with a normal space (in-place)."
        )
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=_DEFAULT_DIR,
        help=f"Directory containing .txt files to clean up (default: {_DEFAULT_DIR})",
    )
    args = parser.parse_args()

    target_dir: Path = args.dir
    if not target_dir.is_dir():
        parser.error(f"Directory not found: {target_dir}")

    txt_files = sorted(target_dir.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {target_dir}")
        return

    total_files = len(txt_files)
    modified_files = 0
    unchanged_files = 0
    total_replacements = 0
    per_file: list[tuple[str, int]] = []

    for path in txt_files:
        text = path.read_text(encoding="utf-8")
        sanitized, count = sanitize_text(text)
        if count:
            path.write_text(sanitized, encoding="utf-8")
            modified_files += 1
            total_replacements += count
            per_file.append((path.name, count))
        else:
            unchanged_files += 1

    # --- Statistics ---
    print(f"Directory : {target_dir}")
    print(f"Files scanned   : {total_files}")
    print(f"Files modified  : {modified_files}")
    print(f"Files unchanged : {unchanged_files}")
    print(f"Total replacements: {total_replacements}")

    if per_file:
        print()
        print("Modified files:")
        for name, count in per_file:
            print(f"  {count:>4} replacement{'s' if count != 1 else ''}  {name}")


if __name__ == "__main__":
    main()
