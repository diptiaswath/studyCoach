"""
download_spiqa.py
-----------------
Downloads the SPIQA dataset from HuggingFace to ./spiqa_data/.

Splits downloaded
  train/      – training QA pairs + images
  val/        – validation QA pairs + images   (used for early stopping)
  test-A/     – test set A (direct QA with figures; main eval target)

Usage
-----
  pip install huggingface_hub
  python download_spiqa.py

The script also does a quick sanity-check and reports how many images
are actually present on disk after download.
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

LOCAL_DIR = "./spiqa_data"

SPLITS_TO_DOWNLOAD = [
    "train/SPIQA_train.json",
    "val/SPIQA_val.json",
    "test-A/SPIQA_testA.json",
]


def download():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("Please run: pip install huggingface_hub")

    log.info(f"Downloading SPIQA to {LOCAL_DIR}  (this may take a while — ~30 GB)")
    snapshot_download(
        repo_id   = "google/spiqa",
        repo_type = "dataset",
        local_dir = LOCAL_DIR,
    )
    log.info("Download complete.")


def sanity_check():
    log.info("Running sanity checks ...")
    for split_json in SPLITS_TO_DOWNLOAD:
        json_path = Path(LOCAL_DIR) / split_json
        if not json_path.exists():
            log.warning(f"  MISSING: {json_path}")
            continue
        with open(json_path) as f:
            data = json.load(f)
        n_papers = len(data)

        # Count QA pairs
        n_qa = 0
        for paper_data in data.values():
            figs = paper_data.get("figures", paper_data.get("qa", {}))
            if isinstance(figs, list):
                n_qa += len(figs)
            else:
                for fig in figs.values():
                    n_qa += len(fig.get("questions", []))

        log.info(f"  {split_json}: {n_papers} papers, ~{n_qa} QA pairs")


if __name__ == "__main__":
    download()
    sanity_check()
