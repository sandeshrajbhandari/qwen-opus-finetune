"""Download Intel AutoRound int4 Qwen3.5-9B locally.

Uses Hugging Face fast downloads (``hf_transfer``) when available; install with
``pip install hf_transfer`` or use ``0.8b-finetune-tests/requirements.txt``.

Usage:
    python download_qwen35_09b_intel_int4.py
    python download_qwen35_09b_intel_int4.py --output-dir 9b-int4-finetune-tests
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Rust-backed parallel downloads; no-op if hf_transfer is not installed.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from huggingface_hub import snapshot_download


DEFAULT_REPO_ID = "Intel/Qwen3.5-9B-int4-AutoRound"
DEFAULT_OUTPUT_ROOT = Path("9b-int4-finetune-tests")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the Intel Qwen3.5-9B int4 (AutoRound) snapshot."
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Model repo on Hugging Face (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_ROOT),
        help=f"Root folder for downloads (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional model revision (branch, tag, or commit SHA).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_root = Path(args.output_dir).resolve()
    model_name = args.repo_id.split("/")[-1]
    local_model_dir = output_root / model_name
    local_model_dir.mkdir(parents=True, exist_ok=True)

    hf_token = os.getenv("HF_TOKEN") or os.getenv("tHF_TOKEN")

    downloaded_path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(local_model_dir),
        revision=args.revision,
        token=hf_token,
    )

    print(f"Downloaded: {args.repo_id}")
    print(f"Local path: {downloaded_path}")


if __name__ == "__main__":
    main()
