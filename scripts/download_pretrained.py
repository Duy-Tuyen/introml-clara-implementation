"""
Download Apple CLaRa pretrained checkpoints from Hugging Face.

Examples:
  python -m scripts.download_pretrained --repo apple/CLaRa-7B-E2E
  python -m scripts.download_pretrained --repo apple/CLaRa-7B-E2E --out ./clara-ckpts/pretrained-e2e
"""

from __future__ import annotations

import argparse
from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download CLaRa pretrained weights")
    parser.add_argument(
        "--repo",
        default="apple/CLaRa-7B-E2E",
        help="Hugging Face repo id (default: apple/CLaRa-7B-E2E)",
    )
    parser.add_argument(
        "--out",
        default="./clara-ckpts/pretrained-e2e",
        help="Output directory (default: ./clara-ckpts/pretrained-e2e)",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional repo revision (branch, tag, or commit)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot_download(
        repo_id=args.repo,
        local_dir=args.out,
        local_dir_use_symlinks=False,
        revision=args.revision,
    )
    print(f"Downloaded {args.repo} -> {args.out}")


if __name__ == "__main__":
    main()
