"""Build fixed-length train/validation/test arrays from extracted frames."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from ml.pipeline.paths import FRAMES_ROOT, PROCESSED_ROOT
from ml.pipeline.sequences import create_split_arrays
from ml.pipeline.summary import build_summary, format_summary, write_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create SEQUENCE_LENGTH windows and signer-aware splits."
    )
    parser.add_argument("--frames-dir", type=Path, default=FRAMES_ROOT)
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_ROOT)
    parser.add_argument(
        "--strict-signer",
        action="store_true",
        help="Fail if there are not enough signers for a leakage-free split",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = create_split_arrays(
        frames_root=args.frames_dir,
        processed_root=args.out_dir,
        strict_signer=args.strict_signer,
    )
    summary = build_summary(result)
    path = write_summary(summary, processed_root=args.out_dir)
    print(format_summary(summary))
    print(f"\nWrote {path}")
    print("No model was trained. Phase 4 will consume these arrays.")


if __name__ == "__main__":
    main()
