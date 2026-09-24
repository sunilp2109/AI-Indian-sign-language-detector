"""Extract normalized landmark feature sequences from collected raw takes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from ml.pipeline.paths import FRAMES_ROOT, RAW_ROOT
from ml.pipeline.sequences import extract_all_takes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize raw landmark takes into (frames, feature_dim) .npy files."
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_ROOT)
    parser.add_argument("--out-dir", type=Path, default=FRAMES_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = extract_all_takes(raw_root=args.raw_dir, frames_root=args.out_dir)
    print(f"Extracted {len(manifest)} takes into {args.out_dir}")
    for row in manifest:
        print(f"  {row['label']:12} {row['signer_id']:12} frames={row['num_frames']:3}  {row['path']}")
    if not manifest:
        print("No raw takes found. Run collect_data.py first (with --i-am-recording).")


if __name__ == "__main__":
    main()
