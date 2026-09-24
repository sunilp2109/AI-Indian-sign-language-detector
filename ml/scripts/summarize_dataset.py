"""Print the latest processed dataset summary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.pipeline.paths import PROCESSED_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Show processed/summary.json")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_ROOT)
    args = parser.parse_args()
    summary_path = args.processed_dir / "summary.json"
    if not summary_path.is_file():
        raise SystemExit("summary.json not found. Run create_sequences.py first.")
    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
