"""Collect labelled ISL landmark sequences from a webcam.

This path is explicit dataset recording. It never runs during inference
and does not save video files — only landmark arrays.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

import numpy as np

from ml.pipeline.labels import collect_label_ids
from ml.pipeline.paths import RAW_ROOT, load_dataset_config, resolve_labels_path
from ml.pipeline.raw_io import detections_to_raw_frame, save_raw_take, take_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DATASET RECORDING: capture hand landmark sequences (no video file)."
    )
    parser.add_argument("--label", help="Sign id from labels.json, e.g. HELLO")
    parser.add_argument("--signer", default="signer_01", help="Signer id used for leakage-safe splits")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--i-am-recording", action="store_true", help="Required confirmation; collection is opt-in")
    parser.add_argument("--list-labels", action="store_true", help="Print collectable labels and exit")
    parser.add_argument("--countdown", type=int, default=3)
    parser.add_argument("--raw-dir", type=Path, default=RAW_ROOT)
    return parser.parse_args()


def list_labels() -> list[str]:
    config = load_dataset_config()
    return collect_label_ids(resolve_labels_path(config))


def _draw_hand(frame, points, color) -> None:
    import cv2

    height, width = frame.shape[:2]
    for x, y, _z in np.asarray(points, dtype=np.float32):
        cx = int(np.clip(x, 0, 1) * (width - 1))
        cy = int(np.clip(y, 0, 1) * (height - 1))
        cv2.circle(frame, (cx, cy), 4, color, -1)


def _overlay(frame, lines: list[str], recording: bool) -> None:
    import cv2

    tinted = frame.copy()
    cv2.rectangle(tinted, (0, 0), (frame.shape[1], 120), (20, 20, 20), -1)
    frame[:] = cv2.addWeighted(tinted, 0.45, frame, 0.55, 0)
    color = (40, 40, 220) if recording else (230, 230, 230)
    y = 28
    for line in lines:
        cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
        y += 28


def run_collector(args: argparse.Namespace) -> int:
    import time

    import cv2

    from app.vision.hand_detector import HandDetector

    labels = list_labels()
    if args.label not in labels:
        raise SystemExit(f"Unknown or inactive label '{args.label}'. Choose from: {', '.join(labels)}")

    config = load_dataset_config()
    min_frames = int(config.get("min_frames", 10))
    max_frames = int(config.get("max_frames", 90))

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    detector = HandDetector()
    recording = False
    pending_start = 0.0
    frames: list[dict] = []
    saved = 0

    window = "ISL Bridge — DATASET RECORDING"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame grab failed.")
                break

            detections = detector.detect(frame)
            if detections.get("left") is not None:
                _draw_hand(frame, detections["left"], (180, 200, 60))
            if detections.get("right") is not None:
                _draw_hand(frame, detections["right"], (60, 140, 240))

            now = time.monotonic()
            if pending_start and now >= pending_start:
                recording = True
                pending_start = 0.0
                frames = []

            if recording:
                frames.append(detections_to_raw_frame(detections))
                if len(frames) >= max_frames:
                    save_raw_take(
                        label=args.label,
                        signer_id=args.signer,
                        frames=frames,
                        raw_root=args.raw_dir,
                    )
                    saved += 1
                    recording = False
                    frames = []

            countdown_left = max(0, int(np.ceil(pending_start - now))) if pending_start else 0
            takes_dir = take_dir(args.label, args.signer, args.raw_dir)
            existing = len(list(takes_dir.glob("take_*.npz"))) if takes_dir.exists() else 0
            status = "RECORDING" if recording else ("COUNTDOWN" if pending_start else "PREVIEW")
            _overlay(
                frame,
                [
                    f"DATASET RECORDING  ({status})  video is NOT saved",
                    f"label={args.label}  signer={args.signer}  takes={existing + saved}",
                    "SPACE=record/stop   Q=quit   landmarks only",
                    f"countdown={countdown_left}  frames={len(frames)}/{max_frames}" if recording or pending_start else f"min_frames={min_frames}",
                ],
                recording=recording,
            )
            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key == 32:  # space
                if recording:
                    if len(frames) < min_frames:
                        print(f"Discarded take ({len(frames)} < min_frames {min_frames})")
                    else:
                        path = save_raw_take(
                            label=args.label,
                            signer_id=args.signer,
                            frames=frames,
                            raw_root=args.raw_dir,
                        )
                        saved += 1
                        print(f"Saved {path}")
                    recording = False
                    frames = []
                elif pending_start:
                    pending_start = 0.0
                    print("Countdown cancelled")
                else:
                    pending_start = time.monotonic() + max(args.countdown, 0)
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()

    print(f"Session finished. New takes this run: {saved}")
    return 0


def main() -> None:
    args = parse_args()
    if args.list_labels:
        print("\n".join(list_labels()))
        return
    if not args.i_am_recording:
        raise SystemExit(
            "Refusing to open the camera.\n"
            "This script is DATASET RECORDING, not inference.\n"
            "Re-run with --i-am-recording --label HELLO --signer signer_01"
        )
    if not args.label:
        raise SystemExit("Provide --label (or --list-labels).")
    raise SystemExit(run_collector(args))


if __name__ == "__main__":
    main()
