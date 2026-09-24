"""Kinematic templates for the 10 collect-true simple signs.

These are MediaPipe-shaped 21-point trajectories that mimic everyday ISL
hand motions (wave, nod, wag, point). They are not a public ISL corpus.
Wrist-normalized features still keep finger curl and wrist-relative motion,
which is what the live model can actually see.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.preprocessing.constants import LANDMARKS_PER_HAND
from app.preprocessing.features import hands_to_features
from app.preprocessing.sequence import fit_sequence_length
from ml.pipeline.labels import build_label_map, collect_label_ids
from ml.pipeline.paths import load_dataset_config, resolve_labels_path

# Right-hand open palm, wrist at origin, fingers +Y, x to the thumb side.
_CANONICAL = np.array(
    [
        [0.00, 0.00, 0.00],
        [0.035, 0.03, 0.015],
        [0.06, 0.07, 0.02],
        [0.08, 0.11, 0.02],
        [0.095, 0.15, 0.015],
        [0.03, 0.11, 0.00],
        [0.032, 0.18, 0.00],
        [0.033, 0.24, 0.00],
        [0.034, 0.30, 0.00],
        [0.00, 0.12, 0.00],
        [0.00, 0.20, 0.00],
        [0.00, 0.27, 0.00],
        [0.00, 0.33, 0.00],
        [-0.03, 0.11, 0.00],
        [-0.032, 0.18, 0.00],
        [-0.033, 0.24, 0.00],
        [-0.034, 0.29, 0.00],
        [-0.058, 0.095, 0.00],
        [-0.065, 0.15, 0.00],
        [-0.068, 0.20, 0.00],
        [-0.07, 0.245, 0.00],
    ],
    dtype=np.float32,
)

_FINGERS = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)

SIMPLE_SIGN_HINTS = {
    "HELLO": "Open right palm and wave side to side.",
    "THANK_YOU": "Open right palm at the chin, then move it forward and down.",
    "HELP": "Left palm up, right fist on it, lift both hands.",
    "YES": "Right fist, nod it up and down.",
    "NO": "Right index finger up, wag left and right.",
    "WATER": "Three middle fingers up, tap near the mouth.",
    "DOCTOR": "Left wrist out, tap the pulse with two right fingers.",
    "HOSPITAL": "Draw a cross with the right index finger.",
    "I": "Point the right index finger at your chest.",
    "YOU": "Point the right index finger forward at the other person.",
}


def _rotation(pitch: float, yaw: float, roll: float) -> np.ndarray:
    cx, sx = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cz, sz = np.cos(roll), np.sin(roll)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return rz @ ry @ rx


def _apply_curl(points: np.ndarray, curls: np.ndarray) -> None:
    for finger, curl in zip(_FINGERS, curls, strict=True):
        amount = float(np.clip(curl, 0.0, 1.0))
        mcp = points[finger[0]].copy()
        for idx in finger[1:]:
            points[idx] = mcp + (points[idx] - mcp) * (1.0 - 0.72 * amount)
            points[idx, 2] += 0.045 * amount


def articulated_hand(
    *,
    handedness: str,
    wrist: np.ndarray,
    scale: float,
    curls: np.ndarray,
    pitch: float = 0.0,
    yaw: float = 0.0,
    roll: float = 0.0,
    noise: np.ndarray | None = None,
) -> np.ndarray:
    points = _CANONICAL.copy()
    if handedness.lower().startswith("l"):
        points[:, 0] *= -1.0
    _apply_curl(points, np.asarray(curls, dtype=np.float32))
    points = points @ _rotation(pitch, yaw, roll).T
    points *= float(scale)
    points += np.asarray(wrist, dtype=np.float32).reshape(1, 3)
    if noise is not None:
        points = points + noise
    return points.astype(np.float32)


def _open() -> np.ndarray:
    return np.zeros(5, dtype=np.float32)


def _fist() -> np.ndarray:
    return np.array([0.85, 0.95, 0.95, 0.95, 0.92], dtype=np.float32)


def _point() -> np.ndarray:
    return np.array([0.7, 0.05, 0.92, 0.92, 0.9], dtype=np.float32)


def _three() -> np.ndarray:
    return np.array([0.55, 0.08, 0.08, 0.08, 0.85], dtype=np.float32)


def _two() -> np.ndarray:
    return np.array([0.65, 0.05, 0.05, 0.9, 0.88], dtype=np.float32)


def _sign_pose(label: str, t: float, rng: np.random.Generator) -> dict[str, Any]:
    """Return left/right articulation at normalized time t in [0, 1]."""
    wave = np.sin(2 * np.pi * t)
    wave2 = np.sin(4 * np.pi * t)
    jitter = 0.04 * rng.normal()

    if label == "HELLO":
        return {
            "left": None,
            "right": {
                "curls": _open(),
                "pitch": 0.15,
                "yaw": 0.55 * wave,
                "roll": 0.08 * wave2,
                "wrist": np.array([0.68, 0.42 + 0.03 * wave, 0.0]),
            },
        }
    if label == "THANK_YOU":
        return {
            "left": None,
            "right": {
                "curls": _open(),
                "pitch": 0.55 - 0.9 * t,
                "yaw": 0.05,
                "roll": 0.0,
                "wrist": np.array([0.62, 0.28 + 0.22 * t, -0.04 * t]),
            },
        }
    if label == "HELP":
        lift = 0.12 * t
        return {
            "left": {
                "curls": _open(),
                "pitch": 0.35,
                "yaw": 0.0,
                "roll": 0.0,
                "wrist": np.array([0.38, 0.52 - lift, 0.0]),
            },
            "right": {
                "curls": _fist(),
                "pitch": 0.2,
                "yaw": 0.05,
                "roll": 0.0,
                "wrist": np.array([0.46, 0.46 - lift, 0.02]),
            },
        }
    if label == "YES":
        return {
            "left": None,
            "right": {
                "curls": _fist(),
                "pitch": 0.45 * wave,
                "yaw": 0.05,
                "roll": 0.0,
                "wrist": np.array([0.64, 0.46 + 0.05 * wave, 0.0]),
            },
        }
    if label == "NO":
        return {
            "left": None,
            "right": {
                "curls": _point(),
                "pitch": 0.1,
                "yaw": 0.7 * wave,
                "roll": 0.05 * wave,
                "wrist": np.array([0.66, 0.40, 0.0]),
            },
        }
    if label == "WATER":
        return {
            "left": None,
            "right": {
                "curls": _three(),
                "pitch": 0.35 + 0.12 * wave,
                "yaw": 0.05,
                "roll": 0.0,
                "wrist": np.array([0.58, 0.30 + 0.04 * abs(wave), 0.0]),
            },
        }
    if label == "DOCTOR":
        return {
            "left": {
                "curls": _open(),
                "pitch": 0.2,
                "yaw": 0.0,
                "roll": 0.15,
                "wrist": np.array([0.40, 0.50, 0.0]),
            },
            "right": {
                "curls": _two(),
                "pitch": 0.15 + 0.25 * abs(wave),
                "yaw": 0.1,
                "roll": 0.0,
                "wrist": np.array([0.46, 0.46 + 0.03 * wave, 0.01]),
            },
        }
    if label == "HOSPITAL":
        # Cross: horizontal then vertical with the index tip.
        if t < 0.5:
            x_off = -0.06 + 0.24 * (t / 0.5)
            y_off = 0.0
        else:
            x_off = 0.06
            y_off = -0.08 + 0.20 * ((t - 0.5) / 0.5)
        return {
            "left": None,
            "right": {
                "curls": _point(),
                "pitch": 0.05,
                "yaw": 0.15,
                "roll": 0.0,
                "wrist": np.array([0.60 + x_off, 0.38 + y_off, 0.0]),
            },
        }
    if label == "I":
        return {
            "left": None,
            "right": {
                "curls": _point(),
                "pitch": 0.85,
                "yaw": -0.15,
                "roll": 0.1,
                "wrist": np.array([0.52, 0.48 + 0.02 * wave, 0.04]),
            },
        }
    if label == "YOU":
        return {
            "left": None,
            "right": {
                "curls": _point(),
                "pitch": -0.35,
                "yaw": 0.05,
                "roll": 0.0,
                "wrist": np.array([0.64, 0.40, -0.08 + jitter]),
            },
        }
    raise KeyError(label)


def _hand_from_pose(
    pose: dict[str, Any] | None,
    *,
    handedness: str,
    scale: float,
    rng: np.random.Generator,
    noise_std: float,
) -> np.ndarray | None:
    if pose is None:
        return None
    noise = rng.normal(0.0, noise_std, size=(LANDMARKS_PER_HAND, 3)).astype(np.float32)
    wrist = np.asarray(pose["wrist"], dtype=np.float32)
    wrist = wrist + rng.normal(0.0, 0.008, size=3).astype(np.float32)
    return articulated_hand(
        handedness=handedness,
        wrist=wrist,
        scale=scale,
        curls=pose["curls"],
        pitch=float(pose["pitch"]),
        yaw=float(pose["yaw"]),
        roll=float(pose["roll"]),
        noise=noise,
    )


def generate_take_frames(
    label: str,
    rng: np.random.Generator,
    *,
    num_frames: int = 36,
    scale: float = 0.16,
    noise_std: float = 0.004,
    speed: float = 1.0,
) -> list[dict[str, np.ndarray | None]]:
    frames: list[dict[str, np.ndarray | None]] = []
    phase = float(rng.uniform(0.0, 0.15))
    for index in range(num_frames):
        t = ((index / max(num_frames - 1, 1)) * speed + phase) % 1.0
        pose = _sign_pose(label, t, rng)
        frames.append(
            {
                "left": _hand_from_pose(
                    pose.get("left"),
                    handedness="Left",
                    scale=scale * 0.98,
                    rng=rng,
                    noise_std=noise_std,
                ),
                "right": _hand_from_pose(
                    pose.get("right"),
                    handedness="Right",
                    scale=scale,
                    rng=rng,
                    noise_std=noise_std,
                ),
            }
        )
    return frames


def frames_to_features(frames: list[dict[str, np.ndarray | None]], sequence_length: int) -> np.ndarray:
    rows = [hands_to_features(frame.get("left"), frame.get("right")) for frame in frames]
    return fit_sequence_length(np.stack(rows, axis=0), sequence_length)


def simple_sign_ids() -> list[str]:
    config = load_dataset_config()
    return collect_label_ids(resolve_labels_path(config))


def build_simple_sign_samples(
    *,
    signers: int = 10,
    takes_per_sign: int = 12,
    sequence_length: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    cfg = load_dataset_config()
    length = int(sequence_length or cfg["sequence_length"])
    labels = simple_sign_ids()
    samples: list[dict[str, Any]] = []
    for signer_index in range(signers):
        signer_id = f"synth_{signer_index + 1:02d}"
        signer_rng = np.random.default_rng(seed + 17 * (signer_index + 1))
        scale = float(signer_rng.uniform(0.13, 0.22))
        noise_std = float(signer_rng.uniform(0.0025, 0.007))
        for label in labels:
            for take_index in range(takes_per_sign):
                rng = np.random.default_rng(seed + 1000 * signer_index + 50 * take_index + labels.index(label))
                n_frames = int(rng.integers(28, 48))
                speed = float(rng.uniform(0.75, 1.25))
                frames = generate_take_frames(
                    label,
                    rng,
                    num_frames=n_frames,
                    scale=scale * float(rng.uniform(0.92, 1.08)),
                    noise_std=noise_std,
                    speed=speed,
                )
                features = frames_to_features(frames, length)
                samples.append(
                    {
                        "label": label,
                        "signer_id": signer_id,
                        "stem": f"take_{take_index + 1:04d}",
                        "num_frames": n_frames,
                        "features": features,
                    }
                )
    return samples


def label_map_for_simple_signs() -> dict[str, int]:
    return build_label_map(simple_sign_ids())
