"""Simple-sign kinematic templates are not a claimed ISL corpus."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from ml.pipeline.simple_signs import (  # noqa: E402
    build_simple_sign_samples,
    frames_to_features,
    generate_take_frames,
    simple_sign_ids,
)


def test_simple_sign_templates_are_separated() -> None:
    rng = np.random.default_rng(0)
    means = {}
    for label in ("HELLO", "YES", "NO", "I", "YOU"):
        frames = generate_take_frames(label, rng, num_frames=30, scale=0.16, noise_std=0.001)
        means[label] = frames_to_features(frames, 30).mean(axis=0)
    gap = float(np.linalg.norm(means["HELLO"] - means["YES"]))
    assert gap > 0.15
    assert float(np.linalg.norm(means["I"] - means["YOU"])) > 0.08
    assert float(np.linalg.norm(means["YES"] - means["NO"])) > 0.1


def test_build_simple_sign_samples_covers_collect_labels() -> None:
    samples = build_simple_sign_samples(signers=2, takes_per_sign=1, sequence_length=30, seed=1)
    labels = {sample["label"] for sample in samples}
    assert labels == set(simple_sign_ids())
    assert all(sample["features"].shape == (30, 128) for sample in samples)
    last = samples[0]["features"][-1]
    assert last[-1] >= 0.5 or last[-2] >= 0.5
