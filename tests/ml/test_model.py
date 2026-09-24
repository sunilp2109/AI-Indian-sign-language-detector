"""Phase 4 model tests. Random tensors are not an ISL accuracy claim."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from app.model.isl_model import ISLSequenceModel
from app.model.model_loader import load_checkpoint, save_checkpoint
from ml.pipeline.engine import evaluate_model, select_device, set_seed, train_model
from ml.pipeline.metrics import compute_classification_metrics
from ml.pipeline.torch_data import make_loader


def test_lstm_forward_shape() -> None:
    model = ISLSequenceModel(input_size=128, num_classes=10, hidden_size=32, num_layers=2, dropout=0.2)
    batch = torch.randn(4, 30, 128)
    lengths = torch.tensor([30, 18, 7, 30])
    logits = model(batch, lengths)
    assert logits.shape == (4, 10)
    probs = model.predict_proba(batch, lengths)
    assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-5)


def test_gru_and_bidirectional() -> None:
    model = ISLSequenceModel(
        input_size=16,
        num_classes=3,
        hidden_size=8,
        num_layers=1,
        dropout=0.1,
        rnn_type="gru",
        bidirectional=True,
        sequence_length=12,
    )
    logits = model(torch.randn(2, 12, 16))
    assert logits.shape == (2, 3)


def test_rejects_mismatched_feature_dim() -> None:
    model = ISLSequenceModel(input_size=8, num_classes=2, hidden_size=8, num_layers=1, dropout=0.0)
    with pytest.raises(ValueError, match="input_size"):
        model(torch.randn(1, 5, 9))


def test_metrics_on_known_predictions() -> None:
    names = ["A", "B"]
    metrics = compute_classification_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 1, 1]),
        class_names=names,
    )
    assert metrics["n_samples"] == 4
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    empty = compute_classification_metrics(np.array([]), np.array([]), class_names=names)
    assert empty["accuracy"] is None


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = ISLSequenceModel(input_size=8, num_classes=2, hidden_size=8, num_layers=1, dropout=0.0)
    path = tmp_path / "best_model.pth"
    save_checkpoint(path, model, label_map={"A": 0, "B": 1}, class_names=["A", "B"])
    loaded, payload = load_checkpoint(path)
    assert payload["class_names"] == ["A", "B"]
    x = torch.randn(1, 6, 8)
    model.eval()
    loaded.eval()
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x), atol=1e-5)


def test_train_loop_smoke_not_isl_accuracy(tmp_path: Path) -> None:
    """Tiny random data only checks that training runs and writes a checkpoint."""
    set_seed(0)
    n_train, n_val, t, f, c = 12, 6, 8, 10, 2
    rng = np.random.default_rng(0)
    train_x = rng.normal(size=(n_train, t, f)).astype(np.float32)
    train_y = np.array([i % c for i in range(n_train)], dtype=np.int64)
    val_x = rng.normal(size=(n_val, t, f)).astype(np.float32)
    val_y = np.array([i % c for i in range(n_val)], dtype=np.int64)
    train_len = np.full(n_train, t, dtype=np.int64)
    val_len = np.full(n_val, t, dtype=np.int64)
    model = ISLSequenceModel(
        input_size=f,
        num_classes=c,
        hidden_size=16,
        num_layers=1,
        dropout=0.1,
        sequence_length=t,
    )
    device = select_device("cpu")
    model.to(device)
    result = train_model(
        model,
        make_loader(train_x, train_y, train_len, batch_size=4, shuffle=True),
        make_loader(val_x, val_y, val_len, batch_size=4, shuffle=False),
        class_names=["A", "B"],
        label_map={"A": 0, "B": 1},
        epochs=2,
        learning_rate=0.01,
        weight_decay=0.0,
        patience=5,
        device=device,
        checkpoint_path=tmp_path / "best_model.pth",
        history_path=tmp_path / "history.json",
    )
    assert (tmp_path / "best_model.pth").is_file()
    assert result["epochs_run"] == 2
    loaded, _payload = load_checkpoint(tmp_path / "best_model.pth", device="cpu")
    metrics = evaluate_model(
        loaded,
        make_loader(val_x, val_y, val_len, batch_size=4, shuffle=False),
        class_names=["A", "B"],
        device=device,
    )
    assert metrics["n_samples"] == n_val
    assert "accuracy" in metrics
    # Explicitly not an ISL result.
    assert "not a published" in metrics["note"].lower() or "split" in metrics["note"].lower()


def test_train_script_missing_dataset() -> None:
    script = ROOT / "ml" / "scripts" / "train.py"
    missing = ROOT / "ml" / "dataset" / "does-not-exist"
    completed = subprocess.run(
        [sys.executable, str(script), "--processed-dir", str(missing)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    combined = (completed.stdout + completed.stderr).lower()
    assert any(
        token in combined
        for token in ("pytorch", "missing", "not found", "create_sequences")
    )
