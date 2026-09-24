"""Training and evaluation loops for ISLSequenceModel. No FastAPI imports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from app.model.isl_model import ISLSequenceModel
from app.model.model_loader import save_checkpoint
from ml.pipeline.metrics import compute_classification_metrics, format_confusion_matrix


def select_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_epoch(
    model: ISLSequenceModel,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    *,
    grad_clip: float = 0.0,
) -> tuple[float, np.ndarray, np.ndarray]:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    n_batches = 0
    truths: list[np.ndarray] = []
    preds: list[np.ndarray] = []

    for features, labels, lengths in loader:
        features = features.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(features, lengths)
        loss = criterion(logits, labels)
        if train:
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
        truths.append(labels.detach().cpu().numpy())
        preds.append(torch.argmax(logits, dim=-1).detach().cpu().numpy())

    y_true = np.concatenate(truths) if truths else np.array([], dtype=np.int64)
    y_pred = np.concatenate(preds) if preds else np.array([], dtype=np.int64)
    mean_loss = total_loss / max(n_batches, 1)
    return mean_loss, y_true, y_pred


def train_model(
    model: ISLSequenceModel,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    *,
    class_names: list[str],
    label_map: dict[str, int],
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    device: torch.device,
    checkpoint_path: Path,
    history_path: Path,
    label_smoothing: float = 0.0,
    grad_clip: float = 0.0,
    class_weights: np.ndarray | None = None,
    use_cosine: bool = False,
) -> dict[str, Any]:
    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.tensor(np.asarray(class_weights, dtype=np.float32), device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=float(label_smoothing))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(epochs), 1))
        if use_cosine
        else None
    )
    history: list[dict[str, Any]] = []
    best_val_f1 = -1.0
    best_epoch = 0
    stalled = 0
    selected_by = "last_epoch_no_validation"

    for epoch in range(1, epochs + 1):
        train_loss, y_true, y_pred = _run_epoch(
            model, train_loader, criterion, optimizer, device, grad_clip=float(grad_clip)
        )
        train_metrics = compute_classification_metrics(y_true, y_pred, class_names=class_names)
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_f1_macro": train_metrics["f1_macro"],
        }

        improved = False
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                val_loss, val_true, val_pred = _run_epoch(
                    model, val_loader, criterion, None, device
                )
            val_metrics = compute_classification_metrics(val_true, val_pred, class_names=class_names)
            row.update(
                {
                    "val_loss": val_loss,
                    "val_accuracy": val_metrics["accuracy"],
                    "val_precision_macro": val_metrics["precision_macro"],
                    "val_recall_macro": val_metrics["recall_macro"],
                    "val_f1_macro": val_metrics["f1_macro"],
                }
            )
            val_f1 = float(val_metrics["f1_macro"] or 0.0)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                stalled = 0
                improved = True
                selected_by = "best_val_f1_macro"
                save_checkpoint(
                    checkpoint_path,
                    model,
                    label_map=label_map,
                    class_names=class_names,
                    extra={
                        "best_epoch": best_epoch,
                        "best_val_f1_macro": best_val_f1,
                        "selected_by": selected_by,
                    },
                )
            else:
                stalled += 1
        else:
            save_checkpoint(
                checkpoint_path,
                model,
                label_map=label_map,
                class_names=class_names,
                extra={"best_epoch": epoch, "selected_by": selected_by},
            )
            best_epoch = epoch

        history.append(row)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        val_part = ""
        if "val_loss" in row:
            val_part = (
                f"  val_loss={row['val_loss']:.4f}  val_acc={row['val_accuracy']:.3f}  "
                f"val_f1={row['val_f1_macro']:.3f}"
            )
        marker = "  saved" if improved or val_loader is None else ""
        print(
            f"epoch {epoch:03d}  train_loss={train_loss:.4f}  "
            f"train_acc={train_metrics['accuracy']:.3f}{val_part}{marker}"
        )
        if scheduler is not None:
            scheduler.step()

        if val_loader is not None and patience > 0 and stalled >= patience:
            print(f"Early stopping at epoch {epoch} (patience={patience}).")
            break

    if val_loader is not None and best_val_f1 < 0:
        save_checkpoint(
            checkpoint_path,
            model,
            label_map=label_map,
            class_names=class_names,
            extra={"best_epoch": best_epoch, "selected_by": "fallback_last_epoch"},
        )

    return {
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_f1_macro": None if best_val_f1 < 0 else best_val_f1,
        "selected_by": selected_by,
        "checkpoint": str(checkpoint_path),
        "history_path": str(history_path),
        "history": history,
        "note": (
            "Training history is not a test-set result. "
            "Run evaluate.py on the held-out test split before quoting accuracy."
        ),
    }


def evaluate_model(
    model: ISLSequenceModel,
    loader: DataLoader,
    *,
    class_names: list[str],
    device: torch.device,
) -> dict[str, Any]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        loss, y_true, y_pred = _run_epoch(model, loader, criterion, None, device)
    metrics = compute_classification_metrics(y_true, y_pred, class_names=class_names)
    metrics["loss"] = float(loss)
    metrics["y_true"] = y_true.astype(int).tolist()
    metrics["y_pred"] = y_pred.astype(int).tolist()
    metrics["confusion_matrix_text"] = format_confusion_matrix(
        metrics["confusion_matrix"], class_names
    )
    return metrics
