"""LSTM/GRU sequence classifier for ISL gloss recognition.

Input: (batch, sequence_length, input_size)
Output: logits (batch, num_classes)

Softmax is applied at inference time, not inside training (CrossEntropyLoss
expects logits). This module does not import FastAPI.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


def _rnn_class(rnn_type: str) -> type[nn.Module]:
    name = rnn_type.strip().lower()
    if name == "lstm":
        return nn.LSTM
    if name == "gru":
        return nn.GRU
    raise ValueError(f"rnn_type must be 'lstm' or 'gru', got {rnn_type!r}")


class ISLSequenceModel(nn.Module):
    """Temporal sign classifier: sequence_length × feature_dim → class logits."""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        rnn_type: str = "lstm",
        bidirectional: bool = False,
        sequence_length: int | None = None,
    ) -> None:
        super().__init__()
        if input_size < 1:
            raise ValueError("input_size must be >= 1")
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        if hidden_size < 1:
            raise ValueError("hidden_size must be >= 1")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.input_size = int(input_size)
        self.num_classes = int(num_classes)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout_p = float(dropout)
        self.rnn_type = rnn_type.strip().lower()
        self.bidirectional = bool(bidirectional)
        self.sequence_length = sequence_length

        rnn_dropout = self.dropout_p if self.num_layers > 1 else 0.0
        self.rnn = _rnn_class(self.rnn_type)(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=rnn_dropout,
            bidirectional=self.bidirectional,
            batch_first=True,
        )
        rnn_out = self.hidden_size * (2 if self.bidirectional else 1)
        self.head_dropout = nn.Dropout(self.dropout_p)
        self.fc = nn.Linear(rnn_out, self.num_classes)

    def config_dict(self) -> dict[str, Any]:
        return {
            "rnn_type": self.rnn_type,
            "input_size": self.input_size,
            "num_classes": self.num_classes,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout_p,
            "bidirectional": self.bidirectional,
            "sequence_length": self.sequence_length,
        }

    def _final_hidden(self, hidden: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        h_n = hidden[0] if isinstance(hidden, tuple) else hidden
        if self.bidirectional:
            return torch.cat([h_n[-2], h_n[-1]], dim=1)
        return h_n[-1]

    def forward(self, sequences: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if sequences.dim() != 3:
            raise ValueError(f"Expected (batch, time, features), got {tuple(sequences.shape)}")
        if sequences.size(-1) != self.input_size:
            raise ValueError(
                f"Expected input_size={self.input_size}, got {sequences.size(-1)}"
            )

        if lengths is None:
            _output, hidden = self.rnn(sequences)
        else:
            lengths_cpu = lengths.detach().cpu().clamp(min=1, max=sequences.size(1)).long()
            packed = pack_padded_sequence(
                sequences,
                lengths_cpu,
                batch_first=True,
                enforce_sorted=False,
            )
            _output, hidden = self.rnn(packed)
        features = self.head_dropout(self._final_hidden(hidden))
        return self.fc(features)

    def predict_proba(self, sequences: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        return torch.softmax(self.forward(sequences, lengths), dim=-1)
