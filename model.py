"""Small CPU-friendly LSTM for next-note-event prediction."""

from __future__ import annotations

import torch
from torch import nn


class MusicLSTM(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_dim: int = 64, hidden_size: int = 96):
        super().__init__()
        if vocabulary_size < 1:
            raise ValueError("The model vocabulary must contain at least one note event.")
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, vocabulary_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)
        sequence_output, _ = self.lstm(embedded)
        return self.output(sequence_output)