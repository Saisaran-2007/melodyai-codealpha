"""Train a next-note LSTM from a directory of MIDI files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Callable, Sequence

import torch
from torch import nn

from model import MusicLSTM
from music_utils import load_dataset

DEFAULT_CHECKPOINT = Path("models/melodyai_lstm.pt")


def _checkpoint_payload(
    model: MusicLSTM,
    token_to_id: dict[str, int],
    seed_sequence: list[int],
    sequence_length: int,
    epochs_completed: int,
    loss_history: list[float],
    completed: bool,
) -> dict:
    return {
        "format_version": 1,
        "completed": completed,
        "model_config": {
            "vocabulary_size": len(token_to_id),
            "embedding_dim": model.embedding.embedding_dim,
            "hidden_size": model.lstm.hidden_size,
        },
        "token_to_id": token_to_id,
        "id_to_token": [token for token, _ in sorted(token_to_id.items(), key=lambda item: item[1])],
        "seed_sequence": seed_sequence,
        "sequence_length": sequence_length,
        "epochs_completed": epochs_completed,
        "loss_history": loss_history,
        "model_state_dict": model.state_dict(),
    }


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def train_model(
    sequences: Sequence[Sequence[str]],
    sequence_length: int = 16,
    epochs: int = 30,
    batch_size: int = 32,
    model_dir: str | Path = "models",
    seed: int = 42,
    on_epoch: Callable[[int, int, float], None] | None = None,
) -> tuple[Path, list[float]]:
    """Fit the LSTM and save a completed checkpoint plus resumable progress state."""
    if sequence_length < 1 or epochs < 1 or batch_size < 1:
        raise ValueError("Sequence length, epochs, and batch size must all be positive.")
    valid_sequences = [list(sequence) for sequence in sequences if sequence]
    if not valid_sequences:
        raise ValueError("No prepared note sequences are available for training.")

    for sequence in valid_sequences:
        for token in sequence:
            pitch_text, duration_text = token.split(":", maxsplit=1)
            pitch, duration_index = int(pitch_text), int(duration_text)
            if not 0 <= pitch <= 127 or not 0 <= duration_index <= 8:
                raise ValueError(f"Invalid note-event token in training data: {token!r}.")

    token_to_id = {token: index for index, token in enumerate(sorted({token for seq in valid_sequences for token in seq}))}
    examples: list[tuple[list[int], list[int]]] = []
    for sequence in valid_sequences:
        encoded = [token_to_id[token] for token in sequence]
        examples.extend(
            (encoded[start : start + sequence_length], encoded[start + 1 : start + sequence_length + 1])
            for start in range(len(encoded) - sequence_length)
        )
    if not examples:
        longest = max(len(sequence) for sequence in valid_sequences)
        raise ValueError(
            f"Not enough notes to train: sequence length {sequence_length} needs at least "
            f"{sequence_length + 1} notes in one MIDI sequence; the longest has {longest}."
        )

    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))
    model = MusicLSTM(len(token_to_id))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()
    inputs = torch.tensor([item[0] for item in examples], dtype=torch.long)
    targets = torch.tensor([item[1] for item in examples], dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    loss_history: list[float] = []
    partial_path = Path(model_dir) / "melodyai_training.pt"
    checkpoint_path = Path(model_dir) / "melodyai_lstm.pt"
    first_encoded = [token_to_id[token] for token in valid_sequences[0]]
    seed_sequence = first_encoded[:sequence_length]

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            permutation = torch.randperm(len(examples), generator=generator)
            total_loss = 0.0
            observed = 0
            for start in range(0, len(examples), batch_size):
                batch_indices = permutation[start : start + batch_size]
                batch_inputs, batch_targets = inputs[batch_indices], targets[batch_indices]
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_inputs)
                loss = criterion(logits.reshape(-1, len(token_to_id)), batch_targets.reshape(-1))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                token_count = batch_targets.numel()
                total_loss += loss.item() * token_count
                observed += token_count

            epoch_loss = total_loss / observed
            loss_history.append(epoch_loss)
            _atomic_save(
                _checkpoint_payload(model, token_to_id, seed_sequence, sequence_length, epoch, loss_history, False),
                partial_path,
            )
            if on_epoch is not None:
                on_epoch(epoch, epochs, epoch_loss)

        _atomic_save(
            _checkpoint_payload(model, token_to_id, seed_sequence, sequence_length, epochs, loss_history, True),
            checkpoint_path,
        )
        partial_path.unlink(missing_ok=True)
    except BaseException:
        _atomic_save(
            _checkpoint_payload(model, token_to_id, seed_sequence, sequence_length, len(loss_history), loss_history, False),
            partial_path,
        )
        raise

    return checkpoint_path, loss_history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MelodyAI's LSTM on local MIDI files.")
    parser.add_argument("--dataset", default="dataset", help="Directory containing .mid/.midi files")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sequences, errors = load_dataset(args.dataset)
    for error in errors:
        print(f"Dataset warning: {error}")
    if not sequences:
        raise SystemExit("No valid MIDI sequences found. Add MIDI files and try again.")
    try:
        checkpoint, losses = train_model(
            [sequence for _, sequence in sequences],
            sequence_length=args.sequence_length,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    except (ValueError, OSError, RuntimeError) as exc:
        raise SystemExit(f"Training failed: {exc}") from exc
    print(f"Training complete. Final loss: {losses[-1]:.4f}. Checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()