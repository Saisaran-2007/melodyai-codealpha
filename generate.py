"""Load a completed MelodyAI checkpoint and sample new note events."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import MusicLSTM
from music_utils import midi_bytes_from_tokens, wav_bytes_from_tokens
from train import DEFAULT_CHECKPOINT


def load_checkpoint(path: str | Path = DEFAULT_CHECKPOINT) -> tuple[MusicLSTM, dict]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"No trained checkpoint found at {checkpoint_path}. Prepare MIDI files and train a model first."
        )
    try:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:
        raise ValueError(f"Could not load model checkpoint {checkpoint_path}: {exc}") from exc
    if not isinstance(checkpoint, dict) or not checkpoint.get("completed", False):
        raise ValueError("The model checkpoint is missing or training did not complete successfully.")

    config = checkpoint.get("model_config", {})
    id_to_token = checkpoint.get("id_to_token", [])
    if not id_to_token or len(id_to_token) != config.get("vocabulary_size"):
        raise ValueError("The checkpoint has an invalid note-event vocabulary.")
    model = MusicLSTM(
        vocabulary_size=len(id_to_token),
        embedding_dim=config["embedding_dim"],
        hidden_size=config["hidden_size"],
    )
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except (KeyError, RuntimeError) as exc:
        raise ValueError(f"Checkpoint model weights do not match its configuration: {exc}") from exc
    model.eval()
    return model, checkpoint


def generate_tokens(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    length: int = 32,
    temperature: float = 0.9,
    seed: int = 42,
    seed_tokens: list[str] | None = None,
) -> list[str]:
    """Generate `length` new notes after the model's (or supplied) seed phrase."""
    if length < 1:
        raise ValueError("Generation length must be at least one note.")
    if temperature <= 0:
        raise ValueError("Sampling temperature must be greater than zero.")
    model, checkpoint = load_checkpoint(checkpoint_path)
    token_to_id = checkpoint["token_to_id"]
    id_to_token = checkpoint["id_to_token"]
    initial_tokens = seed_tokens or [id_to_token[index] for index in checkpoint["seed_sequence"]]
    if not initial_tokens:
        raise ValueError("The checkpoint does not contain a usable seed sequence.")
    missing = [token for token in initial_tokens if token not in token_to_id]
    if missing:
        raise ValueError(f"Seed contains note events not present in the trained vocabulary: {missing[:3]}.")

    context = [token_to_id[token] for token in initial_tokens]
    torch.manual_seed(seed)
    generated: list[int] = []
    max_context = max(1, int(checkpoint.get("sequence_length", len(context))))
    with torch.inference_mode():
        for _ in range(length):
            input_ids = torch.tensor([context[-max_context:]], dtype=torch.long)
            logits = model(input_ids)[0, -1] / temperature
            probabilities = torch.softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probabilities, num_samples=1).item())
            if not 0 <= next_id < len(id_to_token):
                raise ValueError("Model produced a note ID outside its trained vocabulary.")
            generated.append(next_id)
            context.append(next_id)

    return [id_to_token[index] for index in context[: len(initial_tokens)] + generated]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a melody from a trained MelodyAI model.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--length", type=int, default=32, help="Number of new notes to sample")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="generated/melody.mid")
    args = parser.parse_args()

    try:
        tokens = generate_tokens(args.checkpoint, args.length, args.temperature, args.seed)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(midi_bytes_from_tokens(tokens))
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        raise SystemExit(f"Generation failed: {exc}") from exc
    print(f"Generated {len(tokens)} notes using the trained model: {output_path}")


if __name__ == "__main__":
    main()