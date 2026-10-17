"""MIDI parsing, note-event encoding, export, and lightweight audio rendering."""

from __future__ import annotations

import io
import wave
from pathlib import Path
from typing import Iterable

import numpy as np
import pretty_midi

DURATION_BINS = (0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)


def duration_bin_index(duration: float) -> int:
    """Return the closest supported duration bin for a note in seconds."""
    return min(range(len(DURATION_BINS)), key=lambda index: abs(DURATION_BINS[index] - duration))


def encode_event(pitch: int, duration: float) -> str:
    if not 0 <= int(pitch) <= 127:
        raise ValueError(f"MIDI pitch must be between 0 and 127, got {pitch}.")
    return f"{int(pitch)}:{duration_bin_index(float(duration))}"


def decode_event(token: str) -> tuple[int, float]:
    try:
        pitch_text, duration_text = token.split(":", maxsplit=1)
        pitch, duration_index = int(pitch_text), int(duration_text)
        if not 0 <= pitch <= 127 or not 0 <= duration_index < len(DURATION_BINS):
            raise ValueError
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid note-event token: {token!r}.") from exc
    return pitch, DURATION_BINS[duration_index]


def parse_midi_bytes(content: bytes, source: str = "uploaded MIDI") -> list[str]:
    """Parse a MIDI byte stream into ordered pitch-duration event tokens."""
    if not content:
        raise ValueError(f"{source} is empty.")
    try:
        midi = pretty_midi.PrettyMIDI(midi_file=io.BytesIO(content))
    except Exception as exc:
        raise ValueError(f"Could not read {source} as MIDI: {exc}") from exc

    notes = [
        note
        for instrument in midi.instruments
        if not instrument.is_drum
        for note in instrument.notes
        if note.end > note.start
    ]
    notes.sort(key=lambda note: (note.start, note.pitch, note.end))
    if not notes:
        raise ValueError(f"{source} contains no pitched notes.")
    return [encode_event(note.pitch, note.end - note.start) for note in notes]


def load_dataset(directory: str | Path) -> tuple[list[tuple[str, list[str]]], list[str]]:
    """Load supported MIDI files, returning usable sequences and per-file errors."""
    root = Path(directory)
    if not root.is_dir():
        return [], [f"Dataset directory does not exist: {root}"]

    midi_paths = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".mid", ".midi"}
    )
    if not midi_paths:
        return [], [f"No .mid or .midi files found in {root}."]

    sequences: list[tuple[str, list[str]]] = []
    errors: list[str] = []
    for path in midi_paths:
        try:
            sequences.append((str(path), parse_midi_bytes(path.read_bytes(), str(path))))
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
    return sequences, errors


def midi_bytes_from_tokens(tokens: Iterable[str]) -> bytes:
    """Export note events as a standards-compliant, sequential piano MIDI file."""
    events = [decode_event(token) for token in tokens]
    if not events:
        raise ValueError("Cannot export an empty melody.")

    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program("Acoustic Grand Piano"))
    cursor = 0.0
    for pitch, duration in events:
        instrument.notes.append(
            pretty_midi.Note(velocity=88, pitch=pitch, start=cursor, end=cursor + duration)
        )
        cursor += duration
    midi.instruments.append(instrument)
    output = io.BytesIO()
    midi.write(output)
    return output.getvalue()


def wav_bytes_from_tokens(tokens: Iterable[str], sample_rate: int = 16000) -> bytes:
    """Synthesize a simple offline plucked-tone preview playable in modern browsers."""
    events = [decode_event(token) for token in tokens]
    if not events:
        raise ValueError("Cannot render an empty melody.")
    if sample_rate < 8000:
        raise ValueError("Sample rate must be at least 8000 Hz.")

    max_samples = sample_rate * 120
    chunks: list[np.ndarray] = []
    rendered_samples = 0
    for pitch, duration in events:
        count = min(round(duration * sample_rate), max_samples - rendered_samples)
        if count <= 0:
            break
        time = np.arange(count, dtype=np.float32) / sample_rate
        frequency = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
        tone = np.sin(2 * np.pi * frequency * time) + 0.2 * np.sin(4 * np.pi * frequency * time)
        envelope = np.minimum(time / 0.012, 1.0) * np.exp(-2.4 * time / max(duration, 0.05))
        chunks.append(np.asarray(tone * envelope * 0.28 * 32767, dtype=np.int16))
        rendered_samples += count

    pcm = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.int16)
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return output.getvalue()