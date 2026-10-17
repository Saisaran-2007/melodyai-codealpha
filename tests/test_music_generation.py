from __future__ import annotations

import io

import pretty_midi
import pytest
import torch

from generate import generate_tokens
from model import MusicLSTM
from music_utils import decode_event, encode_event, midi_bytes_from_tokens, parse_midi_bytes, wav_bytes_from_tokens
from train import train_model


def make_test_midi() -> bytes:
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    instrument.notes = [
        pretty_midi.Note(velocity=90, pitch=pitch, start=index * 0.5, end=index * 0.5 + 0.3)
        for index, pitch in enumerate((60, 64, 67, 72, 67, 64, 60, 62))
    ]
    midi.instruments.append(instrument)
    output = io.BytesIO()
    midi.write(output)
    return output.getvalue()


def test_midi_preprocessing_extracts_pitch_and_duration() -> None:
    tokens = parse_midi_bytes(make_test_midi(), "test.mid")

    assert len(tokens) == 8
    assert decode_event(tokens[0]) == (60, 0.25)
    assert encode_event(64, 0.3) == tokens[1]


def test_invalid_midi_is_reported() -> None:
    with pytest.raises(ValueError, match="Could not read"):
        parse_midi_bytes(b"not a MIDI file", "broken.mid")


def test_lstm_input_and_output_shapes() -> None:
    model = MusicLSTM(vocabulary_size=11, embedding_dim=8, hidden_size=12)
    output = model(torch.zeros((3, 5), dtype=torch.long))

    assert output.shape == (3, 5, 11)


def test_training_generation_midi_and_audio_export(tmp_path) -> None:
    tokens = parse_midi_bytes(make_test_midi())
    checkpoint, losses = train_model(
        [tokens, tokens],
        sequence_length=3,
        epochs=2,
        batch_size=4,
        model_dir=tmp_path,
        seed=13,
    )
    melody = generate_tokens(checkpoint, length=5, temperature=0.8, seed=9)

    assert checkpoint.is_file()
    assert len(losses) == 2
    assert len(melody) == 8
    assert all(0 <= decode_event(token)[0] <= 127 for token in melody)

    exported = pretty_midi.PrettyMIDI(midi_file=io.BytesIO(midi_bytes_from_tokens(melody)))
    assert len(exported.instruments[0].notes) == len(melody)
    assert wav_bytes_from_tokens(melody).startswith(b"RIFF")


def test_training_rejects_too_short_sequence(tmp_path) -> None:
    with pytest.raises(ValueError, match="Not enough notes"):
        train_model([["60:2", "62:2"]], sequence_length=2, epochs=1, model_dir=tmp_path)