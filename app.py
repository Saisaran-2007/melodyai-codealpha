"""MelodyAI Streamlit music-generation dashboard."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from generate import DEFAULT_CHECKPOINT, generate_tokens
from music_utils import decode_event, load_dataset, midi_bytes_from_tokens, parse_midi_bytes, wav_bytes_from_tokens
from train import train_model

DATASET_DIR = Path("dataset")
GENERATED_DIR = Path("generated")

st.set_page_config(page_title="MelodyAI | Music Generator", page_icon="♫", layout="wide")
st.markdown(
    """
    <style>
    :root { color-scheme: dark; --ink: #f3effa; --muted: #a9a4b2; --panel: #201f27; --line: #383541; --violet: #ad94f5; --coral: #ff8d78; }
    .stApp { background-color: #15151a; background-image: linear-gradient(rgba(255,255,255,.018) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.018) 1px, transparent 1px); background-size: 36px 36px; color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(21, 21, 26, 0.9); }
    [data-testid="stSidebar"] { background: #1b1a20; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] h2 { font-size: 1.08rem; }
    h1, h2, h3 { color: var(--ink); font-family: 'Trebuchet MS', sans-serif; letter-spacing: 0; }
    p, label, span { letter-spacing: 0; }
    .block-container { max-width: 1360px; padding-top: 2rem; padding-bottom: 3rem; }
    .studio-hero { position: relative; overflow: hidden; margin: 1.5rem 0 1.35rem; padding: 1.25rem 1.5rem 1.35rem; border: 1px solid #423b50; border-left: 3px solid var(--coral); border-radius: 8px; background: linear-gradient(112deg, #26212f 0%, #201f27 58%, #1c1b22 100%); }
    .studio-hero::after { position: absolute; top: 0; right: 0; width: 34%; height: 100%; content: ''; pointer-events: none; background: repeating-linear-gradient(135deg, transparent 0 13px, rgba(173,148,245,.055) 13px 14px); mask-image: linear-gradient(90deg, transparent, #000); }
    .studio-kicker { color: var(--violet); font-size: .73rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
    .studio-main { position: relative; z-index: 1; display: flex; align-items: flex-end; justify-content: space-between; gap: 1.5rem; margin-top: .45rem; }
    .studio-title { margin: 0; color: var(--ink); font-family: 'Trebuchet MS', sans-serif; font-size: 2.35rem; font-weight: 800; line-height: 1.1; }
    .studio-title span { color: var(--coral); }
    .studio-copy { margin: .55rem 0 0; color: #bbb5c5; font-size: .96rem; }
    .studio-tags { display: flex; flex-wrap: wrap; gap: .5rem; padding-bottom: .1rem; }
    .studio-tag { padding: .4rem .65rem; border: 1px solid #494255; border-radius: 4px; color: #d4c8f5; font-size: .7rem; font-weight: 700; letter-spacing: .06em; white-space: nowrap; }
    .metric-panel { min-height: 102px; padding: 16px 18px; border: 1px solid var(--line); border-top: 2px solid var(--violet); border-radius: 6px; background: var(--panel); }
    .metric-label { color: var(--muted); font-size: 0.82rem; }
    .metric-value { overflow-wrap: anywhere; margin-top: 7px; color: var(--ink); font-size: 1.3rem; font-weight: 700; }
    .subtle { color: var(--muted); }
    div[data-testid="stFormSubmitButton"] button[kind="primary"], div[data-testid="stButton"] button[kind="primary"] { border: 1px solid #a287ec; border-radius: 5px; background: #7656c9; color: white; transition: background .15s ease, transform .15s ease; }
    div[data-testid="stFormSubmitButton"] button[kind="primary"]:hover, div[data-testid="stButton"] button[kind="primary"]:hover { border-color: #bea8ff; background: #886bd8; transform: translateY(-1px); }
    div[data-testid="stDownloadButton"] button, div[data-testid="stButton"] button { border-radius: 5px; }
    div[data-testid="stTabs"] button[aria-selected="true"] { color: var(--violet); }
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background-color: var(--violet); }
    [data-testid="stProgressBar"] > div > div { background-color: var(--violet); }
    [data-testid="stMetric"] { border: 1px solid var(--line); border-radius: 6px; background: var(--panel); }
    [data-testid="stAudio"] { margin: .6rem 0; }
    :focus-visible { outline: 2px solid var(--violet) !important; outline-offset: 2px; }
    @media (max-width: 700px) { .block-container { padding: 1.2rem 1rem 2rem; } .studio-hero { margin-top: .8rem; padding: 1rem; } .studio-main { align-items: flex-start; flex-direction: column; gap: .85rem; } .studio-title { font-size: 2rem; } .metric-panel { min-height: 88px; padding: 12px; } .metric-value { font-size: 1.12rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


def piano_roll(tokens: list[str], title: str = "Generated piano roll") -> go.Figure:
    events = [decode_event(token) for token in tokens]
    starts: list[float] = []
    durations: list[float] = []
    pitches: list[int] = []
    cursor = 0.0
    for pitch, duration in events:
        starts.append(cursor)
        durations.append(duration)
        pitches.append(pitch)
        cursor += duration

    figure = go.Figure(
        go.Bar(
            x=durations,
            y=pitches,
            base=starts,
            orientation="h",
            width=0.72,
            marker={"color": "#9a7bea", "line": {"color": "#c6b5ff", "width": 0.6}},
            customdata=[[pretty_pitch(pitch), duration] for pitch, duration in events],
            hovertemplate="%{customdata[0]}<br>Start: %{base:.2f}s<br>Duration: %{customdata[1]:.2f}s<extra></extra>",
        )
    )
    if pitches:
        low, high = min(pitches), max(pitches)
        tick_values = list(range((low // 12) * 12, high + 1, 12))
        figure.update_yaxes(
            tickmode="array",
            tickvals=tick_values,
            ticktext=[pretty_pitch(pitch) for pitch in tick_values],
            range=[max(-1, low - 2), min(128, high + 2)],
        )
    figure.update_layout(
        title=title,
        height=340,
        margin={"l": 44, "r": 20, "t": 50, "b": 42},
        paper_bgcolor="#201f27",
        plot_bgcolor="#201f27",
        font={"color": "#e8e4ef", "family": "Trebuchet MS, sans-serif"},
        xaxis={"title": "Time (seconds)", "gridcolor": "#37333f", "zerolinecolor": "#48434f"},
        yaxis={"title": "Pitch", "gridcolor": "#37333f", "zerolinecolor": "#48434f"},
        showlegend=False,
        bargap=0.18,
    )
    return figure


def pretty_pitch(pitch: int) -> str:
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    return f"{names[pitch % 12]}{pitch // 12 - 1}"


def metric_panel(label: str, value: str) -> None:
    st.markdown(
        f'<div class="metric-panel"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


if "prepared_sequences" not in st.session_state:
    initial_sequences, initial_errors = load_dataset(DATASET_DIR)
    st.session_state.prepared_sequences = initial_sequences
    st.session_state.dataset_errors = initial_errors
if "generation_history" not in st.session_state:
    st.session_state.generation_history = []
if "loss_history" not in st.session_state:
    st.session_state.loss_history = []

st.markdown(
    '<section class="studio-hero"><div class="studio-kicker">CodeAlpha / AI Music Generation</div>'
    '<div class="studio-main"><div><h1 class="studio-title">MelodyAI<span>.</span></h1>'
    '<p class="studio-copy">Shape note patterns into new melodies. Train, sample, listen, export.</p></div>'
    '<div class="studio-tags"><span class="studio-tag">PYTORCH / LSTM</span>'
    '<span class="studio-tag">PITCH + DURATION</span></div></div></section>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("MIDI library")
    dataset_path = st.text_input("Local dataset folder", value=str(DATASET_DIR))
    uploads = st.file_uploader("Add MIDI files", type=["mid", "midi"], accept_multiple_files=True)
    if st.button("Prepare MIDI files", type="primary", use_container_width=True):
        loaded, errors = load_dataset(dataset_path)
        for upload in uploads or []:
            try:
                loaded.append((upload.name, parse_midi_bytes(upload.getvalue(), upload.name)))
            except ValueError as exc:
                errors.append(str(exc))
        st.session_state.prepared_sequences = loaded
        st.session_state.dataset_errors = errors
        st.rerun()
    if st.session_state.dataset_errors:
        for error in st.session_state.dataset_errors:
            st.warning(error)
    else:
        st.success(f"{len(st.session_state.prepared_sequences)} MIDI file(s) prepared")
    if st.session_state.prepared_sequences:
        note_count = sum(len(sequence) for _, sequence in st.session_state.prepared_sequences)
        st.caption(f"{note_count:,} note events ready")

checkpoint_exists = DEFAULT_CHECKPOINT.is_file()
trained = False
checkpoint_loss: list[float] = []
if checkpoint_exists:
    try:
        from generate import load_checkpoint

        _, checkpoint_info = load_checkpoint(DEFAULT_CHECKPOINT)
        trained = True
        checkpoint_loss = checkpoint_info.get("loss_history", [])
    except (FileNotFoundError, ValueError, OSError, RuntimeError):
        trained = False

training_status = "Trained model ready" if trained else ("Training incomplete" if Path("models/melodyai_training.pt").exists() else "Not trained")
dataset_count = len(st.session_state.prepared_sequences)
note_total = sum(len(sequence) for _, sequence in st.session_state.prepared_sequences)
column1, column2, column3, column4 = st.columns(4)
with column1:
    metric_panel("Prepared MIDI files", str(dataset_count))
with column2:
    metric_panel("Note events", f"{note_total:,}")
with column3:
    metric_panel("Model status", training_status)
with column4:
    latest_loss = (st.session_state.loss_history or checkpoint_loss)
    metric_panel("Latest training loss", f"{latest_loss[-1]:.4f}" if latest_loss else "Not available")

overview_tab, training_tab, generation_tab = st.tabs(["Overview", "Train model", "Generate"])

with overview_tab:
    st.subheader("Training signal")
    loss_values = st.session_state.loss_history or checkpoint_loss
    if loss_values:
        st.line_chart({"Cross-entropy loss": loss_values}, height=220)
    else:
        st.info("No completed training run yet. Prepare MIDI notes and train a model to see its loss curve.")
    st.subheader("Recent generations")
    if not st.session_state.generation_history:
        st.caption("Generated melodies will appear here after your first sampling run.")
    for item in st.session_state.generation_history[:3]:
        st.markdown(f"**{item['name']}** · {len(item['tokens'])} notes · {item['created']}")
        st.audio(item["wav"], format="audio/wav")
        st.download_button("Download MIDI", item["midi"], file_name=item["name"], mime="audio/midi", key=f"overview_{item['name']}")

with training_tab:
    st.subheader("Train the note-sequence model")
    st.markdown("Each training example predicts the next pitch-and-duration event from the preceding notes.")
    with st.form("training_settings"):
        setting1, setting2, setting3 = st.columns(3)
        with setting1:
            sequence_length = st.number_input("Sequence length", min_value=2, max_value=256, value=16)
        with setting2:
            epochs = st.number_input("Epochs", min_value=1, max_value=500, value=30)
        with setting3:
            batch_size = st.number_input("Batch size", min_value=1, max_value=512, value=32)
        training_seed = st.number_input("Reproducibility seed", min_value=0, max_value=2_147_483_647, value=42)
        start_training = st.form_submit_button("Train on prepared MIDI", type="primary")

    if start_training:
        sequences = [sequence for _, sequence in st.session_state.prepared_sequences]
        if not sequences:
            st.error("No valid MIDI sequences are prepared. Add files in the sidebar first.")
        else:
            progress = st.progress(0.0, text="Starting CPU training…")
            loss_chart = st.empty()
            live_losses: list[float] = []

            def report_epoch(epoch: int, total_epochs: int, loss: float) -> None:
                live_losses.append(loss)
                progress.progress(epoch / total_epochs, text=f"Epoch {epoch}/{total_epochs} · loss {loss:.4f}")
                loss_chart.line_chart({"Cross-entropy loss": live_losses}, height=220)

            try:
                checkpoint_path, losses = train_model(
                    sequences,
                    sequence_length=int(sequence_length),
                    epochs=int(epochs),
                    batch_size=int(batch_size),
                    seed=int(training_seed),
                    on_epoch=report_epoch,
                )
                st.session_state.loss_history = losses
                progress.progress(1.0, text="Training run completed")
                st.session_state.training_complete_message = (
                    f"Training completed. Final loss: {losses[-1]:.4f}. Checkpoint saved to {checkpoint_path}."
                )
                st.rerun()
            except (ValueError, OSError, RuntimeError, KeyboardInterrupt) as exc:
                st.error(f"Training did not complete: {exc}")

    training_message = st.session_state.pop("training_complete_message", None)
    if training_message:
        st.success(training_message)

with generation_tab:
    st.subheader("Sample from your trained model")
    if not trained:
        st.warning("Generation is disabled until a training run completes successfully.")
    with st.form("generation_settings"):
        gen_column1, gen_column2, gen_column3 = st.columns(3)
        with gen_column1:
            generation_length = st.slider("New notes", min_value=1, max_value=128, value=32)
        with gen_column2:
            temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=0.9, step=0.1)
        with gen_column3:
            generation_seed = st.number_input("Sampling seed", min_value=0, max_value=2_147_483_647, value=7)
        generate_button = st.form_submit_button("Generate melody", type="primary", disabled=not trained)

    if generate_button:
        try:
            generated_tokens = generate_tokens(
                DEFAULT_CHECKPOINT,
                length=int(generation_length),
                temperature=float(temperature),
                seed=int(generation_seed),
            )
            midi_content = midi_bytes_from_tokens(generated_tokens)
            wav_content = wav_bytes_from_tokens(generated_tokens)
            GENERATED_DIR.mkdir(parents=True, exist_ok=True)
            filename = f"melody_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mid"
            (GENERATED_DIR / filename).write_bytes(midi_content)
            record = {
                "name": filename,
                "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "tokens": generated_tokens,
                "midi": midi_content,
                "wav": wav_content,
            }
            st.session_state.generation_history.insert(0, record)
            st.success("Melody generated from the trained checkpoint.")
            st.audio(wav_content, format="audio/wav")
            st.download_button("Download MIDI", midi_content, file_name=filename, mime="audio/midi", key=f"generated_{filename}")
            st.plotly_chart(piano_roll(generated_tokens), use_container_width=True)
        except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
            st.error(f"Could not generate a melody: {exc}")