# MelodyAI — AI Music Generator

MelodyAI is an educational, CPU-capable music-generation app. It learns next-note patterns from MIDI files with a PyTorch LSTM, where each event encodes a MIDI pitch and a quantized note duration. It exports generated sequences to MIDI and synthesizes a WAV preview locally for browser playback; no paid service, account, GPU, or browser MIDI support is required.

The model learns from the MIDI files you prepare. The repository intentionally does not download music or include a dataset. Supply MIDI files you own or have permission to use, and verify the license and permitted use of any public dataset before adding it.

## Requirements

- Windows 10/11 and Python 3.11–3.13
- Internet access for the initial package installation
- A few non-drum MIDI files; short files work if the sequence length is set below the number of notes

## Install (Windows PowerShell)

Run these commands in the project directory:

```powershell
py -3.13 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If Python 3.13 is not installed, use an installed supported version, for example `py -3.12 -m venv .venv`.

## Prepare a dataset

Copy permitted `.mid` or `.midi` files into `dataset\`, including subfolders if useful. Alternatively, upload files in the app's sidebar. The loader skips drum tracks, sorts note events by onset, quantizes durations to a small musical vocabulary, and reports unreadable or note-free files. No dataset is bundled or fetched automatically.

For a small CPU training run, use a handful of short, varied melodies. Each training sequence needs at least `sequence length + 1` note events. Start with a sequence length of 4–16, 10–30 epochs, and a batch size of 8–32; increase those values as the dataset grows. Files are treated as ordered monophonic event streams for training; simultaneous notes are ordered by pitch.

## Run the app

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

In the browser, prepare the local and/or uploaded MIDI files, open **Train model**, and start training. The dashboard shows epoch loss and only reports a trained model after the run finishes. Then open **Generate**, choose the number of new notes and sampling temperature, and download the resulting MIDI. The initial notes of each generated melody are the seed phrase saved in the trained checkpoint.

## Publish with GitHub and deploy

### Publish the source to GitHub

In VS Code, open **Source Control**, choose **Initialize Repository** if prompted, then choose **Publish to GitHub** and sign in. Pick a private or public repository as appropriate. The included GitHub Actions workflow runs the test suite on pushes and pull requests to `main`. The local `.venv`, MIDI data, model checkpoints, and generated audio are excluded from Git; add only dataset files you have permission to redistribute.

### Deploy the Streamlit app

Deploy directly from GitHub using [Streamlit Community Cloud](https://share.streamlit.io/): sign in with the GitHub account that owns the repository, choose **Create app**, select the repository and `main` branch, set the app file to `app.py`, then deploy. The cloud app starts without a checkpoint; upload permitted MIDI files in its sidebar and train there. Cloud instances have limited CPU and temporary local storage, so keep datasets small and download any generated MIDI you need to keep.

Vercel does not host Streamlit's persistent web server process. This repository intentionally has no `vercel.json`; deploying this app directly to Vercel will not run it. Use Streamlit Community Cloud for the simplest deployment, or a Python container host if you specifically need another provider.

### Optional container run

```powershell
docker build -t melodyai .
docker run --rm -p 8501:8501 melodyai
```

Then open `http://localhost:8501`.

The built-in preview is an offline, simple plucked-tone synthesizer that produces WAV audio from the same note events, so playback does not depend on whether the browser can play MIDI. It is a functional listening preview, not a sampled piano soundfont. For an optional higher-quality render, install FluidSynth and a soundfont whose license permits your use, then run:

```powershell
fluidsynth -ni -F generated\rendered.wav -r 44100 path\to\licensed-soundfont.sf2 generated\melody.mid
```

The application itself does not require FluidSynth or distribute a soundfont.

## Train and generate from PowerShell

```powershell
python train.py --dataset dataset --epochs 30 --sequence-length 16 --batch-size 32 --seed 42
python generate.py --checkpoint models\melodyai_lstm.pt --length 32 --temperature 0.9 --seed 7 --output generated\melody.mid
```

Training writes the final model, learned vocabulary, seed phrase, configuration, and loss history to `models\melodyai_lstm.pt`. Per-epoch progress is written separately to `models\melodyai_training.pt`; an interrupted run is not accepted for generation and does not replace an earlier completed checkpoint. Generation refuses missing, incomplete, or incompatible checkpoints rather than falling back to random notes.

## Tests

```powershell
python -m pytest -q
```

The tests cover MIDI parsing and pitch-duration encoding, invalid input, LSTM tensor shapes, a real short training run, autoregressive generation, and MIDI/WAV export.

## Troubleshooting

- **`py` or Python is not found:** Install Python 3.11–3.13 from python.org and enable the Python launcher, then reopen PowerShell.
- **Torch installation fails:** Upgrade pip inside the virtual environment and retry `python -m pip install -r requirements.txt`. The project uses CPU PyTorch and does not need CUDA.
- **No valid MIDI files:** Check the folder path, `.mid`/`.midi` extensions, and that files contain pitched notes rather than percussion only.
- **Not enough notes:** Lower sequence length or add longer MIDI sequences; training needs one extra target note beyond the input length.
- **Generation is unavailable:** Complete a training run first. An interrupted training checkpoint is deliberately not considered a trained model.
- **Playback is silent:** Confirm browser audio is enabled and the tab is not muted; download the WAV/MIDI and open it in an audio or MIDI application as an alternative.