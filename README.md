# Real‑time Biosignal Classifier CLI

An interactive CLI to go from biosignal stream to real‑time prediction and robotic control. Built in a research context for a rehab robotic hand prototype, it emphasizes rapid iteration: collect, label, train, and predict live—without leaving the terminal. 

The system has been used to build an ML pipeline capable of recognizing 7 hand gestures with ~95% accuracy, trained in under two minutes of interactive CLI use, and deployed immediately for real-time robotic control.

## What it does

- Data ingest over UDP at user‑defined sampling rates (e.g., EMG 1200 Hz, EEG 250 Hz)
- Interactive CLI to:
  - label and group samples for cross‑validation
  - train a scikit‑learn pipeline
  - run real‑time prediction and (optionally) send serial commands
- Flexible processing per paradigm via configurable windowing, cleaning, and feature extraction (e.g., linear time-domain, CCA, CSP, etc.)
- Reusable artifacts: data, processing, and model can be saved and reloaded
- Optional serial “server” so multiple processes can share a single serial connection. This was used to run multiple prediction tasks simultaneously (e.g., EMG and EEG), and communicate to the same hardware.

## Why it’s helpful

- Agility: immediate loop from collection → training → live prediction
- Structure: labeled, grouped, ML‑ready datasets; sklearn‑friendly processing
- Reproducibility: save/load data, processor, and pipeline for fast iteration
- Hardware loop: send mapped predictions through a serial connection (used to communicate with a robotic hand)

## Quick start

1) Clone the repository

```bash
git clone https://github.com/bruno-rda/biosignal-classifier-cli.git
cd biosignal-classifier-cli
```

2) Environment

- Tested with Python 3.13 on macOS
- Install dependencies:

```bash
pip install -r requirements.txt
```

3) (Optional) Start the shared serial server

```bash
python server.py
```

- Edit `server.py` to set your serial `port` and `baudrate`
- The server exposes a shared manager on port 50000 so multiple scripts can share the same serial connection

4) Run a paradigm script

Scripts contain the paradigm implementation and can be run with the module runner:

```bash
python -m scripts.emg
```

- The CLI opens; use it to collect, train, and predict