# DarkGPT

A GPT style transformer language model with no limitations built from scratch in Python using PyTorch. Runs entirely locally; no API, no internet, no restrictions.
Built as a learning project to explore the capabilities of artificial intellegence.

---
## Models:
_DarkGPT v0.1_alpha:_

```bash
3.6 million params
Steps: 10000
Context length: 256
Rate: 3e-4
Conversational data: 5000 kb
Wikipedia data: 20000 kb
Other data: 2000 kb
Total (training data): 30000 kb.
```

_DarkGPT v0.2_super:_

```bash
6.3 million params
Steps: 15000
Context length: 256
Rate: 2e-4
Conversational data: 60000 kb
Wikipedia data: 25000 kb
Other data: 2500 kb
Total (training data): 87500 kb.
```

---

## What it is

DarkGPT is a decoder only transformer (the same architecture behind ChatGPT, Claude, Gemini etc..) Every component is written by hand. It is not a remake of any existing model.

**Architecture**
- 3.6M parameters
- 4 transformer blocks, 4 attention heads
- 256 token context window

---

## Getting started

**1. Install dependencies**
```bash
pip install torch
pip install torch-directml  # AMD GPU users only
```

**2. Generate training data (if you add more data)**
```bash
python data.py         # scrapes Wikipedia (geopolitics, history)
# If you wish to keep it as in release, skip this step.
```

**3. Train the model (if you add more data)**
```bash
python train.py --data data/training.txt --steps 10000
# If you wish to keep it as in release, skip this step.
```

**4. Chat with it**
```bash
python chat.py
```

---

## Project structure

```
dark_gpt/
├── model.py                 # The model
├── train.py                 # Training loop and checkpointing
├── chat.py                  # Tkinter chat interface
├── data.py            # Wikipedia data scraper
├── data/                    # Training data (generated locally)
└── checkpoints/
    └── model.pt             # Trained weights
```

---

## Hardware

Trained on an AMD RX 7600 using DirectML. Also works on NVIDIA (CUDA) or CPU, the training script detects your hardware automatically.

---

## Limitations

This is a small research/learning model. It may produce incoherent or off topic responses, especially to short prompts.
NOTICE!! This will be fixed in the next release!!

---

## Releases

**Current version:** DarkGPT v0.1 Alpha
