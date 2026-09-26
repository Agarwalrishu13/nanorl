# nanorl — alignment for tiny models

From-scratch SFT (Supervised Fine-Tuning) and DPO (Direct Preference
Optimization) trainers for tiny local language models.

## Why?

Tiny models suffer from *pre-training rot*: trained only to continue text,
they complete stories beautifully but ignore instructions. Alignment is the
fix — two losses, written from scratch on raw PyTorch:

- **SFT** — "here is good text, learn to continue it." Masked next-token
  cross-entropy; padding never teaches the model anything.
- **DPO** — "of these two answers, this one was better." The model is nudged
  toward the chosen answer relative to a frozen copy of itself. The reference
  never moves and never takes a gradient.

## The tokenizer question, answered honestly

Text becomes numbers at the **byte** level (see `nanorl/tokens.py`): 256 byte
values plus three specials (pad, begin, end) — 259 in total. Nothing to
download, nothing to train first, and *any* text on Earth round-trips: English,
Hindi, emoji, code. A trained BPE tokenizer compresses better; bytes never fail
on a character nobody predicted.

## Usage

```bash
# SFT: teach the model to answer, from examples
python scripts/train_sft.py --model my-tiny-llm.pt --data data/instruct.json

# DPO: teach it which answers are better, from preference pairs
python scripts/train_dpo.py --model my-tiny-llm.pt --data data/preferences.json
```

The data files are plain JSON lists — `{"text": "..."}` or
`{"instruction": "...", "response": "..."}` for SFT,
`{"chosen": "...", "rejected": "..."}` for DPO.

Works from any folder:

```bash
python C:\path\to\nanorl\scripts\train_sft.py --model model.pt --data examples.json
```

## What is tested

`python -m unittest discover tests` — 18 tests, CPU, a few seconds:

- both losses produce finite, positive numbers, and **go down** when trained
- padding provably contributes nothing to the loss
- DPO with an unchanged policy sits exactly at log 2 (the neutral point)
- gradients reach the policy, never the frozen reference
- the byte tokenizer round-trips unicode (Hindi, emoji) and pads correctly
- a saved checkpoint loads back as the same model

## Pairs with

- [nanobrain](https://github.com/Agarwalrishu13/nanobrain) — pre-training from scratch
- [nanollama.c](https://github.com/Agarwalrishu13/nanollama.c) — the C inference engine
- [nanoforge](https://github.com/Agarwalrishu13/nanoforge) — the offline studio for tiny models

## Status

- SFT: implemented, tested
- DPO: implemented, tested (sequence-level, frozen reference)
- Dependencies: torch only
- License: MIT
