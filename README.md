# nanorl — Alignment for tiny models

From-scratch SFT (Supervised Fine-Tuning) and DPO (Direct Preference 
Optimization) trainers for your tiny models. 

## Why?
Tiny models (like the ones running on `nanollama.c`) suffer from "pre-training rot" — they are great at completing stories but bad at following instructions. Alignment is the fix.

## Usage
```bash
# SFT on instruct dataset
python scripts/train_sft.py --model my-tiny-llm.pt --data data/instruct.json

# DPO alignment on preference data
python scripts/train_dpo.py --model my-tiny-llm.pt --data data/preferences.json
```

## Status
- SFT: Implemented
- DPO: Implemented
- Dependencies: torch only
- License: MIT
