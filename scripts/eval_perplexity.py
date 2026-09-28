"""How well does a model know this text? One number: perplexity.

    python scripts/eval_perplexity.py --model my-tiny-llm.pt --text data/sample.txt

Lower is better: 1.0 means the model predicts every byte exactly, and the
vocabulary size (259 for byte-level models) means it knows nothing. Works on
any checkpoint saved by the nanorl trainers. Runs on the CPU.
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from anywhere

import torch

from nanorl import tokens
from nanorl.trainer import sft_loss


def main():
    parser = argparse.ArgumentParser(description="Perplexity of a model on a text file")
    parser.add_argument("--model", required=True, help="path to a .pt model checkpoint")
    parser.add_argument("--text", required=True, help="path to a plain-text file")
    parser.add_argument("--seq-len", type=int, default=256)
    args = parser.parse_args()

    if not Path(args.model).is_file():
        raise SystemExit("There is no model at %s." % args.model)
    model = torch.load(args.model, weights_only=False, map_location="cpu")
    model.eval()
    text = Path(args.text).read_text(encoding="utf-8")
    if not text.strip():
        raise SystemExit("The text file is empty.")

    batch, mask = tokens.pad_batch([text], args.seq_len)
    with torch.no_grad():
        loss = sft_loss(model, batch, mask).item()
    perplexity = math.exp(min(loss, 20))  # capped so a broken model prints a number
    print("loss:        %.4f" % loss)
    print("perplexity:  %.1f   (1 is perfect; %.0f means no better than chance)"
          % (perplexity, tokens.VOCAB_SIZE))
    print("read the first %d characters of %s" % (min(len(text), args.seq_len - 2), args.text))


if __name__ == "__main__":
    main()
