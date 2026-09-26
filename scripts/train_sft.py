"""SFT trainer: teach a tiny model to answer, from examples.

    python scripts/train_sft.py --model my-tiny-llm.pt --data data/instruct.json

The data file is a JSON list; each item is either ``{"text": "..."}`` or
``{"instruction": "...", "response": "..."}``. Every example becomes one line
of bytes the model learns to continue, with padding that never teaches it
anything. Works from any working directory; the model file and the data file
can be anywhere.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from anywhere

import torch
from torch.utils.data import DataLoader, Dataset

from nanorl import tokens
from nanorl.trainer import sft_loss


class InstructDataset(Dataset):
    """A JSON list of examples, read once, tokenised in the collate step."""

    def __init__(self, data_path, seq_len=256):
        with open(data_path, encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, list):
            raise SystemExit("The data file should be a JSON list of examples.")
        self.lines = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if text is None and "instruction" in item:
                text = (item.get("instruction", "") + "\n" + item.get("response", "")).strip()
            if text:
                self.lines.append(text)
        if not self.lines:
            raise SystemExit("No usable examples found in %s." % data_path)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.lines[idx]


def main():
    parser = argparse.ArgumentParser(description="SFT trainer for tiny models")
    parser.add_argument("--model", required=True, help="path to a .pt model checkpoint")
    parser.add_argument("--data", required=True, help="path to an instruct dataset JSON")
    parser.add_argument("--output", default="model-sft.pt", help="output checkpoint path")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    args = parser.parse_args()

    model = torch.load(args.model, weights_only=False, map_location="cpu")
    model.train()
    dataset = InstructDataset(args.data, args.seq_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: tokens.pad_batch(batch, args.seq_len),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        total = 0.0
        for batch, mask in loader:
            optimizer.zero_grad()
            loss = sft_loss(model, batch, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        print("Epoch %d/%d, avg loss %.4f" % (epoch + 1, args.epochs, total / max(1, len(loader))))

    # The whole model, because the loader above loads one the same way.
    torch.save(model, args.output)
    print("SFT complete. Model saved to %s" % args.output)


if __name__ == "__main__":
    main()
