"""DPO trainer: teach a tiny model which of two answers is better.

    python scripts/train_dpo.py --model my-tiny-llm.pt --data data/preferences.json

The data file is a JSON list of ``{"chosen": "...", "rejected": "..."}`` pairs.
A frozen copy of the starting model is the reference point: the trainer nudges
the working model toward the chosen answers and away from the rejected ones,
relative to that copy.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from anywhere

import torch
from torch.utils.data import DataLoader, Dataset

from nanorl import tokens
from nanorl.trainer import dpo_loss


class PreferenceDataset(Dataset):
    """A JSON list of chosen/rejected pairs, read once, tokenised in collate."""

    def __init__(self, data_path):
        with open(data_path, encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, list):
            raise SystemExit("The data file should be a JSON list of {chosen, rejected} pairs.")
        self.pairs = [
            (item["chosen"], item["rejected"])
            for item in raw
            if isinstance(item, dict) and item.get("chosen") and item.get("rejected")
        ]
        if not self.pairs:
            raise SystemExit("No usable preference pairs found in %s." % data_path)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def main():
    parser = argparse.ArgumentParser(description="DPO trainer for tiny models")
    parser.add_argument("--model", required=True, help="path to a .pt model checkpoint")
    parser.add_argument("--data", required=True, help="path to a preferences dataset JSON")
    parser.add_argument("--output", default="model-dpo.pt", help="output checkpoint path")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1, help="how strongly to stay near the reference")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    args = parser.parse_args()

    model = torch.load(args.model, weights_only=False, map_location="cpu")
    ref_model = copy.deepcopy(model)  # the starting point, frozen
    ref_model.eval()
    for parameter in ref_model.parameters():
        parameter.requires_grad_(False)
    model.train()

    dataset = PreferenceDataset(args.data)

    def collate(batch):
        chosen_ids, chosen_mask = tokens.pad_batch([pair[0] for pair in batch], args.seq_len)
        rejected_ids, rejected_mask = tokens.pad_batch([pair[1] for pair in batch], args.seq_len)
        return chosen_ids, chosen_mask, rejected_ids, rejected_mask

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        total = 0.0
        for chosen, chosen_mask, rejected, rejected_mask in loader:
            optimizer.zero_grad()
            loss = dpo_loss(model, ref_model, chosen, rejected, beta=args.beta,
                            chosen_mask=chosen_mask, rejected_mask=rejected_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        print("Epoch %d/%d, avg DPO loss %.4f" % (epoch + 1, args.epochs, total / max(1, len(loader))))

    torch.save(model, args.output)
    print("DPO complete. Model saved to %s" % args.output)


if __name__ == "__main__":
    main()
