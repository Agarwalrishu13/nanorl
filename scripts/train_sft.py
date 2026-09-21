"""SFT trainer: Supervised Fine-Tuning for tiny models.

Usage:
    python scripts/train_sft.py --model data/model.pt --data data/instruct.json --output model-sft.pt
"""

import argparse
import torch
import json
import os

from nanorl.trainer import sft_loss
from torch.utils.data import DataLoader, Dataset


class SimpleInstructDataset(Dataset):
    """Very small instruction dataset wrapper."""
    def __init__(self, data_path, seq_len=32):
        with open(data_path, 'r') as f:
            data = json.load(f)
        self.examples = data  # list of {"text": "..."} or {"instruction": "...", "response": "..."}
        self.seq_len = seq_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx].get("text", 
                      self.examples[idx].get("instruction", "")
                      + " " + self.examples[idx].get("response", ""))
        tokens = torch.tensor([ord(c) % 256 for c in text[:self.seq_len]], dtype=torch.long)
        mask = torch.ones(self.seq_len, dtype=torch.bool)
        return tokens, mask


def main():
    parser = argparse.ArgumentParser(description="SFT trainer for tiny models")
    parser.add_argument("--model", required=True, help="Path to .pt model checkpoint")
    parser.add_argument("--data", required=True, help="Path to instruct dataset JSON")
    parser.add_argument("--output", default="model-sft.pt", help="Output checkpoint path")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=32)
    args = parser.parse_args()

    model = torch.load(args.model, weights_only=False)
    model.train()
    dataset = SimpleInstructDataset(args.data, args.seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        total_loss = 0
        for tokens, mask in loader:
            optimizer.zero_grad()
            # Expect model forward with tokens seq_len handling inside loss
            loss = sft_loss(model, tokens, mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs}, Avg Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), args.output)
    print(f"SFT complete. Model saved to {args.output}")


if __name__ == "__main__":
    main()