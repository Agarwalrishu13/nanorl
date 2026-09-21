"""DPO trainer: Direct Preference Optimization for tiny models.

Usage:
    python scripts/train_dpo.py --model data/model.pt --data data/preferences.json --output model-dpo.pt
"""

import argparse
import torch
import json
import os

from nanorl.trainer import dpo_loss
from torch.utils.data import DataLoader, Dataset


class SimplePreferenceDataset(Dataset):
    """Very small preference dataset wrapper."""
    def __init__(self, data_path, seq_len=32):
        with open(data_path, 'r') as f:
            data = json.load(f)
        # data = list of {"chosen": "...", "rejected": "..."}
        self.examples = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        chosen = torch.tensor([ord(c) % 256 for c in self.examples[idx]["chosen"][:self.seq_len]], dtype=torch.long)
        rejected = torch.tensor([ord(c) % 256 for c in self.examples[idx]["rejected"][:self.seq_len]], dtype=torch.long)
        return chosen, rejected


def main():
    parser = argparse.ArgumentParser(description="DPO trainer for tiny models")
    parser.add_argument("--model", required=True, help="Path to .pt model checkpoint")
    parser.add_argument("--data", required=True, help="Path to preference dataset JSON")
    parser.add_argument("--output", default="model-dpo.pt", help="Output checkpoint path")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=32)
    args = parser.parse_args()

    model = torch.load(args.model, weights_only=False)
    ref_model = torch.load(args.model, weights_only=False)  # In practice, use a separate checkpoint
    model.train()
    ref_model.eval()
    
    dataset = SimplePreferenceDataset(args.data, args.seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        total_loss = 0
        for chosen, rejected in loader:
            optimizer.zero_grad()
            loss = dpo_loss(model, ref_model, chosen, rejected, beta=args.beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs}, Avg DPO Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), args.output)
    print(f"DPO complete. Model saved to {args.output}")


if __name__ == "__main__":
    main()