from nanorl.trainer import sft_loss, dpo_loss
import torch
import random

class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size=259, dim=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = torch.nn.Embedding(vocab_size, dim)
        self.linear = torch.nn.Linear(dim, vocab_size)

    def forward(self, tokens):
        embedded = self.embedding(tokens)
        return self.linear(embedded)

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=10, seq_len=10, vocab_size=259):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        tokens = torch.randint(0, self.vocab_size, (self.seq_len,))
        attention_mask = torch.ones(self.seq_len, dtype=torch.bool)
        return tokens, attention_mask

class DummyPreferenceDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=10, seq_len=10, vocab_size=259):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        chosen = torch.randint(0, self.vocab_size, (self.seq_len,))
        rejected = torch.randint(0, self.vocab_size, (self.seq_len,))
        return chosen, rejected


def test_sft_loss():
    model = DummyModel()
    dataset = DummyDataset(num_samples=2, seq_len=5)
    tokens, attention_mask = dataset[0]
    tokens = tokens.unsqueeze(0)
    attention_mask = attention_mask.unsqueeze(0)
    
    loss = sft_loss(model, tokens, attention_mask)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0
    print("SFT Loss Test Passed")

def test_dpo_loss():
    policy_model = DummyModel()
    ref_model = DummyModel()
    dataset = DummyPreferenceDataset(num_samples=2, seq_len=5)
    chosen, rejected = dataset[0]
    chosen = chosen.unsqueeze(0)
    rejected = rejected.unsqueeze(0)

    loss = dpo_loss(policy_model, ref_model, chosen, rejected)
    assert isinstance(loss, torch.Tensor)
    print("DPO Loss Test Passed")


if __name__ == "__main__":
    from nanorl.trainer import sft_loss, dpo_loss
    test_sft_loss()
    test_dpo_loss()
