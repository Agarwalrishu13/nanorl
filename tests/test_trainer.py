"""The two losses, tested the only way that means anything: train on them.

Everything here runs on the CPU in a second or two — a tiny model, a fixed
seed, and assertions about behaviour (the loss goes down, padding is ignored,
the reference model never moves) rather than about exact numbers.
"""

import unittest

import torch

from nanorl import tokens
from nanorl.trainer import dpo_loss, sft_loss


class TinyModel(torch.nn.Module):
    """An embedding and a linear layer — just enough to learn a pattern."""

    def __init__(self, vocab_size=tokens.VOCAB_SIZE, dim=32):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, dim)
        self.linear = torch.nn.Linear(dim, vocab_size)

    def forward(self, tokens_in):
        return self.linear(self.embedding(tokens_in))


def _seed():
    torch.manual_seed(7)


class SftLossTests(unittest.TestCase):
    def test_the_loss_is_a_positive_number(self):
        _seed()
        model = TinyModel()
        batch, mask = tokens.pad_batch(["hello there"], 16)
        loss = sft_loss(model, batch, mask)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)

    def test_padding_never_contributes(self):
        _seed()
        model = TinyModel()
        short, mask = tokens.pad_batch(["hi"], 8)
        # Two batches with the same real tokens but different filler in the
        # padded region must give exactly the same loss.
        padded_a = short.clone()
        padded_b = short.clone()
        real = int(mask[0].sum())
        padded_b[0, real:] = torch.randint(0, 100, (8 - real,))
        loss_a = sft_loss(model, padded_a, mask)
        loss_b = sft_loss(model, padded_b, mask)
        self.assertAlmostEqual(loss_a.item(), loss_b.item(), places=5)

    def test_training_on_it_makes_the_loss_go_down(self):
        _seed()
        model = TinyModel()
        # One sentence repeated: the most learnable thing there is.
        batch, mask = tokens.pad_batch(["the sky is blue today"], 32)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        first = sft_loss(model, batch, mask).item()
        for _ in range(40):
            optimizer.zero_grad()
            loss = sft_loss(model, batch, mask)
            loss.backward()
            optimizer.step()
        last = sft_loss(model, batch, mask).item()
        self.assertLess(last, first * 0.5)


class DpoLossTests(unittest.TestCase):
    def test_an_unchanged_policy_sits_at_log_two(self):
        # When the policy IS the reference, the margin is zero everywhere and
        # -logsigmoid(0) is log(2) — the neutral starting point, by definition.
        _seed()
        model = TinyModel()
        chosen, chosen_mask = tokens.pad_batch(["a good answer"], 16)
        rejected, rejected_mask = tokens.pad_batch(["a bad answer"], 16)
        loss = dpo_loss(model, model, chosen, rejected,
                        chosen_mask=chosen_mask, rejected_mask=rejected_mask)
        self.assertAlmostEqual(loss.item(), 0.6931, places=3)

    def test_training_on_preferences_teaches_the_preference(self):
        _seed()
        model = TinyModel()
        ref = TinyModel()
        ref.load_state_dict(model.state_dict())
        ref.eval()
        for parameter in ref.parameters():
            parameter.requires_grad_(False)
        chosen, chosen_mask = tokens.pad_batch(["please say yes"], 16)
        rejected, rejected_mask = tokens.pad_batch(["please say no!"], 16)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        first = dpo_loss(model, ref, chosen, rejected,
                         chosen_mask=chosen_mask, rejected_mask=rejected_mask).item()
        for _ in range(40):
            optimizer.zero_grad()
            loss = dpo_loss(model, ref, chosen, rejected,
                            chosen_mask=chosen_mask, rejected_mask=rejected_mask)
            loss.backward()
            optimizer.step()
        last = dpo_loss(model, ref, chosen, rejected,
                        chosen_mask=chosen_mask, rejected_mask=rejected_mask).item()
        self.assertLess(last, first * 0.5)

    def test_the_reference_model_never_moves(self):
        _seed()
        model = TinyModel()
        ref = TinyModel()
        ref.load_state_dict(model.state_dict())
        chosen, _ = tokens.pad_batch(["chosen"], 16)
        rejected, _ = tokens.pad_batch(["rejected"], 16)
        loss = dpo_loss(model, ref, chosen, rejected)
        loss.backward()
        for name, parameter in ref.named_parameters():
            self.assertTrue(torch.equal(parameter, model.state_dict()[name]),
                            "the reference model changed: %s" % name)

    def test_no_gradient_reaches_the_reference(self):
        _seed()
        model = TinyModel()
        ref = TinyModel()
        chosen, _ = tokens.pad_batch(["chosen"], 16)
        rejected, _ = tokens.pad_batch(["rejected"], 16)
        dpo_loss(model, ref, chosen, rejected).backward()
        for parameter in ref.parameters():
            self.assertIsNone(parameter.grad)


class CheckpointTests(unittest.TestCase):
    def test_a_saved_model_loads_back_as_the_same_model(self):
        _seed()
        model = TinyModel()
        batch, _ = tokens.pad_batch(["remember me"], 16)
        before = model(batch).detach()
        import os
        import tempfile
        handle, path = tempfile.mkstemp(suffix=".pt")
        os.close(handle)
        try:
            torch.save(model, path)
            loaded = torch.load(path, weights_only=False)
            self.assertTrue(torch.equal(loaded(batch), before))
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
