"""SFT & DPO trainer core — the maths only; data and files live elsewhere.

Two losses, both written from scratch on raw PyTorch:

* :func:`sft_loss` — "here is good text, learn to continue it." Masked
  next-token cross-entropy, so padding never teaches the model anything.
* :func:`dpo_loss` — "of these two answers, this one was better." The policy
  is nudged toward the chosen answer and away from the rejected one, relative
  to a frozen reference model. The per-answer log probabilities are summed
  over the answer's tokens (as the DPO paper intends), and no gradient ever
  flows into the reference model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sft_loss(model, tokens, attention_mask=None):
    """Next-token cross-entropy over the tokens the mask says are real.

    ``tokens`` is ``[batch, time]``; ``attention_mask`` is 1 for real tokens
    and 0 for padding, aligned with ``tokens``. With no mask, every position
    counts.
    """
    logits = model(tokens)
    shift_logits = logits[..., :-1, :]
    shift_labels = tokens[..., 1:]
    flat_logits = shift_logits.reshape(-1, shift_logits.size(-1))
    flat_labels = shift_labels.reshape(-1)
    loss = F.cross_entropy(flat_logits, flat_labels, reduction="none")
    if attention_mask is None:
        return loss.mean()
    mask = attention_mask[..., 1:].to(loss.dtype).reshape(-1)
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def _sequence_log_probs(model, tokens, attention_mask=None):
    """log P(tokens | model), summed over time: one number per example."""
    logits = model(tokens)
    log_probs = F.log_softmax(logits, dim=-1)
    per_token = torch.gather(log_probs, -1, tokens.unsqueeze(-1)).squeeze(-1)
    if attention_mask is not None:
        per_token = per_token * attention_mask.to(per_token.dtype)
    return per_token.sum(-1)


def dpo_loss(policy_model, ref_model, chosen, rejected, beta=0.1,
             chosen_mask=None, rejected_mask=None):
    """Direct Preference Optimisation, sequence-level.

    ``chosen`` and ``rejected`` are ``[batch, time]`` token ids, with optional
    masks of the same shape. The reference model is treated as frozen: it runs
    under ``no_grad``, so it is both correct and cheaper, and optimising never
    moves it.
    """
    with torch.no_grad():
        ref_chosen = _sequence_log_probs(ref_model, chosen, chosen_mask)
        ref_rejected = _sequence_log_probs(ref_model, rejected, rejected_mask)
    policy_chosen = _sequence_log_probs(policy_model, chosen, chosen_mask)
    policy_rejected = _sequence_log_probs(policy_model, rejected, rejected_mask)
    margin = (policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)
    return -F.logsigmoid(beta * margin).mean()
