"""SFT & DPO trainer core."""

import torch
import torch.nn.functional as F

def sft_loss(model, tokens, attention_mask):
    logits = model(tokens)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = tokens[..., 1:].contiguous()
    mask = attention_mask[..., 1:].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), 
                           shift_labels.view(-1), reduction='none')
    return (loss * mask.view(-1)).sum() / mask.sum()

def dpo_loss(policy_model, ref_model, chosen, rejected, beta=0.1):
    """Direct Preference Optimization."""
    def log_probs(model, tokens):
        logits = model(tokens)
        probs = F.softmax(logits, dim=-1)
        return torch.gather(probs, -1, tokens.unsqueeze(-1)).log().squeeze(-1)

    log_policy_chosen = log_probs(policy_model, chosen)
    log_policy_rejected = log_probs(policy_model, rejected)
    log_ref_chosen = log_probs(ref_model, chosen)
    log_ref_rejected = log_probs(ref_model, rejected)

    logits = (log_policy_chosen - log_ref_chosen) - (log_policy_rejected - log_ref_rejected)
    return -F.logsigmoid(beta * logits).mean()
