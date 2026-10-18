"""The two training signals for the noise policy pi(eps | s).

Both are (weighted) maximum likelihood on noise samples; they differ in where the samples and
weights come from.

Importance weights (``train_iw``). For a state s, M noises eps_1..M are pushed through FastWAM
and scored by Q. The softmax of Q / tau over the M samples weights each one,

    w_i = exp(Q(s, F(s, eps_i)) / tau + log r_i) / sum_j exp(Q(s, F(s, eps_j)) / tau + log r_j)
    L_IW = -E_s [ sum_i w_i log pi(eps_i | s) ]

with log r_i = 0 when the noises come from N(0, I) (the notes' setting), and
log r_i = log N(eps_i) - log pi(eps_i | s) when they come from the policy itself, so that both
target the same tilted distribution N(eps) exp(Q / tau).

Flow reversal of improved actions (``train_reversal``). The action is improved under Q first
and mapped back to the noise that produces it, eps* = F^-1(s, a*); the policy then fits those:

    L_a = -E_(s, eps*) [ log pi(eps* | s) ]

Both are reported per free noise dimension (divided by K x A), so their scale -- and a sensible
learning rate -- does not depend on the chunk length or on ``noise_mode``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from noise_policy.modeling import NoiseDistribution


def importance_weights(q: Tensor, temperature: float, log_ratio: Tensor | None = None) -> Tensor:
    """Self-normalized weights over the M samples of each state: ``(B, M)`` -> ``(B, M)``."""
    logits = q.float() / temperature
    if log_ratio is not None:
        logits = logits + log_ratio.float()
    return torch.softmax(logits, dim=-1)


def effective_sample_size(weights: Tensor) -> Tensor:
    """1 / sum w^2 per state: M for uniform weights, 1 when one sample takes all the weight."""
    return 1.0 / weights.pow(2).sum(dim=-1)


def importance_weighted_nll(dist: "NoiseDistribution", noise: Tensor, weights: Tensor) -> Tensor:
    """L_IW for ``(B, M, H, A)`` noise with ``(B, M)`` weights summing to 1 per state."""
    log_prob = dist.log_prob(noise)                                    # (B, M)
    return -(weights.detach() * log_prob).sum(dim=-1).mean() / dist.n_free


def reversal_nll(dist: "NoiseDistribution", noise: Tensor) -> Tensor:
    """L_a for ``(B, M, H, A)`` flow-reversed targets, all weighted equally."""
    return -dist.log_prob(noise).mean() / dist.n_free
