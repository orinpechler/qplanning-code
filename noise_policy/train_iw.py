"""Train the noise policy with Q-based importance weights.

    python -m noise_policy.train_iw --config noise_policy/configs/libero_10_iw.yaml [--iw.temperature=0.02 ...]

For each demonstration state s (see :mod:`noise_policy.training` for the loop):

    eps_1..M ~ pi(.|s)            (iw.proposal=policy, the default; or ~ N(0, I) with iw.proposal=prior)
    a_i      = FastWAM(s, eps_i)
    w_i      = softmax_i( Q(s, a_i) / tau  [+ log N(eps_i) - log pi(eps_i | s)] )
    L_IW     = -sum_i w_i log pi(eps_i | s)

so the policy is pulled towards the noises whose chunks Q prefers, with no success labels and
no flow reversal. Logged per collection round: the spread of Q over the M samples (how much
there is to prefer at all), the effective sample size of the weights, and the Q gained by the
weighting over a uniform pick.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

    from noise_policy.config import NoisePolicyRunConfig
    from noise_policy.training import Collect, Loss, TrainContext
    from qoptimizing.config import QOptConfig


def collect(ctx: "TrainContext", state: dict[str, Any]) -> "tuple[Tensor, Tensor, dict[str, float]]":
    import torch

    from noise_policy.fastwam import prepare
    from noise_policy.losses import effective_sample_size, importance_weights

    cfg = ctx.cfg
    m = cfg.noise_train.samples_per_state
    with torch.no_grad():
        denoiser = prepare(ctx.sampler, state, ctx.denoise_steps)
        prior = ctx.prior()
        if cfg.iw.proposal == "prior":
            noise = prior.sample(m)[0]                                       # (M, H, A)
            log_ratio = None
        else:
            dist = ctx.distribution(state)
            noise = dist.sample(m)[0]
            log_ratio = (prior.log_prob(noise[None]) - dist.log_prob(noise[None]))[0]
        chunks = denoiser(noise)
        context = ctx.planner.encode(state, chunks)
        q = ctx.planner.score(chunks, context)                               # (M,)
        weights = importance_weights(q[None], cfg.iw.temperature,
                                     None if log_ratio is None else log_ratio[None])[0]

    metrics = {
        "q_mean": float(q.mean()),
        "q_spread": float(q.max() - q.min()),
        "q_std": float(q.std()),
        "ess_frac": float(effective_sample_size(weights)) / m,
        "weighted_q_gain": float((weights * q).sum() - q.mean()),
    }
    return noise, weights, metrics


def make(cfg: "NoisePolicyRunConfig") -> "tuple[QOptConfig, Collect, Loss]":
    from noise_policy.losses import importance_weighted_nll
    from qoptimizing.config import QOptConfig

    # Only the planner's Q scoring is used here; no gradient ascent.
    return QOptConfig(steps=0), collect, importance_weighted_nll


def main(argv: Sequence[str] | None = None) -> int:
    from noise_policy.training import main as train_main

    return train_main(argv, prog="python -m noise_policy.train_iw", method="importance_weights", make=make)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
