"""Train the noise policy on flow-reversed, Q-improved actions.

    python -m noise_policy.train_reversal --config noise_policy/configs/libero_10_reversal.yaml \
        [--reversal.improve.anchor_weight=3.0 ...]

For each demonstration state s (see :mod:`noise_policy.training` for the loop):

    eps_0    ~ pi(.|s)                    (reversal.start=policy; or N(0, I) with start=prior)
    a_0      = FastWAM(s, eps_0)
    a*       = a_0 after `improve.steps` normalized-gradient steps on
               J(a) = Q(s, a) - lambda ||a - a_0||^2           (lambda = improve.anchor_weight)
    eps*     = FastWAM^-1(s, a*)          (flow reversal, FastWAMDenoiser.invert)
    L_a      = -log pi(eps* | s)

The improvement is the qopt planner's own (:meth:`QGradientPlanner.improve`), so it is exactly
what the s30 anchored rollouts executed. Logged per collection round, besides the Q gain:
how well the reversal round-trips (FastWAM(eps*) against a*, and the Q it actually reaches),
and whether eps* still looks like N(0, I) noise.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

    from noise_policy.config import NoisePolicyRunConfig
    from noise_policy.training import Collect, Loss, TrainContext
    from qoptimizing.config import QOptConfig


def _rms(x: Any) -> float:
    return float(x.float().pow(2).mean().sqrt())


def collect(ctx: "TrainContext", state: dict[str, Any]) -> "tuple[Tensor, Tensor, dict[str, float]]":
    import torch

    from noise_policy.fastwam import prepare

    cfg = ctx.cfg
    m = cfg.noise_train.samples_per_state
    with torch.no_grad():
        denoiser = prepare(ctx.sampler, state, ctx.denoise_steps)
        start = ctx.distribution(state) if cfg.reversal.start == "policy" else ctx.prior()
        eps0 = start.sample(m)[0]                                            # (M, H, A)
        a0 = denoiser(eps0)
        context = ctx.planner.encode(state, a0)
    a_star, trajectory = ctx.planner.improve(a0, context)                    # autograd inside
    with torch.no_grad():
        eps_star = denoiser.invert(a_star, refine_iters=cfg.reversal.refine_iters)
        # The round trip: what FastWAM actually makes of eps*, and what Q thinks of that.
        roundtrip = denoiser(eps_star)
        q0, q_star = trajectory[0], trajectory[-1]
        q_roundtrip = ctx.planner.score(roundtrip, context)

    metrics = {
        "q_start": float(q0.mean()),
        "q_gain_improved": float((q_star - q0).mean()),
        "q_gain_roundtrip": float((q_roundtrip - q0).mean()),
        "action_change_rms": _rms(a_star - a0),
        "roundtrip_error_rms": _rms(roundtrip - a_star),
        # Relative to how far the improvement moved the action: < 1 means the reversal keeps
        # most of the improvement.
        "roundtrip_error_rel": _rms(roundtrip - a_star) / max(_rms(a_star - a0), 1e-12),
        "noise_change_rms": _rms(eps_star - eps0),
        "target_noise_std": float(eps_star.float().std()),
    }
    return eps_star, torch.full((m,), 1.0 / m), metrics


def make(cfg: "NoisePolicyRunConfig") -> "tuple[QOptConfig, Collect, Loss]":
    from noise_policy.losses import reversal_nll

    return cfg.reversal.improve, collect, lambda dist, noise, weights: reversal_nll(dist, noise)


def main(argv: Sequence[str] | None = None) -> int:
    from noise_policy.training import main as train_main

    return train_main(argv, prog="python -m noise_policy.train_reversal", method="flow_reversal", make=make)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
