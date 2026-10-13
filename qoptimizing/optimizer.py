"""Gradient ascent on the action chunk against a frozen Q-function.

At each planning step:

    a_0      = FastWAM(o, l)                                  (the baseline chunk)
    a_{k+1}  = a_k + eta * grad_a Q(o, l, a_k)                k = 0 .. steps-1
    execute  a_steps

Nothing but ``a`` receives a gradient. The observation context is encoded once under
``no_grad`` -- so the DINOv2 and T5 encoders are never in the graph -- and the decoder's
parameters have ``requires_grad`` switched off; ``torch.autograd.grad`` is then taken with
respect to the action tensor alone.

The chunk lives in FastWAM's normalized action space, while the Q-function reads actions in
its own normalized space (the two were trained with different statistics; see
:mod:`qplanning.planner`). The existing conversion goes through LeRobot processor pipelines,
which are not meant to carry gradients. Both normalizations are per-dimension affine maps,
so their composition is too:

    a_Q = scale * a_BC + shift

``scale`` and ``shift`` are read off the real pipelines once, by pushing a zero and a one
chunk through them, and checked on a random chunk. The optimization then uses this affine
map, which is exact and differentiable, instead of the pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qplanning.planner import PlannerContext, PlanResult, QSpread, QWeightedPlanner, _broadcast

if TYPE_CHECKING:
    import torch
    from torch import Tensor

    from qoptimizing.config import QOptConfig
    from qplanning.policies.base import BCBundle, BCChunkSampler


@dataclass
class QOptStats:
    """What one optimization did to the chunk that was executed."""

    q_init: float                     # Q of the policy's chunk
    q_final: float                    # Q of the executed (optimized) chunk
    delta_rms: float                  # RMS change, BC-normalized units
    delta_max: float                  # largest single-element change, BC-normalized units
    grad_rms_init: float              # RMS of dQ/da at the policy's chunk
    q_trajectory: list[float] = field(default_factory=list)   # steps + 1 values

    @property
    def q_gain(self) -> float:
        return self.q_final - self.q_init


class QGradientPlanner(QWeightedPlanner):
    """Draw the policy's chunk, then push it uphill on Q for a fixed number of steps.

    Reuses :class:`~qplanning.planner.QWeightedPlanner` for loading the Q-function, reading
    its cameras out of the batch and encoding the observation once; replaces the planning
    step itself.
    """

    def __init__(
        self,
        cfg: "QOptConfig",
        ctx: PlannerContext,
        bc_unnormalize: Any,
        denoise_steps: int | None,
    ) -> None:
        super().__init__(cfg=cfg, ctx=ctx, bc_unnormalize=bc_unnormalize)  # type: ignore[arg-type]
        self.denoise_steps = denoise_steps
        self._affine: "tuple[Tensor, Tensor] | None" = None
        self.last_stats: QOptStats | None = None
        #: One entry per planning step since the last reset(), i.e. for the current episode.
        self.episode_stats: list[QOptStats] = []

    # ── construction ───────────────────────────────────────────────────────────────────
    @classmethod
    def build(  # type: ignore[override]
        cls,
        cfg: "QOptConfig",
        q_checkpoint: str | Path,
        bc: "BCBundle",
        device: "torch.device | str" = "cuda",
        denoise_steps: int | None = None,
    ) -> "QGradientPlanner":
        from qplanning.modeling_qfunction import load_q_function

        q_policy, q_pre = load_q_function(q_checkpoint, device=device)
        # Frozen: no parameter of the Q-function can enter the graph.
        q_policy.requires_grad_(False)

        horizon = int(q_policy.config.h)
        if horizon != bc.sampler.chunk_size:
            raise ValueError(
                f"the Q-function scores chunks of length {horizon} but this BC policy emits "
                f"chunks of length {bc.sampler.chunk_size}. They must match."
            )
        ctx = PlannerContext(
            q_policy=q_policy,
            q_pre=q_pre,
            camera_keys=tuple(q_policy.config.camera_keys),
            horizon=horizon,
        )
        return cls(cfg=cfg, ctx=ctx, bc_unnormalize=bc.sampler.unnormalize_actions,
                   denoise_steps=denoise_steps)

    def reset(self) -> None:
        self.episode_stats = []
        self.last_stats = None

    # ── the planning step ──────────────────────────────────────────────────────────────
    def plan(self, bc: "BCChunkSampler", batch: dict[str, "Tensor"]) -> PlanResult:
        import torch

        with torch.no_grad():
            initial = bc.sample_chunks(batch, self.cfg.n_samples, denoise_steps=self.denoise_steps)
            observation = self._observation_inputs(batch)
            context = self._encode_observation_once(initial, observation)
            if self._affine is None:
                self._affine = self._bc_to_q_affine(initial, observation)

        final, trajectory, grad_rms_init = self._ascend(initial, context)

        q_init, q_final = trajectory[0], trajectory[-1]
        best = int(torch.argmax(q_final))
        a0 = initial.to(final.dtype)
        delta = final[best] - a0[best]
        stats = QOptStats(
            q_init=float(q_init[best]),
            q_final=float(q_final[best]),
            delta_rms=float(delta.pow(2).mean().sqrt()),
            delta_max=float(delta.abs().max()),
            grad_rms_init=float(grad_rms_init[best]) if grad_rms_init is not None else 0.0,
            q_trajectory=[float(q[best]) for q in trajectory],
        )
        self.last_stats = stats
        self.episode_stats.append(stats)

        spread = QSpread(
            min=float(q_final.min()),
            max=float(q_final.max()),
            mean=float(q_final.mean()),
            std=float(q_final.std()) if q_final.numel() > 1 else 0.0,
        )
        self.last_spread = spread
        chunk = final[best : best + 1].to(dtype=initial.dtype, device=initial.device)
        return PlanResult(chunk=chunk, q_values=q_final, spread=spread, candidates=final)

    # ── internals ──────────────────────────────────────────────────────────────────────
    def _ascend(
        self, initial: "Tensor", context: "Tensor"
    ) -> "tuple[Tensor, list[Tensor], Tensor | None]":
        """Run the gradient ascent. Returns the final chunks, Q at every iterate, and the
        initial gradient RMS per chunk."""
        import torch

        cfg = self.cfg
        n_samples = initial.shape[0]
        scale, shift = self._affine  # type: ignore[misc]

        # The rollout calls select_action under no_grad, so autograd has to be switched back
        # on locally. inference_mode(False) + clone() guards against inputs created under
        # torch.inference_mode, which can never enter an autograd graph.
        with torch.inference_mode(False), torch.enable_grad():
            context_n = context.detach().clone().expand(-1, n_samples, -1).contiguous()
            a0 = initial.detach().clone().float()
            actions = a0.clone().requires_grad_(True)
            optimizer = torch.optim.Adam([actions], lr=cfg.step_size) if cfg.optimizer == "adam" else None

            trajectory: list[Tensor] = []
            grad_rms_init: "Tensor | None" = None
            for step in range(cfg.steps + 1):
                q = self._q_values(actions, context_n, scale, shift)       # (N,)
                trajectory.append(q.detach())
                if step == cfg.steps:
                    break
                # Chunks are scored independently, so the gradient of the sum is the
                # per-chunk gradient. Taken w.r.t. `actions` only.
                (grad,) = torch.autograd.grad(q.sum(), actions)
                grad_rms = grad.pow(2).mean(dim=(1, 2), keepdim=True).sqrt()
                if grad_rms_init is None:
                    grad_rms_init = grad_rms.flatten()
                with torch.no_grad():
                    if optimizer is not None:
                        actions.grad = -grad          # the optimizer minimizes; we ascend
                        optimizer.step()
                        actions.grad = None
                    else:
                        if cfg.normalize_grad:
                            grad = grad / grad_rms.clamp_min(1e-12)
                        actions.add_(cfg.step_size * grad)
                    if cfg.max_delta is not None:
                        actions.copy_(a0 + (actions - a0).clamp(-cfg.max_delta, cfg.max_delta))

        return actions.detach(), trajectory, grad_rms_init

    def _q_values(self, actions_bc: "Tensor", context_n: "Tensor", scale: "Tensor",
                  shift: "Tensor") -> "Tensor":
        """Q(o, l, a) for BC-normalized chunks, differentiable in ``actions_bc``."""
        from qplanning.modeling_qfunction import expected_value

        q_policy = self.ctx.q_policy
        actions_q = actions_bc * scale + shift
        actions_q = q_policy._truncate_action(actions_q[:, : self.ctx.horizon, :])
        dtype = q_policy.q_online.action_proj.weight.dtype
        logits = q_policy.q_online.forward_with_context(context_n.to(dtype), actions_q.to(dtype))
        return expected_value(logits.float(), q_policy.bin_centers.float())

    def _bc_to_q_affine(self, like: "Tensor", observation: dict[str, Any]) -> "tuple[Tensor, Tensor]":
        """Read the BC-normalized -> Q-normalized map off the real processor pipelines."""
        import torch
        from lerobot.utils.constants import ACTION

        _, horizon, action_dim = like.shape
        shape = (horizon, action_dim)
        generator = torch.Generator(device="cpu").manual_seed(0)
        check = torch.randn(shape, generator=generator).to(like.device, like.dtype)
        probe = torch.stack([torch.zeros_like(check), torch.ones_like(check), check])

        n = probe.shape[0]
        raw = self._bc_unnormalize(probe.reshape(n * horizon, action_dim))
        raw = raw.reshape(n, horizon, action_dim).to(like.device)
        q_batch: dict[str, Any] = {ACTION: raw}
        for key, value in observation.items():
            q_batch[key] = _broadcast(value, n)
        mapped = self.ctx.q_pre(q_batch)[ACTION].float().to(like.device)

        shift = mapped[0, 0]
        scale = mapped[1, 0] - shift
        expected = check.float() * scale + shift
        error = (expected - mapped[2]).abs().max().item()
        tolerance = 1e-3 * max(1.0, float(mapped[2].abs().max()))
        if error > tolerance:
            raise RuntimeError(
                "the BC -> Q action conversion is not a per-dimension affine map (max error "
                f"{error:.2e}). The differentiable shortcut used for the gradient would be "
                "wrong; check the normalization modes of both checkpoints."
            )
        if bool((scale.abs() < 1e-8).any()):
            raise RuntimeError(f"degenerate BC -> Q action scale {scale.tolist()}")
        return scale.detach(), shift.detach()

    def __repr__(self) -> str:  # pragma: no cover
        cfg = self.cfg
        return (
            f"QGradientPlanner(steps={cfg.steps}, step_size={cfg.step_size}, "
            f"optimizer={cfg.optimizer}, normalize_grad={cfg.normalize_grad}, "
            f"max_delta={cfg.max_delta}, N={cfg.n_samples}, denoise_steps={self.denoise_steps})"
        )
