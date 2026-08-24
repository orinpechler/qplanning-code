"""Value-guided action selection.

At each planning step the frozen BC policy draws N candidate action chunks from a short
flow-matching pass. The Q-function scores all N in a single batched decoder pass, and the
chunk actually executed is the softmax-Q-weighted average of the best K:

    w_i  =  softmax( Q(o, l, a_i) / lambda )         over the top-K candidates
    a_bar = sum_i w_i * a_i

Two details are what make this affordable rather than merely correct:

* The observation is encoded **once** per planning step. The DINOv2 and T5 encoders are the
  expensive part of the Q-function and do not depend on the candidate, so their output is
  computed once and expanded across all N. Only the ~500M-parameter decoder scales with N,
  at a few milliseconds per candidate.
* Candidates round-trip through the BC policy's unnormalizer and the Q-function's
  normalizer before scoring. The two networks were trained on different data and therefore
  hold different action statistics; scoring BC-normalized actions directly gives Q-values
  that look reasonable and rank candidates wrongly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import torch
    from torch import Tensor

    from lerobot.processor import PolicyProcessorPipeline
    from qplanning.config.schema import PlannerConfig
    from qplanning.modeling_qfunction import QFunctionPolicy
    from qplanning.policies.base import BCBundle, BCChunkSampler


@dataclass(frozen=True)
class QSpread:
    """Summary of the Q-values across one step's candidates.

    Worth logging: if ``std`` collapses to zero the Q-function is returning a constant, the
    weighting degenerates to a plain mean, and the planner silently stops doing anything.
    That failure is invisible in a success rate but obvious here.
    """

    min: float
    max: float
    mean: float
    std: float

    @property
    def is_degenerate(self) -> bool:
        return self.std < 1e-4


@dataclass
class PlanResult:
    """What one planning step produced."""

    chunk: "Tensor"                     # (1, H, action_dim), in the BC's normalized space
    q_values: "Tensor | None" = None    # (N,) raw scores, for logging and diagnostics
    spread: QSpread | None = None
    candidates: "Tensor | None" = None  # (N, H, action_dim)


class Planner(Protocol):
    """What :class:`~qplanning.policy.QPlanningPolicy` needs from a planner."""

    def plan(self, bc: "BCChunkSampler", batch: dict[str, "Tensor"]) -> PlanResult: ...
    def reset(self) -> None: ...


@dataclass
class PlannerContext:
    """The Q-side objects a planning step needs."""

    q_policy: "QFunctionPolicy"
    q_pre: "PolicyProcessorPipeline"
    camera_keys: tuple[str, ...]
    horizon: int


# ---------------------------------------------------------------------------------------
# The unguided baseline
# ---------------------------------------------------------------------------------------
class NoPlanner:
    """Execute the policy's own chunk, with no value guidance.

    Deliberately routed through the same ``sample_chunks`` call as Q-Planning (with N=1 and
    the full denoising budget) so that the baseline and the guided runs share one execution
    path, one action queue and one seeding scheme. Running the baseline through a separate
    code path is how protocol differences creep in between columns of a results table.
    """

    def __init__(self, denoise_steps: int | None = None) -> None:
        self.denoise_steps = denoise_steps

    def reset(self) -> None:
        return None

    def plan(self, bc: "BCChunkSampler", batch: dict[str, "Tensor"]) -> PlanResult:
        chunk = bc.sample_chunks(batch, 1, denoise_steps=self.denoise_steps)
        return PlanResult(chunk=chunk[:1])

    def __repr__(self) -> str:  # pragma: no cover
        return f"NoPlanner(denoise_steps={self.denoise_steps})"


# ---------------------------------------------------------------------------------------
# Q-weighted selection
# ---------------------------------------------------------------------------------------
class QWeightedPlanner:
    """Softmax-Q-weighted average over the top-K of N policy draws."""

    def __init__(
        self,
        cfg: "PlannerConfig",
        ctx: PlannerContext,
        bc_unnormalize: Any,
        generator: "torch.Generator | None" = None,
    ) -> None:
        self.cfg = cfg
        self.ctx = ctx
        self._bc_unnormalize = bc_unnormalize
        self._generator = generator
        self.last_spread: QSpread | None = None

    # ── construction ───────────────────────────────────────────────────────────────────
    @classmethod
    def build(
        cls,
        cfg: "PlannerConfig",
        q_checkpoint: str | Path,
        bc: "BCBundle",
        device: "torch.device | str" = "cuda",
        seed: int | None = None,
    ) -> "QWeightedPlanner":
        """Load the Q-function and check it is compatible with this BC policy."""
        import torch

        from qplanning.modeling_qfunction import load_q_function

        q_policy, q_pre = load_q_function(q_checkpoint, device=device)

        horizon = int(q_policy.config.h)
        if horizon != bc.sampler.chunk_size:
            raise ValueError(
                f"the Q-function scores chunks of length {horizon} but this BC policy emits "
                f"chunks of length {bc.sampler.chunk_size}. They must match: either use a "
                "Q-function trained with the same horizon, or a policy with that chunk size."
            )

        ctx = PlannerContext(
            q_policy=q_policy,
            q_pre=q_pre,
            camera_keys=tuple(q_policy.config.camera_keys),
            horizon=horizon,
        )

        planner_seed = cfg.seed if cfg.seed is not None else seed
        generator = (
            torch.Generator(device=torch.device(device)).manual_seed(planner_seed)
            if planner_seed is not None
            else None
        )
        return cls(cfg=cfg, ctx=ctx, bc_unnormalize=bc.sampler.unnormalize_actions,
                   generator=generator)

    def reset(self) -> None:
        return None

    # ── the planning step ──────────────────────────────────────────────────────────────
    def plan(self, bc: "BCChunkSampler", batch: dict[str, "Tensor"]) -> PlanResult:
        import torch

        with torch.no_grad():
            candidates = bc.sample_chunks(
                batch,
                self.cfg.n_samples,
                denoise_steps=self.cfg.denoise_steps,
                generator=self._generator,
            )
            observation = self._observation_inputs(batch)
            context = self._encode_observation_once(candidates, observation)
            q_values = self._score(candidates, context, observation)
            chunk = self._aggregate(candidates, q_values)

        spread = QSpread(
            min=float(q_values.min()),
            max=float(q_values.max()),
            mean=float(q_values.mean()),
            std=float(q_values.std()) if q_values.numel() > 1 else 0.0,
        )
        self.last_spread = spread
        return PlanResult(chunk=chunk, q_values=q_values, spread=spread, candidates=candidates)

    # ── internals ──────────────────────────────────────────────────────────────────────
    def _observation_inputs(self, batch: dict[str, "Tensor"]) -> dict[str, Any]:
        """The observation entries the Q-function reads: its cameras, plus the instruction."""
        missing = [key for key in self.ctx.camera_keys if key not in batch]
        if missing:
            raise KeyError(
                f"the Q-function expects camera(s) {missing} which the environment did not "
                f"provide. Available observation keys: "
                f"{sorted(k for k in batch if k.startswith('observation'))}."
            )
        inputs: dict[str, Any] = {key: batch[key] for key in self.ctx.camera_keys}
        if "task" in batch:
            inputs["task"] = batch["task"]
        return inputs

    def _encode_observation_once(self, candidates: "Tensor", observation: dict[str, Any]) -> "Tensor":
        """Run the visual and language encoders a single time for this step.

        A one-candidate batch is passed through the Q preprocessor first because that
        pipeline also normalizes pixels -- encoding raw images would feed DINOv2 different
        inputs than it saw in training.
        """
        from lerobot.utils.constants import ACTION

        single = {ACTION: candidates[:1], **observation}
        return self.ctx.q_policy.encode_obs_context(self.ctx.q_pre(single))

    def _score(self, candidates: "Tensor", context: "Tensor", observation: dict[str, Any]) -> "Tensor":
        """Score all N candidates in one decoder pass, reusing the encoded observation."""
        from lerobot.utils.constants import ACTION

        from qplanning.modeling_qfunction import expected_value

        n_samples, horizon, action_dim = candidates.shape

        # BC-normalized -> raw units -> Q-normalized.
        raw = self._bc_unnormalize(candidates.reshape(n_samples * horizon, action_dim))
        raw = raw.reshape(n_samples, horizon, action_dim).to(candidates.device)

        q_batch: dict[str, Any] = {ACTION: raw}
        for key, value in observation.items():
            q_batch[key] = _broadcast(value, n_samples)
        q_batch = self.ctx.q_pre(q_batch)

        actions = self.ctx.q_policy._truncate_action(q_batch[ACTION][:, : self.ctx.horizon, :])
        context_n = context.expand(-1, n_samples, -1).contiguous()
        logits = self.ctx.q_policy.q_online.forward_with_context(context_n, actions)
        return expected_value(logits, self.ctx.q_policy.bin_centers).to(
            device=candidates.device, dtype=candidates.dtype
        )

    def _aggregate(self, candidates: "Tensor", q_values: "Tensor") -> "Tensor":
        """Softmax-weighted mean of the top-K candidates."""
        import torch

        n_samples = candidates.shape[0]
        k = min(self.cfg.n_elites, n_samples) if self.cfg.n_elites > 0 else n_samples
        top = torch.topk(q_values, k)
        # Subtracting the max before the exponential is the usual softmax stabilisation and
        # leaves the weights unchanged.
        weights = torch.softmax((top.values - top.values.max()) / self.cfg.temperature, dim=0)
        selected = candidates[top.indices]
        return (weights.view(k, 1, 1) * selected).sum(dim=0, keepdim=True)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"QWeightedPlanner(N={self.cfg.n_samples}, K={self.cfg.n_elites}, "
            f"lambda={self.cfg.temperature}, denoise_steps={self.cfg.denoise_steps})"
        )


def _broadcast(value: Any, n_samples: int) -> Any:
    """Repeat a single observation across the candidate dimension."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.expand(n_samples, *value.shape[1:]).contiguous()
    if isinstance(value, (list, tuple)):
        return list(value[:1]) * n_samples if len(value) == 1 else list(value) * n_samples
    return value


def build_planner(
    cfg: "PlannerConfig",
    bc: "BCBundle",
    q_checkpoint: str | Path | None,
    device: "torch.device | str" = "cuda",
    seed: int | None = None,
    baseline_denoise_steps: int | None = None,
) -> Planner:
    """Build the planner named by ``cfg.type``."""
    if cfg.type == "none":
        return NoPlanner(denoise_steps=baseline_denoise_steps)
    if not q_checkpoint:
        raise ValueError(
            "planner.type is 'q_weighted' but no Q-function checkpoint is configured. "
            "Set paths.q_checkpoint, or use planner.type: none for the unguided baseline."
        )
    return QWeightedPlanner.build(cfg=cfg, q_checkpoint=q_checkpoint, bc=bc, device=device, seed=seed)
