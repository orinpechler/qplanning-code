"""The policy that gets rolled out: a BC sampler, a planner, and an action queue.

Bundling these into one ``nn.Module`` means the same object serves the frozen baseline, the
Q-guided planner and the self-improvement loop, and satisfies the rollout contract LeRobot's
evaluation loop expects (``reset()`` then repeated ``select_action()``).

It also keeps the queue in one place. A planning step emits a chunk of ``H`` actions, of
which only the first ``n_action_steps`` are executed before replanning -- that ratio is the
replanning cadence, and having it live here rather than inside each policy is what lets a
new BC policy be plugged in without reimplementing it.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    from torch import Tensor

    from qplanning.planner import PlanResult, Planner
    from qplanning.policies.base import BCBundle


class QPlanningPolicy(nn.Module):
    """A frozen BC policy under value-guided action selection."""

    def __init__(self, bc: "BCBundle", planner: "Planner", n_action_steps: int | None = None) -> None:
        super().__init__()
        self.bc = bc
        self.planner = planner
        self.n_action_steps = int(n_action_steps if n_action_steps is not None else bc.n_action_steps)
        if self.n_action_steps > bc.sampler.chunk_size:
            raise ValueError(
                f"execution.n_action_steps ({self.n_action_steps}) exceeds the policy's chunk "
                f"size ({bc.sampler.chunk_size}); it cannot execute more steps than it plans."
            )
        self._queue: deque["Tensor"] = deque()
        self.last_plan: "PlanResult | None" = None
        self.steps_planned = 0
        self.episode_seed: int | None = None
        self._plans_this_episode = 0

    # ── rollout interface ──────────────────────────────────────────────────────────────
    def set_episode_seed(self, seed: int | None) -> None:
        """Seed the policy's noise for the coming episode(s).

        Planning step ``k`` of an episode with seed ``s`` then draws its noise from an RNG
        seeded by ``(s, k)`` alone, independent of anything that ran before it. Two runs
        that differ only in what is done *with* the policy's chunks (baseline, Q-weighting,
        Q-gradient refinement) therefore see identical noise at every planning step they
        both reach, even after their trajectories diverge. ``None`` falls back to the global
        RNG. Survives :meth:`reset`, which LeRobot's rollout calls on its own.
        """
        self.episode_seed = seed

    def reset(self) -> None:
        """Start a new episode: drop any unexecuted actions."""
        self._queue.clear()
        self.last_plan = None
        self._plans_this_episode = 0
        self.bc.sampler.reset()
        self.planner.reset()

    @torch.no_grad()
    def select_action(self, batch: dict[str, "Tensor"]) -> "Tensor":
        """Return the next action, replanning when the queue runs dry."""
        self._assert_single_observation(batch)
        if not self._queue:
            self._replan(batch)
        return self._queue.popleft()

    # ── internals ──────────────────────────────────────────────────────────────────────
    def _replan(self, batch: dict[str, "Tensor"]) -> None:
        with self._plan_noise():
            result = self.planner.plan(self.bc.sampler, batch)
        self.last_plan = result
        self.steps_planned += 1
        self._plans_this_episode += 1
        chunk = result.chunk[0, : self.n_action_steps]     # (n_action_steps, action_dim)
        self._queue.extend(chunk[i : i + 1] for i in range(chunk.shape[0]))

    @contextmanager
    def _plan_noise(self) -> Iterator[None]:
        """Reseed the global RNGs for this planning step, restoring them afterwards.

        The global RNG is reseeded, rather than a generator handed down, because FastWAM
        draws its initial noise from the global CPU RNG and has no generator argument.
        Candidate ``i`` of an N-sample draw gets the same noise as the single draw of an
        N=1 run whenever torch's CPU ``randn`` is prefix-stable for the chunk shape, which
        it is for chunk_size * action_dim divisible by 16 (LIBERO: 32 * 7).
        """
        if self.episode_seed is None:
            yield
            return
        import numpy as np

        seed = int(np.random.SeedSequence([self.episode_seed, self._plans_this_episode]).generate_state(1)[0])
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            yield

    @staticmethod
    def _assert_single_observation(batch: dict[str, Any]) -> None:
        """Q-Planning scores candidates for one state, so the batch must hold one state.

        Checked here, with the fix in the message, because the alternative is a shape error
        several frames deeper into the Q-function.
        """
        for key, value in batch.items():
            if not isinstance(value, torch.Tensor) or value.dim() == 0:
                continue
            if value.shape[0] != 1:
                raise NotImplementedError(
                    f"Q-Planning runs with one environment at a time, but {key!r} has batch "
                    f"size {value.shape[0]}. Q-Planning is single-environment; the "
                    "batch size is fixed at 1 and is not configurable."
                )

    # ── convenience ────────────────────────────────────────────────────────────────────
    @property
    def chunk_size(self) -> int:
        return self.bc.sampler.chunk_size

    def to(self, *args: Any, **kwargs: Any) -> "QPlanningPolicy":
        self.bc.sampler.to(*args, **kwargs)
        return self

    def parameters(self, recurse: bool = True):  # noqa: D102 - inherited meaning
        return self.bc.sampler.parameters()

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"QPlanningPolicy(planner={self.planner!r}, chunk_size={self.chunk_size}, "
            f"n_action_steps={self.n_action_steps})"
        )
