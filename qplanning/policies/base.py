"""The interface a behaviour-cloning policy must satisfy to be planned over.

Q-Planning is deliberately policy-agnostic: the Q-function scores whatever action chunks it
is given, and never needs to know how they were produced. Concretely, a policy has to be
able to do two things -- draw N candidate chunks for one observation, and say what those
chunks mean in raw environment units.

That second one is easy to overlook and is where a plug-in most often goes wrong. The BC
policy and the Q-function were trained on overlapping but distinct data, so their action
normalization statistics differ. Candidates therefore have to be converted out of the BC's
normalized space and back into the Q-function's before scoring; skipping that step produces
Q-values that look plausible and rank candidates wrongly.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch
    from torch import Tensor

    from lerobot.processor import PolicyProcessorPipeline


@runtime_checkable
class BCChunkSampler(Protocol):
    """A frozen BC policy Q-Planning can draw and score candidates from."""

    #: H -- the length of an action chunk. Must equal the Q-function's ``h``.
    chunk_size: int
    #: d_a -- the action dimension.
    action_dim: int

    def reset(self) -> None:
        """Clear any per-episode state. Called at the start of every episode."""
        ...

    def sample_chunks(
        self,
        batch: dict[str, "Tensor"],
        n_samples: int,
        *,
        denoise_steps: int | None = None,
        generator: "torch.Generator | None" = None,
    ) -> "Tensor":
        """Draw ``n_samples`` independent candidate chunks for a single observation.

        Args:
            batch: One observation, already through this policy's own preprocessor. The
                planner guarantees a batch size of 1.
            n_samples: N, the number of candidates to draw. Implementations should amortize
                observation encoding across all N rather than encoding N times -- that
                amortization is what keeps a planning step inside the replanning budget.
            denoise_steps: Sampler steps per candidate, if the policy has such a notion.
                Q-Planning deliberately passes a small value so the draws stay diverse.
                ``None`` means the policy's own default.
            generator: Optional RNG, for reproducible candidate draws.

        Returns:
            ``(n_samples, chunk_size, action_dim)`` in this policy's normalized action space.
        """
        ...

    def unnormalize_actions(self, actions: "Tensor") -> "Tensor":
        """Map actions from this policy's normalized space into raw environment units.

        Applied to any trailing shape; only the last dimension is interpreted as the action.
        """
        ...

    def to(self, device: Any) -> "BCChunkSampler":
        """Move the policy to a device."""
        ...

    def parameters(self) -> Iterable[Any]:
        """The policy's parameters. Q-Planning never puts these in an optimizer."""
        ...


@dataclass
class BCBundle:
    """A BC policy together with everything needed to run it in an environment.

    The sampler alone is not enough to roll out: observations have to be converted into the
    policy's input format, and the actions it emits back into what the environment expects.
    Keeping those alongside the sampler means the planner, the evaluator and the
    self-improvement loop all agree on the conversion.
    """

    sampler: BCChunkSampler
    #: Environment observation -> policy batch.
    preprocessor: "PolicyProcessorPipeline"
    #: Policy action -> raw environment action.
    postprocessor: "PolicyProcessorPipeline"
    #: The policy's own config, needed to build the matching environment processors.
    policy_config: Any
    #: How many steps of each chunk are executed before replanning.
    n_action_steps: int

    def __post_init__(self) -> None:
        if self.n_action_steps < 1:
            raise ValueError(f"n_action_steps must be >= 1, got {self.n_action_steps}")
        if self.n_action_steps > self.sampler.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) exceeds the chunk size "
                f"({self.sampler.chunk_size}): the policy cannot execute more steps than it plans."
            )
