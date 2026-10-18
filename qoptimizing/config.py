"""Configuration for Q-gradient action optimization.

Extends :class:`qplanning.config.schema.RunConfig` with one ``qopt`` block, so every existing
benchmark config can be reused unchanged and only the optimization settings are added. The
inherited ``planner`` block is ignored by this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from qplanning.config.schema import RunConfig


@dataclass
class QOptConfig:
    """How the policy's chunk is refined against the Q-function.

    The optimization runs in the BC policy's *normalized* action space -- the space the
    executed chunk lives in -- so ``step_size`` and ``max_delta`` are in units of the
    policy's action normalization (roughly one standard deviation per unit).
    """

    # What the ascent moves. "action": the chunk itself. "noise": FastWAM's initial
    # flow-matching noise; dQ/d(noise) is taken back through every denoising step and the
    # chunk is regenerated from the updated noise, so it stays a FastWAM sample.
    target: str = "action"
    steps: int = 10                   # gradient-ascent steps; 0 executes the policy's chunk as-is
    step_size: float = 0.01           # learning rate
    optimizer: str = "sgd"            # sgd | adam
    # sgd only: rescale each chunk's gradient to unit RMS, so every step moves the chunk by
    # `step_size` per element regardless of how flat Q is at that point.
    normalize_grad: bool = True
    # Optional trust region: clamp each element of (optimized - initial) to [-max_delta, max_delta],
    # in the units of whatever `target` names.
    max_delta: Optional[float] = None
    # Soft trust region: ascend J(a) = Q(a) - lambda * ||a - a_0||^2 instead of Q alone, so the
    # gradient gains a -2 * lambda * (a_k - a_0) pull back towards the policy's chunk a_0. In
    # BC-normalized action units; with target=noise the penalty is still on the denoised chunk,
    # and its gradient is taken back through the denoiser. 0 disables it.
    anchor_weight: float = 0.0
    # Keep the gripper (the last action dimension) at the policy's value throughout. It is
    # executed as open/close only, so continuous gripper changes do nothing on the robot and
    # would only inflate Q. Applies to target=noise too: the regenerated chunk keeps the
    # starting chunk's gripper.
    freeze_gripper: bool = True
    # noise only: after each step, rescale every chunk's noise back to its original norm, so it
    # stays on the shell where Gaussian noise of this size lives and the denoiser stays in
    # distribution.
    renorm_noise: bool = True
    # Chunks drawn and optimized in parallel; `select` picks which one is executed.
    n_samples: int = 1
    # Which of the N chunks to execute, by final Q: "best" (highest) or "worst" (lowest).
    # "worst" is a diagnostic: if Q ranks actions by how well they do, best should beat it.
    select: str = "best"
    # Flow-matching steps for the initial chunk. None uses bc.baseline_denoise_steps, so that
    # with steps=0 this reproduces the unguided baseline exactly.
    denoise_steps: Optional[int] = None
    # Write every planning step's Q trajectory to <output_dir>/qopt_steps.jsonl.
    log_steps: bool = True

    def __post_init__(self) -> None:
        if self.target not in ("action", "noise"):
            raise ValueError(f"qopt.target must be 'action' or 'noise', got {self.target!r}")
        if self.select not in ("best", "worst"):
            raise ValueError(f"qopt.select must be 'best' or 'worst', got {self.select!r}")
        if self.steps < 0:
            raise ValueError(f"qopt.steps must be >= 0, got {self.steps}")
        if self.step_size <= 0:
            raise ValueError(f"qopt.step_size must be > 0, got {self.step_size}")
        if self.optimizer not in ("sgd", "adam"):
            raise ValueError(f"qopt.optimizer must be 'sgd' or 'adam', got {self.optimizer!r}")
        if self.max_delta is not None and self.max_delta <= 0:
            raise ValueError(f"qopt.max_delta must be > 0 or null, got {self.max_delta}")
        if self.anchor_weight < 0:
            raise ValueError(f"qopt.anchor_weight must be >= 0, got {self.anchor_weight}")
        if self.n_samples <= 0:
            raise ValueError(f"qopt.n_samples must be > 0, got {self.n_samples}")
        if self.denoise_steps is not None and self.denoise_steps <= 0:
            raise ValueError(f"qopt.denoise_steps must be > 0 or null, got {self.denoise_steps}")


@dataclass
class QOptRunConfig(RunConfig):
    """A normal run config plus the ``qopt`` block."""

    qopt: QOptConfig = field(default_factory=QOptConfig)
