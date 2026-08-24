"""Q-Planning: value-guided action selection and self-improvement for frozen policies.

Importing this package prepares the process environment (see :mod:`qplanning._env_setup`)
**before** anything can pull in LeRobot, which is what stops the pinned fork's built-in
cluster path for the Wan2.2 weights from taking effect. Everything else is imported lazily,
so ``import qplanning`` stays cheap and never drags in torch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qplanning._env_setup import (
    EnvironmentAlreadyImported,
    add_robotwin_to_path,
    describe_environment,
    prepare_environment,
)

# Apply whatever the user already exported. A config-driven call in qplanning.cli refines
# this once a run config has been resolved; because prepare_environment() never overwrites
# an existing variable, the two calls compose rather than fight.
prepare_environment(strict=False)

__version__ = "1.0.0"

# The pinned commit of https://github.com/varungiridhar/lerobot this package builds on.
# Kept here so `qplanning doctor` can report drift between the installed fork and this pin.
LEROBOT_PIN = "qplanning-release-v1"

if TYPE_CHECKING:  # pragma: no cover - import-time cost is the whole point of deferring
    from qplanning.planner import NoPlanner, QWeightedPlanner
    from qplanning.policies.base import BCBundle, BCChunkSampler
    from qplanning.policy import QPlanningPolicy

_LAZY: dict[str, str] = {
    "QWeightedPlanner": "qplanning.planner",
    "NoPlanner": "qplanning.planner",
    "QPlanningPolicy": "qplanning.policy",
    "BCChunkSampler": "qplanning.policies.base",
    "BCBundle": "qplanning.policies.base",
}


def __getattr__(name: str) -> Any:
    """Resolve the heavy public symbols on first use (PEP 562)."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module), name)


def __dir__() -> list[str]:
    return sorted([*globals(), *_LAZY])


__all__ = [
    "LEROBOT_PIN",
    "BCBundle",
    "BCChunkSampler",
    "EnvironmentAlreadyImported",
    "NoPlanner",
    "QPlanningPolicy",
    "QWeightedPlanner",
    "__version__",
    "add_robotwin_to_path",
    "describe_environment",
    "prepare_environment",
]
