"""Process environment preparation, which must happen before LeRobot is imported.

Why this module exists
----------------------
``lerobot.policies.fastwam.modeling_fastwam`` sets ``DIFFSYNTH_MODEL_BASE_PATH`` at
*import* time via :func:`os.environ.setdefault`, pointing at a directory that will not
exist on your machine. ``setdefault`` is the saving grace: whoever writes the variable first wins. So as
long as we export our value before that module is imported, the upstream default never
takes effect and no patch to the pinned fork is needed.

The corollary is that importing LeRobot before calling :func:`prepare_environment` is a
silent misconfiguration -- FastWAM would look for Wan2.2 weights in a directory that does
not exist on this machine, and fail much later with an opaque model-hash error. So we
detect that ordering and refuse.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Modules that latch environment state at import time. If any of these is already in
# sys.modules when prepare_environment() runs, our exports are too late to matter.
_LATCHING_MODULES = (
    "lerobot.policies.fastwam.modeling_fastwam",
    "lerobot.policies.fastwam.planning_vis_trajectories",
)

_DEFAULTS = {
    # LIBERO/RoboTwin render headlessly through EGL; without this MuJoCo tries GLFW and
    # dies on a node with no display.
    "MUJOCO_GL": "egl",
    # HuggingFace tokenizers warn and can deadlock when forked by DataLoader workers.
    "TOKENIZERS_PARALLELISM": "false",
    # FastWAM + Q + a planning batch fragments the allocator badly without this.
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}


class EnvironmentAlreadyImported(RuntimeError):
    """Raised when LeRobot was imported before the environment could be prepared."""


def _assert_not_latched(strict: bool) -> None:
    latched = [m for m in _LATCHING_MODULES if m in sys.modules]
    if not latched or not strict:
        return
    raise EnvironmentAlreadyImported(
        "qplanning.prepare_environment() ran after "
        + ", ".join(latched)
        + " was already imported, so DIFFSYNTH_MODEL_BASE_PATH may still hold the pinned "
        "fork's built-in default instead of your configured path. Import qplanning (or "
        "call prepare_environment()) before importing lerobot."
    )


def prepare_environment(
    *,
    wan22_weights: str | os.PathLike[str] | None = None,
    hf_home: str | os.PathLike[str] | None = None,
    robotwin_root: str | os.PathLike[str] | None = None,
    strict: bool = True,
) -> dict[str, str]:
    """Export the variables LeRobot latches at import time, and return what was set.

    Every argument is optional and falls back to the matching environment variable, so a
    bare ``prepare_environment()`` at package import is enough to neutralize the upstream
    default when the user has exported ``QPLANNING_WAN22_DIR`` themselves. Nothing already
    present in ``os.environ`` is overwritten -- the caller's explicit choice always wins.

    Returns:
        The variables this call actually set (absent ones were already defined).
    """
    _assert_not_latched(strict)
    applied: dict[str, str] = {}

    def _set(key: str, value: str | None) -> None:
        if value is None or key in os.environ:
            return
        os.environ[key] = str(value)
        applied[key] = str(value)

    for key, value in _DEFAULTS.items():
        _set(key, value)

    # The one that actually matters: Wan2.2 base weights for FastWAM's VAE and T5.
    _set("DIFFSYNTH_MODEL_BASE_PATH", wan22_weights or os.environ.get("QPLANNING_WAN22_DIR"))

    hf_home = hf_home or os.environ.get("QPLANNING_HF_HOME")
    if hf_home is not None:
        _set("HF_HOME", hf_home)
        _set("HF_HUB_CACHE", str(Path(hf_home) / "hub"))

    robotwin_root = robotwin_root or os.environ.get("ROBOTWIN_ROOT")
    if robotwin_root is not None:
        add_robotwin_to_path(robotwin_root)

    return applied


def add_robotwin_to_path(robotwin_root: str | os.PathLike[str]) -> None:
    """Put a RoboTwin checkout on ``sys.path``, replacing the usual PYTHONPATH exports.

    RoboTwin's task modules are imported as top-level ``envs.<task>`` packages, and its
    vendored curobo lives under ``envs/curobo/src``. Both must be importable before any
    ``lerobot.envs.robotwin`` call.
    """
    root = Path(robotwin_root).expanduser()
    for entry in (root / "envs" / "curobo" / "src", root):
        text = str(entry)
        if text not in sys.path:
            sys.path.insert(0, text)


def describe_environment() -> dict[str, str | None]:
    """Return the environment variables that affect where artifacts are loaded from."""
    keys = (
        "DIFFSYNTH_MODEL_BASE_PATH",
        "HF_HOME",
        "HF_HUB_CACHE",
        "ROBOTWIN_ROOT",
        "MUJOCO_GL",
        "TOKENIZERS_PARALLELISM",
        "PYTORCH_CUDA_ALLOC_CONF",
    )
    return {key: os.environ.get(key) for key in keys}
