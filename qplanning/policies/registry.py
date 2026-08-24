"""Registry of behaviour-cloning policy adapters.

``bc.type`` in a config names an entry here. FastWAM is the default, but nothing in the
planner or the loops depends on it: an adapter satisfying
:class:`~qplanning.policies.base.BCChunkSampler` is all that is required.

Third-party packages can register without editing this repository by exposing an entry point
in the ``qplanning.bc_adapters`` group.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qplanning.config.schema import BCConfig
    from qplanning.policies.base import BCBundle

_REGISTRY: dict[str, type] = {}
_ENTRY_POINT_GROUP = "qplanning.bc_adapters"
_entry_points_loaded = False


def register_bc(name: str) -> Callable[[type], type]:
    """Register a BC adapter class under ``name``."""

    def decorate(cls: type) -> type:
        existing = _REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(f"BC adapter {name!r} is already registered to {existing.__name__}")
        _REGISTRY[name] = cls
        return cls

    return decorate


def _load_entry_points() -> None:
    global _entry_points_loaded
    if _entry_points_loaded:
        return
    _entry_points_loaded = True
    from importlib.metadata import entry_points

    for entry in entry_points(group=_ENTRY_POINT_GROUP):
        try:
            register_bc(entry.name)(entry.load())
        except Exception:  # noqa: BLE001 - a broken plug-in must not break the built-ins
            continue


def _load_builtins() -> None:
    # Imported for the side effect of registering. Kept lazy so that `qplanning doctor`
    # does not import torch.
    from qplanning.policies import fastwam  # noqa: F401


def available() -> list[str]:
    """Names that can be used as ``bc.type``."""
    _load_builtins()
    _load_entry_points()
    return sorted(_REGISTRY)


def get_adapter(name: str) -> type:
    """Look up an adapter class by name."""
    _load_builtins()
    _load_entry_points()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown bc.type {name!r}. Available: {', '.join(sorted(_REGISTRY))}. "
            "To add your own, see docs/configuration.md."
        ) from None


def build_bc(cfg: "BCConfig", env_cfg: Any = None, **kwargs: Any) -> "BCBundle":
    """Instantiate the BC policy named by ``cfg.type``."""
    return get_adapter(cfg.type).build(cfg, env_cfg=env_cfg, **kwargs)
