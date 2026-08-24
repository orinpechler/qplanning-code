"""Loading a run config: ``extends``, interpolation, then draccus.

draccus gives typed decoding and the ``--a.b=c`` override grammar for free, and it is the
same grammar ``lerobot-train`` uses, so a user only has to learn one. What it does not give
is config inheritance or environment interpolation, which is what this module adds:

1. read the YAML and follow its ``extends:`` chain, deep-merging parents under children;
2. substitute ``${env:VAR}``, ``${env:VAR:default}`` and ``${cfg:dotted.path}``;
3. hand the fully-resolved mapping to draccus along with any CLI overrides.

Precedence, lowest to highest: ``base.yaml`` -> the ``extends`` chain -> the named config ->
values from ``.env`` -> real environment variables -> CLI overrides.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, TypeVar

import yaml

from qplanning.config.paths import repository_root
from qplanning.config.schema import RunConfig

T = TypeVar("T")

_INTERPOLATION = re.compile(r"\$\{(env|cfg):([^}:]+)(?::([^}]*))?\}")
_MAX_INTERPOLATION_PASSES = 10


class ConfigError(RuntimeError):
    """A config could not be loaded, merged, or interpolated."""


# --------------------------------------------------------------------------------------
# .env
# --------------------------------------------------------------------------------------
def load_dotenv(path: Path | None = None) -> dict[str, str]:
    """Populate ``os.environ`` from a ``.env`` file without overwriting real variables.

    Real environment variables always win, so exporting a value in the shell (or via
    ``sbatch --export``) overrides the file, which is what makes one checkout usable for
    several concurrent runs.
    """
    path = path or (repository_root() / ".env")
    if not path.exists():
        return {}
    loaded: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        # An unfilled placeholder is not a value; leave it unset so the error message
        # names the variable rather than a nonexistent directory.
        if value.startswith("<") and value.endswith(">"):
            continue
        loaded[key] = value
        os.environ.setdefault(key, value)
    return loaded


# --------------------------------------------------------------------------------------
# extends
# --------------------------------------------------------------------------------------
def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge ``override`` onto ``base``. Mappings merge; scalars and lists replace."""
    merged = dict(deepcopy(base))
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _read_with_extends(path: Path, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in _seen:
        chain = " -> ".join(p.name for p in (*_seen, path))
        raise ConfigError(f"circular `extends` chain: {chain}")
    if not path.exists():
        raise ConfigError(f"config not found: {path}")

    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping, got {type(data).__name__}")

    parents = data.pop("extends", None)
    if parents is None:
        return data
    if isinstance(parents, str):
        parents = [parents]

    merged: dict[str, Any] = {}
    for parent in parents:
        parent_path = (path.parent / parent).resolve()
        merged = _deep_merge(merged, _read_with_extends(parent_path, (*_seen, path)))
    return _deep_merge(merged, data)


# --------------------------------------------------------------------------------------
# interpolation
# --------------------------------------------------------------------------------------
def _lookup(data: Mapping[str, Any], dotted: str) -> Any:
    node: Any = data
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            raise ConfigError(f"${{cfg:{dotted}}} does not resolve: no key {part!r}")
        node = node[part]
    return node


def _substitute_scalar(value: str, data: Mapping[str, Any]) -> Any:
    """Substitute every ``${...}`` in one string, preserving type for a whole-string match."""

    def replace(match: re.Match[str]) -> str:
        kind, name, default = match.group(1), match.group(2).strip(), match.group(3)
        if kind == "env":
            resolved = os.environ.get(name)
            if resolved is None:
                resolved = default
            if resolved is None:
                raise ConfigError(
                    f"${{env:{name}}} is not set and has no default. Set {name} in your .env "
                    "(see .env.example), export it, or pass the field explicitly on the CLI."
                )
            return resolved
        looked_up = _lookup(data, name)
        return "" if looked_up is None else str(looked_up)

    whole = _INTERPOLATION.fullmatch(value)
    if whole is not None:
        kind, name, default = whole.group(1), whole.group(2).strip(), whole.group(3)
        if kind == "cfg":
            # A whole-string ${cfg:...} keeps the referenced value's type (int stays int).
            return _lookup(data, name)
        resolved = os.environ.get(name, default)
        if resolved is None:
            raise ConfigError(
                f"${{env:{name}}} is not set and has no default. Set {name} in your .env "
                "(see .env.example), export it, or pass the field explicitly on the CLI."
            )
        return resolved or None
    return _INTERPOLATION.sub(replace, value)


def _walk(node: Any, data: Mapping[str, Any]) -> Any:
    if isinstance(node, Mapping):
        return {k: _walk(v, data) for k, v in node.items()}
    if isinstance(node, list):
        return [_walk(v, data) for v in node]
    if isinstance(node, str) and "${" in node:
        return _substitute_scalar(node, data)
    return node


def _unresolved(node: Any, trail: str = "") -> list[str]:
    """Locations still holding a ``${...}`` reference after interpolation."""
    if isinstance(node, Mapping):
        return [x for k, v in node.items() for x in _unresolved(v, f"{trail}.{k}" if trail else str(k))]
    if isinstance(node, list):
        return [x for i, v in enumerate(node) for x in _unresolved(v, f"{trail}[{i}]")]
    if isinstance(node, str) and _INTERPOLATION.search(node):
        return [f"{trail} = {node}"]
    return []


def interpolate(data: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve interpolations to a fixpoint, so ``${cfg:}`` may reference other references.

    Reaching a fixpoint is not on its own success: a reference cycle such as
    ``a: ${cfg:b}`` / ``b: ${cfg:a}`` is stable under substitution while leaving the
    placeholders in place, which would then flow verbatim into the typed config. So the
    result is also checked for leftovers, which catches cycles and any other reference that
    never resolved.
    """
    current = dict(data)
    for _ in range(_MAX_INTERPOLATION_PASSES):
        nxt = _walk(current, current)
        if nxt == current:
            break
        current = nxt
    else:
        raise ConfigError(
            "interpolation did not converge after "
            f"{_MAX_INTERPOLATION_PASSES} passes -- check for a ${{cfg:...}} cycle."
        )

    leftovers = _unresolved(current)
    if leftovers:
        raise ConfigError(
            "these references never resolved (a ${cfg:...} cycle, or a key that does not "
            "exist):\n  " + "\n  ".join(leftovers)
        )
    return current


# --------------------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------------------
def _apply_overrides(mapping: dict[str, Any], overrides: Sequence[str] | None) -> dict[str, Any]:
    """Fold ``--dotted.key=value`` overrides into the mapping before interpolation.

    draccus applies overrides after decoding, which is too late for ``${cfg:...}``: a
    reference is resolved from the file's value and never sees the override. Overriding
    ``run_name`` would then rename the run while ``output_dir`` still pointed at the old
    directory, quietly overwriting the previous run's results. Applying them here means a
    reference resolves against the value the user actually asked for.

    Only unambiguous ``--a.b=c`` forms are folded in; anything else is left for draccus.
    """
    for item in overrides or []:
        if not item.startswith("--") or "=" not in item:
            continue
        key, _, raw = item[2:].partition("=")
        parts = key.split(".")
        node = mapping
        for part in parts[:-1]:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                node = None
                break
            node = nxt
        if node is None or parts[-1] not in node:
            continue        # unknown key: let draccus report it
        node[parts[-1]] = yaml.safe_load(raw)
    return mapping


def load_mapping(config_path: str | Path, overrides: Sequence[str] | None = None) -> dict[str, Any]:
    """Read a config to a fully-merged, fully-interpolated mapping (no typing yet)."""
    load_dotenv()
    merged = _apply_overrides(_read_with_extends(Path(config_path)), overrides)
    return interpolate(merged)


def load(
    config_path: str | Path,
    overrides: Sequence[str] | None = None,
    config_class: type[T] = RunConfig,  # type: ignore[assignment]
) -> T:
    """Load a config into its typed dataclass, applying CLI overrides last."""
    import tempfile

    import draccus

    mapping = load_mapping(config_path, overrides)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(mapping, handle, sort_keys=False)
        resolved_path = handle.name
    try:
        return draccus.parse(
            config_class=config_class,
            config_path=resolved_path,
            args=list(overrides or []),
        )
    finally:
        os.unlink(resolved_path)


def dump(config: Any, path: str | Path) -> Path:
    """Write the fully-resolved config next to a run's outputs, for provenance."""
    import draccus

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(draccus.dump(config))
    return path
