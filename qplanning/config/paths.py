"""Turning configured strings into real locations.

This module is the reason the repository contains no absolute paths. A configured value may
be any of:

===========================  ==========================================================
``null`` / empty             optional; consumers raise a named error if they need it
``/abs/path`` or ``~/path``  used as-is -- a *user's* absolute path is perfectly fine
``org/name``                 a Hugging Face repo id, downloaded on first use
anything else                relative to the repository root, never to the cwd
===========================  ==========================================================

Resolving against the repository root rather than the cwd matters because SLURM jobs and
interactive shells start in different directories, and a config that silently means two
different things depending on where it was launched from is worse than one that fails.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

# `org/name` with the characters the Hub allows, and no path separators beyond the single
# slash. Deliberately strict: `configs/base.yaml` must not look like a repo id.
_HF_REPO_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")


class PathResolutionError(RuntimeError):
    """A required path could not be resolved."""


def repository_root() -> Path:
    """The root of this checkout (the directory containing ``qplanning/``)."""
    return Path(__file__).resolve().parents[2]


@dataclass
class ResolvedPath:
    """The outcome of resolving one configured value."""

    raw: str
    kind: str                 # "local" | "hub" | "missing" | "unset"
    path: Path | None = None
    repo_id: str | None = None

    @property
    def exists(self) -> bool:
        return self.kind == "local" and self.path is not None and self.path.exists()

    def require(self, field_name: str) -> Path:
        """Return a local directory, downloading from the Hub if needed."""
        if self.kind == "unset":
            raise PathResolutionError(
                f"{field_name} is not set. Set it in your .env (see .env.example) or pass "
                f"--{field_name}=<path or hf-repo-id>."
            )
        if self.kind == "hub":
            return download_from_hub(self.repo_id or self.raw)
        if self.path is None or not self.path.exists():
            raise PathResolutionError(
                f"{field_name} points at {self.raw!r}, which does not exist. "
                "If you meant a Hugging Face repo, use the `org/name` form."
            )
        return self.path


class PathResolver:
    """Resolves configured strings, with the repository root as the relative base."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else repository_root()

    def resolve(self, value: str | os.PathLike[str] | None) -> ResolvedPath:
        if value is None:
            return ResolvedPath(raw="", kind="unset")
        raw = str(value).strip()
        # An unfilled .env placeholder is 'unset', not a directory named "<TODO: ...>".
        if not raw or (raw.startswith("<") and raw.endswith(">")):
            return ResolvedPath(raw=raw, kind="unset")

        expanded = Path(raw).expanduser()
        if expanded.is_absolute():
            return ResolvedPath(raw=raw, kind="local", path=expanded)

        # A repo id only if it looks like one *and* no such local directory exists, so a
        # local `checkpoints/libero` always wins over a same-named Hub repo.
        local = (self.root / expanded).resolve()
        if _HF_REPO_ID.match(raw) and not local.exists():
            return ResolvedPath(raw=raw, kind="hub", repo_id=raw)
        return ResolvedPath(raw=raw, kind="local", path=local)

    def require(self, value: str | None, field_name: str) -> Path:
        return self.resolve(value).require(field_name)


def download_from_hub(repo_id: str, *, cache_dir: str | os.PathLike[str] | None = None) -> Path:
    """Snapshot a Hub repo and return the local directory."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise PathResolutionError(
            f"{repo_id!r} looks like a Hugging Face repo id but huggingface_hub is not "
            "installed. Install it, or point the field at a local directory."
        ) from exc
    return Path(snapshot_download(repo_id=repo_id, cache_dir=cache_dir))

