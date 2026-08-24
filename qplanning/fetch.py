"""``qplanning fetch-checkpoints`` -- download the released weights.

Reads ``configs/checkpoints.yaml`` and snapshots each repository into the Hugging Face
cache (``QPLANNING_HF_HOME`` if set). The
registry is the single place a maintainer edits when the weights are published; nothing
else in the repository names a model location.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from qplanning.config.loader import ConfigError, load_dotenv, load_mapping
from qplanning.config.paths import repository_root

log = logging.getLogger("qplanning.fetch")


def main(config_path: str | None = None, overrides: Sequence[str] | None = None,
         show_help: bool = False) -> int:
    if show_help:
        print(
            "usage: qplanning fetch-checkpoints [--config configs/checkpoints.yaml] [--only NAME]\n\n"
            "Downloads the released weights into paths.root and prints the environment\n"
            "variables to put in your .env."
        )
        return 0

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    load_dotenv()
    config_path = config_path or str(repository_root() / "configs" / "checkpoints.yaml")

    only = None
    for i, arg in enumerate(list(overrides or [])):
        if arg == "--only" and i + 1 < len(overrides or []):
            only = overrides[i + 1]  # type: ignore[index]
        elif arg.startswith("--only="):
            only = arg.split("=", 1)[1]

    try:
        spec = load_mapping(config_path)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 1

    entries = spec.get("checkpoints", {})
    if only:
        if only not in entries:
            print(f"unknown checkpoint {only!r}; available: {', '.join(entries)}", file=sys.stderr)
            return 2
        entries = {only: entries[only]}

    # A repo id of the form "/name" means `org` is unset: those weights are not published
    # yet. Fetch whatever *is* available and say plainly what is not, rather than refusing
    # the whole command.
    unresolved = {n: e for n, e in entries.items() if str(e.get("repo_id", "")).startswith("/")}
    entries = {n: e for n, e in entries.items() if n not in unresolved}

    if unresolved:
        print(
            "Not yet published, so not downloaded: " + ", ".join(unresolved) + ".\n"
            "Point these at local directories in your .env: "
            + ", ".join(str(e.get("env_var")) for e in unresolved.values() if e.get("env_var"))
            + "\n(a checkpoint field takes a path as readily as a repo id). "
            "`qplanning doctor` reports which are still unresolved.\n",
            file=sys.stderr,
        )
    if not entries:
        return 0

    total = sum(float(e.get("size_gb", 0)) for e in entries.values())
    log.info("fetching %d checkpoint(s), about %.0f GB in total", len(entries), total)

    from huggingface_hub import snapshot_download

    cache_dir = os.environ.get("QPLANNING_HF_HOME") or None
    resolved: dict[str, Path] = {}
    for name, entry in entries.items():
        repo_id = entry["repo_id"]
        log.info("  %-14s %-52s (~%s GB)", name, repo_id, entry.get("size_gb", "?"))
        resolved[name] = Path(snapshot_download(repo_id=repo_id, cache_dir=cache_dir))

    print("\nAdd these to your .env:\n")
    for name, entry in entries.items():
        if entry.get("env_var"):
            print(f"{entry['env_var']}={resolved[name]}")
    return 0
