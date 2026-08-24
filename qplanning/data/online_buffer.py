"""The growing replay buffer of deployment rollouts.

Rollouts are written as LeRobot datasets, one directory per iteration (and per shard), and
read back lazily. The buffer is cumulative: iteration N trains on everything collected up to
N, not just the newest batch, which is what lets the Q-function keep the earlier failure
modes in view instead of forgetting them.

Both successful and failed episodes are kept. That asymmetry is the point of the method -- a
behaviour-cloning policy could only learn from the successes, while a value function learns
from a failure too, because a failed episode is simply one whose return is zero.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

from lerobot.policies.q_function.online_dataset import (
    OnlineQDataset,
    save_episodes_lerobot,
)

log = logging.getLogger("qplanning.data")

__all__ = ["OnlineQDataset", "save_episodes_lerobot", "save_iteration", "validate_shards",
           "iteration_dirs", "write_manifest"]

#: Leaf directory name every collection job writes into. ``OnlineQDataset`` globs for it, so
#: a shard that names its directory anything else is silently invisible to training.
EPISODES_DIRNAME = "online_episodes"


def iteration_dir(output_dir: Path, iteration: int, shard: int | None = None) -> Path:
    """Where one collection job writes, following the layout the buffer globs for."""
    name = f"iter_{iteration:03d}" if shard is None else f"iter_{iteration:03d}_shard{shard:02d}"
    return Path(output_dir) / name


def iteration_dirs(output_dir: Path) -> list[Path]:
    """Every episode directory currently in the buffer, oldest first."""
    return sorted(Path(output_dir).glob(f"iter_*/{EPISODES_DIRNAME}"))


def write_manifest(episodes_dir: Path, **fields: object) -> Path:
    """Record how a shard was produced, next to the data itself.

    Without this, a buffer is a pile of frames with no record of which policy or which
    Q-function generated them, which makes a surprising result impossible to attribute
    after the fact.
    """
    path = Path(episodes_dir).parent / "manifest.json"
    path.write_text(json.dumps(fields, indent=2, default=str))
    return path


def save_iteration(
    episode_dicts: Sequence[dict],
    episodes_dir: Path,
    *,
    camera_keys: Sequence[str],
    image_shape: tuple[int, int, int] | None,
    action_dim: int,
    manifest: dict | None = None,
) -> Path:
    """Write one job's episodes into the buffer."""
    episodes_dir = Path(episodes_dir)
    kwargs: dict = {"action_dim": action_dim, "camera_keys": tuple(camera_keys)}
    if image_shape is not None:
        kwargs["image_shape"] = image_shape
    save_episodes_lerobot(list(episode_dicts), episodes_dir, **kwargs)
    if manifest is not None:
        write_manifest(episodes_dir, **manifest)
    return episodes_dir


def validate_shards(output_dir: Path, iteration: int | None = None) -> tuple[list[Path], list[Path]]:
    """Separate readable episode directories from damaged ones.

    A collection job killed mid-write leaves a truncated parquet file behind. Training then
    fails partway through an epoch with an error that points at the reader rather than at
    the shard, and the whole iteration is lost. Checking up front costs a directory listing
    and turns that into a named, skippable shard.

    Returns:
        ``(good, quarantined)`` directories.
    """
    candidates = iteration_dirs(output_dir)
    if iteration is not None:
        prefix = f"iter_{iteration:03d}"
        candidates = [p for p in candidates if p.parent.name.startswith(prefix)]

    good: list[Path] = []
    quarantined: list[Path] = []
    for episodes_dir in candidates:
        problem = _shard_problem(episodes_dir)
        if problem is None:
            good.append(episodes_dir)
        else:
            quarantined.append(episodes_dir)
            log.warning("quarantining %s: %s", episodes_dir.parent.name, problem)
    return good, quarantined


def _shard_problem(episodes_dir: Path) -> str | None:
    """Why this directory cannot be trained on, or None if it is fine."""
    info = episodes_dir / "meta" / "info.json"
    if not info.exists():
        return "no meta/info.json (the job probably died before finalising)"
    try:
        total = json.loads(info.read_text()).get("total_episodes", 0)
    except json.JSONDecodeError:
        return "meta/info.json is not valid JSON"
    if not total:
        return "contains no episodes"

    parquets = list((episodes_dir / "data").rglob("*.parquet"))
    if not parquets:
        return "no parquet files"
    try:
        import pyarrow.parquet as pq

        for path in parquets:
            pq.read_metadata(path)          # reads the footer; a torn file fails here
    except ImportError:
        return None                          # cannot check without pyarrow; assume fine
    except Exception as exc:  # noqa: BLE001
        return f"unreadable parquet ({type(exc).__name__})"
    return None
