"""Per-episode results.

One row per episode, written by both the standalone evaluator and the self-improvement
loop's held-out block, so a results table can be assembled from either.

``episode_length`` is counted in environment steps and recorded for **every** episode, so
that success rates and episode lengths are always computed over the same set. Deriving
lengths from rendered video instead would restrict them to whichever episodes happened to be
rendered.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path

#: Core columns. Extra columns may follow; consumers read by name.
PAPER_COLUMNS: tuple[str, ...] = ("task_id", "task_name", "episode", "success", "episode_length")


@dataclass
class EpisodeRecord:
    """The outcome of one episode."""

    task_id: int
    task_name: str
    episode: int
    success: bool
    episode_length: int
    seed: int | None = None
    sum_reward: float | None = None
    wall_s: float | None = None
    #: Mean Q-value across the candidates at the episode's final planning step, when a
    #: Q-function was used. A single step rather than an episode average -- enough to spot a
    #: Q-function returning constants, not a summary of the episode.
    q_final: float | None = None

    @classmethod
    def column_names(cls) -> list[str]:
        extra = [f.name for f in fields(cls) if f.name not in PAPER_COLUMNS]
        return [*PAPER_COLUMNS, *extra]


def write_episodes(records: Iterable[EpisodeRecord], path: str | Path) -> Path:
    """Write records to CSV, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = EpisodeRecord.column_names()
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({k: asdict(record).get(k) for k in columns})
    return path


def read_episodes(paths: Sequence[str | Path]) -> list[dict]:
    """Read and concatenate per-episode CSVs, de-duplicating on (task_name, episode).

    De-duplication matters when a sharded run is resumed: a shard that was re-submitted
    after a preemption can legitimately produce the same episodes twice, and counting them
    twice would quietly skew a success rate.
    """
    seen: set[tuple[str, str]] = set()
    rows: list[dict] = []
    for path in paths:
        with Path(path).open(newline="") as handle:
            for row in csv.DictReader(handle):
                key = (row.get("task_name", ""), row.get("episode", ""))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
    return rows
