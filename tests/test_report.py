"""Result aggregation.

The synthetic tests below always run. An optional golden check against a directory of
reference CSVs is enabled by setting ``QPLANNING_REFERENCE_DATA``.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest

from qplanning.report import (
    Row,
    agg,
    build_row,
    fmt_el,
    fmt_sr,
    per_task,
    render,
    restrict_shared,
)
from qplanning.utils.records import EpisodeRecord, read_episodes, write_episodes

# Golden reference CSVs, if the user has any. Point QPLANNING_REFERENCE_DATA at a directory
# of per-episode CSVs plus a `reference.json` mapping each file to its expected
# {"success": float, "episode_length": float}, and the aggregation is checked against them.
REFERENCE_DATA = Path(os.environ["QPLANNING_REFERENCE_DATA"]) if os.environ.get(
    "QPLANNING_REFERENCE_DATA") else None

needs_reference_data = pytest.mark.skipif(
    REFERENCE_DATA is None or not REFERENCE_DATA.exists(),
    reason="set QPLANNING_REFERENCE_DATA to a directory of reference CSVs to enable",
)


@needs_reference_data
def test_reproduces_reference_numbers():
    """Aggregation must reproduce known-good figures for a set of reference runs."""
    import json

    expected = json.loads((REFERENCE_DATA / "reference.json").read_text())
    for filename, want in expected.items():
        task_map = per_task(read_episodes([REFERENCE_DATA / filename]))
        success, length = agg(task_map)
        assert 100 * success == pytest.approx(want["success"], abs=0.05), filename
        assert length == pytest.approx(want["episode_length"], abs=0.5), filename


# ── synthetic ──────────────────────────────────────────────────────────────────────────
def _write(tmp_path: Path, name: str, rows: list[tuple[str, bool, int]]) -> Path:
    records = [
        EpisodeRecord(task_id=i, task_name=task, episode=i, success=ok, episode_length=length)
        for i, (task, ok, length) in enumerate(rows)
    ]
    return write_episodes(records, tmp_path / name)


def test_length_uses_successful_episodes_only():
    """A failure runs to the step limit; including it would inflate the reported length."""
    rows = [
        {"task_name": "t", "success": "True", "episode_length": "100"},
        {"task_name": "t", "success": "False", "episode_length": "520"},
    ]
    stats = per_task(rows)["t"]
    assert stats["success"] == 0.5
    assert stats["ep_len"] == 100


def test_tasks_are_weighted_equally_regardless_of_episode_count():
    """Per-task rates are averaged, not per-episode outcomes."""
    rows = [{"task_name": "a", "success": "True", "episode_length": "10"}]
    rows += [{"task_name": "b", "success": "False", "episode_length": "10"}] * 99
    success, _ = agg(per_task(rows))
    assert success == pytest.approx(0.5), "a 99-episode task must not dominate a 1-episode task"


def test_restrict_shared_uses_the_intersection():
    a = {"t1": {"success": 1.0, "ep_len": 10.0}, "t2": {"success": 0.0, "ep_len": 20.0}}
    b = {"t1": {"success": 0.5, "ep_len": 12.0}}
    ra, rb = restrict_shared(a, b)
    assert set(ra) == set(rb) == {"t1"}


def test_a_task_missing_from_one_column_is_dropped_from_all(tmp_path):
    """A crashed task must not change another column's denominator."""
    full = _write(tmp_path, "full.csv", [("t1", True, 10), ("t2", False, 20)])
    partial = _write(tmp_path, "partial.csv", [("t1", True, 12)])
    row = build_row("bench", {"baseline": [str(full)], "offline": [str(partial)],
                              "self_improved": [str(partial)]})
    assert row.n_tasks == 1
    assert row.cells["baseline"][0] == 1.0, "t2 must be excluded, not counted as a failure"


def test_duplicate_episodes_are_counted_once(tmp_path):
    """Re-submitting a preempted shard can legitimately re-emit episodes."""
    path = _write(tmp_path, "a.csv", [("t1", True, 10), ("t1", False, 20)])
    assert len(read_episodes([path, path])) == 2


def test_missing_columns_are_reported_not_guessed(tmp_path):
    path = _write(tmp_path, "a.csv", [("t1", True, 10)])
    row = build_row("bench", {"baseline": [str(path)]})
    assert set(row.missing) == {"offline", "self_improved"}
    assert row.cells["offline"][0] != row.cells["offline"][0], "missing column must be NaN"


@pytest.mark.parametrize("fmt", ["markdown", "latex", "csv"])
def test_renders_without_error(fmt):
    rows = [Row("LIBERO-10", 10, {"baseline": (0.90, 274.0), "offline": (0.93, 261.0),
                                  "self_improved": (0.99, 224.0)}, [])]
    out = render(rows, fmt)
    assert "90.0" in out and "99.0" in out


def test_formatting():
    assert fmt_sr(0.905) == "90.5"
    assert fmt_el(273.6) == "274"
    assert fmt_sr(float("nan")) == "--"
