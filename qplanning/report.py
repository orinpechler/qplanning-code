"""Assemble per-episode results into a table.

Per-task success rate, then per-task mean length over *successful* episodes, then an
unweighted mean across tasks -- so a task with many episodes cannot outweigh one with few.

Columns are restricted to the intersection of task names before aggregating, so a task that
crashed under one condition cannot change another condition's denominator. The task count in
each row label is computed from that intersection rather than asserted, so a row that
silently lost tasks says so.
"""

from __future__ import annotations

import glob
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qplanning.utils.records import read_episodes

#: Column order of the rendered table.
COLUMN_KEYS: tuple[str, ...] = ("baseline", "offline", "self_improved")
COLUMN_LABELS: dict[str, str] = {
    "baseline": "BC policy",
    "offline": "Q-Planning (offline)",
    "self_improved": "Q-Planning (self-improved)",
}


# ── aggregation ────────────────────────────────────────────────────────────────────────
def per_task(rows: Sequence[dict]) -> dict[str, dict[str, float]]:
    """Per-task success rate, and mean episode length over successful episodes."""
    by_task: dict[str, list[dict]] = {}
    for row in rows:
        by_task.setdefault(row["task_name"], []).append(row)

    out: dict[str, dict[str, float]] = {}
    for task, task_rows in by_task.items():
        successes = [_as_bool(r["success"]) for r in task_rows]
        lengths = [
            float(r["episode_length"])
            for r in task_rows
            if _as_bool(r["success"]) and str(r.get("episode_length", "")).strip() not in ("", "None")
        ]
        out[task] = {
            "success": sum(successes) / len(successes),
            "ep_len": (sum(lengths) / len(lengths)) if lengths else float("nan"),
            "n": len(task_rows),
        }
    return out


def restrict_shared(*per_task_maps: dict[str, dict[str, float]]) -> list[dict[str, dict[str, float]]]:
    """Restrict every column to the tasks all of them evaluated."""
    populated = [m for m in per_task_maps if m]
    if not populated:
        return list(per_task_maps)
    shared = set(populated[0])
    for m in populated[1:]:
        shared &= set(m)
    return [{k: v for k, v in m.items() if k in shared} for m in per_task_maps]


def agg(task_map: dict[str, dict[str, float]]) -> tuple[float, float]:
    """Mean success rate and mean successful-episode length across tasks."""
    if not task_map:
        return float("nan"), float("nan")
    successes = [v["success"] for v in task_map.values()]
    lengths = [v["ep_len"] for v in task_map.values() if v["ep_len"] == v["ep_len"]]
    mean_len = sum(lengths) / len(lengths) if lengths else float("nan")
    return sum(successes) / len(successes), mean_len


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes")


def fmt_sr(x: float) -> str:
    return "--" if x != x else f"{100 * x:.1f}"


def fmt_el(x: float) -> str:
    return "--" if x != x else f"{x:.0f}"


# ── table assembly ─────────────────────────────────────────────────────────────────────
@dataclass
class Row:
    label: str
    n_tasks: int
    cells: dict[str, tuple[float, float]]
    missing: list[str]


def build_row(label: str, sources: dict[str, list[str]]) -> Row:
    """Aggregate one benchmark's columns onto a shared task set."""
    maps: dict[str, dict[str, dict[str, float]]] = {}
    missing: list[str] = []
    for key in COLUMN_KEYS:
        paths = sources.get(key) or []
        if not paths:
            missing.append(key)
            maps[key] = {}
            continue
        maps[key] = per_task(read_episodes(paths))

    restricted = restrict_shared(*(maps[k] for k in COLUMN_KEYS))
    n_tasks = max((len(m) for m in restricted), default=0)
    cells = {key: agg(m) for key, m in zip(COLUMN_KEYS, restricted)}
    return Row(label=label, n_tasks=n_tasks, cells=cells, missing=missing)


def build_table(spec: dict[str, Any]) -> list[Row]:
    """Build every row declared in a table spec, plus the mean row."""
    root = Path(spec.get("root", "."))
    rows: list[Row] = []
    for entry in spec.get("benchmarks", []):
        sources = {
            key: sorted(glob.glob(str(root / pattern)))
            for key, pattern in (entry.get("columns") or {}).items()
        }
        rows.append(build_row(entry["label"], sources))

    if rows:
        rows.append(_mean_row(rows))
    return rows


def _mean_row(rows: Sequence[Row]) -> Row:
    cells: dict[str, tuple[float, float]] = {}
    for key in COLUMN_KEYS:
        values = [r.cells[key][0] for r in rows if r.cells[key][0] == r.cells[key][0]]
        cells[key] = (sum(values) / len(values) if values else float("nan"), float("nan"))
    return Row(label="Mean", n_tasks=0, cells=cells, missing=[])


# ── rendering ──────────────────────────────────────────────────────────────────────────
def render(rows: Sequence[Row], fmt: str = "markdown") -> str:
    if fmt == "csv":
        lines = ["benchmark,column,success,episode_length,n_tasks"]
        for row in rows:
            for key in COLUMN_KEYS:
                sr, el = row.cells[key]
                lines.append(f"{row.label},{key},{fmt_sr(sr)},{fmt_el(el)},{row.n_tasks}")
        return "\n".join(lines)

    header = ["Benchmark"] + [f"{COLUMN_LABELS[k]} {s}" for k in COLUMN_KEYS for s in ("Succ.", "Len.")]

    if fmt == "latex":
        out = [
            r"\begin{tabular}{lcccccc}", r"\toprule",
            " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{COLUMN_LABELS[k]}}}" for k in COLUMN_KEYS) + r" \\",
            r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
            "Benchmark & " + " & ".join(["Succ. & Len."] * len(COLUMN_KEYS)) + r" \\",
            r"\midrule",
        ]
        for row in rows:
            if row.label == "Mean":
                out.append(r"\midrule")
            label = _labelled(row)
            cells = " & ".join(f"{fmt_sr(row.cells[k][0])} & {fmt_el(row.cells[k][1])}" for k in COLUMN_KEYS)
            out.append(f"{label} & {cells} " + r"\\")
        out += [r"\bottomrule", r"\end{tabular}"]
        return "\n".join(out)

    widths = [max(len(_labelled(r)) for r in rows) if rows else 9] + [len(h) for h in header[1:]]
    widths[0] = max(widths[0], len(header[0]))
    lines = ["| " + " | ".join(h.ljust(w) for h, w in zip(header, widths)) + " |",
             "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    for row in rows:
        cells = [_labelled(row).ljust(widths[0])]
        for i, key in enumerate(COLUMN_KEYS):
            sr, el = row.cells[key]
            cells.append(fmt_sr(sr).rjust(widths[1 + 2 * i]))
            cells.append(fmt_el(el).rjust(widths[2 + 2 * i]))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _labelled(row: Row) -> str:
    return f"{row.label} ({row.n_tasks} tasks)" if row.n_tasks > 10 else row.label


# ── CLI ────────────────────────────────────────────────────────────────────────────────
def main(config_path: str | None = None, overrides: Sequence[str] | None = None,
         show_help: bool = False) -> int:
    if show_help or not config_path:
        print(
            "usage: qplanning report --config configs/report.yaml [--format markdown|latex|csv]\n\n"
            "Aggregates per-episode CSVs from completed runs into the main results table."
        )
        return 0 if show_help else 2

    fmt = "markdown"
    for i, arg in enumerate(list(overrides or [])):
        if arg == "--format" and i + 1 < len(overrides or []):
            fmt = overrides[i + 1]  # type: ignore[index]
        elif arg.startswith("--format="):
            fmt = arg.split("=", 1)[1]
    if fmt not in ("markdown", "latex", "csv"):
        print(f"unknown --format {fmt!r}; expected markdown, latex or csv", file=sys.stderr)
        return 2

    from qplanning.config.loader import load_mapping

    spec = load_mapping(config_path)
    rows = build_table(spec)
    if not rows:
        print("no benchmarks declared in the table spec", file=sys.stderr)
        return 1

    print(render(rows, fmt))

    incomplete = [(r.label, r.missing) for r in rows if r.missing]
    if incomplete:
        print("\nMissing results (run these, then re-run report):", file=sys.stderr)
        for label, missing in incomplete:
            print(f"  {label}: {', '.join(missing)}", file=sys.stderr)
    return 0
