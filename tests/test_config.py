"""Config loading: inheritance, interpolation, precedence, and validation."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from qplanning.config.loader import ConfigError, _deep_merge, interpolate, load, load_mapping
from qplanning.config.paths import PathResolver, repository_root
from qplanning.config.schema import PlannerConfig

CONFIGS = repository_root() / "configs"


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch):
    """A fully-populated environment, so configs resolve without a real .env."""
    for key, value in {
        "QPLANNING_ROOT": "/tmp/qp-artifacts",
        "QPLANNING_BC_CKPT_LIBERO": "/tmp/bc-libero",
        "QPLANNING_Q_CKPT_LIBERO": "/tmp/q-libero",
        "QPLANNING_BC_CKPT_ROBOTWIN": "/tmp/bc-robotwin",
        "QPLANNING_Q_CKPT_ROBOTWIN": "/tmp/q-robotwin",
        "QPLANNING_WAN22_DIR": "/tmp/wan22",
        "QPLANNING_DATA_ROOT": "/tmp/data",
        "QPLANNING_ROBOTWIN_DATASET_ROOT": "/tmp/robotwin-data",
        "ROBOTWIN_ROOT": "/tmp/robotwin",
    }.items():
        monkeypatch.setenv(key, value)
    return monkeypatch


# ── inheritance ────────────────────────────────────────────────────────────────────────
def test_deep_merge_merges_mappings_and_replaces_scalars():
    base = {"a": {"x": 1, "y": 2}, "b": [1, 2], "c": 3}
    got = _deep_merge(base, {"a": {"y": 20}, "b": [9], "c": 30})
    assert got == {"a": {"x": 1, "y": 20}, "b": [9], "c": 30}
    assert base["a"]["y"] == 2, "the base mapping must not be mutated"


def test_extends_chain_is_applied(tmp_path: Path, env):
    (tmp_path / "parent.yaml").write_text(yaml.safe_dump({"seed": 1, "planner": {"n_samples": 64}}))
    (tmp_path / "child.yaml").write_text(
        "extends: parent.yaml\n" + yaml.safe_dump({"planner": {"n_samples": 8}})
    )
    got = load_mapping(tmp_path / "child.yaml")
    assert got["seed"] == 1, "inherited from the parent"
    assert got["planner"]["n_samples"] == 8, "child wins"


def test_circular_extends_is_reported(tmp_path: Path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n")
    with pytest.raises(ConfigError, match="circular"):
        load_mapping(tmp_path / "a.yaml")


# ── interpolation ──────────────────────────────────────────────────────────────────────
def test_env_interpolation_uses_default_when_unset(monkeypatch):
    monkeypatch.delenv("QP_TEST_VAR", raising=False)
    assert interpolate({"k": "${env:QP_TEST_VAR:fallback}"})["k"] == "fallback"


def test_env_interpolation_prefers_the_real_variable(monkeypatch):
    monkeypatch.setenv("QP_TEST_VAR", "real")
    assert interpolate({"k": "${env:QP_TEST_VAR:fallback}"})["k"] == "real"


def test_missing_env_without_default_names_the_variable(monkeypatch):
    monkeypatch.delenv("QP_TEST_VAR", raising=False)
    with pytest.raises(ConfigError, match="QP_TEST_VAR"):
        interpolate({"k": "${env:QP_TEST_VAR}"})


def test_cfg_interpolation_preserves_type_and_composes():
    got = interpolate({"n": 7, "copy": "${cfg:n}", "joined": "x-${cfg:n}"})
    assert got["copy"] == 7 and isinstance(got["copy"], int), "whole-string ref keeps its type"
    assert got["joined"] == "x-7", "embedded ref is stringified"


def test_cfg_interpolation_resolves_chains():
    got = interpolate({"root": "/r", "run": "demo", "out": "${cfg:root}/runs/${cfg:run}"})
    assert got["out"] == "/r/runs/demo"


def test_interpolation_cycle_is_reported():
    with pytest.raises(ConfigError, match="converge|resolve"):
        interpolate({"a": "${cfg:b}", "b": "${cfg:a}"})


# ── the shipped configs ────────────────────────────────────────────────────────────────
#: Files under configs/ that are not run configs (see doctor._SPEC_KINDS).
NON_RUN_CONFIGS = {"base.yaml", "report.yaml", "checkpoints.yaml"}


def _shipped():
    return sorted(p for p in CONFIGS.rglob("*.yaml") if p.name not in NON_RUN_CONFIGS)


@pytest.mark.parametrize("path", _shipped(), ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_every_shipped_config_parses(path: Path, env):
    cfg = load(path)
    assert cfg.run_name
    assert cfg.qfunction.h == 32, "H must match FastWAM's chunk size"


def test_baseline_differs_from_qplanning_only_by_planner_type(env):
    """The one-field claim the README makes: guided and unguided differ only by planner.type."""
    for suite in ("libero_10", "libero_spatial", "libero_object", "libero_goal", "robotwin"):
        base = load(CONFIGS / suite / "baseline.yaml")
        qplan = load(CONFIGS / suite / "qplanning_offline.yaml")
        assert base.planner.type == "none"
        assert qplan.planner.type == "q_weighted"
        for field in ("n_samples", "n_elites", "temperature", "denoise_steps"):
            assert getattr(base.planner, field) == getattr(qplan.planner, field), field
        assert base.execution.n_action_steps == qplan.execution.n_action_steps
        assert base.eval.episodes_per_task == qplan.eval.episodes_per_task


def test_shipped_planner_settings(env):
    """The published planner settings for each benchmark."""
    libero = load(CONFIGS / "libero_10" / "qplanning_offline.yaml")
    assert (libero.planner.n_samples, libero.planner.n_elites) == (64, 16)
    assert libero.execution.n_action_steps == 10
    robotwin = load(CONFIGS / "robotwin" / "qplanning_offline.yaml")
    assert (robotwin.planner.n_samples, robotwin.planner.n_elites) == (32, 8)
    assert robotwin.execution.n_action_steps == 24
    assert robotwin.benchmark.tasks == "paper47"
    for cfg in (libero, robotwin):
        assert cfg.planner.temperature == 1.0
        assert cfg.planner.denoise_steps == 3
        assert cfg.qfunction.gamma == 0.99
        assert (cfg.qfunction.num_bins, cfg.qfunction.v_min, cfg.qfunction.v_max) == (101, -0.01, 1.01)


def test_cli_overrides_win(env):
    cfg = load(CONFIGS / "libero_10" / "qplanning_offline.yaml", ["--planner.n_samples=8"])
    assert cfg.planner.n_samples == 8


def test_overrides_are_visible_to_interpolation(env):
    """An overridden field must be seen by anything that references it.

    `output_dir` is built from `${cfg:run_name}`. If the override landed after
    interpolation, renaming a run would leave it writing into the previous run's directory
    and silently overwrite those results.
    """
    default = load(CONFIGS / "libero_10" / "baseline.yaml")
    renamed = load(CONFIGS / "libero_10" / "baseline.yaml", ["--run_name=ablation"])
    assert renamed.run_name == "ablation"
    assert renamed.paths.output_dir.endswith("ablation")
    assert renamed.paths.output_dir != default.paths.output_dir


# ── validation ─────────────────────────────────────────────────────────────────────────
def test_n_elites_is_clamped_not_rejected():
    """Upstream computes K = min(K, N); lowering N alone must stay valid."""
    assert PlannerConfig(n_samples=8, n_elites=16).n_elites == 8


def test_planner_type_is_validated():
    with pytest.raises(ValueError, match="q_weighted"):
        PlannerConfig(type="mppi")


# ── path resolution ────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "value,kind",
    [
        (None, "unset"),
        ("", "unset"),
        ("<TODO: fill me in>", "unset"),
        ("/absolute/path", "local"),
        ("HuggingFaceVLA/libero", "hub"),
        ("configs", "local"),
    ],
)
def test_path_resolution_kinds(value, kind):
    assert PathResolver().resolve(value).kind == kind


def test_relative_paths_resolve_against_the_repo_not_the_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    resolved = PathResolver().resolve("configs")
    assert resolved.path == repository_root() / "configs"


def test_non_run_configs_still_parse_as_mappings(env):
    """The table spec and checkpoint registry are validated too, just not as RunConfig."""
    from qplanning.config.loader import load_mapping

    for name in ("report.yaml", "checkpoints.yaml"):
        mapping = load_mapping(CONFIGS / name)
        assert mapping, name


def test_report_spec_declares_every_benchmark_and_column(env):
    """The report spec must cover every benchmark and column; a missing one is a silent gap."""
    from qplanning.config.loader import load_mapping
    from qplanning.report import COLUMN_KEYS

    spec = load_mapping(CONFIGS / "report.yaml")
    labels = [b["label"] for b in spec["benchmarks"]]
    assert labels == ["LIBERO-Spatial", "LIBERO-Object", "LIBERO-Goal", "LIBERO-10", "RoboTwin"]
    for entry in spec["benchmarks"]:
        assert set(entry["columns"]) == set(COLUMN_KEYS), entry["label"]
