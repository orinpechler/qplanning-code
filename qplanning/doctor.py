"""``qplanning doctor`` -- resolve everything a run needs, before the run.

The failure this exists to prevent is a job that queues for four hours, loads a 25 GB
policy, and only then discovers that a checkpoint path was never set. Every check here runs
on a login node in seconds and touches no GPU.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from qplanning.config.loader import ConfigError, load, load_dotenv
from qplanning.config.paths import PathResolver
from qplanning.config.schema import RunConfig

_OK = "OK"
_MISSING = "MISSING"
_HUB = "HUB"
_UNSET = "UNSET"
_PENDING = "PENDING"   # an output location that does not exist yet, which is fine


@dataclass
class Check:
    field: str
    raw: str
    status: str
    detail: str = ""
    required: bool = True

    @property
    def failed(self) -> bool:
        return self.required and self.status in (_MISSING, _UNSET)


def _check_path(resolver: PathResolver, field: str, value: str | None, *, required: bool,
                expect: Sequence[str] = (), output: bool = False) -> Check:
    """Resolve one configured path.

    ``output=True`` marks a location this run *writes* to, so its absence is expected on a
    first run and is reported as pending rather than as a failure.
    """
    resolved = resolver.resolve(value)
    if resolved.kind == "unset":
        return Check(field, value or "", _UNSET, "not set", required)
    if resolved.kind == "hub":
        return Check(field, resolved.raw, _HUB, "downloads on first use", required)
    path = resolved.path
    assert path is not None
    if not path.exists():
        if output:
            return Check(field, resolved.raw, _PENDING, f"will be created: {path}", False)
        return Check(field, resolved.raw, _MISSING, f"no such path: {path}", required)
    missing = [name for name in expect if not (path / name).exists()]
    if missing:
        return Check(field, resolved.raw, _MISSING, f"missing {', '.join(missing)}", required)
    return Check(field, resolved.raw, _OK, str(path), required)


# A LeRobot-format checkpoint directory always carries these.
_CHECKPOINT_FILES = ("config.json", "model.safetensors")


def collect_checks(cfg: RunConfig) -> list[Check]:
    """Every path this config depends on, with whether it is required for this run."""
    resolver = PathResolver()
    needs_q = cfg.planner.type != "none"
    checks: list[Check] = [
        _check_path(resolver, "paths.bc_checkpoint", cfg.bc.checkpoint,
                    required=True, expect=_CHECKPOINT_FILES),
        _check_path(resolver, "paths.q_checkpoint", cfg.qfunction.checkpoint,
                    required=needs_q, expect=_CHECKPOINT_FILES),
        # FastWAM loads its VAE and text encoder from here; it is not inside the checkpoint.
        _check_path(resolver, "paths.wan22_weights", cfg.paths.wan22_weights, required=True),
        _check_path(resolver, "paths.root", cfg.paths.root, required=False, output=True),
    ]
    if cfg.benchmark.env_type == "robotwin":
        checks.append(
            _check_path(resolver, "paths.robotwin_root", cfg.paths.robotwin_root,
                        required=True, expect=("envs", "assets"))
        )
    if cfg.paths.dataset_root:
        checks.append(_check_path(resolver, "paths.dataset_root", cfg.paths.dataset_root,
                                  required=False))
    return checks


def check_dataset_identity(cfg: RunConfig) -> Check | None:
    """Guard against the truncated RoboTwin copy that is structurally identical to the real one.

    Two RoboTwin datasets exist in the wild with the same layout and the same camera keys;
    one has 27,500 episodes and one has 300. Training on the small one silently produces a
    much worse Q-function, and nothing downstream notices. ``total_episodes`` separates them.
    """
    if cfg.benchmark.env_type != "robotwin" or not cfg.paths.dataset_root:
        return None
    import json

    resolved = PathResolver().resolve(cfg.paths.dataset_root)
    if resolved.kind != "local" or resolved.path is None:
        return None
    info = resolved.path / "meta" / "info.json"
    if not info.exists():
        return Check("dataset.total_episodes", str(info), _MISSING, "no meta/info.json", False)
    total = json.loads(info.read_text()).get("total_episodes")
    if total is not None and total < 1000:
        return Check("dataset.total_episodes", str(total), _MISSING,
                     "this looks like the small RoboTwin subset, not the full dataset", False)
    return Check("dataset.total_episodes", str(total), _OK, "full dataset", False)


def check_lerobot_pin() -> Check:
    """Report which LeRobot the process will actually import."""
    import qplanning

    try:
        import lerobot
    except ImportError as exc:
        return Check("lerobot", "", _MISSING, f"not importable ({exc})", True)
    location = Path(lerobot.__file__).resolve()
    return Check("lerobot", getattr(lerobot, "__version__", "?"), _OK,
                 f"{location} (pinned to {qplanning.LEROBOT_PIN})", True)


def _render(checks: Sequence[Check]) -> str:
    width = max((len(c.field) for c in checks), default=10)
    lines = []
    for check in checks:
        marker = {
            _OK: "  ok  ",
            _HUB: " hub  ",
            _PENDING: " new  ",
            _MISSING: " FAIL " if check.required else " warn ",
            _UNSET: " unset" if not check.required else " FAIL ",
        }[check.status]
        lines.append(f"[{marker}] {check.field:<{width}}  {check.detail or check.raw}")
    return "\n".join(lines)


def main(config_path: str | None = None, overrides: Sequence[str] | None = None,
         show_help: bool = False) -> int:
    if show_help or not config_path:
        print(
            "usage: qplanning doctor --config <config.yaml> [--field.name=value ...]\n\n"
            "Resolves every path the config needs and reports what is missing. Run this "
            "before submitting anything long."
        )
        return 0 if show_help else 2

    load_dotenv()
    try:
        cfg = load(config_path, overrides)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 1

    checks = [check_lerobot_pin(), *collect_checks(cfg)]
    identity = check_dataset_identity(cfg)
    if identity is not None:
        checks.append(identity)

    print(f"config    : {config_path}")
    print(f"run_name  : {cfg.run_name}")
    print(f"benchmark : {cfg.benchmark.name} ({cfg.benchmark.env_type}), tasks={cfg.benchmark.tasks}")
    print(f"planner   : {cfg.planner.type}"
          + (f"  N={cfg.planner.n_samples} K={cfg.planner.n_elites} "
             f"lambda={cfg.planner.temperature} denoise={cfg.planner.denoise_steps}"
             if cfg.planner.type != "none" else "  (unguided BC baseline)"))
    print(f"output    : {cfg.paths.output_dir}\n")
    print(_render(checks))

    failures = [c for c in checks if c.failed]
    if failures:
        print(
            f"\n{len(failures)} required path(s) unresolved: "
            + ", ".join(c.field for c in failures)
            + "\nFill them in your .env (copy .env.example) or pass them on the CLI."
        )
        return 1
    print("\nall required paths resolved.")
    return 0


def validate_main(config_path: str | None = None, overrides: Sequence[str] | None = None,
                  show_help: bool = False) -> int:
    """Parse configs without resolving paths or importing torch."""
    if show_help:
        print("usage: qplanning validate-config [--config <config.yaml>]\n\n"
              "Parses every shipped config (or just one) and reports syntax or type errors.")
        return 0

    import glob

    from qplanning.config.paths import repository_root

    if config_path:
        targets = [config_path]
    else:
        root = repository_root()
        targets = sorted(
            p for p in glob.glob(str(root / "configs" / "**" / "*.yaml"), recursive=True)
            if not p.endswith("base.yaml")
        )

    failed = 0
    for target in targets:
        try:
            kind, label = _validate_one(target, overrides)
        except Exception as exc:  # noqa: BLE001 - report every file, do not abort the sweep
            failed += 1
            print(f"[ FAIL ] {target}\n         {type(exc).__name__}: {exc}")
        else:
            print(f"[  ok  ] {target}  ({kind}: {label})")
    print(f"\n{len(targets) - failed}/{len(targets)} configs parsed.")
    return 1 if failed else 0


#: Top-level keys that identify a YAML as something other than a run config.
_SPEC_KINDS = {"benchmarks": "table spec", "checkpoints": "checkpoint registry"}


def _validate_one(target: str, overrides: Sequence[str] | None) -> tuple[str, str]:
    """Parse one config, dispatching on what kind of file it is.

    Not every YAML under ``configs/`` describes a run: the table spec and the checkpoint
    registry have their own shapes and are validated as mappings.
    """
    from qplanning.config.loader import load_mapping

    mapping = load_mapping(target)
    for key, kind in _SPEC_KINDS.items():
        if key in mapping:
            entries = mapping[key]
            return kind, f"{len(entries)} entries"
    cfg = load(target, overrides)
    return "run config", cfg.run_name
