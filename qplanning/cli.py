"""The ``qplanning`` command-line entry point.

One executable, one subcommand per stage:

=====================  ===================================================================
``doctor``             Resolve every path a config needs and report what is missing.
``probe``              Load the models and run one planning step, with no simulator.
``fetch-checkpoints``  Download the released checkpoints.
``train-q``            Train the Q-function on demonstrations.
``eval``               Roll out a policy, optionally under Q-guided selection.
``self-improve``       The Q-only self-improvement loop.
``report``             Assemble per-episode CSVs into a results table.
``validate-config``    Parse configs without loading anything (no GPU required).
=====================  ===================================================================

Subcommand modules are imported lazily so that ``qplanning doctor`` -- the command a new
user runs first, and the one most likely to run in a half-configured environment -- does
not pay for importing torch.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import qplanning  # noqa: F401  -- prepares the process environment before LeRobot loads

# subcommand -> (module, function, help text)
_COMMANDS: dict[str, tuple[str, str, str]] = {
    "doctor": ("qplanning.doctor", "main", "check that every configured path resolves"),
    "fetch-checkpoints": (
        "qplanning.fetch",
        "main",
        "download the released checkpoints",
    ),
    "train-q": ("qplanning.train_q", "main", "train the Q-function on demonstrations"),
    "eval": ("qplanning.eval", "main", "evaluate a BC policy, with or without Q-Planning"),
    "self-improve": ("qplanning.self_improve", "main", "run the Q-only self-improvement loop"),
    "report": ("qplanning.report", "main", "assemble per-episode CSVs into a results table"),
    "probe": ("qplanning.probe", "main", "run one planning step without a simulator"),
    "validate-config": (
        "qplanning.doctor",
        "validate_main",
        "parse configs without loading models (no GPU needed)",
    ),
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qplanning",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Any config field can be overridden inline, e.g.\n"
            "  qplanning eval --config configs/libero_10/baseline.yaml "
            "--eval.episodes_per_task=2 --planner.n_samples=8\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"qplanning {qplanning.__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    for name, (_, _, help_text) in _COMMANDS.items():
        sub = subparsers.add_parser(name, help=help_text, add_help=False)
        sub.add_argument(
            "-h",
            "--help",
            action="store_true",
            dest="show_help",
            help="show help for this command",
        )
        sub.add_argument(
            "--config",
            metavar="PATH",
            help="path to a run config YAML (see configs/)",
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch to a subcommand. Unrecognised ``--a.b=c`` flags are passed through."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()

    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    if argv[0] == "--version":
        parser.parse_args(argv)
        return 0

    command = argv[0]
    if command not in _COMMANDS:
        parser.print_usage(sys.stderr)
        print(f"qplanning: unknown command {command!r}. Choose from: "
              f"{', '.join(_COMMANDS)}", file=sys.stderr)
        return 2

    known, overrides = parser.parse_known_args(argv)
    module_name, function_name, _ = _COMMANDS[command]

    import importlib

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:  # pragma: no cover - only during development
        print(f"qplanning: {command!r} is unavailable ({exc}).", file=sys.stderr)
        return 3

    entry = getattr(module, function_name)
    return int(entry(config_path=known.config, overrides=overrides, show_help=known.show_help) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
