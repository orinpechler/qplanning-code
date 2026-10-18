"""Evaluate FastWAM with its chunks refined by gradient ascent on the frozen Q-function.

    python -m qoptimizing.eval --config qoptimizing/configs/libero_10.yaml [--qopt.steps=20 ...]

Same rollout, seeding and initial-state protocol as ``qplanning eval``, so the resulting
``episodes.csv`` is directly comparable with the baseline's (``qopt.steps=0`` reproduces the
baseline). Adds per-episode columns for what the optimization did, and optionally one JSON
line per planning step with the full Q trajectory.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import qoptimizing  # noqa: F401  -- prepares the process environment before LeRobot loads
from qplanning.config.loader import ConfigError, dump, load, load_dotenv
from qplanning.envs import (
    apply_shard,
    init_state_schedule,
    num_init_states,
    open_task_envs,
    resolve_tasks,
    set_init_state,
    task_label,
)
from qplanning.utils.records import EpisodeRecord

if TYPE_CHECKING:
    from qoptimizing.config import QOptRunConfig
    from qoptimizing.optimizer import QGradientPlanner, QOptStats
    from qplanning.policy import QPlanningPolicy

log = logging.getLogger("qoptimizing.eval")


@dataclass
class QOptEpisodeRecord(EpisodeRecord):
    """An episode row, plus episode means over its planning steps."""

    n_plans: int = 0
    q_init_mean: float | None = None
    q_final_mean: float | None = None
    q_gain_mean: float | None = None
    delta_rms_mean: float | None = None
    grad_rms_init_mean: float | None = None


def build_policy(cfg: "QOptRunConfig", env_cfg: Any) -> "QPlanningPolicy":
    """FastWAM, the Q-gradient planner, and the usual action queue.

    The noise of every planning step is seeded per (episode seed, plan index) by
    :meth:`QPlanningPolicy.set_episode_seed`, so this run sees the baseline's noise draws
    and, with ``qopt.steps=0``, takes the baseline's actions.
    """
    from qoptimizing.optimizer import QGradientPlanner
    from qplanning.configuration_qfunction import assert_matches_checkpoint
    from qplanning.policies.registry import build_bc
    from qplanning.policy import QPlanningPolicy

    if not cfg.qfunction.checkpoint:
        raise ConfigError("Q-gradient optimization needs a Q-function: set paths.q_checkpoint")

    bc = build_bc(cfg.bc, env_cfg=env_cfg)
    denoise_steps = (cfg.qopt.denoise_steps if cfg.qopt.denoise_steps is not None
                     else cfg.bc.baseline_denoise_steps)
    planner = QGradientPlanner.build(
        cfg=cfg.qopt,
        q_checkpoint=cfg.qfunction.checkpoint,
        bc=bc,
        device=cfg.bc.device,
        denoise_steps=denoise_steps,
    )
    assert_matches_checkpoint(cfg.qfunction, planner.ctx.q_policy.config)
    return QPlanningPolicy(bc=bc, planner=planner, n_action_steps=cfg.execution.n_action_steps)


def _mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def episode_means(stats: Sequence["QOptStats"]) -> dict[str, Any]:
    return {
        "n_plans": len(stats),
        "q_init_mean": _mean([s.q_init for s in stats]),
        "q_final_mean": _mean([s.q_final for s in stats]),
        "q_gain_mean": _mean([s.q_gain for s in stats]),
        "delta_rms_mean": _mean([s.delta_rms for s in stats]),
        "grad_rms_init_mean": _mean([s.grad_rms_init for s in stats]),
    }


def write_records(records: Sequence[QOptEpisodeRecord], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = QOptEpisodeRecord.column_names()
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            writer.writerow({k: row.get(k) for k in columns})
    return path


def evaluate(cfg: "QOptRunConfig", steps_log: Path | None = None,
             build: Callable[[Any, Any], "QPlanningPolicy"] | None = None) -> list[QOptEpisodeRecord]:
    """Evaluate every task in this run's (possibly sharded) task list.

    ``build(cfg, env_cfg)`` makes the policy (default :func:`build_policy`); its planner must
    keep ``episode_stats``, a list of :class:`QOptStats`, like :class:`QGradientPlanner`.
    """
    from lerobot.envs.factory import make_env_pre_post_processors

    from qplanning.eval import run_episode

    tasks = apply_shard(resolve_tasks(cfg), cfg.eval.shard)
    log.info("evaluating %d task group(s): %s", len(tasks), ", ".join(tasks))
    is_libero = cfg.benchmark.env_type == "libero"
    if not is_libero and cfg.eval.init_state_ids is not None:
        raise ConfigError("eval.init_state_ids applies to LIBERO only; RoboTwin varies by seed")

    records: list[QOptEpisodeRecord] = []
    policy: "QPlanningPolicy | None" = None
    processors: dict[str, Any] = {}
    steps_handle = steps_log.open("w") if steps_log is not None else None

    try:
        for task in tasks:
            with open_task_envs(cfg, task) as (env_cfg, per_task_envs):
                if policy is None:
                    policy = (build or build_policy)(cfg, env_cfg)
                    env_pre, env_post = make_env_pre_post_processors(
                        env_cfg=env_cfg, policy_cfg=policy.bc.policy_config
                    )
                    processors = {
                        "env_pre": env_pre,
                        "env_post": env_post,
                        "pre": policy.bc.preprocessor,
                        "post": policy.bc.postprocessor,
                    }
                    log.info("policy ready: %r", policy)
                planner: "QGradientPlanner" = policy.planner  # type: ignore[assignment]

                for task_id, env in sorted(per_task_envs.items()):
                    label = task_label(cfg, task, task_id, env)
                    states = (init_state_schedule(cfg.eval.episodes_per_task, cfg.eval.init_state_ids,
                                                  num_init_states(env))
                              if is_libero else [None] * cfg.eval.episodes_per_task)
                    for episode in range(cfg.eval.episodes_per_task):
                        seed = cfg.eval.start_seed + task_id * cfg.eval.episodes_per_task + episode
                        if states[episode] is not None:
                            set_init_state(env, states[episode])
                        started = time.time()
                        try:
                            success, length, reward, final_q = run_episode(env, policy, processors, seed)
                        except Exception:
                            log.exception("task %s episode %d failed; skipping the rest of this task",
                                          label, episode)
                            break
                        stats = list(planner.episode_stats)
                        means = episode_means(stats)
                        records.append(
                            QOptEpisodeRecord(
                                task_id=task_id,
                                task_name=label,
                                episode=episode,
                                success=success,
                                episode_length=length,
                                seed=seed,
                                init_state=states[episode],
                                sum_reward=reward,
                                wall_s=round(time.time() - started, 2),
                                q_final=final_q,
                                **means,
                            )
                        )
                        if steps_handle is not None:
                            for plan_index, s in enumerate(stats):
                                steps_handle.write(json.dumps({
                                    "task_name": label, "episode": episode, "plan": plan_index,
                                    **asdict(s),
                                }) + "\n")
                            steps_handle.flush()
                        log.info("%-38s ep %3d  state %-4s  %-7s  %4d steps  %.1fs  "
                                 "Q %.4f -> %.4f (%+.4f)  |da| %.4f",
                                 label, episode,
                                 "-" if states[episode] is None else states[episode],
                                 "success" if success else "failure", length,
                                 records[-1].wall_s or 0.0,
                                 means["q_init_mean"] or 0.0, means["q_final_mean"] or 0.0,
                                 means["q_gain_mean"] or 0.0, means["delta_rms_mean"] or 0.0)
    finally:
        if steps_handle is not None:
            steps_handle.close()
    return records


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m qoptimizing.eval",
        description="Roll out FastWAM with Q-gradient-refined action chunks.",
        epilog="Any config field can be overridden inline, e.g. --qopt.steps=20 "
               "--qopt.step_size=0.02 --eval.episodes_per_task=2",
    )
    parser.add_argument("--config", required=True, help="run config YAML")
    known, overrides = parser.parse_known_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    load_dotenv()

    from qoptimizing.config import QOptRunConfig
    from qplanning.eval import prepare, summarise

    try:
        cfg = load(known.config, overrides, config_class=QOptRunConfig)
    except ConfigError as exc:
        print(f"config error: {exc}")
        return 1

    prepare(cfg)

    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dump(cfg, output_dir / "resolved.yaml")

    suffix = f".shard{cfg.eval.shard.replace('/', 'of')}" if cfg.eval.shard else ""
    steps_log = output_dir / f"qopt_steps{suffix}.jsonl" if cfg.qopt.log_steps else None

    started = time.time()
    records = evaluate(cfg, steps_log=steps_log)
    if not records:
        log.error("no episodes completed; nothing written")
        return 1

    csv_path = write_records(records, output_dir / f"episodes{suffix}.csv")

    summary = summarise(records)
    summary["elapsed_s"] = round(time.time() - started, 1)
    summary["config"] = known.config
    summary["qopt"] = asdict(cfg.qopt)
    for key in ("q_init_mean", "q_final_mean", "q_gain_mean", "delta_rms_mean"):
        summary[key] = _mean([getattr(r, key) for r in records if getattr(r, key) is not None])
    import torch

    if torch.cuda.is_available():
        summary["peak_gpu_mem_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 1)
        log.info("peak GPU memory reserved        %.1f GiB", summary["peak_gpu_mem_gib"])
    (output_dir / f"summary{suffix}.json").write_text(json.dumps(summary, indent=2))

    log.info("")
    log.info("%d episodes over %d tasks", summary["n_episodes"], summary["n_tasks"])
    log.info("success rate                    %.1f%%", 100 * summary["success_rate"])
    log.info("mean successful episode length  %.0f steps", summary["mean_successful_episode_length"])
    log.info("mean Q per plan                 %.4f -> %.4f (%+.4f)",
             summary["q_init_mean"] or 0.0, summary["q_final_mean"] or 0.0,
             summary["q_gain_mean"] or 0.0)
    log.info("wrote %s", csv_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
