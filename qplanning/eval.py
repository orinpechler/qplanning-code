"""Evaluate a policy, with or without Q-guided action selection.

Produces one row per episode, with episode length counted in environment steps.

The rollout itself is LeRobot's, called unmodified; what this module adds is the bookkeeping
around it.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
from qplanning.utils.records import EpisodeRecord, write_episodes

if TYPE_CHECKING:
    from qplanning.config.schema import RunConfig
    from qplanning.policy import QPlanningPolicy

log = logging.getLogger("qplanning.eval")


def prepare(cfg: "RunConfig") -> None:
    """Export environment variables and set determinism, before LeRobot is touched."""
    from qplanning._env_setup import prepare_environment

    prepare_environment(
        wan22_weights=cfg.paths.wan22_weights or None,
        hf_home=cfg.paths.hf_home or None,
        robotwin_root=cfg.paths.robotwin_root or None,
        strict=False,
    )
    from qplanning.utils import determinism

    determinism.apply(deterministic=cfg.eval.deterministic, seed=cfg.seed)


def build_policy(cfg: "RunConfig", env_cfg: Any) -> "QPlanningPolicy":
    """Assemble the BC policy, the planner, and the queue that drives them."""
    from qplanning.planner import build_planner
    from qplanning.policies.registry import build_bc
    from qplanning.policy import QPlanningPolicy

    bc = build_bc(cfg.bc, env_cfg=env_cfg)
    planner = build_planner(
        cfg=cfg.planner,
        bc=bc,
        q_checkpoint=cfg.qfunction.checkpoint,
        device=cfg.bc.device,
        seed=cfg.seed,
        baseline_denoise_steps=cfg.bc.baseline_denoise_steps,
    )
    if cfg.planner.type != "none":
        from qplanning.configuration_qfunction import assert_matches_checkpoint

        assert_matches_checkpoint(cfg.qfunction, planner.ctx.q_policy.config)

    return QPlanningPolicy(bc=bc, planner=planner, n_action_steps=cfg.execution.n_action_steps)


def run_episode(
    env: Any,
    policy: "QPlanningPolicy",
    processors: dict[str, Any],
    seed: int,
) -> tuple[bool, int, float, float | None]:
    """Run one episode and return ``(success, env_steps, sum_reward, final_q)``."""
    import torch
    from lerobot.scripts.lerobot_eval import rollout

    # Same seed as the environment, so every run sees the same noise at each planning step.
    policy.set_episode_seed(seed)
    policy.reset()

    outcome = rollout(
        env,
        policy,
        env_preprocessor=processors["env_pre"],
        env_postprocessor=processors["env_post"],
        preprocessor=processors["pre"],
        postprocessor=processors["post"],
        seeds=[seed],
    )

    # `done` is cumulative, so the first True marks the final step of the episode.
    done = outcome["done"][0]
    if bool(done.any()):
        length = int(torch.argmax(done.to(torch.int)).item()) + 1
    else:
        length = int(done.shape[0])

    success = bool(outcome["success"][0, :length].any())
    sum_reward = float(outcome["reward"][0, :length].sum())

    # `rollout` runs the whole episode internally, so only the final plan is still
    # reachable. Report that one honestly rather than presenting it as an episode mean.
    plan = policy.last_plan
    final_q = plan.spread.mean if plan is not None and plan.spread is not None else None
    return success, length, sum_reward, final_q


def evaluate(cfg: "RunConfig") -> list[EpisodeRecord]:
    """Evaluate every task in this run's (possibly sharded) task list."""
    from lerobot.envs.factory import make_env_pre_post_processors

    tasks = apply_shard(resolve_tasks(cfg), cfg.eval.shard)
    log.info("evaluating %d task group(s): %s", len(tasks), ", ".join(tasks))
    is_libero = cfg.benchmark.env_type == "libero"
    if not is_libero and cfg.eval.init_state_ids is not None:
        raise ConfigError("eval.init_state_ids applies to LIBERO only; RoboTwin varies by seed")

    records: list[EpisodeRecord] = []
    policy: "QPlanningPolicy | None" = None
    processors: dict[str, Any] = {}

    for task in tasks:
        with open_task_envs(cfg, task) as (env_cfg, per_task_envs):
            # The policy is expensive to load (tens of GB) and does not depend on the task,
            # so it is built once and reused across every environment in the run.
            if policy is None:
                policy = build_policy(cfg, env_cfg)
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
                        # One task failing must not lose the results of the others; three
                        # RoboTwin tasks are known to raise under any learned policy.
                        log.exception("task %s episode %d failed; skipping the rest of this task",
                                      label, episode)
                        break
                    records.append(
                        EpisodeRecord(
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
                        )
                    )
                    log.info("%-38s ep %3d  state %-4s  %-7s  %4d steps  %.1fs",
                             label, episode, "-" if states[episode] is None else states[episode],
                             "success" if success else "failure", length,
                             records[-1].wall_s or 0.0)
    return records


def summarise(records: Sequence[EpisodeRecord]) -> dict[str, Any]:
    """Per-task and overall figures."""
    from qplanning.report import agg, per_task

    rows = [
        {"task_name": r.task_name, "success": r.success, "episode_length": r.episode_length}
        for r in records
    ]
    task_map = per_task(rows)
    success, length = agg(task_map)
    return {
        "n_episodes": len(records),
        "n_tasks": len(task_map),
        "success_rate": success,
        "mean_successful_episode_length": length,
        "per_task": task_map,
    }


def main(config_path: str | None = None, overrides: Sequence[str] | None = None,
         show_help: bool = False) -> int:
    if show_help or not config_path:
        print(
            "usage: qplanning eval --config <config.yaml> [--field.name=value ...]\n\n"
            "Rolls out a BC policy, optionally under Q-guided selection, and writes one row\n"
            "per episode to <output_dir>/episodes.csv.\n\n"
            "Examples:\n"
            "  qplanning eval --config configs/libero_10/baseline.yaml\n"
            "  qplanning eval --config configs/libero_10/qplanning_offline.yaml \\\n"
            "      --eval.episodes_per_task=2 --planner.n_samples=8\n"
        )
        return 0 if show_help else 2

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    load_dotenv()
    try:
        cfg = load(config_path, overrides)
    except ConfigError as exc:
        print(f"config error: {exc}")
        return 1

    prepare(cfg)

    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dump(cfg, output_dir / "resolved.yaml")

    started = time.time()
    records = evaluate(cfg)
    if not records:
        log.error("no episodes completed; nothing written")
        return 1

    suffix = f".shard{cfg.eval.shard.replace('/', 'of')}" if cfg.eval.shard else ""
    csv_path = write_episodes(records, output_dir / f"episodes{suffix}.csv")

    summary = summarise(records)
    summary["elapsed_s"] = round(time.time() - started, 1)
    summary["config"] = config_path
    (output_dir / f"summary{suffix}.json").write_text(json.dumps(summary, indent=2))

    log.info("")
    log.info("%d episodes over %d tasks", summary["n_episodes"], summary["n_tasks"])
    log.info("success rate                    %.1f%%", 100 * summary["success_rate"])
    log.info("mean successful episode length  %.0f steps", summary["mean_successful_episode_length"])
    log.info("wrote %s", csv_path)
    return 0
