"""``qplanning probe`` -- exercise the whole model stack without a simulator.

Loads the behaviour-cloning policy and the Q-function, synthesises one observation with the
shapes their configs declare, and runs a full planning step. That covers checkpoint loading,
the Wan2.2 base weights, the horizon agreement between the two models, candidate sampling,
the encoder amortisation, scoring and aggregation -- everything except the environment.

Worth having as its own command because the simulators are the hardest part of the install
and the slowest thing to fail. If this passes, a later failure is the environment's; if it
fails, there is no point debugging inside a simulator.

It also reports the spread of Q-values across candidates. A spread of ~0 means the
Q-function is returning a constant, the weighted average degenerates to a plain mean, and
value guidance is silently doing nothing -- a failure that a success rate would hide.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from qplanning.config.loader import ConfigError, load, load_dotenv

if TYPE_CHECKING:
    from qplanning.config.schema import RunConfig

log = logging.getLogger("qplanning.probe")


def synthetic_observation(cfg: "RunConfig", policy_config: Any, device: str) -> dict[str, Any]:
    """One batch of plausible observations, shaped as the checkpoints expect."""
    import torch
    from lerobot.configs.types import FeatureType

    batch: dict[str, Any] = {}
    for key, feature in policy_config.input_features.items():
        if feature.type is FeatureType.VISUAL:
            batch[key] = torch.rand(1, *feature.shape, device=device)
        else:
            batch[key] = torch.zeros(1, *feature.shape, device=device)

    # The Q-function reads its own cameras, which for RoboTwin are the raw views rather
    # than the concatenated frame the policy consumes.
    for key in cfg.qfunction.camera_keys:
        if key not in batch:
            reference = next(
                (v for k, v in batch.items() if k.startswith("observation.images.")), None
            )
            shape = tuple(reference.shape[1:]) if reference is not None else (3, 224, 224)
            batch[key] = torch.rand(1, *shape, device=device)

    batch["task"] = ["pick up the object and place it in the basket"]
    return batch


def main(config_path: str | None = None, overrides: Sequence[str] | None = None,
         show_help: bool = False) -> int:
    if show_help or not config_path:
        print(
            "usage: qplanning probe --config <config.yaml> [--field.name=value ...]\n\n"
            "Loads the policy and the Q-function and runs one planning step on a synthetic\n"
            "observation. No simulator required -- use this to check an install before\n"
            "submitting anything long.\n\n"
            "Example:\n"
            "  qplanning probe --config configs/libero_10/qplanning_offline.yaml\n"
        )
        return 0 if show_help else 2

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    load_dotenv()
    try:
        cfg = load(config_path, overrides)
    except ConfigError as exc:
        print(f"config error: {exc}")
        return 1

    from qplanning.eval import build_policy, prepare

    prepare(cfg)

    import torch

    log.info("device: %s", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")

    # The policy factory derives its input/output features from either a dataset or an
    # environment config. Building the config is cheap and starts no simulator -- it is a
    # dataclass -- so the probe can stay simulator-free while still describing the right
    # observation and action spaces.
    from qplanning.envs import build_env_config, resolve_tasks

    env_cfg = build_env_config(cfg, resolve_tasks(cfg)[0])

    started = time.time()
    policy = build_policy(cfg, env_cfg=env_cfg)
    log.info("loaded policy and Q-function in %.0fs", time.time() - started)
    log.info("  chunk size H         %d", policy.chunk_size)
    log.info("  executed per plan    %d steps", policy.n_action_steps)
    log.info("  planner              %r", policy.planner)

    batch = synthetic_observation(cfg, policy.bc.policy_config, cfg.bc.device)
    log.info("  observation keys     %s", ", ".join(sorted(k for k in batch if k != "task")))

    policy.reset()
    action = policy.select_action(batch)

    # Time the planning step itself, not the first call. The first plan pays for CUDA
    # context setup and kernel autotuning, which is several times the steady-state cost and
    # would misrepresent the replanning budget. Time `plan` directly rather than
    # `select_action`, since most calls to the latter just pop an already-planned action.
    timings: list[float] = []
    for _ in range(3):
        step_started = time.time()
        policy.planner.plan(policy.bc.sampler, batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings.append(time.time() - step_started)
    median = sorted(timings)[len(timings) // 2]

    plan = policy.last_plan
    log.info("")
    log.info("planning step: %.0f ms median of %d, warm (%d candidate draws)",
             1000 * median, len(timings),
             cfg.planner.n_samples if cfg.planner.type != "none" else 1)
    log.info("action shape : %s", tuple(action.shape))

    if plan is not None and plan.spread is not None:
        s = plan.spread
        log.info("Q over %d candidates: min=%.4f max=%.4f mean=%.4f std=%.4f",
                 cfg.planner.n_samples, s.min, s.max, s.mean, s.std)
        if s.is_degenerate:
            log.error(
                "the Q-values are effectively constant (std=%.2e). Value guidance is a "
                "no-op in this state: every candidate scores the same, so the weighted "
                "average is just their mean. Check that the Q-function matches this "
                "benchmark's cameras and action space.", s.std
            )
            return 1
        log.info("")
        log.info("PASS: the planner discriminates between candidates.")
    else:
        log.info("")
        log.info("PASS: the policy produced an action (no Q-function in this config).")
    return 0
