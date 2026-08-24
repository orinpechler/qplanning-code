"""Self-improvement: deploy, collect, refine the Q-function, repeat.

Each iteration rolls out the current planner, appends every episode -- successes and
failures alike -- to a growing replay buffer, and takes S gradient steps on the Q-function.
The behaviour-cloning policy is never updated. That is what makes an iteration cheap: the
gradient step touches a ~1B-parameter critic rather than a multi-billion-parameter policy.

Measurement is folded into the same rollout pass. Every task runs a *held-out* block whose
seeds are fixed across iterations and which is never written to the buffer, followed by a
*training* block whose seeds advance and which is. Training on the held-out seeds would let
the loop fit the very states its score is computed on, so "iteration N improved" would be
partly memorisation with no way to separate it out. The held-out numbers at iteration 0 and
at the final iteration are the before-and-after measurements of the loop.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qplanning.config.loader import ConfigError, dump, load, load_dotenv
from qplanning.data.online_buffer import (
    EPISODES_DIRNAME,
    OnlineQDataset,
    iteration_dir,
    save_iteration,
    validate_shards,
)
from qplanning.envs import apply_shard, open_task_envs, resolve_tasks, task_label
from qplanning.utils.records import EpisodeRecord, write_episodes

if TYPE_CHECKING:
    import torch

    from qplanning.config.schema import RunConfig

log = logging.getLogger("qplanning.self_improve")

#: Training seeds start this far past the held-out block so the two can never collide,
#: however many tasks or iterations a run has.
TRAIN_SEED_STRIDE = 1_000_000


# ---------------------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------------------
def run_episode(env: Any, policy: Any, processors: dict[str, Any], seed: int) -> dict:
    """Roll out one episode, keeping the frames needed to train the Q-function on it."""
    import numpy as np
    import torch
    from lerobot.envs.utils import add_envs_task, preprocess_observation
    from lerobot.utils.constants import ACTION

    policy.reset()
    observation, _ = env.reset(seed=[seed])

    frames: list[dict[str, "torch.Tensor"]] = []
    actions: list["torch.Tensor"] = []
    successes: list[bool] = []
    done = np.array([False])
    step = 0
    max_steps = env.call("_max_episode_steps")[0]

    while not np.all(done) and step < max_steps:
        obs = preprocess_observation(observation)
        # Kept before the environment processors run, so what is stored is the camera image
        # itself rather than a policy-specific view of it.
        frames.append({
            k: deepcopy(v) for k, v in obs.items()
            if k.startswith("observation.images.") and isinstance(v, torch.Tensor)
        })

        obs = processors["pre"](processors["env_pre"](add_envs_task(env, obs)))
        with torch.no_grad():
            action = policy.select_action(obs)
        action = processors["env_post"]({ACTION: processors["post"](action)})[ACTION]
        action_np = action.cpu().numpy()

        observation, _reward, terminated, truncated, info = env.step(action_np)
        step_success = info["final_info"]["is_success"].tolist() if "final_info" in info else [False]

        done = terminated | truncated | done
        actions.append(torch.from_numpy(action_np))
        successes.append(bool(step_success[0]))
        step += 1

    episode: dict[str, Any] = {
        "action": torch.cat(actions, dim=0),
        "success": any(successes),
        "next.success": torch.tensor(successes, dtype=torch.bool),
        "task": _task_description(env),
    }
    for key in {k for frame in frames for k in frame}:
        stacked = [frame[key][0] for frame in frames if key in frame]
        if len(stacked) == len(actions):
            episode[key] = torch.stack(stacked, dim=0)
    return episode


def _task_description(env: Any) -> str:
    inner = getattr(env, "envs", None)
    if not inner:
        return ""
    for attribute in ("task_description", "task"):
        value = getattr(inner[0], attribute, None)
        if isinstance(value, str):
            return value
    return ""


def collect(cfg: "RunConfig", iteration: int, policy: Any, processors: dict[str, Any],
            camera_key_map: dict[str, str]) -> tuple[list[dict], list[EpisodeRecord], list[str]]:
    """Roll out the held-out and training blocks for this job's tasks."""
    si = cfg.self_improve
    all_tasks = resolve_tasks(cfg)
    shard = cfg.eval.shard
    my_tasks = apply_shard(all_tasks, shard)

    # Seeds are derived from a task's index in the FULL list, so a sharded run reproduces
    # exactly the episodes an unsharded one would have run.
    index_of = {task: i for i, task in enumerate(all_tasks)}
    train_seed_base = cfg.seed + iteration * max(1, si.train_episodes_per_task) * len(all_tasks)

    saved: list[dict] = []
    heldout: list[EpisodeRecord] = []
    failed: list[str] = []

    for task in my_tasks:
        task_index = index_of[task]
        try:
            with open_task_envs(cfg, task) as (_env_cfg, per_task_envs):
                for task_id, env in sorted(per_task_envs.items()):
                    label = task_label(cfg, task, task_id, env)
                    offset = task_index if cfg.benchmark.env_type == "robotwin" else task_id

                    for i in range(si.heldout_episodes_per_task):
                        seed = si.heldout_seed_base + offset * si.heldout_episodes_per_task + i
                        episode = run_episode(env, policy, processors, seed)
                        heldout.append(EpisodeRecord(
                            task_id=task_id, task_name=label, episode=i,
                            success=bool(episode["success"]),
                            episode_length=int(episode["action"].shape[0]), seed=seed,
                        ))

                    for j in range(si.train_episodes_per_task):
                        seed = (train_seed_base + TRAIN_SEED_STRIDE
                                + offset * si.train_episodes_per_task + j)
                        episode = run_episode(env, policy, processors, seed)
                        for src, dst in camera_key_map.items():
                            if src in episode:
                                episode[dst] = episode.pop(src)
                        saved.append(episode)

                    n_ok = sum(r.success for r in heldout if r.task_name == label)
                    log.info("  %-38s held-out %d/%d", label, n_ok, si.heldout_episodes_per_task)
        except Exception as exc:  # noqa: BLE001
            # One task must not cost the whole iteration; three RoboTwin tasks raise from
            # their own success check under any learned policy.
            failed.append(task)
            log.warning("  %-38s SKIPPED (%s: %s)", task, type(exc).__name__, exc)

    if failed:
        log.warning("%d task(s) skipped: %s", len(failed), ", ".join(failed))
    if not saved:
        raise RuntimeError("collected no training episodes; every task errored")
    return saved, heldout, failed


# ---------------------------------------------------------------------------------------
# Q-only refinement
# ---------------------------------------------------------------------------------------
def finetune(cfg: "RunConfig", q_policy: Any, output_dir: Path, checkpoint_dir: Path,
             source_checkpoint: str, global_step: int = 0, iteration: int | None = None
             ) -> tuple[float, int]:
    """Take S gradient steps on the Q-function over demonstrations + online rollouts."""
    import shutil

    import torch
    from torch.optim import AdamW

    from qplanning.data.mixture import build_mixture_loader, load_demonstrations
    from qplanning.processor_qfunction import make_q_train_preprocessor

    si = cfg.self_improve
    good, quarantined = validate_shards(output_dir, iteration)
    if quarantined:
        log.warning("%d shard(s) quarantined as unreadable: %s", len(quarantined),
                    ", ".join(p.parent.name for p in quarantined))
    minimum = si.resolved_min_shards()
    if len(good) < minimum:
        raise RuntimeError(
            f"only {len(good)} readable shard(s) in {output_dir}, need at least {minimum}. "
            "Re-run the collection stage for the missing shards rather than training on a "
            "partial iteration."
        )

    online = OnlineQDataset(
        output_dir=output_dir,
        h=int(q_policy.config.h),
        terminal_bonus=1.0,
        camera_keys=tuple(q_policy.config.camera_keys),
    )
    if quarantined:
        # OnlineQDataset globs the buffer itself, so a torn shard would be loaded anyway
        # and fail mid-epoch. Refuse instead: the fix is to re-collect that shard.
        raise RuntimeError(
            f"{len(quarantined)} shard(s) are unreadable and would fail partway through "
            "training: " + ", ".join(p.parent.name for p in quarantined)
            + ". Delete them and re-run the collect stage for those shards."
        )
    offline = load_demonstrations(si.offline.repo_ids, si.offline.root or "", q_policy)
    log.info("buffer: %d online frames over %d shard(s); %d demonstration frames",
             len(online), len(good), len(offline))

    loader = build_mixture_loader(
        offline, online,
        steps=si.finetune_steps, batch_size=si.batch_size, online_fraction=si.online_fraction,
    )
    preprocess = make_q_train_preprocessor(source_checkpoint)

    # The encoders were trained at a lower rate than the head; preserving that ratio at the
    # fine-tuning rate keeps their relative speed unchanged. Setting a single learning rate
    # instead leaves the backbone on its original (much larger) value and lets it move
    # several times faster than the head it is supposed to support.
    q_cfg = q_policy.config
    backbone_ratio = q_cfg.optimizer_lr_backbone / q_cfg.optimizer_lr
    groups = q_policy.get_optim_params()
    groups[0]["lr"] = si.finetune_lr
    groups[1]["lr"] = si.finetune_lr * backbone_ratio
    optimizer = AdamW(groups, weight_decay=1e-4)

    q_policy.train()
    losses: list[float] = []
    iterator = iter(loader)
    for step in range(si.finetune_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        optimizer.zero_grad()
        loss, loss_dict = q_policy.forward(preprocess(batch))
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(q_policy.parameters(), si.grad_clip_norm)
        optimizer.step()
        q_policy.update()          # EMA update of the target parameters
        losses.append(float(loss_dict["td_ce_loss"]))
        global_step += 1

        if (step + 1) % 10 == 0:
            log.info("  step %4d/%d  td_ce=%.4f  grad_norm=%.3f",
                     step + 1, si.finetune_steps, sum(losses[-10:]) / 10, float(grad_norm))

    q_policy.eval()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    q_policy.save_pretrained(str(checkpoint_dir))
    # Copy the processor files across so the result is a self-contained checkpoint the next
    # iteration (or a later evaluation) can load on its own.
    for pattern in ("policy_pre*", "policy_post*"):
        for path in Path(source_checkpoint).glob(pattern):
            shutil.copy2(path, checkpoint_dir / path.name)
    log.info("  saved Q checkpoint to %s", checkpoint_dir)

    return (sum(losses) / len(losses) if losses else float("nan")), global_step


# ---------------------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------------------
def _camera_key_map(cfg: "RunConfig") -> dict[str, str]:
    """Environment camera names -> the keys the Q-function indexes by."""
    if cfg.benchmark.env_type != "robotwin":
        return {}
    return {
        "observation.images.head_camera": "observation.images.cam_high",
        "observation.images.left_camera": "observation.images.cam_left_wrist",
        "observation.images.right_camera": "observation.images.cam_right_wrist",
    }


def _offline_image_shape(root: str | None) -> tuple[int, int, int] | None:
    """The demonstration dataset's image shape, so online frames are stored to match."""
    if not root:
        return None
    info = Path(root) / "meta" / "info.json"
    if not info.exists():
        return None
    features = json.loads(info.read_text()).get("features", {})
    for key, spec in features.items():
        if key.startswith("observation.images.") and "shape" in spec:
            return tuple(spec["shape"])
    return None


def run(cfg: "RunConfig", stage: str = "all") -> int:
    """Execute the requested stage(s) of the loop."""
    from qplanning.eval import build_policy, prepare

    prepare(cfg)
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dump(cfg, output_dir / "resolved.yaml")

    si = cfg.self_improve
    source_checkpoint = str(cfg.qfunction.checkpoint)
    global_step = si.start_iteration * si.finetune_steps
    policy = None
    processors: dict[str, Any] = {}

    # Only the combined stage iterates. A `collect` or `finetune` job is one step of a
    # loop driven from outside, and looping here would silently repeat that step -- for
    # `finetune`, re-training on an unchanged buffer while logging it as new iterations.
    iterations = si.iterations if stage == "all" else 1
    for offset in range(iterations):
        iteration = si.start_iteration + offset
        started = time.time()
        log.info("")
        log.info("═══ iteration %d ═══", iteration)

        shard = _shard_index(cfg.eval.shard)
        iter_out = iteration_dir(output_dir, iteration, shard)
        metrics: dict[str, Any] = {"iteration": iteration}

        # ── collect ────────────────────────────────────────────────────────────────────
        if stage in ("collect", "all"):
            if policy is None:
                from lerobot.envs.factory import make_env_pre_post_processors

                from qplanning.envs import build_env_config

                env_cfg = build_env_config(cfg, resolve_tasks(cfg)[0])
                policy = build_policy(cfg, env_cfg)
                env_pre, env_post = make_env_pre_post_processors(
                    env_cfg=env_cfg, policy_cfg=policy.bc.policy_config
                )
                processors = {"env_pre": env_pre, "env_post": env_post,
                              "pre": policy.bc.preprocessor, "post": policy.bc.postprocessor}

            episodes, heldout, failed = collect(cfg, iteration, policy, processors,
                                                _camera_key_map(cfg))
            iter_out.mkdir(parents=True, exist_ok=True)
            write_episodes(heldout, iter_out / "heldout_episodes.csv")
            save_iteration(
                episodes, iter_out / EPISODES_DIRNAME,
                camera_keys=tuple(cfg.qfunction.camera_keys),
                image_shape=_offline_image_shape(si.offline.root),
                action_dim=int(episodes[0]["action"].shape[1]),
                manifest={
                    "iteration": iteration, "shard": cfg.eval.shard,
                    "q_checkpoint": source_checkpoint, "planner": cfg.planner.type,
                    "n_saved": len(episodes), "n_heldout": len(heldout),
                    "failed_tasks": failed,
                },
            )
            rate = 100 * sum(r.success for r in heldout) / max(len(heldout), 1)
            metrics.update({"heldout_success": rate, "n_heldout": len(heldout),
                            "n_collected": len(episodes), "failed_tasks": failed})
            log.info("held-out success: %.1f%% over %d episodes", rate, len(heldout))

            if stage == "collect":
                (iter_out / "metrics.json").write_text(json.dumps(metrics, indent=2))
                log.info("collection complete: %s", iter_out)
                return 0

        # ── refine ─────────────────────────────────────────────────────────────────────
        if stage in ("finetune", "all"):
            q_policy = _q_for_finetune(cfg, policy, source_checkpoint)
            if policy is not None:
                # The BC policy is idle during the update and is the larger of the two
                # models by far; parking it on the CPU is what lets the Q-function's
                # gradients, optimiser state and activations fit alongside it.
                policy.bc.sampler.to("cpu")
                _empty_cuda_cache()

            checkpoint_dir = iteration_dir(output_dir, iteration) / "q_checkpoint"
            loss, global_step = finetune(cfg, q_policy, output_dir, checkpoint_dir,
                                         source_checkpoint, global_step, iteration)
            metrics["finetune_loss"] = loss

            if policy is not None:
                _empty_cuda_cache()
                policy.bc.sampler.to(cfg.bc.device)
            # The next iteration plans with the checkpoint just written.
            source_checkpoint = str(checkpoint_dir)
            cfg.qfunction.checkpoint = source_checkpoint

        metrics["elapsed_s"] = round(time.time() - started, 1)
        target = iteration_dir(output_dir, iteration)
        target.mkdir(parents=True, exist_ok=True)
        (target / "metrics.json").write_text(json.dumps(metrics, indent=2))
        with (output_dir / "loop_summary.jsonl").open("a") as handle:
            handle.write(json.dumps(metrics) + "\n")

    return 0


def _q_for_finetune(cfg: "RunConfig", policy: Any, checkpoint: str) -> Any:
    """The Q-function to update: the planner's own, or freshly loaded when only refining.

    A finetune-only job deliberately does not build the BC policy or a simulator. Loading a
    multi-billion-parameter policy that will never be called costs minutes and tens of
    gigabytes, and on a single device it is the difference between the update fitting and
    not.
    """
    if policy is not None and hasattr(policy.planner, "ctx"):
        return policy.planner.ctx.q_policy
    from qplanning.modeling_qfunction import load_q_function

    log.info("finetune-only: loading the Q-function alone (no BC policy, no simulator)")
    q_policy, _ = load_q_function(checkpoint, device=cfg.bc.device, eval_mode=False)
    return q_policy


def _shard_index(shard: str | None) -> int | None:
    return int(shard.split("/")[0]) if shard else None


def _empty_cuda_cache() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _extract_stage(argv: list[str]) -> tuple[str, list[str]]:
    """Pull ``--stage`` out of the argument list, leaving the config overrides intact.

    Consumes both ``--stage collect`` and ``--stage=collect``, and removes exactly the
    tokens it consumed -- filtering by value would also delete an unrelated override that
    happened to equal the stage name.
    """
    stage = "all"
    remaining: list[str] = []
    skip_next = False
    for i, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if arg == "--stage":
            if i + 1 < len(argv):
                stage = argv[i + 1]
                skip_next = True
            continue
        if arg.startswith("--stage="):
            stage = arg.split("=", 1)[1]
            continue
        remaining.append(arg)
    return stage, remaining


def main(config_path: str | None = None, overrides: Sequence[str] | None = None,
         show_help: bool = False) -> int:
    if show_help or not config_path:
        print(
            "usage: qplanning self-improve --config <config.yaml> [--stage all|collect|finetune]\n\n"
            "Runs the Q-only self-improvement loop. Iteration 0's held-out block is the\n"
            "offline Q-Planning result; the final iteration's is the self-improved one.\n\n"
            "Stages let collection be fanned out across jobs:\n"
            "  --stage collect  --eval.shard=k/n   roll out this job's slice of the tasks\n"
            "  --stage finetune                    update Q on everything collected so far\n\n"
            "Example:\n"
            "  qplanning self-improve --config configs/libero_10/self_improve.yaml\n"
        )
        return 0 if show_help else 2

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    stage, config_overrides = _extract_stage(list(overrides or []))
    if stage not in ("all", "collect", "finetune"):
        print(f"unknown --stage {stage!r}; expected all, collect or finetune")
        return 2

    load_dotenv()
    try:
        cfg = load(config_path, config_overrides)
    except ConfigError as exc:
        print(f"config error: {exc}")
        return 1
    return run(cfg, stage=stage)
