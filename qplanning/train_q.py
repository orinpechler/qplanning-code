"""Train the Q-function on demonstrations.

The training itself is LeRobot's -- the fork already wires the Q-function's reward labelling
into its dataset factory -- so this module's job is to turn one YAML into the right command
and run it. Building the argv here rather than in a shell script is what makes the same
settings reachable from a laptop, a SLURM job, and a test.

The Q-function trains on the same demonstrations as the policy -- no extra data is
collected for it. What it can additionally absorb, later, are the failed deployment rollouts
that imitation has no way to use.
"""

from __future__ import annotations

import logging
import shlex
import socket
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from qplanning.config.loader import ConfigError, dump, load, load_dotenv

if TYPE_CHECKING:
    from qplanning.config.schema import RunConfig

log = logging.getLogger("qplanning.train_q")


def _free_port() -> int:
    """An unused TCP port for the distributed rendezvous."""
    with socket.socket() as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def build_argv(cfg: "RunConfig") -> list[str]:
    """The full command that trains the Q-function for this config."""
    from qplanning.config.paths import PathResolver

    train = cfg.train_q
    resolver = PathResolver()
    output_dir = Path(cfg.paths.output_dir)

    dataset_root = train.dataset.root or cfg.paths.dataset_root
    if not dataset_root:
        raise ConfigError(
            "train_q.dataset.root is not set; point it at the LeRobotDataset holding the "
            "demonstrations (see .env.example)."
        )
    if not train.dataset.repo_ids:
        raise ConfigError("train_q.dataset.repo_ids is empty; name the dataset(s) to train on.")

    q = cfg.qfunction
    if train.init_from and train.resume_from:
        raise ConfigError("train_q.init_from and train_q.resume_from are mutually exclusive.")
    if train.init_from:
        # Fine-tuning: architecture, reward and bins are read off the checkpoint itself.
        init = resolver.resolve(train.init_from).path or train.init_from
        policy_args = [f"--policy.path={init}"]
    else:
        policy_args = ["--policy.type=q_function", *_architecture_args(cfg)]

    args: list[str] = [
        *policy_args,
        "--policy.push_to_hub=false",
        f"--job_name={cfg.run_name}",
        f"--output_dir={output_dir}",
        *_ranking_args(train.ranking),
        # Optimisation.
        f"--policy.optimizer_lr={train.optimizer_lr}",
        f"--policy.optimizer_lr_backbone={train.optimizer_lr_backbone}",
        f"--policy.optimizer_weight_decay={train.weight_decay}",
        f"--policy.lr_scheduler={train.lr_scheduler}",
        f"--policy.lr_warmup_steps={train.lr_warmup_steps}",
        f"--policy.lr_decay_steps={train.lr_decay_steps}",
        f"--policy.lr_decay_min={train.lr_decay_min}",
        f"--policy.device={cfg.bc.device}",
        # Data.
        f"--dataset.repo_ids=[{','.join(train.dataset.repo_ids)}]",
        f"--dataset.root={resolver.resolve(dataset_root).path or dataset_root}",
        f"--dataset.use_imagenet_stats={str(train.dataset.use_imagenet_stats).lower()}",
        # Schedule.
        f"--batch_size={train.batch_size}",
        f"--steps={train.steps}",
        f"--save_freq={train.save_freq}",
        f"--log_freq={train.log_freq}",
        f"--num_workers={train.num_workers}",
        f"--seed={cfg.seed}",
        f"--cudnn_deterministic={str(train.deterministic).lower()}",
        # A held-out split of episodes, so the logged loss is not purely on training frames.
        f"--test_split_ratio={train.test_split_ratio}",
        f"--test_freq={train.test_freq}",
        f"--test_n_batches={train.test_n_batches}",
        # The Q-function is scored by planning, not by an environment rollout during training.
        "--eval_freq=0",
    ]

    if train.wandb.enabled:
        args += [
            "--wandb.enable=true",
            f"--wandb.project={train.wandb.project}",
            "--wandb.disable_artifact=true",
        ]
        if train.wandb.entity:
            args.append(f"--wandb.entity={train.wandb.entity}")
    else:
        args.append("--wandb.enable=false")

    if train.resume_from:
        # LeRobot resumes from a saved train_config.json, not from a weights path.
        resume = Path(str(resolver.resolve(train.resume_from).path or train.resume_from))
        args += [f"--config_path={resume / 'train_config.json'}", "--resume=true"]

    launcher = [sys.executable, "-m", "qplanning.lerobot_train"]
    if train.num_gpus > 1:
        launcher = [
            "accelerate", "launch",
            f"--num_processes={train.num_gpus}",
            "--multi_gpu",
            "--mixed_precision=bf16",
            f"--main_process_port={_free_port()}",
            "-m", "qplanning.lerobot_train",
        ]
    return [*launcher, *args]


def _ranking_args(ranking) -> list[str]:
    """The fork's ranking-margin loss settings; none at all when the loss is off."""
    if ranking.weight <= 0:
        return []
    return [
        f"--policy.neg_margin_weight={ranking.weight}",
        f"--policy.neg_margin_delta={ranking.delta}",
        f"--policy.neg_tube_sigmas=[{','.join(str(s) for s in ranking.tube_sigmas)}]",
        f"--policy.neg_tube_smooth_sigma_t={ranking.tube_smooth_sigma_t}",
        f"--policy.neg_use_swap={str(ranking.swap).lower()}",
        f"--policy.neg_use_matched_swap={str(ranking.matched_swap).lower()}",
        f"--policy.neg_use_temporal={str(ranking.temporal).lower()}",
        f"--policy.neg_use_shift={str(ranking.shift).lower()}",
        f"--policy.neg_shift_frames=[{','.join(str(k) for k in ranking.shift_frames)}]",
    ]


def _architecture_args(cfg: "RunConfig") -> list[str]:
    """Architecture, Bellman target and value head of a Q-function trained from scratch."""
    train, q = cfg.train_q, cfg.qfunction
    return [
        # Architecture -- must match the checkpoint any downstream planner will load.
        f"--policy.dino_model_name={q.dino_model_name}",
        f"--policy.text_encoder_model={q.text_encoder_model}",
        "--policy.use_text_conditioning=true",
        f"--policy.dim_model={q.dim_model}",
        f"--policy.n_heads={q.n_heads}",
        f"--policy.dim_feedforward={q.dim_feedforward}",
        f"--policy.n_decoder_layers={q.n_decoder_layers}",
        # Chunked Bellman target and the categorical value head.
        f"--policy.camera_keys=[{','.join(q.camera_keys)}]",
        f"--policy.h={q.h}",
        f"--policy.gamma={q.gamma}",
        f"--policy.num_bins={q.num_bins}",
        f"--policy.v_min={q.v_min}",
        f"--policy.v_max={q.v_max}",
        f"--policy.hl_gauss_sigma={q.hl_gauss_sigma}",
        f"--policy.target_tau={q.target_tau}",
        f"--policy.reward_mode={q.reward_mode}",
        f"--policy.step_reward={q.step_reward}",
        f"--policy.terminal_bonuses={_as_mapping(q.terminal_bonuses)}",
        f"--policy.bucket_overrides={_as_mapping(train.dataset.bucket_overrides or q.bucket_overrides)}",
    ]


def _as_mapping(value: dict) -> str:
    """Render a dict the way draccus parses it on the command line."""
    return "{" + ", ".join(f"{k}: {v}" for k, v in (value or {}).items()) + "}"


def main(config_path: str | None = None, overrides: Sequence[str] | None = None,
         show_help: bool = False) -> int:
    if show_help or not config_path:
        print(
            "usage: qplanning train-q --config <config.yaml> [--field.name=value ...]\n\n"
            "Trains the Q-function on demonstrations. Only needed if you are not using a\n"
            "released checkpoint. Pass --dry-run to print the command without running it.\n\n"
            "Example:\n"
            "  qplanning train-q --config configs/libero_10/train_q.yaml --train_q.num_gpus=1\n"
        )
        return 0 if show_help else 2

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    overrides = list(overrides or [])
    dry_run = "--dry-run" in overrides
    overrides = [o for o in overrides if o != "--dry-run"]

    load_dotenv()
    try:
        cfg = load(config_path, overrides)
    except ConfigError as exc:
        print(f"config error: {exc}")
        return 1

    from qplanning._env_setup import prepare_environment

    prepare_environment(hf_home=cfg.paths.hf_home or None, strict=False)

    argv = build_argv(cfg)
    log.info("running:\n  %s", " \\\n    ".join(shlex.quote(a) for a in argv))
    if dry_run:
        return 0

    # LeRobot must create output_dir itself: when it already exists (and this is not a
    # resume), LeRobot silently trains into "<output_dir>_<timestamp>" instead, and nothing
    # downstream can find the checkpoints. So never create it first, and refuse a collision.
    output_dir = Path(cfg.paths.output_dir)
    if output_dir.exists() and not cfg.train_q.resume_from:
        print(f"output directory {output_dir} already exists; pick a new run_name "
              "(LeRobot would otherwise train into a timestamped copy of it).")
        return 1
    returncode = subprocess.run(argv, check=False).returncode
    output_dir.mkdir(parents=True, exist_ok=True)
    dump(cfg, output_dir / "resolved.yaml")
    return returncode
