"""The loop both noise-policy training scripts share.

    warm-up: label `warmup_states` demonstration states
    repeat `steps` times:
        every `updates_per_round` updates, label `states_per_round` more states
        one AdamW step on the loss over `batch_size` states drawn from the replay buffer

Labelling a state is what differs between the scripts: each one passes a ``collect`` function
that turns one state into M noises and their weights (see :mod:`noise_policy.train_iw` and
:mod:`noise_policy.train_reversal`), and the loss to fit them with (:mod:`noise_policy.losses`).
FastWAM and the Q-function are frozen throughout; only the noise policy's head side trains.

The buffer keeps each state's policy inputs (images as uint8 at the policy's input size,
~0.3 MB a state), so updates never go back to the dataset: random reads from it cost ~0.15 s a
frame, which would dominate every update.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from collections import defaultdict, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from torch import Tensor

    from noise_policy.config import NoisePolicyRunConfig
    from noise_policy.data import DemoStates
    from noise_policy.modeling import NoiseDistribution, NoisePolicy
    from qoptimizing.config import QOptConfig
    from qoptimizing.optimizer import QGradientPlanner

log = logging.getLogger("noise_policy.training")


@dataclass
class Labelled:
    """One state's policy inputs and training targets: M noises and their weights (sum 1)."""

    images: "Tensor"                  # (V, 3, s, s) uint8, CPU -- resized to the policy's input size
    state: "Tensor"                   # (S,) float32, CPU -- FastWAM-normalized proprioception
    task: str
    noise: "Tensor"                   # (M, H, A) float32, CPU
    weights: "Tensor"                 # (M,) float32, CPU


@dataclass
class TrainContext:
    """Everything a ``collect`` function may use."""

    cfg: "NoisePolicyRunConfig"
    sampler: Any                      # FastWAMAdapter
    planner: "QGradientPlanner"       # the Q-function, its encoder, scoring and improvement
    policy: "NoisePolicy"
    states: "DemoStates"
    denoise_steps: int | None

    def distribution(self, batch: dict[str, Any]) -> "NoiseDistribution":
        from noise_policy.fastwam import policy_inputs

        return self.policy(*policy_inputs(batch, self.cfg.noise_policy.camera_keys))

    def prior(self) -> "NoiseDistribution":
        """N(0, I) in the policy's own parameterization (time-shared when it is)."""
        import torch

        from noise_policy.modeling import NoiseDistribution

        shape = (1, self.policy.n_queries, self.policy.action_dim)
        device = self.policy.head.weight.device
        return NoiseDistribution(mean=torch.zeros(shape, device=device),
                                 log_std=torch.zeros(shape, device=device), horizon=self.policy.horizon)


#: One state (a FastWAM batch of one) -> its M noises (M, H, A), their weights (M,), and metrics.
Collect = Callable[[TrainContext, dict[str, Any]], "tuple[Tensor, Tensor, dict[str, float]]"]
Loss = Callable[["NoiseDistribution", "Tensor", "Tensor"], "Tensor"]


class ReplayBuffer:
    """FIFO over labelled states."""

    def __init__(self, capacity: int) -> None:
        self._items: deque[Labelled] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._items)

    def add(self, items: Sequence[Labelled]) -> None:
        self._items.extend(items)

    def sample(self, n: int, rng: "np.random.Generator") -> list[Labelled]:
        picks = rng.integers(0, len(self._items), size=n)
        return [self._items[int(i)] for i in picks]


# ── construction ───────────────────────────────────────────────────────────────────────
def build(cfg: "NoisePolicyRunConfig", planner_cfg: "QOptConfig") -> TrainContext:
    """FastWAM, the Q-function (behind the qopt planner, configured by ``planner_cfg``), a fresh
    noise policy, and the demonstration states."""
    from noise_policy.data import DemoStates
    from noise_policy.fastwam import noise_shape
    from noise_policy.modeling import NoisePolicy
    from qoptimizing.optimizer import QGradientPlanner
    from qplanning.config.loader import ConfigError
    from qplanning.config.paths import PathResolver
    from qplanning.configuration_qfunction import assert_matches_checkpoint
    from qplanning.envs import build_env_config, resolve_tasks
    from qplanning.policies.registry import build_bc

    if not cfg.qfunction.checkpoint:
        raise ConfigError("training the noise policy needs a Q-function: set paths.q_checkpoint")
    root = cfg.noise_train.data.root or cfg.paths.dataset_root
    if not root:
        raise ConfigError("no demonstration dataset: set paths.dataset_root (QPLANNING_DATA_ROOT)")

    device = cfg.bc.device
    bc = build_bc(cfg.bc, env_cfg=build_env_config(cfg, resolve_tasks(cfg)[0]))
    # Frozen: FastWAM only ever runs under no_grad here and is in no optimizer; this makes it
    # impossible for it to enter a graph at all. (The Q-function is frozen by the planner.)
    bc.sampler.policy.requires_grad_(False)
    denoise_steps = (cfg.noise_train.denoise_steps if cfg.noise_train.denoise_steps is not None
                     else cfg.bc.baseline_denoise_steps)
    planner = QGradientPlanner.build(cfg=planner_cfg, q_checkpoint=cfg.qfunction.checkpoint, bc=bc,
                                     device=device, denoise_steps=denoise_steps)
    assert_matches_checkpoint(cfg.qfunction, planner.ctx.q_policy.config)

    shape = noise_shape(bc.sampler)
    policy = NoisePolicy.build(cfg.noise_policy, shape, device=device)
    states = DemoStates(cfg.noise_train.data, str(PathResolver().resolve(root).path or root),
                        cfg.noise_policy.camera_keys, bc.preprocessor,
                        image_size=bc.policy_config.image_size)
    log.info("FastWAM noise %s x %s, denoise steps %s; noise policy %s, %.1fM trainable parameters",
             shape[0], shape[1], denoise_steps, cfg.noise_policy.noise_mode,
             sum(p.numel() for p in policy.trainable_parameters()) / 1e6)
    return TrainContext(cfg=cfg, sampler=bc.sampler, planner=planner, policy=policy, states=states,
                        denoise_steps=denoise_steps)


# ── the loop ───────────────────────────────────────────────────────────────────────────
def run(cfg: "NoisePolicyRunConfig", ctx: TrainContext, collect: Collect, loss_fn: Loss,
        method: str) -> None:
    import numpy as np
    import torch

    train = cfg.noise_train
    output_dir = Path(cfg.paths.output_dir)
    rng = np.random.default_rng(cfg.seed)
    buffer = ReplayBuffer(train.buffer_states)
    policy = ctx.policy
    params = policy.trainable_parameters()
    optimizer = torch.optim.AdamW(params, lr=train.lr, weight_decay=train.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / max(1, train.lr_warmup_steps)))
    wandb_run = _init_wandb(cfg, method)

    def collect_round(n_states: int) -> dict[str, float]:
        started = time.time()
        batch = ctx.states.batch(ctx.states.sample(n_states, rng))
        read_s = time.time() - started
        images, tasks, proprio = _stored_inputs(batch, cfg.noise_policy)
        sums: dict[str, list[float]] = defaultdict(list)
        policy.eval()
        labelled = []
        for i in range(n_states):
            noise, weights, metrics = collect(ctx, ctx.states.row(batch, i))
            labelled.append(Labelled(images=images[i], state=proprio[i], task=tasks[i],
                                     noise=noise.float().cpu(), weights=weights.float().cpu()))
            for key, value in metrics.items():
                sums[key].append(value)
        policy.train()
        buffer.add(labelled)
        out = {f"collect/{k}": float(np.mean(v)) for k, v in sums.items()}
        out["collect/seconds_per_state"] = (time.time() - started) / n_states
        out["collect/read_seconds_per_state"] = read_s / n_states
        out["collect/buffer_states"] = float(len(buffer))
        return out

    # FastWAM run from given noise must be FastWAM: check once against its own sampler.
    from noise_policy.fastwam import verify_against_sampler

    error = verify_against_sampler(ctx.sampler, ctx.states.batch(ctx.states.sample(1, rng)),
                                   denoise_steps=ctx.denoise_steps)
    log.info("FastWAM from given noise vs its own sampler: max |diff| %.2e", error)
    if error > 1e-2:
        raise RuntimeError(f"the noise-driven FastWAM chain disagrees with FastWAM's sampler ({error:.2e})")

    log.info("labelling %d warm-up states", train.warmup_states)
    collected: dict[str, float] = {}
    while len(buffer) < train.warmup_states:
        collected = collect_round(min(train.states_per_round, train.warmup_states - len(buffer)))
    log.info("warm-up done: %s", _fmt(collected))
    if wandb_run is not None:
        wandb_run.log(collected, step=0)

    policy.train()
    window: dict[str, list[float]] = defaultdict(list)
    started = time.time()
    for step in range(1, train.steps + 1):
        if step > 1 and (step - 1) % train.updates_per_round == 0:
            collected = collect_round(train.states_per_round)

        items = buffer.sample(train.batch_size, rng)
        device = policy.head.weight.device
        images = torch.stack([it.images for it in items]).to(device).float() / 255.0
        proprio = torch.stack([it.state for it in items]).to(device)
        noise = torch.stack([it.noise for it in items]).to(device)          # (B, M, H, A)
        weights = torch.stack([it.weights for it in items]).to(device)      # (B, M)
        dist = policy(images, [it.task for it in items], proprio)
        loss = loss_fn(dist, noise, weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, train.grad_clip_norm)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            window["train/loss"].append(float(loss))
            window["train/grad_norm"].append(float(grad_norm))
            window["policy/mean_abs_mu"].append(float(dist.mean.abs().mean()))
            window["policy/mean_std"].append(float(dist.std.mean()))
            window["policy/entropy_per_dim"].append(float(dist.entropy().mean() / dist.n_free))
            # The fit: how far the targets sit from the policy mean, in policy stds.
            window["policy/target_z_rms"].append(float(
                ((noise[:, :, : dist.mean.shape[1]] - dist.mean.unsqueeze(1)) / dist.std.unsqueeze(1))
                .pow(2).mean().sqrt()))

        if step % train.log_freq == 0 or step == train.steps:
            metrics = {k: float(np.mean(v)) for k, v in window.items()}
            metrics["train/lr"] = scheduler.get_last_lr()[0]
            metrics["train/seconds_per_step"] = (time.time() - started) / train.log_freq
            if torch.cuda.is_available():
                metrics["train/peak_gpu_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
            metrics.update(collected)
            log.info("step %6d  %s", step, _fmt(metrics))
            if wandb_run is not None:
                wandb_run.log(metrics, step=step)
            window.clear()
            started = time.time()
        if step % train.save_freq == 0 or step == train.steps:
            save_checkpoint(ctx, output_dir / "checkpoints" / f"{step:06d}", step)

    if wandb_run is not None:
        wandb_run.finish()


def _stored_inputs(batch: dict[str, Any], policy_cfg: Any) -> "tuple[Tensor, list[str], Tensor]":
    """The policy's inputs for a batch, compacted for the buffer: images resized to the
    policy's input size and stored as uint8 (the demonstrations are uint8 to begin with)."""
    import torch
    from torch.nn import functional as F

    from noise_policy.fastwam import policy_inputs

    images, tasks, proprio = policy_inputs(batch, policy_cfg.camera_keys)
    b, v = images.shape[:2]
    size = policy_cfg.image_size
    flat = images.reshape(b * v, *images.shape[2:]).float()
    if flat.shape[-2:] != (size, size):
        flat = F.interpolate(flat, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    stored = (flat.clamp(0, 1) * 255).round().to(torch.uint8).reshape(b, v, 3, size, size).cpu()
    return stored, tasks, proprio.float().cpu()


def save_checkpoint(ctx: TrainContext, path: Path, step: int) -> None:
    import torch

    path.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "noise_shape": [ctx.policy.horizon, ctx.policy.action_dim],
        "config": dataclasses.asdict(ctx.cfg.noise_policy),
        "state_dict": ctx.policy.trainable_state_dict(),
    }, path / "noise_policy.pt")
    last = path.parent / "last"
    if last.is_symlink() or last.exists():
        last.unlink()
    last.symlink_to(path.name)
    log.info("saved %s", path)


def _init_wandb(cfg: "NoisePolicyRunConfig", method: str) -> Any:
    settings = cfg.noise_train.wandb
    if not settings.enabled:
        log.info("W&B off (set noise_train.wandb.project, or QPLANNING_WANDB_PROJECT, to log)")
        return None
    import wandb

    return wandb.init(
        project=settings.project,
        entity=settings.entity or None,
        name=settings.run_name or cfg.run_name,
        group=method,
        config=json.loads(json.dumps(dataclasses.asdict(cfg), default=str)),
        dir=str(Path(cfg.paths.output_dir)),
    )


def _fmt(metrics: dict[str, float]) -> str:
    return "  ".join(f"{k.split('/')[-1]} {v:.4g}" for k, v in metrics.items())


def main(argv: Sequence[str] | None, prog: str, method: str,
         make: Callable[["NoisePolicyRunConfig"], "tuple[QOptConfig, Collect, Loss]"]) -> int:
    """Shared entry point: parse the config, build everything, train."""
    import argparse

    parser = argparse.ArgumentParser(prog=prog, description=f"Train the noise policy ({method}).",
                                     epilog="Any config field can be overridden inline, e.g. --noise_train.steps=100")
    parser.add_argument("--config", required=True, help="run config YAML")
    known, overrides = parser.parse_known_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    import qoptimizing  # noqa: F401  -- prepares the process environment before LeRobot loads
    from noise_policy.config import NoisePolicyRunConfig
    from qplanning.config.loader import ConfigError, dump, load, load_dotenv

    load_dotenv()
    try:
        cfg = load(known.config, overrides, config_class=NoisePolicyRunConfig)
        planner_cfg, collect, loss_fn = make(cfg)
    except (ConfigError, ValueError) as exc:
        print(f"config error: {exc}")
        return 1

    from qplanning.eval import prepare

    prepare(cfg)
    from qplanning import lerobot_patches

    lerobot_patches.apply()

    output_dir = Path(cfg.paths.output_dir)
    if (output_dir / "checkpoints").exists():
        print(f"{output_dir} already has checkpoints; pick a new run_name")
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)
    dump(cfg, output_dir / "resolved.yaml")

    ctx = build(cfg, planner_cfg)
    run(cfg, ctx, collect, loss_fn, method)
    return 0
