"""Evaluate FastWAM with its initial noise drawn from a trained noise policy.

    python -m noise_policy.eval --config noise_policy/configs/libero_10_eval.yaml \
        --noise_eval.checkpoint=libero_10_noise_reversal_anc05_plain --run_name=...

Same rollout, seeding and initial-state protocol as ``qplanning eval`` and ``qoptimizing.eval``
(the loop is :func:`qoptimizing.eval.evaluate`), so ``episodes.csv`` compares row for row with
the baseline's. At every planning step, eps_0 is the noise FastWAM's own sampler would have
drawn there (seeded by episode seed and plan index), and the executed chunk is

    FastWAM(s, mu(s) + sigma(s) * eps_0)          (noise_eval.use_mean: FastWAM(s, mu(s)))

so with pi = N(0, I) this is exactly the baseline. Time-shared policies take eps_0's first row.

Per plan, the baseline's chunk FastWAM(s, eps_0) is also denoised and both are scored by Q.
The ``q_init`` / ``q_final`` columns are Q of the baseline's / the executed chunk and
``delta_rms`` their RMS difference (BC-normalized), so the qopt analysis applies unchanged.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import qoptimizing  # noqa: F401  -- prepares the process environment before LeRobot loads
from qoptimizing.optimizer import QGradientPlanner, QOptStats
from qplanning.planner import PlanResult, QSpread

if TYPE_CHECKING:
    from torch import Tensor

    from noise_policy.config import NoisePolicyRunConfig
    from noise_policy.modeling import NoisePolicy
    from qplanning.policies.base import BCChunkSampler
    from qplanning.policy import QPlanningPolicy

log = logging.getLogger("noise_policy.eval")


@dataclass
class NoisePlanStats(QOptStats):
    """:class:`QOptStats` of one plan, plus what pi looked like there."""

    mu_rms: float = 0.0               # RMS of mu(s)
    sigma_mean: float = 1.0           # mean of sigma(s)
    noise_shift_rms: float = 0.0      # RMS of eps - eps_0


class NoisePolicyPlanner(QGradientPlanner):
    """Draw the baseline's noise, map it through pi, denoise; no gradient ascent.

    Built by :meth:`QGradientPlanner.build` with ``target=noise`` (for its noise draw and the
    one-time check against FastWAM's sampler), then given the policy with :meth:`attach`.
    """

    noise_policy: "NoisePolicy"
    camera_keys: list[str]
    use_mean: bool = False

    def attach(self, noise_policy: "NoisePolicy", use_mean: bool) -> "NoisePolicyPlanner":
        self.noise_policy = noise_policy.eval()
        self.camera_keys = list(noise_policy.cfg.camera_keys)
        self.use_mean = use_mean
        return self

    def plan(self, bc: "BCChunkSampler", batch: dict[str, "Tensor"]) -> PlanResult:
        import torch

        from noise_policy.fastwam import policy_inputs
        from qoptimizing.noise import FastWAMDenoiser

        with torch.no_grad():
            denoiser = FastWAMDenoiser.prepare(bc, batch, self.denoise_steps)
            base = self._draw_noise(bc, batch, denoiser)                        # (1, H, A)
            dist = self.noise_policy(*policy_inputs(batch, self.camera_keys))
            k = dist.mean.shape[1]
            draw = torch.zeros_like(base[:, :k]) if self.use_mean else base[:, :k]
            noise = dist.transform(draw.unsqueeze(1).to(dist.mean))[:, 0]        # (1, H, A)

            # Denoised separately, each as a batch of one like the baseline's own draw, so
            # that pi = N(0, I) gives the baseline's chunk bit for bit.
            chunk = denoiser(noise)
            reference = denoiser(base)
            both = torch.cat([reference, chunk])
            q_ref, q_pi = self.score(both, self.encode(batch, both)).float().tolist()

        delta = (chunk - reference).float()
        stats = NoisePlanStats(
            q_init=q_ref, q_final=q_pi,
            delta_rms=float(delta.pow(2).mean().sqrt()), delta_max=float(delta.abs().max()),
            grad_rms_init=0.0, q_trajectory=[q_ref, q_pi],
            mu_rms=float(dist.mean.pow(2).mean().sqrt()), sigma_mean=float(dist.std.mean()),
            noise_shift_rms=float((noise.float() - base.to(noise).float()).pow(2).mean().sqrt()),
        )
        self.last_stats = stats
        self.episode_stats.append(stats)
        q = torch.tensor([q_pi])
        self.last_spread = QSpread(min=q_pi, max=q_pi, mean=q_pi, std=0.0)
        return PlanResult(chunk=chunk, q_values=q, spread=self.last_spread, candidates=chunk)

    def __repr__(self) -> str:  # pragma: no cover
        return (f"NoisePolicyPlanner(noise_mode={self.noise_policy.cfg.noise_mode}, "
                f"use_mean={self.use_mean}, denoise_steps={self.denoise_steps})")


def resolve_checkpoint(spec: str, runs_root: str | Path) -> Path:
    """A ``noise_policy.pt``, a directory holding one, a training run's output directory, or
    the bare run name of one under ``<paths.root>/runs``."""
    for base in (Path(spec), Path(runs_root) / "runs" / spec):
        for candidate in (base, base / "noise_policy.pt", base / "checkpoints" / "last" / "noise_policy.pt"):
            if candidate.is_file():
                return candidate.resolve()
    raise FileNotFoundError(f"no noise_policy.pt at or under {spec!r} (nor under {runs_root}/runs/{spec})")


def load_noise_policy(path: Path, noise_shape: tuple[int, int], device: Any) -> "NoisePolicy":
    import torch

    from noise_policy.config import NoisePolicyConfig
    from noise_policy.modeling import NoisePolicy

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if tuple(checkpoint["noise_shape"]) != tuple(noise_shape):
        raise ValueError(f"{path} was trained for noise {checkpoint['noise_shape']}, FastWAM's is {noise_shape}")
    policy = NoisePolicy.build(NoisePolicyConfig(**checkpoint["config"]), noise_shape, device=device)
    policy.load_trainable_state_dict(checkpoint["state_dict"])
    log.info("noise policy %s (step %s, %s)", path, checkpoint["step"], policy.cfg.noise_mode)
    return policy.eval()


def build_policy(cfg: "NoisePolicyRunConfig", env_cfg: Any) -> "QPlanningPolicy":
    from noise_policy.fastwam import noise_shape
    from qoptimizing.config import QOptConfig
    from qplanning.config.loader import ConfigError
    from qplanning.configuration_qfunction import assert_matches_checkpoint
    from qplanning.policies.registry import build_bc
    from qplanning.policy import QPlanningPolicy

    if not cfg.noise_eval.checkpoint:
        raise ConfigError("set noise_eval.checkpoint to the noise policy to evaluate")
    if not cfg.qfunction.checkpoint:
        raise ConfigError("scoring the chunks needs a Q-function: set paths.q_checkpoint")
    path = resolve_checkpoint(cfg.noise_eval.checkpoint, cfg.paths.root)

    bc = build_bc(cfg.bc, env_cfg=env_cfg)
    planner = NoisePolicyPlanner.build(cfg=QOptConfig(target="noise", steps=0, n_samples=1),
                                       q_checkpoint=cfg.qfunction.checkpoint, bc=bc,
                                       device=cfg.bc.device, denoise_steps=cfg.bc.baseline_denoise_steps)
    assert_matches_checkpoint(cfg.qfunction, planner.ctx.q_policy.config)
    planner.attach(load_noise_policy(path, noise_shape(bc.sampler), cfg.bc.device), cfg.noise_eval.use_mean)
    return QPlanningPolicy(bc=bc, planner=planner, n_action_steps=cfg.execution.n_action_steps)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m noise_policy.eval",
        description="Roll out FastWAM with its noise drawn from a trained noise policy.",
        epilog="Any config field can be overridden inline, e.g. --noise_eval.use_mean=true",
    )
    parser.add_argument("--config", required=True, help="run config YAML")
    known, overrides = parser.parse_known_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

    from noise_policy.config import NoisePolicyRunConfig
    from qoptimizing.eval import _mean, evaluate, write_records
    from qplanning.config.loader import ConfigError, dump, load, load_dotenv
    from qplanning.eval import prepare, summarise

    load_dotenv()
    try:
        cfg = load(known.config, overrides, config_class=NoisePolicyRunConfig)
    except ConfigError as exc:
        print(f"config error: {exc}")
        return 1

    prepare(cfg)
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dump(cfg, output_dir / "resolved.yaml")

    suffix = f".shard{cfg.eval.shard.replace('/', 'of')}" if cfg.eval.shard else ""
    steps_log = output_dir / f"qopt_steps{suffix}.jsonl" if cfg.noise_eval.log_steps else None

    started = time.time()
    records = evaluate(cfg, steps_log=steps_log, build=build_policy)  # type: ignore[arg-type]
    if not records:
        log.error("no episodes completed; nothing written")
        return 1

    csv_path = write_records(records, output_dir / f"episodes{suffix}.csv")
    summary = summarise(records)
    summary["elapsed_s"] = round(time.time() - started, 1)
    summary["config"] = known.config
    summary["noise_eval"] = asdict(cfg.noise_eval)
    summary["noise_policy_checkpoint"] = str(resolve_checkpoint(cfg.noise_eval.checkpoint, cfg.paths.root))
    for key in ("q_init_mean", "q_final_mean", "q_gain_mean", "delta_rms_mean"):
        summary[key] = _mean([getattr(r, key) for r in records if getattr(r, key) is not None])
    import torch

    if torch.cuda.is_available():
        summary["peak_gpu_mem_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 1)
    (output_dir / f"summary{suffix}.json").write_text(json.dumps(summary, indent=2))

    log.info("")
    log.info("%d episodes over %d tasks", summary["n_episodes"], summary["n_tasks"])
    log.info("success rate                    %.1f%%", 100 * summary["success_rate"])
    log.info("mean successful episode length  %.0f steps", summary["mean_successful_episode_length"])
    log.info("mean Q per plan, baseline -> pi %.4f -> %.4f (%+.4f)",
             summary["q_init_mean"] or 0.0, summary["q_final_mean"] or 0.0, summary["q_gain_mean"] or 0.0)
    log.info("wrote %s", csv_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
