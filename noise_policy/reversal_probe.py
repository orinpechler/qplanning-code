"""How well does flow reversal invert FastWAM? A diagnostic: FastWAM only, no Q, no training.

    python -m noise_policy.reversal_probe --config noise_policy/configs/libero_10_reversal.yaml \
        [--states 4 --samples 8]

For a few demonstration states and M noises eps_0 each, a_0 = F_K(eps_0) with K denoising steps;
then eps_hat = F^-1(a_0) by FastWAMDenoiser.invert and the round trip F(eps_hat). Since a_0 is
FastWAM's own output, a perfect reversal gives back eps_0 exactly. Two sweeps, each in
bfloat16 (as FastWAM runs) and float32:

    matched     forward = reverse steps K in {10, 20, 50}, refine_iters in {0, 1, 2, 5, 10, 20}
    finer       forward K = 10 (what rollouts and the noise policy use), reversed with
                {10, 20, 50, 100} plain backward-Euler steps, round trip with K = 10

Reported (means over states): the round-trip action error rms(F(eps_hat) - a_0), the noise error
rms(eps_hat - eps_0), std(eps_hat) (1 for N(0, I) noise), and seconds per state. For the plain
matched reversal, also the error at every reverse step against the true forward path, by sigma
-- where along the flow the reversal goes wrong. Results: <output_dir>/reversal_probe.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

log = logging.getLogger("noise_policy.reversal_probe")

MATCHED_STEPS = (10, 20, 50)
REFINE = (0, 1, 2, 5, 10, 20)
FINER_STEPS = (10, 20, 50, 100)
ROLLOUT_STEPS = 10


def _rms(x: Any) -> float:
    return float(x.float().pow(2).mean().sqrt())


def _forward_path(denoiser: Any, noise: Any) -> list[Any]:
    """x_0 = noise, ..., x_K = chunk, exactly as FastWAMDenoiser.__call__ steps."""
    model, p = denoiser.model, denoiser._p
    x = noise.to(device=model.device, dtype=model.torch_dtype)
    path = [x.float()]
    for t, delta in zip(p["timesteps"], p["deltas"]):
        x = model.infer_action_scheduler.step(denoiser._velocity(x, t), delta, x)
        path.append(x.float())
    return path


def _set_precision(model: Any, dtype: Any) -> None:
    model.to(dtype)
    for module in model.modules():
        if hasattr(module, "torch_dtype"):
            module.torch_dtype = dtype


def probe(sampler: Any, states: list[dict[str, Any]], n_samples: int, seed: int) -> dict[str, Any]:
    import torch

    from qoptimizing.noise import FastWAMDenoiser

    matched: dict[str, list[dict[str, float]]] = defaultdict(list)
    finer: dict[str, list[dict[str, float]]] = defaultdict(list)
    per_step: dict[str, list[list[float]]] = defaultdict(list)
    sigmas: dict[str, list[float]] = {}
    for i, state in enumerate(states):
        generator = torch.Generator().manual_seed(seed + i)
        shape = (n_samples, sampler.chunk_size, int(sampler.policy.model.action_expert.action_dim))
        eps0 = torch.randn(shape, generator=generator)
        denoisers = {k: FastWAMDenoiser.prepare(sampler, state, k)
                     for k in sorted(set(MATCHED_STEPS) | set(FINER_STEPS))}
        for k in MATCHED_STEPS:
            den = denoisers[k]
            path = _forward_path(den, eps0)
            a0 = path[-1]
            for refine in REFINE:
                started = time.time()
                trace: list[Any] = []
                eps_hat = den.invert(a0, refine_iters=refine, trace=trace)
                seconds = time.time() - started
                roundtrip = den(eps_hat)
                matched[f"K{k}_r{refine}"].append({
                    "roundtrip_action_rms": _rms(roundtrip - a0),
                    "noise_error_rms": _rms(eps_hat - eps0.to(eps_hat.device)),
                    "noise_std": float(eps_hat.std()),
                    "seconds": seconds,
                })
                if refine == 0:
                    # trace holds x_{K-1} .. x_0; path holds x_0 .. x_K.
                    errors = [_rms(trace[j] - path[k - 1 - j]) for j in range(k)]
                    per_step[f"K{k}"].append(errors)
                    sched = den._p
                    t_all = [float(t) for t in sched["timesteps"]]
                    sigmas[f"K{k}"] = [t_all[k - 1 - j] / t_all[0] for j in range(k)]
        a0_rollout = _forward_path(denoisers[ROLLOUT_STEPS], eps0)[-1]
        for k in FINER_STEPS:
            started = time.time()
            eps_hat = denoisers[k].invert(a0_rollout, refine_iters=0)
            seconds = time.time() - started
            roundtrip = denoisers[ROLLOUT_STEPS](eps_hat)
            finer[f"F{ROLLOUT_STEPS}_R{k}"].append({
                "roundtrip_action_rms": _rms(roundtrip - a0_rollout),
                "noise_error_rms": _rms(eps_hat - eps0.to(eps_hat.device)),
                "noise_std": float(eps_hat.std()),
                "seconds": seconds,
            })
        log.info("state %d/%d done", i + 1, len(states))

    def mean(rows: list[dict[str, float]]) -> dict[str, float]:
        return {key: sum(r[key] for r in rows) / len(rows) for key in rows[0]}

    return {
        "matched": {name: mean(rows) for name, rows in matched.items()},
        "finer": {name: mean(rows) for name, rows in finer.items()},
        "per_step": {name: {"sigma_from": sigmas[name],
                            "error_rms": [sum(col) / len(col) for col in zip(*rows)]}
                     for name, rows in per_step.items()},
    }


def report(results: dict[str, dict[str, Any]], action_rms: float) -> str:
    lines = [f"(for scale: the actions themselves have rms {action_rms:.3f}; the anchored ascent "
             "moves them by ~0.01)"]
    for precision, res in results.items():
        if "error" in res:
            lines += ["", f"== {precision}: failed: {res['error']}"]
            continue
        lines += ["", f"== {precision}",
                  f"{'setting':14s} {'roundtrip a':>12s} {'noise err':>10s} {'noise std':>10s} {'s/state':>8s}"]
        for table in ("matched", "finer"):
            for name, row in res[table].items():
                lines.append(f"{name:14s} {row['roundtrip_action_rms']:12.4f} {row['noise_error_rms']:10.4f} "
                             f"{row['noise_std']:10.3f} {row['seconds']:8.2f}")
        for name, row in res["per_step"].items():
            steps = "  ".join(f"s{s:.2f}:{e:.3f}" for s, e in zip(row["sigma_from"], row["error_rms"]))
            lines.append(f"plain reversal {name}, error vs true path after each reverse step: {steps}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m noise_policy.reversal_probe",
                                     description="How well flow reversal inverts FastWAM.")
    parser.add_argument("--config", required=True, help="a noise-policy run config (for FastWAM and the data)")
    parser.add_argument("--states", type=int, default=4, help="demonstration states to test")
    parser.add_argument("--samples", type=int, default=8, help="noises per state")
    parser.add_argument("--no-fp32", action="store_true", help="skip the float32 pass")
    known, overrides = parser.parse_known_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    import qoptimizing  # noqa: F401  -- prepares the process environment before LeRobot loads
    from noise_policy.config import NoisePolicyRunConfig
    from qplanning.config.loader import ConfigError, load, load_dotenv

    load_dotenv()
    try:
        cfg = load(known.config, overrides, config_class=NoisePolicyRunConfig)
    except ConfigError as exc:
        print(f"config error: {exc}")
        return 1
    from qplanning.eval import prepare

    prepare(cfg)
    from qplanning import lerobot_patches

    lerobot_patches.apply()

    import numpy as np
    import torch

    from noise_policy.data import DemoStates
    from qplanning.config.paths import PathResolver
    from qplanning.envs import build_env_config, resolve_tasks
    from qplanning.policies.registry import build_bc

    bc = build_bc(cfg.bc, env_cfg=build_env_config(cfg, resolve_tasks(cfg)[0]))
    bc.sampler.policy.requires_grad_(False)
    root = cfg.noise_train.data.root or cfg.paths.dataset_root
    data = DemoStates(cfg.noise_train.data, str(PathResolver().resolve(root).path or root),
                      cfg.noise_policy.camera_keys, bc.preprocessor, image_size=bc.policy_config.image_size)
    batch = data.batch(data.sample(known.states, np.random.default_rng(cfg.seed)))
    states = [data.row(batch, i) for i in range(known.states)]

    model = bc.sampler.policy.model
    precisions = [("bfloat16", model.torch_dtype)] + ([] if known.no_fp32 else [("float32", torch.float32)])
    results: dict[str, dict[str, Any]] = {}
    action_rms = 0.0
    with torch.no_grad():
        for name, dtype in precisions:
            log.info("precision %s", name)
            try:
                _set_precision(model, dtype)
                results[name] = probe(bc.sampler, states, known.samples, cfg.seed)
            except Exception as exc:  # noqa: BLE001 -- report it and keep the other precision's results
                log.exception("precision %s failed", name)
                results[name] = {"error": repr(exc)}
        from qoptimizing.noise import FastWAMDenoiser

        _set_precision(model, precisions[0][1])
        den = FastWAMDenoiser.prepare(bc.sampler, states[0], ROLLOUT_STEPS)
        action_rms = _rms(den(torch.randn(known.samples, *den.noise_shape)))

    text = report(results, action_rms)
    log.info("\n%s", text)
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reversal_probe.json").write_text(json.dumps(results, indent=2))
    log.info("wrote %s", output_dir / "reversal_probe.json")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
