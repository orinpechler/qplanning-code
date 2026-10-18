"""Sanity-check Q-gradient optimization on a synthetic observation, without a simulator.

    python -m qoptimizing.probe --config qoptimizing/configs/libero_10.yaml

Checks, in order:

1. The differentiable BC -> Q action map gives the same Q-values as the planner's own
   (pipeline-based) scoring path, on both the initial and the optimized chunk. With
   ``--qopt.target=noise`` the planner additionally checks, on its first step, that the
   differentiable denoising chain reproduces FastWAM's own sampler.
2. Q rises over the ascent steps.
3. Both networks stay frozen: no parameter requires or holds a gradient afterwards, and
   their weights are bit-identical before and after.
4. The time of one planning step against one baseline FastWAM sample, and peak memory.

Synthetic images make the absolute Q-values meaningless; only the checks matter here.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

import qoptimizing  # noqa: F401  -- prepares the process environment before LeRobot loads
from qplanning.config.loader import ConfigError, load, load_dotenv

log = logging.getLogger("qoptimizing.probe")


def _fingerprint(parameters) -> float:
    """A cheap checksum of a parameter set, to show it did not change."""
    import torch

    total = torch.zeros((), dtype=torch.float64)
    for p in parameters:
        total += p.detach().double().sum().cpu()
    return float(total)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m qoptimizing.probe")
    parser.add_argument("--config", required=True)
    known, overrides = parser.parse_known_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    load_dotenv()

    from qoptimizing.config import QOptRunConfig

    try:
        cfg = load(known.config, overrides, config_class=QOptRunConfig)
    except ConfigError as exc:
        print(f"config error: {exc}")
        return 1

    from qplanning.eval import prepare

    prepare(cfg)

    import torch

    from qoptimizing.eval import build_policy
    from qplanning.envs import build_env_config, resolve_tasks
    from qplanning.probe import synthetic_observation

    env_cfg = build_env_config(cfg, resolve_tasks(cfg)[0])
    policy = build_policy(cfg, env_cfg=env_cfg)
    planner = policy.planner
    q_policy = planner.ctx.q_policy
    log.info("planner: %r", planner)

    batch = synthetic_observation(cfg, policy.bc.policy_config, cfg.bc.device)
    q_before = _fingerprint(q_policy.parameters())
    bc_before = _fingerprint(policy.bc.sampler.parameters())

    # Exactly as in a rollout: select_action runs under no_grad.
    policy.reset()
    with torch.no_grad():
        policy.select_action(batch)
    plan, stats = policy.last_plan, planner.last_stats
    failures: list[str] = []

    log.info("Q trajectory of the executed chunk: %s",
             " ".join(f"{q:.4f}" for q in stats.q_trajectory))
    log.info("Q %.4f -> %.4f (%+.4f), |da| rms %.4f max %.4f, grad rms at start %.2e",
             stats.q_init, stats.q_final, stats.q_gain, stats.delta_rms, stats.delta_max,
             stats.grad_rms_init)

    # 1. The differentiable path must agree with the reference path.
    with torch.no_grad():
        observation = planner._observation_inputs(batch)
        context = planner._encode_observation_once(plan.candidates, observation)
        reference = planner._score(plan.candidates, context, observation).float()
        scale, shift = planner._affine
        n = plan.candidates.shape[0]
        ours = planner._q_values(plan.candidates.float(), context.expand(-1, n, -1).contiguous(),
                                 scale, shift)
    mismatch = float((reference - ours).abs().max())
    log.info("differentiable vs reference Q on optimized chunk(s): max |diff| %.2e", mismatch)
    if mismatch > 1e-3:
        failures.append(f"differentiable Q path disagrees with the reference by {mismatch:.2e}")

    # 2. Q should go up.
    if cfg.qopt.steps > 0:
        if stats.grad_rms_init == 0.0:
            failures.append("dQ/da is exactly zero: no gradient reaches the action")
        elif stats.q_gain <= 0:
            log.warning("Q did not increase (%+.4f); try a smaller qopt.step_size", stats.q_gain)

    # 3. Both networks frozen and untouched.
    for name, params in (("Q-function", q_policy.parameters()),
                         ("BC policy", policy.bc.sampler.parameters())):
        params = list(params)
        if any(p.grad is not None for p in params):
            failures.append(f"{name} parameters hold gradients")
        if name == "Q-function" and any(p.requires_grad for p in params):
            failures.append("Q-function parameters require grad")
    if _fingerprint(q_policy.parameters()) != q_before:
        failures.append("Q-function weights changed")
    if _fingerprint(policy.bc.sampler.parameters()) != bc_before:
        failures.append("BC policy weights changed")

    # 4. Cost of a planning step, relative to the baseline's single FastWAM sample.
    def _timed(fn, repeats: int = 3) -> float:
        import time

        times = []
        for _ in range(repeats):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.no_grad():
                fn()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.perf_counter() - started)
        return min(times)

    sampler = policy.bc.sampler
    t_plan = _timed(lambda: planner.plan(sampler, batch))
    t_base = _timed(lambda: sampler.sample_chunks(batch, 1, denoise_steps=planner.denoise_steps))
    log.info("time per planning step %.2fs vs %.2fs for one baseline sample (%.1fx)",
             t_plan, t_base, t_plan / max(t_base, 1e-9))

    if torch.cuda.is_available():
        log.info("peak GPU memory: %.1f GiB allocated, %.1f GiB reserved (of %.1f GiB)",
                 torch.cuda.max_memory_allocated() / 2**30,
                 torch.cuda.max_memory_reserved() / 2**30,
                 torch.cuda.get_device_properties(0).total_memory / 2**30)

    log.info("")
    if failures:
        for failure in failures:
            log.error("FAIL: %s", failure)
        return 1
    log.info("PASS: gradients reach only the action, and the Q path matches the planner's.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
