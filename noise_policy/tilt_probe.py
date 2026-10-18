"""Is Q's preference among FastWAM's noises a consistent direction, or noise? No training.

    python -m noise_policy.tilt_probe --config noise_policy/configs/libero_10_iw.yaml \
        [--states 32 --samples 512]

Importance weighting pulls pi towards the Q-weighted mean of the sampled noises. For one state s
with noises eps_1..M ~ N(0, I) (the same M for every state) and their chunks' Q:

    d_tau(s) = sum_i softmax(Q / tau)_i eps_i  -  mean_i eps_i        (tau = inf: cov(Q, eps))

pi can only learn d if it is (a) a property of the state rather than of the particular draws,
and (b) predictable from the observation. So, per tau:

    within    cosine between d from the first and from the second half of the noises, same state
    next1/5   cosine between d(s) and d(s') for the frame 1 / 5 steps later in the same episode
    other     cosine between d(s) and d of an unrelated state (what "no relation" looks like)

each next to the same cosine with Q shuffled across the noises of each state (the chance level:
~0 +- 1/sqrt(H*A) per state). Each half estimates a 224-dim direction from M/2 noises, so even a
real preference gives a within cosine well below 1: read it against chance, not against 1.
Within-state cosines clearly above chance mean the preference is real for that state;
neighbour cosines above chance mean it is smooth enough in the state to be learned.
Results: <output_dir>/tilt_probe.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

log = logging.getLogger("noise_policy.tilt_probe")

TAUS = (0.002, 0.005, 0.01, math.inf)
NEIGHBOURS = (1, 5)


def tilt_direction(noise: Any, q: Any, tau: float) -> Any:
    """``(M, H, A)`` noises with ``(M,)`` Q -> the flattened importance-weighting pull ``(H*A,)``."""
    import torch

    flat = noise.flatten(1).double()
    q = q.double()
    if math.isinf(tau):
        weights = (q - q.mean()) / max(float(q.std()), 1e-12) / len(q)     # cov(Q, eps), scale-free
        return (weights[:, None] * flat).sum(0)
    weights = torch.softmax(q / tau, dim=0)
    return (weights[:, None] * flat).sum(0) - flat.mean(0)


def cosine(a: Any, b: Any) -> float:
    import torch

    return float(torch.nn.functional.cosine_similarity(a[None], b[None]).item())


def consistency(noise: Any, qs: dict[str, Any], pairs: dict[str, list[tuple[str, str]]], tau: float,
                generator: Any) -> dict[str, float]:
    """Mean cosines for one tau. ``qs`` maps state ids to their ``(M,)`` Q on the shared noises;
    ``pairs`` lists the state pairs to compare, by name ("next1", "other", ...)."""
    import torch

    m = noise.shape[0]
    half = m // 2
    out: dict[str, list[float]] = defaultdict(list)
    shuffled = {sid: q[torch.randperm(m, generator=generator)] for sid, q in qs.items()}
    for label, source in (("", qs), ("_chance", shuffled)):
        for sid, q in source.items():
            out["within" + label].append(cosine(tilt_direction(noise[:half], q[:half], tau),
                                                tilt_direction(noise[half:], q[half:], tau)))
        directions = {sid: tilt_direction(noise, q, tau) for sid, q in source.items()}
        for name, members in pairs.items():
            for a, b in members:
                out[name + label].append(cosine(directions[a], directions[b]))
    return {k: sum(v) / len(v) for k, v in out.items()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m noise_policy.tilt_probe",
                                     description="Is Q's preference among FastWAM's noises consistent?")
    parser.add_argument("--config", required=True, help="a noise-policy run config")
    parser.add_argument("--states", type=int, default=32, help="base demonstration frames")
    parser.add_argument("--samples", type=int, default=512, help="shared noises per state")
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
    from noise_policy.fastwam import noise_shape
    from qoptimizing.config import QOptConfig
    from qoptimizing.noise import FastWAMDenoiser
    from qoptimizing.optimizer import QGradientPlanner
    from qplanning.config.paths import PathResolver
    from qplanning.envs import build_env_config, resolve_tasks
    from qplanning.policies.registry import build_bc

    bc = build_bc(cfg.bc, env_cfg=build_env_config(cfg, resolve_tasks(cfg)[0]))
    bc.sampler.policy.requires_grad_(False)
    steps = cfg.noise_train.denoise_steps or cfg.bc.baseline_denoise_steps
    planner = QGradientPlanner.build(cfg=QOptConfig(steps=0), q_checkpoint=cfg.qfunction.checkpoint,
                                     bc=bc, device=cfg.bc.device, denoise_steps=steps)
    root = cfg.noise_train.data.root or cfg.paths.dataset_root
    data = DemoStates(cfg.noise_train.data, str(PathResolver().resolve(root).path or root),
                      cfg.noise_policy.camera_keys, bc.preprocessor, image_size=bc.policy_config.image_size)

    # Base frames whose +1 and +5 neighbours are in the same episode.
    episode = np.asarray(data.dataset.hf_dataset.select_columns(["episode_index"])
                         .with_format(None)["episode_index"]).reshape(-1)
    far = max(NEIGHBOURS)
    candidates = np.asarray([f for f in data.frames if f + far < len(episode) and episode[f + far] == episode[f]])
    rng = np.random.default_rng(cfg.seed)
    bases = rng.choice(candidates, size=known.states, replace=False)
    ids = {f"{b}+{k}": int(b) + k for b in bases for k in (0, *NEIGHBOURS)}

    generator = torch.Generator().manual_seed(cfg.seed)
    noise = torch.randn((known.samples, *noise_shape(bc.sampler)), generator=generator)
    qs: dict[str, Any] = {}
    q_std: list[float] = []
    with torch.no_grad():
        for n, (sid, frame) in enumerate(ids.items(), start=1):
            state = data.batch([frame])
            denoiser = FastWAMDenoiser.prepare(bc.sampler, state, steps)
            chunks = torch.cat([denoiser(part) for part in noise.split(128)])
            q = planner.score(chunks, planner.encode(state, chunks)).float().cpu()
            qs[sid] = q
            q_std.append(float(q.std()))
            if n % 12 == 0 or n == len(ids):
                log.info("%d/%d states scored", n, len(ids))

    pairs = {f"next{k}": [(f"{b}+0", f"{b}+{k}") for b in bases] for k in NEIGHBOURS}
    pairs["other"] = [(f"{b}+0", f"{bases[(i + 1) % len(bases)]}+0") for i, b in enumerate(bases)]
    shuffle = torch.Generator().manual_seed(cfg.seed + 1)
    results = {("inf" if math.isinf(t) else f"{t:g}"): consistency(noise, qs, pairs, t, shuffle) for t in TAUS}

    lines = [f"{len(bases)} base frames (+1, +5 neighbours), {known.samples} shared noises; "
             f"Q std across noises: median {float(np.median(q_std)):.4f}; chance level ~ +-{1 / math.sqrt(noise[0].numel()):.3f}",
             "", f"{'tau':>6s} " + " ".join(f"{k:>9s} {'(chance)':>9s}" for k in ("within", "next1", "next5", "other"))]
    for tau, row in results.items():
        lines.append(f"{tau:>6s} " + " ".join(f"{row[k]:9.3f} {row[k + '_chance']:9.3f}"
                                              for k in ("within", "next1", "next5", "other")))
    log.info("\n%s", "\n".join(lines))
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tilt_probe.json").write_text(json.dumps({"q_std": q_std, "results": results}, indent=2))
    log.info("wrote %s", output_dir / "tilt_probe.json")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
