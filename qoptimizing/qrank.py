"""Offline check: does the Q-function rank the demonstrated action above alternatives?

    python -m qoptimizing.qrank --config configs/libero_10/finetune_q_rank.yaml \
        [--checkpoint PATH] [--label NAME] [--stride 8] [--max-episodes N] [--paths.x=...]

Selecting or optimizing actions with Q only helps if, at a given state, Q scores better
actions higher. This walks the held-out demonstration episodes and, every ``stride`` frames,
compares Q of the demonstrated chunk with Q of alternatives at the same state:

- ``tube<s>``   the true chunk plus temporally smoothed Gaussian noise of ``s`` per-dimension
                stds (gripper held). Small ``s`` is the scale at which FastWAM's own samples
                differ, so it is the planning-relevant case.
- ``mswap``     the true chunk of another held-out episode of the same task, at the same
                fraction of the episode: a sensible action for the task phase, wrong for this
                state. A Q that cannot rank the true chunk above this cannot steer a planner.
- ``shift<k>``  the right actions at the wrong time: the true trajectory k frames later.
- ``trev``      the true chunk reversed in time.

For each it reports P(Q_true > Q_neg) (0.5 is chance) and the mean gap, over all held-out
episodes and over the LIBERO-10 tasks alone, plus how well Q_true tracks the Monte-Carlo
value gamma^(steps to success).

The held-out episodes are the fork's own test split (``train_q.test_split_ratio``, seeded
with ``seed``). The fork seeds that split with ``hash()``, which Python randomizes per process
unless PYTHONHASHSEED is fixed, so this module re-executes itself with PYTHONHASHSEED=0 --
the value the fine-tuning job uses, so both see the same held-out episodes. The released
checkpoint was trained without a fixed hash seed: for it, these episodes are mostly ones its
TD loss did see -- but never with negatives, which is what is tested here.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from torch import Tensor

log = logging.getLogger("qoptimizing.qrank")

#: The PYTHONHASHSEED the held-out split is defined under; the fine-tuning job sets the same.
SPLIT_HASH_SEED = "0"
TUBE_SIGMAS = (0.05, 0.1, 0.25, 0.5, 1.0)
TUBE_DRAWS = 2
TUBE_SMOOTH_SIGMA_T = 2.0
SHIFT_FRAMES = (8, 16)
FRAMES_PER_BATCH = 16


# ── pure helpers (tested on CPU) ───────────────────────────────────────────────────────
def chunk_at(actions: "np.ndarray", ep_from: int, ep_to: int, frame: int, h: int) -> "np.ndarray":
    """The h-step action chunk starting at global ``frame``, padded with the episode's last
    action past its end -- as LeRobot pads a delta-timestamp window."""
    import numpy as np

    idx = np.clip(np.arange(frame, frame + h), ep_from, ep_to - 1)
    return actions[idx]


def summarize(gaps: Sequence[float]) -> dict[str, float]:
    """P(Q_true > Q_neg) with ties counted half, and the gap's mean and median."""
    import numpy as np

    g = np.asarray(gaps, dtype=np.float64)
    if g.size == 0:
        return {"n": 0, "rank_acc": float("nan"), "gap_mean": float("nan"), "gap_median": float("nan")}
    return {
        "n": int(g.size),
        "rank_acc": float((g > 0).mean() + 0.5 * (g == 0).mean()),
        "gap_mean": float(g.mean()),
        "gap_median": float(np.median(g)),
    }


def _ensure_hash_seed(argv: Sequence[str]) -> None:
    if os.environ.get("PYTHONHASHSEED") != SPLIT_HASH_SEED:
        env = {**os.environ, "PYTHONHASHSEED": SPLIT_HASH_SEED}
        os.execve(sys.executable, [sys.executable, "-m", "qoptimizing.qrank", *argv], env)


# ── data ───────────────────────────────────────────────────────────────────────────────
def build_heldout(cfg: Any, q_config: Any) -> Any:
    """The demonstrations wrapped exactly as ``lerobot.datasets.factory`` wraps them for
    training, so the held-out split is the training run's."""
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, MultiLeRobotDataset
    from lerobot.policies.q_function.q_value_labels import QValueLabelDataset

    from qplanning.config.loader import ConfigError
    from qplanning.config.paths import PathResolver

    train = cfg.train_q
    root = train.dataset.root or cfg.paths.dataset_root
    if not root:
        raise ConfigError("no demonstration dataset: set paths.dataset_root (QPLANNING_DATA_ROOT)")
    root = str(PathResolver().resolve(root).path or root)
    repo_ids = list(train.dataset.repo_ids)
    if len(repo_ids) != 1:
        raise ConfigError(f"qrank supports a single demonstration dataset, got {repo_ids}")

    meta = LeRobotDatasetMetadata(repo_ids[0], root=str(Path(root) / repo_ids[0]))
    dataset = MultiLeRobotDataset(
        repo_ids, root=root, delta_timestamps=resolve_delta_timestamps(q_config, meta)
    )
    return QValueLabelDataset(
        dataset,
        h=q_config.h,
        step_reward=q_config.step_reward,
        terminal_bonuses=q_config.terminal_bonuses,
        reward_mode=getattr(q_config, "reward_mode", "sparse"),
        quality_scalars=getattr(q_config, "quality_scalars", None),
        precache_root=getattr(q_config, "precache_root", None),
        terminal_bonus_uniform=getattr(q_config, "terminal_bonus_uniform", 1.0),
        bucket_overrides=getattr(q_config, "bucket_overrides", None),
        holdout_fraction=train.test_split_ratio,
        holdout_seed=cfg.seed,
    )


def _all_actions(wrapped: Any) -> "np.ndarray":
    """Every frame's action, read from the parquet column without decoding any image."""
    import numpy as np

    sub = wrapped.dataset._datasets[0]
    return np.asarray(sub.hf_dataset.with_format(None)["action"], dtype=np.float32)


def _libero10_instructions() -> set[str]:
    try:
        from libero.libero import benchmark

        suite = benchmark.get_benchmark_dict()["libero_10"]()
        return {suite.get_task(i).language for i in range(suite.n_tasks)}
    except Exception:  # pragma: no cover - LIBERO not installed
        log.warning("LIBERO not importable; the LIBERO-10 subset will be empty")
        return set()


# ── scoring ────────────────────────────────────────────────────────────────────────────
class Scorer:
    """Encodes a batch of states once, then scores any number of chunks per state."""

    def __init__(self, q_policy: Any, q_pre: Any) -> None:
        self.q_policy = q_policy
        self.q_pre = q_pre
        self.dtype = q_policy.q_online.action_proj.weight.dtype
        self._affine: "tuple[Tensor, Tensor] | None" = None

    def encode(self, images: dict[str, "Tensor"], tasks: list[str], raw_true: "Tensor") -> "Tensor":
        from lerobot.utils.constants import ACTION

        batch = {**images, "task": tasks, ACTION: raw_true}
        pre = self.q_pre(batch)
        if self._affine is None:
            self._affine = self._read_affine(images, tasks, raw_true, pre[ACTION])
        return self.q_policy.encode_obs_context(pre)                     # (S, B, D)

    def normalize(self, raw: "Tensor") -> "Tensor":
        scale, shift = self._affine  # type: ignore[misc]
        return raw.float() * scale + shift

    def score(self, context: "Tensor", chunks: "Tensor") -> "Tensor":
        """Q of ``(B, K, h, A)`` Q-normalized chunks at the B encoded states -> ``(B, K)``."""
        import torch

        from qplanning.modeling_qfunction import expected_value

        n_states, k = chunks.shape[:2]
        ctx = context.repeat_interleave(k, dim=1).to(self.dtype)
        actions = self.q_policy._truncate_action(chunks.reshape(n_states * k, *chunks.shape[2:]))
        with torch.no_grad():
            logits = self.q_policy.q_online.forward_with_context(ctx, actions.to(self.dtype))
        return expected_value(logits.float(), self.q_policy.bin_centers.float()).view(n_states, k)

    def _read_affine(self, images, tasks, raw_true, normalized_true) -> "tuple[Tensor, Tensor]":
        """The per-dimension raw -> Q-normalized action map, read off the preprocessor."""
        import torch
        from lerobot.utils.constants import ACTION

        one = {k: v[:1].expand(2, *v.shape[1:]) for k, v in images.items()}
        probe = torch.stack([torch.zeros_like(raw_true[0]), torch.ones_like(raw_true[0])])
        mapped = self.q_pre({**one, "task": tasks[:1] * 2, ACTION: probe})[ACTION].float()
        shift, scale = mapped[0, 0], mapped[1, 0] - mapped[0, 0]
        error = float((raw_true.float() * scale + shift - normalized_true.float()).abs().max())
        if error > 1e-3 * max(1.0, float(normalized_true.abs().max())):
            raise RuntimeError(f"the Q action normalization is not per-dimension affine (error {error:.2e})")
        return scale, shift


def _tube(true_n: "Tensor", sigma: float, draws: int, generator: Any) -> "Tensor":
    """``(B, draws, h, A)`` smoothed perturbations of ``(B, h, A)`` chunks, gripper held."""
    import torch
    from lerobot.policies.act_simple.planning import _smooth_time

    n_states, h, a = true_n.shape
    noise = torch.randn(n_states * draws, h, a, generator=generator).to(true_n.device) * sigma
    noise = _smooth_time(noise, TUBE_SMOOTH_SIGMA_T)
    noise[..., -1] = 0.0
    return true_n.unsqueeze(1) + noise.view(n_states, draws, h, a)


# ── the walk ───────────────────────────────────────────────────────────────────────────
def evaluate(cfg: Any, checkpoint: str, stride: int, max_episodes: int | None) -> dict[str, Any]:
    import numpy as np
    import torch

    from qplanning.modeling_qfunction import load_q_function

    device = cfg.bc.device
    q_policy, q_pre = load_q_function(checkpoint, device=device)
    q_policy.requires_grad_(False)
    h, gamma = int(q_policy.config.h), float(q_policy.config.gamma)
    camera_keys = tuple(q_policy.config.camera_keys)

    wrapped = build_heldout(cfg, q_policy.config)
    test_ids = [int(e) for e in wrapped.test_episode_ids]
    if not test_ids:
        raise RuntimeError("no held-out episodes: train_q.test_split_ratio is 0")
    if max_episodes is not None and max_episodes < len(test_ids):
        keep = np.linspace(0, len(test_ids) - 1, num=max_episodes).round().astype(int)
        test_ids = [test_ids[i] for i in keep]
    actions = _all_actions(wrapped)
    libero10 = _libero10_instructions()
    scorer = Scorer(q_policy, q_pre)
    generator = torch.Generator().manual_seed(0)

    # Pass 1: episode bounds, task, and the strided true chunks (for the matched swap).
    episodes = []
    for ep in test_ids:
        f0, f1 = int(wrapped._ep_from[ep]), int(wrapped._ep_to[ep])
        first = wrapped[f0]
        task = first["task"] if isinstance(first["task"], str) else first["task"][0]
        if not np.allclose(first["action"][:h].numpy(), chunk_at(actions, f0, f1, f0, h), atol=1e-5):
            raise RuntimeError(f"episode {ep}: the action column disagrees with the dataset items")
        frames = list(range(f0, f1, stride))
        episodes.append({"ep": ep, "from": f0, "to": f1, "task": task, "frames": frames,
                         "fracs": np.array([(f - f0) / max(1, f1 - f0 - 1) for f in frames])})
    by_task: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(episodes):
        by_task[e["task"]].append(i)

    gaps: dict[str, dict[str, list[float]]] = {"all": defaultdict(list), "libero_10": defaultdict(list)}
    calib_q, calib_mc, calib_l10 = [], [], []
    mswap_same_task = mswap_total = 0

    for n_done, e in enumerate(episodes, start=1):
        partners = [j for j in by_task[e["task"]] if episodes[j]["ep"] != e["ep"]]
        same_task = bool(partners)
        if not partners:
            partners = [j for j in range(len(episodes)) if episodes[j]["ep"] != e["ep"]]
        partner = episodes[partners[n_done % len(partners)]]
        for start in range(0, len(e["frames"]), FRAMES_PER_BATCH):
            frames = e["frames"][start : start + FRAMES_PER_BATCH]
            items = [wrapped[f] for f in frames]
            images = {k: torch.stack([it[k][0] if it[k].dim() == 4 else it[k] for it in items]).to(device)
                      for k in camera_keys}
            tasks = [e["task"]] * len(frames)
            raw_true = torch.as_tensor(np.stack([chunk_at(actions, e["from"], e["to"], f, h) for f in frames]))
            with torch.no_grad():
                context = scorer.encode(images, tasks, raw_true.to(device))
            true_n = scorer.normalize(raw_true.to(device))                       # (B, h, A)

            cands, names = [true_n.unsqueeze(1)], ["true"]
            for s in TUBE_SIGMAS:
                cands.append(_tube(true_n, s, TUBE_DRAWS, generator))
                names += [f"tube{s:g}"] * TUBE_DRAWS
            swap_frames = [partner["frames"][int(np.argmin(np.abs(partner["fracs"] - (f - e["from"]) / max(1, e["to"] - e["from"] - 1))))]
                           for f in frames]
            swap = np.stack([chunk_at(actions, partner["from"], partner["to"], f, h) for f in swap_frames])
            cands.append(scorer.normalize(torch.as_tensor(swap).to(device)).unsqueeze(1))
            names.append("mswap")
            valid_shift = {}
            for k in SHIFT_FRAMES:
                shifted = np.stack([chunk_at(actions, e["from"], e["to"], f + k, h) for f in frames])
                cands.append(scorer.normalize(torch.as_tensor(shifted).to(device)).unsqueeze(1))
                names.append(f"shift{k}")
                valid_shift[f"shift{k}"] = [f + k + h <= e["to"] for f in frames]
            cands.append(torch.flip(true_n, dims=[1]).unsqueeze(1))
            names.append("trev")

            with torch.no_grad():
                q = scorer.score(context, torch.cat(cands, dim=1)).cpu().numpy()   # (B, K)
            in_l10 = e["task"] in libero10
            for b, f in enumerate(frames):
                calib_q.append(float(q[b, 0]))
                calib_mc.append(gamma ** (e["to"] - 1 - f))
                calib_l10.append(in_l10)
                for col, name in enumerate(names[1:], start=1):
                    if name.startswith("shift") and not valid_shift[name][b]:
                        continue
                    gap = float(q[b, 0] - q[b, col])
                    gaps["all"][name].append(gap)
                    if in_l10:
                        gaps["libero_10"][name].append(gap)
            mswap_total += len(frames)
            mswap_same_task += len(frames) if same_task else 0
        if n_done % 20 == 0 or n_done == len(episodes):
            log.info("%d/%d held-out episodes", n_done, len(episodes))

    def calibration(mask: "np.ndarray") -> dict[str, float]:
        qv, mc = np.asarray(calib_q)[mask], np.asarray(calib_mc)[mask]
        if qv.size < 3:
            return {"n": int(qv.size)}
        rank = lambda x: np.argsort(np.argsort(x))  # noqa: E731
        return {
            "n": int(qv.size),
            "spearman": float(np.corrcoef(rank(qv), rank(mc))[0, 1]),
            "pearson": float(np.corrcoef(qv, mc)[0, 1]),
            "mae": float(np.abs(qv - mc).mean()),
        }

    l10 = np.asarray(calib_l10, dtype=bool)
    return {
        "checkpoint": str(checkpoint),
        "heldout_episodes": len(episodes),
        "libero_10_episodes": sum(e["task"] in libero10 for e in episodes),
        "stride": stride,
        "mswap_same_task_frac": mswap_same_task / max(1, mswap_total),
        "ranking": {subset: {name: summarize(v) for name, v in per.items()} for subset, per in gaps.items()},
        "calibration": {"all": calibration(np.ones_like(l10)), "libero_10": calibration(l10)},
    }


def report(result: dict[str, Any]) -> str:
    lines = [f"checkpoint: {result['checkpoint']}",
             f"held-out episodes: {result['heldout_episodes']} ({result['libero_10_episodes']} LIBERO-10), "
             f"every {result['stride']} frames; matched swap from the same task for "
             f"{result['mswap_same_task_frac']:.0%} of frames", "",
             f"{'negative':10s} {'P(Q_true>Q_neg)':>16s} {'mean gap':>9s}   {'LIBERO-10':>10s} {'mean gap':>9s}     n"]
    rank = result["ranking"]
    for name in rank["all"]:
        a, b = rank["all"][name], rank["libero_10"].get(name, summarize([]))
        lines.append(f"{name:10s} {a['rank_acc']:16.3f} {a['gap_mean']:+9.4f}   {b['rank_acc']:10.3f} "
                     f"{b['gap_mean']:+9.4f} {a['n']:6d}")
    lines.append("")
    for subset, c in result["calibration"].items():
        if "spearman" in c:
            lines.append(f"Q_true vs gamma^(steps to success), {subset}: Spearman {c['spearman']:.3f}, "
                         f"Pearson {c['pearson']:.3f}, MAE {c['mae']:.3f} (n={c['n']})")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    _ensure_hash_seed(argv)

    parser = argparse.ArgumentParser(prog="python -m qoptimizing.qrank", description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, help="a train_q config: dataset, split ratio and seed")
    parser.add_argument("--checkpoint", default=None, help="Q checkpoint to test (default: qfunction.checkpoint)")
    parser.add_argument("--label", default="q", help="name of this result: <output_dir>/qrank_<label>.json")
    parser.add_argument("--stride", type=int, default=8, help="score every Nth frame")
    parser.add_argument("--max-episodes", type=int, default=None, help="evenly subsample the held-out episodes")
    known, overrides = parser.parse_known_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    import qoptimizing  # noqa: F401  -- prepares the process environment before LeRobot loads
    from qplanning.config.loader import ConfigError, load, load_dotenv

    load_dotenv()
    try:
        cfg = load(known.config, overrides)
    except ConfigError as exc:
        print(f"config error: {exc}")
        return 1
    from qplanning.eval import prepare

    prepare(cfg)
    # Only now: the patch imports LeRobot, and with it huggingface_hub, which fixes its cache
    # location at import -- so HF_HOME has to be set (by prepare) first.
    from qplanning import lerobot_patches

    lerobot_patches.apply()   # without it, every dataset item decodes ~140 images
    checkpoint = known.checkpoint or cfg.qfunction.checkpoint
    if not checkpoint:
        print("no Q checkpoint: pass --checkpoint or set paths.q_checkpoint")
        return 1

    result = evaluate(cfg, checkpoint, known.stride, known.max_episodes)
    out = Path(cfg.paths.output_dir) / f"qrank_{known.label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(report(result))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
