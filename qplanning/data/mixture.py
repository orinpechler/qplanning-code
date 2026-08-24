"""Mixing demonstrations with deployment rollouts for the Q-only update.

Each minibatch is drawn half from the original demonstrations and half from the accumulated
online rollouts (``online_fraction``). Training on the online data alone would let the
Q-function drift away from the demonstrations that define the task; training on the
demonstrations alone would discard exactly the failure signal the loop exists to capture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch.utils.data import DataLoader, Dataset


def make_shared_key_collate(reference: "Dataset") -> Any:
    """Collate using only the keys the reference dataset provides.

    The demonstration dataset carries fields the online buffer does not (proprioceptive
    state, padding masks), and the default collate raises when the two are concatenated.
    Restricting to the online dataset's keys works because those are a superset of what the
    Q-function's loss actually reads.
    """
    shared_keys = set(reference[0].keys())

    def collate(batch: list[dict]) -> dict:
        from torch.utils.data._utils.collate import default_collate

        return default_collate([{k: v for k, v in item.items() if k in shared_keys} for item in batch])

    return collate


def build_mixture_loader(
    offline_dataset: "Dataset",
    online_dataset: "Dataset",
    *,
    steps: int,
    batch_size: int,
    online_fraction: float,
    num_workers: int = 4,
) -> "DataLoader":
    """A loader whose every batch is ``online_fraction`` online data, in expectation."""
    from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

    n_offline, n_online = len(offline_dataset), len(online_dataset)
    if n_online == 0:
        raise ValueError(
            "the online buffer is empty; there is nothing to self-improve from. "
            "Run the collection stage first."
        )

    # Per-sample weights, so each source contributes its configured share of a batch
    # regardless of how much data it holds.
    weights = [(1.0 - online_fraction) / n_offline] * n_offline + [online_fraction / n_online] * n_online
    sampler = WeightedRandomSampler(weights, num_samples=steps * batch_size, replacement=True)

    return DataLoader(
        ConcatDataset([offline_dataset, online_dataset]),
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=make_shared_key_collate(online_dataset),
    )


def load_demonstrations(repo_ids: list[str], root: str, q_policy: Any) -> Any:
    """Load the demonstration dataset with the Q-function's reward labelling applied."""
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.q_function.q_value_labels import QValueLabelDataset

    if not repo_ids:
        raise ValueError(
            "no demonstration dataset configured. Set self_improve.offline.repo_ids -- the "
            "offline half of each minibatch comes from it."
        )
    repo_id = repo_ids[0]
    cfg = q_policy.config

    meta = LeRobotDatasetMetadata(repo_id=repo_id, root=root)
    dataset = LeRobotDataset(
        repo_id=repo_id, root=root, delta_timestamps=resolve_delta_timestamps(cfg, meta)
    )
    return QValueLabelDataset(
        dataset,
        h=int(cfg.h),
        step_reward=float(getattr(cfg, "step_reward", 0.0)),
        terminal_bonuses=dict(cfg.terminal_bonuses) if cfg.terminal_bonuses else {"q5": 1.0},
        reward_mode="sparse",
        bucket_overrides=dict(cfg.bucket_overrides) if cfg.bucket_overrides else {repo_id: "q5"},
        load_preencoded=False,
    )
