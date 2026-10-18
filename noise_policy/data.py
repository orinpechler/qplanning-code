"""Training states: single frames of the LIBERO demonstrations, as FastWAM sees them.

The frames come from HuggingFaceVLA/libero: the same LIBERO demonstrations FastWAM was trained
on, but LeRobot's copy of them (1693 episodes at 256 x 256, against the 1712 episodes at 224 x 224
in FastWAM's own training statistics). Only the observations are used, never the actions.

A frame is the observation half of a dataset item -- both cameras resized to FastWAM's input
size, the proprioceptive state and the instruction -- passed through FastWAM's own
preprocessor, the same one a rollout uses. From
that one batch FastWAM samples, the Q-function reads its inputs (as in the qopt rollouts), and
:func:`noise_policy.fastwam.policy_inputs` builds the noise policy's.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from torch import Tensor

    from noise_policy.config import DemoStatesConfig

log = logging.getLogger("noise_policy.data")

STATE_KEY = "observation.state"


def suite_instructions(suite: str) -> set[str]:
    from libero.libero import benchmark

    tasks = benchmark.get_benchmark_dict()[suite]()
    return {tasks.get_task(i).language for i in range(tasks.n_tasks)}


class DemoStates:
    """The frames of one LeRobot dataset, optionally restricted to one LIBERO suite's tasks."""

    def __init__(self, cfg: "DemoStatesConfig", root: str, camera_keys: list[str],
                 preprocessor: Any, image_size: "Sequence[int]") -> None:
        import numpy as np
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.camera_keys = list(camera_keys)
        self.preprocessor = preprocessor
        # FastWAM does not resize: rollouts hand it the environment's 224 x 224 renders, while
        # the dataset stores 256 x 256 frames. Resize to what FastWAM was trained on and sees.
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.dataset = LeRobotDataset(cfg.repo_id, root=str(Path(root) / cfg.repo_id))
        # The task of every frame, from the parquet column alone (no image is decoded).
        task_of_frame = np.asarray(
            self.dataset.hf_dataset.select_columns(["task_index"]).with_format(None)["task_index"]
        ).reshape(-1)
        if cfg.suite:
            wanted = suite_instructions(cfg.suite)
            names = self.dataset.meta.tasks.index
            keep = {int(self.dataset.meta.tasks.loc[name, "task_index"]) for name in names if name in wanted}
            if len(keep) != len(wanted):
                raise ValueError(f"only {len(keep)} of the {len(wanted)} {cfg.suite} instructions are in {cfg.repo_id}")
            self.frames = np.flatnonzero(np.isin(task_of_frame, sorted(keep)))
        else:
            self.frames = np.arange(len(task_of_frame))
        log.info("%d training frames (%s) of %d in %s", len(self.frames), cfg.suite or "all tasks",
                 len(task_of_frame), cfg.repo_id)

    def __len__(self) -> int:
        return len(self.frames)

    def sample(self, n: int, rng: "np.random.Generator") -> "np.ndarray":
        """``n`` dataset indices, uniformly over the kept frames (with replacement)."""
        return self.frames[rng.integers(0, len(self.frames), size=n)]

    def batch(self, indices: "Sequence[int]") -> dict[str, Any]:
        """The frames at ``indices`` as one FastWAM-preprocessed batch."""
        import torch

        # One batched read of just these columns: several times faster than item by item.
        columns = [*self.camera_keys, STATE_KEY, "task_index"]
        rows = self.dataset.hf_dataset.select_columns(columns)[[int(i) for i in indices]]
        raw: dict[str, Any] = {key: torch.stack(list(rows[key])) for key in [*self.camera_keys, STATE_KEY]}
        for key in self.camera_keys:
            if tuple(raw[key].shape[-2:]) != self.image_size:
                raw[key] = torch.nn.functional.interpolate(
                    raw[key].float(), size=self.image_size, mode="bilinear", align_corners=False, antialias=True
                ).clamp(0.0, 1.0)
        raw["task"] = [str(self.dataset.meta.tasks.iloc[int(t)].name) for t in rows["task_index"]]
        return self.preprocessor(raw)

    @staticmethod
    def row(batch: dict[str, Any], i: int) -> dict[str, Any]:
        """State ``i`` of a batch, as a batch of one."""
        import torch

        return {k: (v[i : i + 1] if isinstance(v, torch.Tensor) else [v[i]] if isinstance(v, list) else v)
                for k, v in batch.items()}
