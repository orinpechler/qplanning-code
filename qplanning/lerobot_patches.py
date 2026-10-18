"""Runtime fixes to the pinned LeRobot fork, applied by this package's own entry points.

``LeRobotDataset._query_hf_dataset`` fetches the delta-timestamp window of each key with
``hf_dataset[indices][key]``. Row-first indexing formats *every* column of those rows, so for
a Q-function sample -- a 2h = 64-frame action window -- it decodes both camera images of all
64 frames just to read their actions: ~140 image decodes per sample instead of ~4. Reading
only the requested column returns the same values; it is the difference between seconds and
a fraction of a second per sample, i.e. between a data-bound and a GPU-bound training run.

Applied in-process by :mod:`qoptimizing.qrank`, and for training through
:mod:`qplanning.lerobot_train`, which wraps LeRobot's training script.
"""

from __future__ import annotations

_applied = False


def apply() -> None:
    """Install the fixes. Idempotent."""
    global _applied
    if _applied:
        return
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        result: dict = {}
        for key, q_idx in query_indices.items():
            if key in self.meta.video_keys:
                continue
            relative_indices = (
                q_idx
                if self._absolute_to_relative_idx is None
                else [self._absolute_to_relative_idx[idx] for idx in q_idx]
            )
            # select_columns is a zero-copy view: only `key` is formatted and decoded.
            values = self.hf_dataset.select_columns([key])[relative_indices][key]
            # As upstream: skip keys where any row lacks a value (optional features).
            if any(v is None for v in values):
                continue
            result[key] = torch.stack(values)
        return result

    LeRobotDataset._query_hf_dataset = _query_hf_dataset
    _applied = True
