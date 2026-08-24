"""Evaluation-parity settings.

The reported success rates were measured with deterministic kernels, TF32 disabled and
cuDNN autotuning off. Those settings cost throughput but make a run reproducible, and mixing
them with the faster deployment settings across conditions would make two columns of a
results table incomparable. So they are the default here, and turning them off is an
explicit choice recorded in the config.
"""

from __future__ import annotations

import os


def apply(deterministic: bool = True, seed: int | None = None) -> None:
    """Configure PyTorch for reproducible evaluation."""
    import torch

    if deterministic:
        # Required by cuBLAS before deterministic algorithms can be enabled; it must be set
        # before the first CUDA context is created, which is why this runs early.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if seed is not None:
        from lerobot.utils.random_utils import set_seed

        set_seed(seed)
