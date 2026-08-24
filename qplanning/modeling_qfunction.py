"""The Q-function model, plus the one loading helper the fork does not provide.

``QFunctionPolicy`` is re-exported from the pinned LeRobot fork -- see
:mod:`qplanning.configuration_qfunction` for why the model is not redeclared here.

Architecture: a DINOv2 visual encoder and a T5 text encoder, both parameter-disjoint from
the policy, feed a transformer decoder that cross-attends to the observation tokens while
taking the candidate action chunk as query tokens. The head emits ``num_bins`` logits over a
grid of return values, and the scalar Q is the expectation under their softmax -- that
expectation is :func:`expected_value`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from lerobot.policies.q_function.modeling_q_function import (
    DINOv2ImageEncoder,
    QFunction,
    QFunctionPolicy,
    T5TextEncoder,
    _expected_value,
)

if TYPE_CHECKING:
    import torch
    from lerobot.processor import PolicyProcessorPipeline

# The fork spells this private; every caller here needs it.
expected_value = _expected_value

__all__ = [
    "DINOv2ImageEncoder",
    "QFunction",
    "QFunctionPolicy",
    "T5TextEncoder",
    "expected_value",
    "load_q_function",
]


def load_q_function(
    checkpoint: str | Path,
    device: "torch.device | str" = "cuda",
    *,
    eval_mode: bool = True,
) -> tuple[QFunctionPolicy, "PolicyProcessorPipeline"]:
    """Load a Q-function and the preprocessor it was trained with.

    The preprocessor has to come from the checkpoint rather than from the policy config:
    it carries the dataset normalization statistics as saved tensors, and those are what
    make an action scored at inference comparable to the actions seen during training.
    Building a fresh preprocessor instead would silently normalize with different stats.

    Returns:
        The policy (moved to ``device``) and its preprocessing pipeline.
    """
    import torch
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import batch_to_transition, transition_to_batch
    from lerobot.utils.constants import POLICY_PREPROCESSOR_DEFAULT_NAME

    checkpoint = str(checkpoint)
    policy = QFunctionPolicy.from_pretrained(checkpoint)
    policy = policy.to(torch.device(device) if isinstance(device, str) else device)
    if eval_mode:
        policy = policy.eval()

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=checkpoint,
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    return policy, preprocessor
