"""Q-function pre/post-processing.

Re-exported from the pinned LeRobot fork, with two names promoted from private to public.

The Q-function needs more than the usual observation/action pair: its loss reads the
per-chunk reward, a padding mask, a bootstrap-validity flag and a bucket index. LeRobot's
default ``batch_to_transition`` drops any key that is not an observation or an action, so
the fork carries a Q-specific converter pair that smuggles those through the transition's
complementary data. Training code has to use that pair rather than the default -- the
self-improvement loop in the fork already reaches into the underscored names to do so, which
is exactly why they are published here instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from lerobot.policies.q_function.processor_q_function import (
    DINOv2ImagePreprocessor,
    _q_batch_to_transition,
    _q_transition_to_batch,
    make_q_function_pre_post_processors,
)

if TYPE_CHECKING:
    from lerobot.processor import PolicyProcessorPipeline

q_batch_to_transition = _q_batch_to_transition
q_transition_to_batch = _q_transition_to_batch

__all__ = [
    "DINOv2ImagePreprocessor",
    "make_q_function_pre_post_processors",
    "make_q_train_preprocessor",
    "q_batch_to_transition",
    "q_transition_to_batch",
]


def make_q_train_preprocessor(checkpoint: str | Path) -> "PolicyProcessorPipeline":
    """Build the preprocessor used when *training* the Q-function.

    Identical to the inference preprocessor except for the converters: this one preserves
    the reward, padding, bootstrap and bucket keys the loss consumes. Using the inference
    preprocessor here instead produces a batch that is missing them, and the failure is a
    ``KeyError`` deep inside the loss rather than anything that names the cause.
    """
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.utils.constants import POLICY_PREPROCESSOR_DEFAULT_NAME

    return PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(checkpoint),
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        to_transition=q_batch_to_transition,
        to_output=q_transition_to_batch,
    )
