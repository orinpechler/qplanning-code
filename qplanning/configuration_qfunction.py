"""Q-function configuration.

The dataclass itself is re-exported from the pinned LeRobot fork rather than redeclared.
That is deliberate: ``QFunctionPolicy.from_pretrained`` resolves a checkpoint's
``type: q_function`` through LeRobot's config registry, so a second declaration here would
either collide on registration or quietly fork the on-disk checkpoint format -- which
becomes unfixable once weights are published.

What this module adds is a check that a config and a checkpoint actually agree.
"""

from __future__ import annotations

from lerobot.policies.q_function.configuration_q_function import QFunctionConfig

__all__ = ["QFunctionConfig", "CheckpointMismatch", "assert_matches_checkpoint"]


class CheckpointMismatch(RuntimeError):
    """A config disagrees with the checkpoint it is being used with."""


# Fields baked into trained weights: disagreeing with the checkpoint means the config
# describes a model that does not exist.
_STRUCTURAL_FIELDS = (
    "h",
    "num_bins",
    "v_min",
    "v_max",
    "dim_model",
    "n_heads",
    "dim_feedforward",
    "n_decoder_layers",
    "dino_model_name",
    "text_encoder_model",
)


def assert_matches_checkpoint(yaml_cfg, ckpt_cfg: QFunctionConfig) -> None:
    """Fail loudly when a config and a loaded checkpoint disagree on architecture.

    Without this the mismatch surfaces much later as a shape error inside the decoder, or --
    worse -- not at all, because ``PreTrainedPolicy.from_pretrained`` loads with
    ``strict=False`` and will happily leave mismatched tensors at their initial values.
    """
    problems = []
    for field in _STRUCTURAL_FIELDS:
        want = getattr(yaml_cfg, field, None)
        if want is None:
            continue
        got = getattr(ckpt_cfg, field, None)
        if got is not None and want != got:
            problems.append(f"  qfunction.{field}: config says {want!r}, checkpoint has {got!r}")

    want_cams = tuple(getattr(yaml_cfg, "camera_keys", ()) or ())
    got_cams = tuple(getattr(ckpt_cfg, "camera_keys", ()) or ())
    if want_cams and got_cams and want_cams != got_cams:
        problems.append(
            f"  qfunction.camera_keys: config says {list(want_cams)}, checkpoint has {list(got_cams)}"
        )

    if problems:
        raise CheckpointMismatch(
            "the Q-function config does not describe this checkpoint:\n"
            + "\n".join(problems)
            + "\n\nThese fields are properties of the trained weights. Either point at a "
            "different checkpoint or drop the overrides from your config."
        )
