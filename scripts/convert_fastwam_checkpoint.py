#!/usr/bin/env python
"""Convert an official FastWAM ``.pt`` checkpoint into the pinned LeRobot fork's format.

    python scripts/convert_fastwam_checkpoint.py \\
        --checkpoint    libero_uncond_2cam224.pt \\
        --dataset-stats libero_uncond_2cam224_dataset_stats.json \\
        --output-dir    fastwam_libero

Why this script exists
----------------------
The fork ships ``scripts/convert_fastwam_checkpoint.py`` too, but it calls
``FastWAMPolicy.from_fastwam_checkpoint``, which is defined on no branch or tag of the fork,
so it cannot run. This is a standalone replacement.

What it writes
--------------
``config.json``, ``model.safetensors`` and the pre/post-processor files -- the directory
``bc.checkpoint`` expects. Only the fine-tuned weights go into ``model.safetensors`` (the MoT
video + action experts and the proprio encoder, ~12 GB). The VAE, T5 text encoder and
tokenizer are deliberately left out: ``FastWAMPolicy.__init__`` loads them from the Wan2.2
directory (``QPLANNING_WAN22_DIR``) every time, and the fork loads checkpoints with
``strict=False``.

The architecture and scheduler settings are the fork's ``FastWAMConfig`` defaults, which
match the released LIBERO checkpoint (uploaded 2026-03-23, trained with action shift 5.0 --
upstream changed new runs to 1.0 only in August). Two fields are overridden:
``redirect_common_files=False``, because the DiffSynth safetensors mirror the default points
at is not publicly accessible, and ``load_text_encoder=True``, because LIBERO tasks are only
distinguishable by their instruction.

Nothing here needs a GPU or the Wan2.2 weights. Every tensor name and shape is checked
against the fork's MoT built on the meta device, so a mismatch fails here rather than as a
silently random-initialised layer at evaluation time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# The fine-tuned part of the model, as named inside FastWAMPolicy (`policy.model.<...>`).
_MOT_PREFIX = "model.mot."
_PROPRIO_PREFIX = "model.proprio_encoder."

# Upstream treats a dimension whose range is below this as constant (normalizer.py).
_RANGE_TOL = 1e-4


def _min_max(stats: dict, section: str, expected_dim: int) -> dict[str, torch.Tensor]:
    """Pull global min/max for `section` out of upstream's dataset_stats.json.

    Upstream nests stats as ``{section: {key: {"global_min": [...], ...}}}``, with one key
    (``default``) for LIBERO's single merged action/state vector.
    """
    if section not in stats:
        raise ValueError(f"dataset stats have no '{section}' section; top-level keys: {sorted(stats)}")
    fields = stats[section]
    if len(fields) != 1:
        raise ValueError(f"expected a single merged '{section}' key, found {sorted(fields)}")
    (key, entry), = fields.items()

    out = {}
    for name in ("min", "max"):
        value = entry.get(f"global_{name}", entry.get(name))
        if value is None:
            raise ValueError(f"'{section}.{key}' has no global_{name}; keys: {sorted(entry)}")
        out[name] = torch.tensor(value, dtype=torch.float32).flatten()
        if out[name].numel() != expected_dim:
            raise ValueError(f"'{section}.{key}.{name}' has {out[name].numel()} dims, expected {expected_dim}")

    flat = (out["max"] - out["min"]) < _RANGE_TOL
    if flat.any():
        # LeRobot's MIN_MAX and upstream's normalizer disagree on constant dimensions.
        print(f"warning: {section} dims {flat.nonzero().flatten().tolist()} are (near-)constant; "
              "LeRobot normalises these differently from upstream FastWAM", file=sys.stderr)
    return out


def _expected_tensors(config) -> dict[str, torch.Size]:
    """Every fine-tuned tensor the fork's FastWAM expects, built without allocating weights."""
    from lerobot.policies.fastwam.wan22.action_dit import ActionDiT
    from lerobot.policies.fastwam.wan22.helpers.loader import _validate_dit_config
    from lerobot.policies.fastwam.wan22.mot import MoT
    from lerobot.policies.fastwam.wan22.wan_video_dit import WanVideoDiT

    with torch.device("meta"):
        video = WanVideoDiT(**_validate_dit_config(config.get_video_dit_config()))
        action = ActionDiT(**config.get_action_dit_config())
        mot = MoT(mixtures={"video": video, "action": action}, mot_checkpoint_mixed_attn=False)
        proprio = torch.nn.Linear(config.state_dim, config.video_dit_text_dim)

    expected = {_MOT_PREFIX + k: t.shape for k, t in mot.state_dict().items()}
    expected.update({_PROPRIO_PREFIX + k: t.shape for k, t in proprio.state_dict().items()})
    return expected


def _load_upstream_weights(path: Path) -> dict[str, torch.Tensor]:
    """Read upstream's ``{"mot": ..., "proprio_encoder": ...}`` payload under the fork's names."""
    # weights_only: the payload is tensors plus a step count and a dtype string, so the safe
    # unpickler suffices and nothing in a downloaded file gets to execute.
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(payload, dict) or "mot" not in payload:
        keys = sorted(payload) if isinstance(payload, dict) else type(payload).__name__
        raise ValueError(f"not an upstream FastWAM checkpoint (no 'mot' entry): {keys}")
    if "proprio_encoder" not in payload:
        raise ValueError("checkpoint has no 'proprio_encoder' weights, but LIBERO FastWAM uses proprio")

    print(f"checkpoint: step={payload.get('step')} dtype={payload.get('torch_dtype')}")
    tensors = {_MOT_PREFIX + k: v for k, v in payload["mot"].items()}
    tensors.update({_PROPRIO_PREFIX + k: v for k, v in payload["proprio_encoder"].items()})
    return tensors


def _check_against_fork(tensors: dict[str, torch.Tensor], expected: dict[str, torch.Size]) -> None:
    missing = sorted(set(expected) - set(tensors))
    unexpected = sorted(set(tensors) - set(expected))
    wrong_shape = sorted(k for k in set(expected) & set(tensors) if tensors[k].shape != expected[k])
    if missing or unexpected or wrong_shape:
        lines = ["checkpoint does not match the fork's FastWAM architecture:"]
        for label, keys in (("missing", missing), ("unexpected", unexpected), ("wrong shape", wrong_shape)):
            if keys:
                lines.append(f"  {label} ({len(keys)}): {', '.join(keys[:5])}{' ...' if len(keys) > 5 else ''}")
        raise ValueError("\n".join(lines))
    print(f"architecture check: all {len(expected)} tensors match by name and shape")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="official FastWAM .pt, e.g. libero_uncond_2cam224.pt")
    parser.add_argument("--dataset-stats", type=Path, required=True,
                        help="the matching *_dataset_stats.json from the same release")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="where to write the converted checkpoint (becomes bc.checkpoint)")
    parser.add_argument("--device", default="cuda",
                        help="device recorded in config.json for evaluation (conversion itself runs on CPU)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output directory's files")
    args = parser.parse_args()

    for path in (args.checkpoint, args.dataset_stats):
        if not path.is_file():
            parser.error(f"no such file: {path}")
    if (args.output_dir / "model.safetensors").exists() and not args.overwrite:
        parser.error(f"{args.output_dir} already holds a checkpoint; pass --overwrite to replace it")

    from safetensors.torch import save_file

    from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig
    from lerobot.policies.fastwam.processor_fastwam import make_fastwam_pre_post_processors
    from lerobot.utils.constants import ACTION, OBS_STATE

    config = FastWAMConfig(device=args.device, redirect_common_files=False, load_text_encoder=True)
    config.validate_features()

    stats_json = json.loads(args.dataset_stats.read_text())
    dataset_stats = {
        ACTION: _min_max(stats_json, "action", config.action_dim),
        OBS_STATE: _min_max(stats_json, "state", config.state_dim),
    }

    tensors = _load_upstream_weights(args.checkpoint)
    _check_against_fork(tensors, _expected_tensors(config))

    dtype = getattr(torch, config.dtype)
    tensors = {k: v.to(dtype).contiguous() for k, v in tensors.items()}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output_dir / "model.safetensors"), metadata={"format": "pt"})
    config._save_pretrained(args.output_dir)
    preprocessor, postprocessor = make_fastwam_pre_post_processors(config, dataset_stats=dataset_stats)
    preprocessor.save_pretrained(args.output_dir)
    postprocessor.save_pretrained(args.output_dir)

    print(f"wrote {args.output_dir}")
    print(f"  chunk_size={config.chunk_size} n_action_steps={config.n_action_steps} "
          f"action_infer_shift={config.action_infer_shift} cameras={config.num_cameras}x{config.image_size}")
    print(f"\nset in .env:\n  QPLANNING_BC_CKPT_LIBERO={args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
