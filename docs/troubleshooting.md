# Troubleshooting

Failures seen in practice, and what they actually mean.

**`Cannot detect model type ... Model hash:`**
The Wan2.2 base weights are missing or incomplete. They are *not* inside the policy
checkpoint — see [setup.md](setup.md#4-weights-and-data). Check `QPLANNING_WAN22_DIR`.

**`no kernel image is available for execution on the device`**
Your PyTorch build does not include your GPU's architecture. Compare
`torch.cuda.get_arch_list()` with `torch.cuda.get_device_capability()`. A card newer than
the build (a Blackwell card on a cu126 build, say) needs a newer PyTorch.

**`the Q-function scores chunks of length X but this BC policy emits chunks of length Y`**
The two were trained with different horizons and cannot be combined. Use a matching pair.

**`Q-Planning runs with one environment at a time`**
The planner scores candidates for a single state. Batch size is fixed at 1.

**`the Q-values are effectively constant`** (from `qplanning probe`)
The Q-function returns the same score for every candidate, so the weighted average is just
a mean and value guidance is doing nothing. Usually the Q-function does not match the
benchmark — check `qfunction.camera_keys` and that the checkpoint was trained for this
environment. This one is worth taking seriously: it does not crash, and a success rate
alone will not reveal it.

**`only N readable shard(s) ... need at least M`**
A collection job died mid-write, leaving a truncated file. Re-run the missing shards rather
than training on a partial iteration.

**`RoboTwin setup_demo failed (attempt k/12, seed N)`**
Benign. The scene was unstable and is being re-seeded. Only `after 12 attempts` is fatal —
do not treat a bare traceback in RoboTwin logs as failure.

**`AttributeError: ... 'arm_tag'` on `open_laptop`, `place_object_scale`,
`put_object_cabinet`**
Known upstream limitation; these three tasks cannot be evaluated with a learned policy and
are excluded by default; see `qplanning/benchmarks/robotwin.py`.

**`Trying to resize storage that is not resizable`**
Online and demonstration images have different shapes. The buffer is written to match the
demonstration dataset, so check `self_improve.offline.root` points at the right dataset.

**Out of memory during self-improvement**
The policy is parked on the CPU during the update, but the Q-function's gradients, optimiser
state and activations still need room. Lower `self_improve.batch_size`, or run the
`collect` and `finetune` stages as separate jobs — a finetune-only job never loads the
policy at all.

**A read-only Hugging Face cache**
`datasets` needs to take file locks. `HF_HOME` must be writable.
