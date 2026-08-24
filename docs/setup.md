# Setup

Budget roughly **70 GB of downloads and 2–4 hours** before the first evaluation runs. Most
of that is model weights: the policy checkpoints are 13 GB (LIBERO) and 25 GB (RoboTwin),
the two Q-functions are 5 GB each, and the Wan2.2 base weights are another 24 GB.

## Requirements

- Linux, Python 3.10+
- One CUDA GPU with **≥ 32 GB** for evaluation (policy ≈ 25 GB + Q-function ≈ 4 GB + a
  planning batch). Q-function training at the default batch size wants ≥ 80 GB.
- Compute capability **8.0 or newer**. The policy runs in bfloat16, which older cards do
  not support natively, and a PyTorch build must include your card's architecture — check
  with `python -c "import torch; print(torch.cuda.get_arch_list())"` before submitting
  anything long. A card newer than the build (for example a Blackwell card on a cu126
  build) fails with `no kernel image is available for execution on the device`.
- A working Vulkan driver, for RoboTwin only.

## 1. Base environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[libero]"
```

This pulls the pinned LeRobot fork, which supplies the frozen policy, the dataset format,
the processor pipelines and the environment wrappers. It is pinned to an exact commit rather
than a branch, because the branch keeps moving.

`sentencepiece` comes with it and is required — the Q-function's text encoder uses a slow
tokenizer and fails with an unhelpful `ImportError` without it.

## 2. LIBERO

Installed by the `libero` extra. LIBERO renders headlessly through EGL:

```bash
export MUJOCO_GL=egl        # qplanning sets this itself, but tools you run directly need it
python -c "from libero.libero import benchmark; print(sorted(benchmark.get_benchmark_dict()))"
```

## 3. RoboTwin

RoboTwin needs **its own environment**. Its stack predates NumPy 2 and cannot be resolved by
pip alongside LeRobot's dependencies.

```bash
python -m venv .venv-robotwin && source .venv-robotwin/bin/activate
pip install -e ".[robotwin]"
```

Then, in that environment:

1. `sapien==3.0.0b1`, `mplib==0.2.1`, `toppra`, `transforms3d==0.4.2`, `trimesh==4.4.3`,
   `open3d`, `pyglet<2`.
2. **curobo v0.7.8 from source**, built against your CUDA (`--no-build-isolation`,
   `FORCE_CUDA=1`, and `TORCH_CUDA_ARCH_LIST` matching your GPU: `8.0` A100, `8.9` L40S,
   `9.0` H100/H200). This is not optional — RoboTwin's robot module imports its planner
   unconditionally, so without it no task will import at all.
3. `warp-lang`: curobo 0.7.8 needs the auto-imported `warp.torch` attribute that later
   releases removed. Verify with `python -c "import warp as wp; wp.torch"`.
4. Two upstream patches RoboTwin's own installer applies, both silently required: SAPIEN's
   URDF loader must open files as UTF-8 (robot descriptions contain non-ASCII characters),
   and mplib's planner must not bail out of inverse kinematics on a collision check, which
   otherwise makes contact-rich tasks unsolvable.
5. Assets: `python assets/_download.py` (~16 GB).
6. `export ROBOTWIN_ROOT=/path/to/RoboTwin` and confirm rendering works —
   `python script/test_render.py` should print `Render Well`.

## 4. Weights and data

```bash
qplanning fetch-checkpoints
```

Reads `configs/checkpoints.yaml` and downloads each repository. Until the weights are
published, point the variables in your `.env` at local directories instead — every
checkpoint field accepts a path as readily as a repo id.

The **Wan2.2 base weights are separate from the policy checkpoint** and are easy to miss.
Fetch `Wan-AI/Wan2.2-TI2V-5B`, `DiffSynth-Studio/Wan-Series-Converted-Safetensors` and
`Wan-AI/Wan2.1-T2V-1.3B` (tokenizer only) into one directory and point
`QPLANNING_WAN22_DIR` at it. A partial download surfaces as
`Cannot detect model type ... Model hash:`, which does not obviously mean "incomplete
download" unless you already know.

Datasets are LeRobotDataset directories. **Check the RoboTwin one:** a 300-episode subset
exists with the same layout and camera keys as the full 27,500-episode dataset, and training
on it silently produces a much worse Q-function. `qplanning doctor` checks this for you.

## 5. Verify

```bash
qplanning doctor --config configs/libero_10/qplanning_offline.yaml   # paths, no GPU
qplanning probe  --config configs/libero_10/qplanning_offline.yaml   # models, no simulator
qplanning eval   --config configs/libero_10/baseline.yaml \
    --eval.episodes_per_task=1 --benchmark.episode_length=20         # a real rollout
```

Take them in that order. Each depends on the previous one, and a failure in the earlier,
cheaper step is far easier to read.

## Without a scheduler

Nothing here requires SLURM. `scripts/run.sh <subcommand> --config <config>` is the primary
path; the SLURM wrapper execs that same script.

On a single GPU the loop runs in-process — leave `self_improve.collect_shards` at 1 and let
`--stage all` iterate. Expect a full RoboTwin self-improvement run to take on the order of
GPU-weeks and ~250 GB of accumulated rollouts; LIBERO suites are far cheaper. Host RAM
matters as much as VRAM during collection: budget 64 GB.
