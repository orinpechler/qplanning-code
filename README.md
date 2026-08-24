# Q-Planning

Value-guided action selection and self-improvement for frozen behaviour-cloning policies.

Q-Planning pairs a behaviour-cloning policy with an off-policy Q-function over action
chunks. At inference the policy proposes several candidate chunks and the Q-function decides
which to execute. Over time the Q-function is refined on deployment rollouts — including
failed ones, which imitation cannot use — while the policy stays frozen.

The policy is treated as a black box that can draw action chunks, so any policy satisfying a
small interface can be used. FastWAM is provided as the default.

## Install

```bash
git clone <this-repo> && cd qplanning_release
python -m venv .venv && source .venv/bin/activate
pip install -e ".[libero]"          # or ".[robotwin]"
cp .env.example .env && $EDITOR .env
```

`.env` holds every path the tool needs — checkpoints, datasets, an output directory.
Nothing is hardcoded, and a checkpoint field accepts a local directory or a Hugging Face
repo id interchangeably.

RoboTwin needs a separate environment (SAPIEN, curobo); see [docs/setup.md](docs/setup.md).

Then check the install:

```bash
qplanning doctor --config configs/libero_10/qplanning_offline.yaml   # resolve paths, no GPU
qplanning probe  --config configs/libero_10/qplanning_offline.yaml   # load models, no simulator
```

`doctor` reports which paths are missing before you queue anything long. `probe` loads the
policy and Q-function and runs one planning step on a synthetic observation, so a failure
here is a setup problem rather than an environment problem.

## Usage

### Evaluate

```bash
# Policy alone.
qplanning eval --config configs/libero_10/baseline.yaml

# Policy under Q-guided selection.
qplanning eval --config configs/libero_10/qplanning_offline.yaml
```

Writes one row per episode to `<output_dir>/episodes.csv` — task, success, episode length in
environment steps — plus a `summary.json` and the fully-resolved config used.

Split a long evaluation across jobs with `--eval.shard=k/n`; the shards write separate CSVs
that the reporting step concatenates.

### Self-improvement

```bash
qplanning self-improve --config configs/libero_10/self_improve.yaml
```

Each iteration rolls out the planner, appends every episode to a replay buffer, and updates
only the Q-function. Two blocks run per task per iteration: a held-out block on fixed seeds
that is never trained on and serves as the metric, and a training block on advancing seeds
that feeds the buffer.

Collection can be fanned out and the update run separately:

```bash
qplanning self-improve --config configs/robotwin/self_improve.yaml --stage collect --eval.shard=0/16
qplanning self-improve --config configs/robotwin/self_improve.yaml --stage finetune
```

### Train a Q-function

```bash
qplanning train-q --config configs/libero_10/train_q.yaml
```

Trains on demonstrations — the same ones the policy was trained on; no extra data is
collected. Multi-GPU by default; `--train_q.num_gpus=1` for a single device, `--dry-run` to
print the command without running it.

### Report

```bash
qplanning report --config configs/report.yaml --format markdown   # or latex, csv
```

Aggregates per-episode CSVs into a table: per-task success rate, per-task mean length over
successful episodes, averaged across tasks. Columns are restricted to the tasks all of them
evaluated, so a task that failed in one condition cannot change another's denominator.

Nothing requires a scheduler. `scripts/run.sh` is the primary entry point; `scripts/slurm/`
holds optional wrappers whose job script carries no site-specific settings, so adapting them
means writing one small profile file.

Running a full multi-condition comparison, including sharding across jobs, is covered in
[docs/evaluation.md](docs/evaluation.md).

## Configuration

One YAML describes a run. Benchmark configs extend `configs/base.yaml` and override only
what differs. Any field can be overridden inline:

```bash
qplanning eval --config configs/libero_10/qplanning_offline.yaml \
    --planner.n_samples=16 --eval.episodes_per_task=5
```

The planner is the main thing to tune:

```yaml
planner:
  type: q_weighted    # or `none` to run the policy unguided
  n_samples: 64       # candidates drawn per planning step
  n_elites: 16        # how many of the best are averaged (0 = all)
  temperature: 1.0    # softmax temperature over Q-values
  denoise_steps: 3    # sampler steps per candidate; low keeps candidates diverse
```

Full reference in [docs/configuration.md](docs/configuration.md); common failures and what
they mean are in [docs/troubleshooting.md](docs/troubleshooting.md).

## Using your own policy

Implement the interface and register the class:

```python
from qplanning.policies.registry import register_bc

@register_bc("my_policy")
class MyAdapter:
    chunk_size: int   # must match the Q-function's horizon
    action_dim: int

    def reset(self): ...              # called at the start of every episode
    def to(self, device): ...
    def parameters(self): ...         # never optimized; Q-Planning leaves the policy frozen

    def sample_chunks(self, batch, n_samples, *, denoise_steps=None, generator=None):
        """(n_samples, chunk_size, action_dim), in your policy's normalized action space.
        Encode the observation once and repeat only the sampling — otherwise planning cost
        grows linearly with n_samples."""

    def unnormalize_actions(self, actions):
        """Into raw environment units. Required: your policy and the Q-function hold
        different action statistics, and scoring un-converted actions ranks candidates
        wrongly while looking plausible."""
```

Set `bc.type: my_policy`. Third-party packages can register through the
`qplanning.bc_adapters` entry-point group without editing this repository. The full protocol
is in `qplanning/policies/base.py`; `qplanning/policies/fastwam.py` is a worked example.

## Repository layout

```
configs/            one YAML per benchmark and stage; base.yaml holds shared defaults
scripts/
  run.sh            the entry point; scripts/slurm/ wraps it for a scheduler
qplanning/
  cli.py            subcommand dispatch
  planner.py        candidate scoring and the Q-weighted average
  policy.py         the rollout policy: sampler + planner + action queue
  policies/         the policy interface, and the FastWAM adapter
  eval.py           rollouts, one row per episode
  self_improve.py   collect, refine the Q-function, repeat
  train_q.py        Q-function training on demonstrations
  report.py         per-episode results to an aggregate table
  doctor.py         path resolution; probe.py loads the models without a simulator
  config/           schema, inheritance, interpolation, path resolution
  data/             replay buffer and the demonstration/online mixture
  envs.py           benchmark environments; benchmarks/ holds the task inventories
  *_qfunction.py    configuration, model and processor for the Q-function
docs/               setup, configuration, evaluation, troubleshooting
tests/
```

## Acknowledgements

Built on [LeRobot](https://github.com/huggingface/lerobot). FastWAM provides the default
policy; LIBERO and RoboTwin provide the benchmarks.

## Citation

```bibtex
@inproceedings{giridhar2026qplanning,
  title     = {Beyond Imitation: Self-Improving Robot Policies via Off-Policy Q-Planning},
  author    = {Giridhar, Varun and Khandelwal, Anant and Collins, Jeremy A.
               and Georgiev, Ignat and Garg, Animesh},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026},
}
```

Apache-2.0. See [LICENSE](LICENSE).
