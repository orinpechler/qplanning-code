# Configuration

One YAML describes a run. `configs/base.yaml` holds the defaults; each benchmark config
extends it and overrides only what differs.

## Precedence

Lowest to highest: `base.yaml` → the `extends` chain → the named config → `.env` → real
environment variables → CLI overrides. So a value exported in your shell beats `.env`, and
`--planner.n_samples=8` beats everything.

Every run writes `resolved.yaml` into its output directory, recording exactly what it ran
with.

## Inheritance

```yaml
extends: ../base.yaml       # relative to this file; a list is allowed
```

Mappings merge, scalars and lists replace. Cycles are detected and reported.

## Interpolation

| Form | Meaning |
|---|---|
| `${env:VAR}` | required environment variable; a clear error names it if unset |
| `${env:VAR:default}` | with a fallback |
| `${cfg:paths.root}` | another field of this config; keeps its type when it is the whole value |

A reference that never resolves is an error rather than a string left verbatim in the
config — that failure mode is silent and expensive.

## Paths

Any path field accepts:

- a local directory, absolute or `~`-relative;
- `org/name`, treated as a Hugging Face repo and downloaded on first use;
- anything else, resolved **against the repository root**, never the working directory —
  so a config means the same thing from an interactive shell and a batch job.

## The fields that matter

```yaml
planner:
  type: q_weighted    # or `none` for the unguided policy — this one field is the
                      # difference between the first two columns of the results table
  n_samples: 64       # N: candidates per planning step
  n_elites: 16        # K: how many are averaged (0 = all N); clamped to N
  temperature: 1.0    # lambda in the softmax weighting
  denoise_steps: 3    # sampler steps per candidate; low on purpose, for diversity

execution:
  n_action_steps: 10  # steps executed before replanning

eval:
  episodes_per_task: 20
  init_state_ids: null  # LIBERO: null = episode e starts from initial state e (of 50);
                        # a list = cycle through those states, e.g. [30, 31, ..., 49]

self_improve:
  iterations: 10
  heldout_episodes_per_task: 20   # fixed seeds, never trained on: this is the measurement
  train_episodes_per_task: 10     # advancing seeds, appended to the buffer
  heldout_init_state_ids: null    # LIBERO: null = states 0..19
  train_init_state_ids: null      # LIBERO: null = all other states, advancing per iteration
  finetune_steps: 200             # S, per iteration
  online_fraction: 0.5            # online share of each minibatch
  collect_shards: 1               # raise to fan collection across jobs
```

Fields under `qfunction` that describe architecture (`h`, `num_bins`, `dim_model`,
`camera_keys`, …) are properties of a trained checkpoint. They are listed so a config is
self-documenting, and are checked against the checkpoint at load time — a mismatch is
reported by name rather than surfacing later as a shape error.

## Plugging in your own policy

Implement `sample_chunks` and `unnormalize_actions` (see `qplanning/policies/base.py`),
register with `@register_bc("name")`, and set `bc.type: name`. Two things to get right:

- **Amortise the observation encoding across the `n_samples` draws.** Re-encoding per
  candidate is correct but makes planning cost grow with `N`.
- **`unnormalize_actions` must be real.** Your policy and the Q-function were trained on
  different data and hold different action statistics; returning actions unchanged produces
  Q-values that look plausible and rank candidates wrongly.
