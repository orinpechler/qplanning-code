# Q-gradient action optimization

FastWAM draws its action chunk as in the unguided baseline. That chunk is then refined by a
fixed number of gradient-ascent steps on the frozen Q-function, and the refined chunk is
executed instead of the policy's own output.

    a_0     = FastWAM(o, l)
    a_{k+1} = a_k + eta * grad_a Q(o, l, a_k)      k = 0 .. steps-1
    execute a_steps

Both networks stay frozen. The Q-function's parameters have `requires_grad=False`, the
observation context is encoded under `no_grad`, and the gradient is taken with respect to
the action tensor only. The ascent runs in FastWAM's normalized action space. A
per-dimension affine map, read once from the two checkpoints' normalizers, converts actions
into the Q-function's space.

```bash
# No simulator: checks that gradients reach only the action, that the differentiable Q path
# matches the planner's scoring, and that Q rises.
python -m qoptimizing.probe --config qoptimizing/configs/libero_10.yaml

# Full evaluation; any field can be overridden inline.
python -m qoptimizing.eval --config qoptimizing/configs/libero_10.yaml \
    --qopt.steps=10 --qopt.step_size=0.01 --run_name=libero_10_qopt_s10
```

SLURM wrappers are in `jobs/`. Outputs go to `<output_dir>/`:

- `episodes.csv`: the baseline columns plus per-episode means of `q_init`, `q_final`,
  `q_gain`, `delta_rms` and `grad_rms_init`.
- `qopt_steps.jsonl`: one line per planning step, with the full Q trajectory.
- `summary.json`

Settings (`qopt.*`, see `config.py`):

| field | default | meaning |
|---|---|---|
| `steps` | 10 | ascent steps; `0` reproduces the baseline |
| `step_size` | 0.01 | learning rate, in BC-normalized action units |
| `optimizer` | sgd | `sgd` or `adam` |
| `normalize_grad` | true | sgd: rescale the gradient to unit RMS per chunk |
| `max_delta` | null | per-element trust region around the policy's chunk |
| `n_samples` | 1 | chunks optimized in parallel; the one with the highest final Q is executed |
| `denoise_steps` | null | null uses `bc.baseline_denoise_steps` |
