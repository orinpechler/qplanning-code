# Running a full evaluation

A complete comparison has three conditions per benchmark: the policy alone, the policy under
Q-guided selection, and the policy after self-improvement.

## Where each column comes from

The two Q-guided columns both come from a single self-improvement run. Each iteration rolls
out a **held-out block** whose seeds are fixed across iterations and which is never added to
the replay buffer. Iteration 0 measures the Q-function before any online updates; the final
iteration measures it after. Measuring this way means the two columns share a protocol
exactly and that no reported episode was ever trained on.

`configs/report.yaml` declares which files feed which column.

## Initial states (LIBERO)

Every LIBERO task ships 50 fixed initial states, and the environment restores one of them on
every reset, so the reset seed does not change the layout. Each episode therefore picks its
state explicitly: by default episode *e* starts from state *e*, so 50 episodes per task
cover every layout once (the standard LIBERO protocol), and 20 cover states 0–19. Set
`eval.init_state_ids` to evaluate on a chosen subset, e.g. a held-out split. The state each
episode used is recorded in the `init_state` column of `episodes.csv`.

The first release did not do this: every episode of a task started from state 0, and only
the policy's sampling noise varied. `--eval.init_state_ids=[0]` reproduces that.

In self-improvement the held-out and training blocks draw from disjoint states (by default
0–19 held out, the remaining 30 for training), and an overlapping split is rejected.

## Commands

```bash
# The unguided policy, per benchmark.
qplanning eval --config configs/libero_spatial/baseline.yaml
qplanning eval --config configs/libero_object/baseline.yaml
qplanning eval --config configs/libero_goal/baseline.yaml
qplanning eval --config configs/libero_10/baseline.yaml
qplanning eval --config configs/robotwin/baseline.yaml

# Q-guided, before and after self-improvement.
qplanning self-improve --config configs/libero_spatial/self_improve.yaml
# ... and so on

qplanning report --config configs/report.yaml --format latex
```

A standalone Q-guided evaluation is also available and is the cheaper way to try planner
settings without running a full loop:

```bash
qplanning eval --config configs/libero_10/qplanning_offline.yaml
```

## Sharding

RoboTwin's tasks are slow, and holding many simulators open exhausts host memory, so
collection is meant to be split across jobs:

```bash
for k in $(seq 0 15); do
  qplanning self-improve --config configs/robotwin/self_improve.yaml \
      --stage collect --eval.shard=$k/16
done
qplanning self-improve --config configs/robotwin/self_improve.yaml --stage finetune
```

Seeds derive from a task's index in the full task list, so a sharded run evaluates exactly
the episodes an unsharded one would.

If you chain these as scheduler jobs, make the update job depend on the collection jobs
**completing rather than succeeding**, and let the shard-coverage check decide whether there
is enough data. A dependency requiring success will strand the chain the first time a
preemptible job is preempted.

## A cheap smoke run first

```bash
qplanning eval --config configs/libero_10/qplanning_offline.yaml \
    --eval.episodes_per_task=2 --planner.n_samples=8 --benchmark.episode_length=60
```

Short episodes over a few tasks. It will not produce a meaningful success rate, but it
exercises the same path end to end.

## What a small run can and cannot tell you

Cheap runs detect breakage, not effect sizes. Differences between conditions are typically a
few percentage points, which is close to the binomial standard error at a couple of hundred
episodes per benchmark; a twenty-episode run cannot resolve them. Use the episode counts in
the configs when the number matters.
