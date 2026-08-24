#!/usr/bin/env bash
# Run one Q-Planning command. This is the primary entry point: nothing here needs SLURM.
#
#   scripts/run.sh <subcommand> --config <config.yaml> [--field.name=value ...]
#   scripts/run.sh eval --config configs/libero_10/baseline.yaml
#
# The SLURM path (scripts/slurm/job.sbatch) execs this same script, so a batch job and an
# interactive run execute identical code and cannot drift apart.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ "$#" -lt 1 ]; then
    echo "usage: scripts/run.sh <subcommand> --config <config.yaml> [overrides...]" >&2
    echo "subcommands: doctor, probe, train-q, eval, self-improve, report," >&2
    echo "             fetch-checkpoints, validate-config" >&2
    exit 2
fi

# Which interpreter to use. Point QPLANNING_PYTHON at a specific one when the environment
# is not already active (conda activate is unreliable in non-interactive shells).
PYTHON="${QPLANNING_PYTHON:-python}"

# Log to a file without `tee`, so the exit status is the command's own rather than tee's.
if [ -n "${QPLANNING_LOG:-}" ]; then
    mkdir -p "$(dirname "$QPLANNING_LOG")"
    exec > >(cat >"$QPLANNING_LOG") 2>&1
fi

echo "repo    : $REPO_ROOT"
echo "python  : $($PYTHON -c 'import sys; print(sys.executable)')"
echo "command : qplanning $*"
echo

exec "$PYTHON" -m qplanning.cli "$@"
