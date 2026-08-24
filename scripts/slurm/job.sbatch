#!/usr/bin/env bash
# Generic SLURM wrapper. Deliberately carries NO site-specific directives -- no account, no
# partition, no QoS, no mail address. A stale account makes `sbatch` fail outright for
# anyone outside it, and a stale mail address quietly emails a stranger. All routing is
# supplied by scripts/slurm/submit.sh from a profile, on the sbatch command line.
#
# Do not run this directly; use:
#     scripts/slurm/submit.sh --profile <name> <subcommand> --config <config.yaml>
#
#SBATCH -N1
#SBATCH --output=slurm_out/%x-%j.out
#SBATCH --error=slurm_out/%x-%j.out
set -euo pipefail

REPO_ROOT="${QPLANNING_REPO_ROOT:?submit.sh must export QPLANNING_REPO_ROOT}"
echo "job     : ${SLURM_JOB_ID:-none} on $(hostname)"
echo "gpus    : ${SLURM_GPUS_ON_NODE:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo

# The SLURM path and the plain-bash path run the same script, by construction.
exec bash "$REPO_ROOT/scripts/run.sh" ${QPLANNING_ARGS}
