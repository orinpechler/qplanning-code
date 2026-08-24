#!/usr/bin/env bash
# Submit a Q-Planning command to SLURM.
#
#   scripts/slurm/submit.sh --profile <name> [--name <job>] [--dry-run] \
#       <subcommand> --config <config.yaml> [overrides...]
#
# Cluster routing lives in scripts/slurm/profiles/<name>.env, never in the job script, so
# adapting this to another cluster means writing one small profile rather than editing a
# batch file.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROFILE=""; JOB_NAME=""; DRY_RUN=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --profile) PROFILE="$2"; shift 2 ;;
        --profile=*) PROFILE="${1#*=}"; shift ;;
        --name) JOB_NAME="$2"; shift 2 ;;
        --name=*) JOB_NAME="${1#*=}"; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) break ;;
    esac
done

if [ -z "$PROFILE" ]; then
    cat >&2 <<MSG
No SLURM profile given.

Copy scripts/slurm/profiles/generic.env, set QP_ACCOUNT / QP_PARTITION / QP_QOS /
QP_GRES / QP_TIME for your cluster, and pass --profile <yours>.

--profile also accepts a path, so the file can live outside this repository.

Available: $(ls "$REPO_ROOT/scripts/slurm/profiles" 2>/dev/null | sed 's/\.env$//' | tr '\n' ' ')
MSG
    exit 2
fi

# A profile may be a name under scripts/slurm/profiles/, or a path to a file anywhere --
# the latter lets you keep cluster credentials and account names outside the repository.
if [ -f "$PROFILE" ]; then
    PROFILE_PATH="$PROFILE"
else
    PROFILE_PATH="$REPO_ROOT/scripts/slurm/profiles/${PROFILE}.env"
fi
[ -f "$PROFILE_PATH" ] || { echo "no such profile: $PROFILE_PATH" >&2; exit 2; }
# shellcheck disable=SC1090
source "$PROFILE_PATH"

[ "$#" -ge 1 ] || { echo "usage: submit.sh --profile <p> <subcommand> --config <cfg>" >&2; exit 2; }
JOB_NAME="${JOB_NAME:-qp_$1}"

mkdir -p "$REPO_ROOT/slurm_out"

SBATCH_ARGS=(
    --job-name="$JOB_NAME"
    --chdir="$REPO_ROOT"
    --export="ALL,QPLANNING_REPO_ROOT=$REPO_ROOT,QPLANNING_ARGS=$*"
)
[ -n "${QP_ACCOUNT:-}"   ] && SBATCH_ARGS+=(--account="$QP_ACCOUNT")
[ -n "${QP_PARTITION:-}" ] && SBATCH_ARGS+=(--partition="$QP_PARTITION")
[ -n "${QP_QOS:-}"       ] && SBATCH_ARGS+=(--qos="$QP_QOS")
[ -n "${QP_GRES:-}"      ] && SBATCH_ARGS+=(--gres="$QP_GRES")
[ -n "${QP_TIME:-}"      ] && SBATCH_ARGS+=(--time="$QP_TIME")
[ -n "${QP_MEM:-}"       ] && SBATCH_ARGS+=(--mem-per-gpu="$QP_MEM")
[ -n "${QP_CPUS:-}"      ] && SBATCH_ARGS+=(--cpus-per-gpu="$QP_CPUS")
[ -n "${QP_EXTRA:-}"     ] && read -r -a EXTRA <<< "$QP_EXTRA" && SBATCH_ARGS+=("${EXTRA[@]}")

if [ "$DRY_RUN" -eq 1 ]; then
    printf 'sbatch'; printf ' %q' "${SBATCH_ARGS[@]}" "$REPO_ROOT/scripts/slurm/job.sbatch"; echo
    exit 0
fi
exec sbatch "${SBATCH_ARGS[@]}" "$REPO_ROOT/scripts/slurm/job.sbatch"
