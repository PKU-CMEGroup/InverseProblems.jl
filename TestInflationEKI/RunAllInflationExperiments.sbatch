#!/bin/bash

#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --job-name=inflation-eki-studies
#SBATCH --output=inflation-eki-studies-%j.out

set -euo pipefail

# If required on the cluster, load its Julia module here.
# module load julia/1.10.5

# Submit from either the repository root or TestInflationEKI.
# SLURM_SUBMIT_DIR is used because Slurm copies this script to its spool directory.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -f "$SUBMIT_DIR/TestInflationEKI/RunNonlinearSeedStudy.jl" ]]; then
    REPO_ROOT="$(cd "$SUBMIT_DIR" && pwd)"
elif [[ -f "$SUBMIT_DIR/RunNonlinearSeedStudy.jl" && -d "$SUBMIT_DIR/../Inversion" ]]; then
    REPO_ROOT="$(cd "$SUBMIT_DIR/.." && pwd)"
else
    echo "Submit this job from the repository root or TestInflationEKI." >&2
    exit 1
fi

export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg

JULIA_BIN="${JULIA_BIN:-julia}"
command -v "$JULIA_BIN" >/dev/null || {
    echo "Julia executable not found: $JULIA_BIN. Load Julia or set JULIA_BIN." >&2
    exit 1
}
JOB_ID="${SLURM_JOB_ID:-manual}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/TestInflationEKI/Logs}"
mkdir -p "$LOG_DIR"

echo "Repository: $REPO_ROOT"
echo "Julia threads: $JULIA_NUM_THREADS"
"$JULIA_BIN" --version

run_study() {
    local name="$1" work_dir="$2" script="$3"
    echo "[$(date -Is)] Starting $name"
    (
        cd "$work_dir"
        "$JULIA_BIN" --startup-file=no --optimize=3 --check-bounds=no "$script"
    ) 2>&1 | tee "$LOG_DIR/${name}-${JOB_ID}.log"
    echo "[$(date -Is)] Finished $name"
}

# Keep the experiment parameters and output locations defined by each Julia entry point.
run_study nonlinear "$REPO_ROOT" "TestInflationEKI/RunNonlinearSeedStudy.jl"
run_study square-condition "$REPO_ROOT" "TestInflationEKI/InflationEKI_SquareConditionNumber_Test.jl"

# InflatedEKI_Barotropic.jl defines the solver; this driver starts its paired experiment.
run_study barotropic "$REPO_ROOT/Climate/Barotropic" "run_inflated_eki.jl"

echo "[$(date -Is)] All three experiments completed."
