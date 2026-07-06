#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH -p C064M0256G
#SBATCH -J "inflation-eki-experiment"
#SBATCH -o "inflation-eki-experiment-%j.out"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

JULIA_PROJECT_DIR="${JULIA_PROJECT_DIR:-${REPO_DIR}}"

cd "${SCRIPT_DIR}"

export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "SCRIPT_DIR: ${SCRIPT_DIR}"
echo "REPO_DIR: ${REPO_DIR}"
echo "JULIA_PROJECT_DIR: ${JULIA_PROJECT_DIR}"
echo "JULIA_NUM_THREADS: ${JULIA_NUM_THREADS}"

julia --project="${JULIA_PROJECT_DIR}" --optimize=3 --check-bounds=no run_inflation_eki.jl |& tee run_inflation_eki.log
