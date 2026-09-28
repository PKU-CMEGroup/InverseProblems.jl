#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1         
#SBATCH -p C064M0256G 
#SBATCH -J "inflation-eki-experiment"
#SBATCH -o "inflation-eki-experiment-%j.out"

# 加载 Julia 模块（如果你的集群需要，取消注释）
# module load julia/1.10.5

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "JULIA_NUM_THREADS: $JULIA_NUM_THREADS"

julia --project=. --optimize=3 --check-bounds=no run_inflation_eki.jl |& tee run_inflation_eki.log