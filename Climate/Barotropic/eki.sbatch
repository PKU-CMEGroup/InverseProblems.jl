#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=32        # number of processor cores (i.e. tasks)
#SBATCH --nodes=1         # number of nodes
#SBATCH -p C064M0256G 
#SBATCH -J "eki-experiment"
#SBATCH -o "eki-experiment-%j.out"


# 加载 Julia 模块
# module load julia/1.7.1

export JULIA_NUM_THREADS=32

echo "JULIA_NUM_THREADS: $JULIA_NUM_THREADS"

julia run_eki.jl |& tee run_eki.log