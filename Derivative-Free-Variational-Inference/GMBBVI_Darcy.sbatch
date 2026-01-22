#!/bin/bash

#SBATCH --time=24:00:00 
#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH -p C064M1024G
#SBATCH -J "par_Darcy"
#SBATCH -o "par_Darcy%j.log"


export JULIA_NUM_THREADS=2

source /etc/profile.d/modules.sh

module load gcc/12.2.0

module load matlab/R2022a

export PATH="/lustre/home/2501110040/software/julia-1.10.0/bin:$PATH"
export LD_PRELOAD="/lustre/software/gcc/12.2.0/lib64/libstdc++.so.6"

echo "Julia path: $(which julia)"
echo "Julia version: $(julia --version)"
echo "Running Julia with $JULIA_NUM_THREADS threads on $SLURM_NTASKS tasks."

julia par_Darcy.jl |& tee par_Darcy.log