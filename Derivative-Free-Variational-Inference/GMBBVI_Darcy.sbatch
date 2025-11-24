#!/bin/bash

#SBATCH --time=8:00:00 
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH -p C064M1024G
#SBATCH -J "GMBBVI_Darcy"
#SBATCH -o "GMBBVI_Darcy_%j.log"

### reference run time:  [1+(0.03*N_iter*N_ens*N_modes)/3600] hours

export JULIA_NUM_THREADS=1

source /etc/profile.d/modules.sh

module load gcc/12.2.0

module load matlab/R2022a

export PATH="/lustre/home/2501110040/software/julia-1.10.0/bin:$PATH"
export LD_PRELOAD="/lustre/software/gcc/12.2.0/lib64/libstdc++.so.6"

echo "Julia path: $(which julia)"
echo "Julia version: $(julia --version)"
echo "Running Julia with $JULIA_NUM_THREADS threads on $SLURM_NTASKS tasks."

julia GMBBVI_Darcy.jl |& tee GMBBVI_Darcy.log