#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=2       # number of processor cores (i.e. tasks)
#SBATCH --nodes=1         # number of nodes
#SBATCH -p C064M1024G 
#SBATCH -J "MultiModal-Comparison"   # job name
#SBATCH -o "MultiModal-Comparison"

# Size of the ensemble
# module load julia/1.7.1

export JULIA_NUM_THREADS=1 #${SLURM_CPUS_PER_TASK:=1}

echo $JULIA_NUM_THREADS

# julia MultiModal-Comparison-2D.jl |& tee MultiModal-Comparison-2D.log
julia MultiModal-GMBBVI-nD.jl |& tee MultiModal-GMBBVI-nD.log
# julia MultiModal-Comparison-100D.jl |& tee MultiModal-Comparison-100D.log
