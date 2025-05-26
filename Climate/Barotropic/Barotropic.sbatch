#!/bin/bash

#SBATCH --time=96:00:00   # walltime
#SBATCH --ntasks=12       # number of processor cores (i.e. tasks)
#SBATCH --nodes=1         # number of nodes
#SBATCH -J "barotropic_calibration"   # job name
#SBATCH -o "barotropic_calibration"

# Size of the ensemble
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:=1}

echo $JULIA_NUM_THREADS

julia DF_GMVI_Barotropic.jl |& tee df_gmvi.log
