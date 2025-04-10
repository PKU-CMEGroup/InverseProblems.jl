#!/bin/bash

#SBATCH --time=168:00:00   # walltime
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G  # 您需要的内存量
#SBATCH -J "Darcy-Plot"    # job name
#SBATCH -o "Darcy-Plot"


# number of tasks


julia  Darcy-Plot.jl 