#!/bin/bash

#SBATCH --time=168:00:00   # walltime
#SBATCH --nodes=1         # number of nodes
#SBATCH --ntasks=2       # number of processor cores (i.e. tasks)
#SBATCH --mem=128G
#SBATCH -J "Gaussian"    # job name
#SBATCH -o "Gaussian"


# number of tasks

jupyter nbconvert --to script Gaussian-density.ipynb 
mv Gaussian-density.jl Gaussian.jl 
julia  Gaussian.jl 
