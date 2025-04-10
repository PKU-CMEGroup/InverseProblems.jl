#!/bin/bash

#SBATCH --time=168:00:00   # walltime
#SBATCH --nodes=1         # number of nodes
#SBATCH --ntasks=2       # number of processor cores (i.e. tasks)
#SBATCH --mem=128G
#SBATCH -J "Banana"    # job name
#SBATCH -o "Banana"


# number of tasks

jupyter nbconvert --to script Banana-density.ipynb 
mv Banana-density.jl Banana.jl 
julia  Banana.jl 
