#!/bin/bash

#SBATCH --time=168:00:00   # walltime
#SBATCH --nodes=1         # number of nodes
#SBATCH --ntasks=2       # number of processor cores (i.e. tasks)
#SBATCH --mem=128G
#SBATCH -J "Logconcave"    # job name
#SBATCH -o "Logconcave"


# number of tasks
jupyter nbconvert --to script Logconcave-density.ipynb 
mv Logconcave-density.jl Logconcave.jl 
julia  Logconcave.jl 
