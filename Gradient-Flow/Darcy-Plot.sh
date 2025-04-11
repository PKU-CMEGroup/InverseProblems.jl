#!/bin/bash

#SBATCH --time=120:00:00   # walltime
#SBATCH --nodes=1
#SBATCH --ntasks=24

#SBATCH -J "Darcy-Plot"    # job name
#SBATCH -o "Darcy-Plot"


# number of tasks


julia  Darcy-Plot.jl 
