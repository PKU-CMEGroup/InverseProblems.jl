#!/bin/bash

#SBATCH --time=120:00:00   # walltime
#SBATCH --nodes=1         # number of nodes
#SBATCH --ntasks=24

#SBATCH -J "Darcy-MCMC"    # job name
#SBATCH -o "Darcy-MCMC"


# number of tasks

module purge

julia  Darcy-MCMC.jl 
