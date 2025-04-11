#!/bin/bash

#SBATCH --time=120:00:00   # walltime
#SBATCH --nodes=1          # number of nodes
#SBATCH --ntasks=6

#SBATCH -J "Darcy-GF"      # job name
#SBATCH -o "Darcy-GF"




julia  Darcy-GF.jl 
