#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --ntasks=5                      # Request 5 CPU cores (for 5 MATLAB workers)
#SBATCH --nodes=1
#SBATCH -p C064M1024G                   # Submit to partition C064M1024G
#SBATCH -J "test_mean_init"             # Job name
#SBATCH -o "test_mean_init%j.log"       # Output log file (with job ID)

source /etc/profile.d/modules.sh

module load gcc/12.2.0
module load matlab/R2022a

export MLNUM_THREADS=1                  # Limit MATLAB internal threading to avoid oversubscription

cd $SLURM_SUBMIT_DIR                    # Switch to submission directory

echo "Running MATLAB script test_with_mean_init.m with 5 parallel workers."
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"

# Run MATLAB in batch mode: no GUI, no splash, no JVM
# Execute script, catch errors, and exit cleanly
matlab -nodisplay -nosplash -nojvm -r "try, test_with_mean_init; catch ME, disp(ME.getReport); exit(1); end; exit(0);"