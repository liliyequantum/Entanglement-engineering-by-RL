#!/bin/bash

#SBATCH -N 1           # number of nodes
#SBATCH -c 1          # number of cores 
#SBATCH -t 7-00:00:00   # time in d-hh:mm:ss
#SBATCH -p general      # partition 
#SBATCH -o ./output/slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e ./output/slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --export=NONE   # Purge the job-submitting shell environment

module purge  # Always purge modules to ensure consistent environments
# Load required modules for job's environment
module load mamba/latest
# Using python, so source activate an appropriate environment
source activate qutip_RL

python3 training_recurrentPPO.py 5 1 2600

source deactivate qutip_RL