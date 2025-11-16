#!/bin/bash
#SBATCH --job-name=xyr   # A name for your job. Visible in squeue.
#SBATCH --account=large-sc-2         # The account you are charged for the job
#SBATCH --nodes=1                    # Number of compute nodes to request.
#SBATCH --ntasks-per-node=1          # Tasks (processes) per node
#SBATCH --time=03:00:00              # HH:MM:SS, set a time limit for this job (here 10min)
#SBATCH --partition=normal            # Partition to use; "debug" is usually for quick tests
#SBATCH --mem=460000                 # Memory needed (simply set the mem of a node)
#SBATCH --cpus-per-task=288          # CPU cores per task (simply set the number of cpus a node has)
#SBATCH --output=/users/msayfiddinov/scratch/antibiogo_data/logs/xyr_%j.out  # log file for stdout, prints, et cetera
#SBATCH --error=/users/msayfiddinov/scratch/antibiogo_data/logs/xyr_%j.err # log file for stderr, errors

# Exit immediately if a command exits with a non-zero status (good practice)
set -eo pipefail

# installing for checker script -> latex_parser needs it
source ~/miniconda3/etc/profile.d/conda.sh
conda activate anti


python /users/msayfiddinov/antibiogo/xyr/main.py
