#!/bin/bash
#SBATCH --job-name=xyr   # A name for your job. Visible in squeue.
#SBATCH --account=a127         # The account you are charged for the job
#SBATCH --nodes=1                    # Number of compute nodes to request.
#SBATCH --ntasks-per-node=1          # Tasks (processes) per node
#SBATCH --time=11:59:00              # HH:MM:SS, set a time limit for this job (here 10min)
#SBATCH --partition=normal            # Partition to use; "debug" is usually for quick tests
#SBATCH --mem=460000                 # Memory needed (simply set the mem of a node)
#SBATCH --cpus-per-task=288          # CPU cores per task (simply set the number of cpus a node has)
#SBATCH --output=$HOME/scratch/antibiogo/logs/xyr_%j.out  # log file for stdout, prints, et cetera
#SBATCH --error=$HOME/scratch/antibiogo/logs/xyr_%j.err # log file for stderr, errors

# Exit immediately if a command exits with a non-zero status (good practice)
set -eo pipefail

echo "==========================================="
echo "The run has started at $(date)"
echo "==========================================="

# installing for checker script -> latex_parser needs it
source ~/miniconda3/etc/profile.d/conda.sh
conda activate anti

# to separate model names so that they don't overwrite at the end
# model is saved in $PREFIX/xyr.keras form inside ExperimentModels
export FOLDER_PREFIX="ConvOnlyNorm"

echo "==========================================="
echo "Starting experiments on ${FOLDER_PREFIX}"
echo "==========================================="

python $HOME/antibiogo/xyr/main.py

# Visualize model predictions on a few batches of validation split
python $HOME/antibiogo/xyr/val_viz.py

echo "==========================================="
echo "The run has finished at $(date)"
echo "==========================================="

