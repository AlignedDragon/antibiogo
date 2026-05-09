#!/bin/bash
#SBATCH --job-name=xyr
#SBATCH --account=a127
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=11:59:00
#SBATCH --partition=normal
#SBATCH --mem=460000
#SBATCH --cpus-per-task=288
#SBATCH --output=data/logs/%x_%j.out
#SBATCH --error=data/logs/%x_%j.err

set -eo pipefail

# Resolve the repo root from this script's location, regardless of submit CWD.
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_DIR"
mkdir -p data/logs

echo "==========================================="
echo "The run has started at $(date)"
echo "Repo: $REPO_DIR"
echo "==========================================="

# Activate conda from whatever conda is on PATH.
eval "$(conda shell.bash hook)"
conda activate biogo

# Folder prefix separates model checkpoints under data/ExperimentModels/$FOLDER_PREFIX/.
export FOLDER_PREFIX="${FOLDER_PREFIX:-MNv3_1.0_MCD}"

echo "==========================================="
echo "Starting experiments on ${FOLDER_PREFIX}"
echo "==========================================="

python xyr/main.py

# Visualize model predictions on a few batches of validation split
python xyr/val_viz.py

echo "==========================================="
echo "The run has finished at $(date)"
echo "==========================================="
