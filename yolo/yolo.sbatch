#!/bin/bash
#SBATCH --job-name=yolo
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

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"
mkdir -p data/logs

echo "==========================================="
echo "The run has started at $(date)"
echo "Repo: $REPO_DIR"
echo "==========================================="

eval "$(conda shell.bash hook)"
conda activate biogo

export FOLDER_PREFIX="${FOLDER_PREFIX:-yolo_pelletOnly}"

echo "==========================================="
echo "Starting experiments on ${FOLDER_PREFIX}"
echo "==========================================="

python yolo/main.py

# Visualize model predictions on a few batches of validation split
python yolo/val_viz.py

echo "==========================================="
echo "The run has finished at $(date)"
echo "==========================================="
