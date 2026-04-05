#!/bin/bash
# The current baseline does not require a GPU.
# Keep this template as a starting point for future heavier models.

#SBATCH --job-name=titanic-baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/logs/%x-%j.out
#SBATCH --error=artifacts/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_ROOT"

python run/train.py --config configs/baseline.json
python run/predict.py --model-path artifacts/checkpoints/baseline.joblib

