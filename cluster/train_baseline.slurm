#!/bin/bash
# The current score-oriented model does not require a GPU.
# Keep this template as a starting point for future heavier models.

#SBATCH --job-name=titanic-gradient-boosting
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

python run/train.py --config configs/gradient_boosting.json --model-output artifacts/checkpoints/gradient_boosting.joblib --metrics-output results/metrics/gradient_boosting_metrics.json
python run/predict.py --model-path artifacts/checkpoints/gradient_boosting.joblib --output-path results/submissions/gradient_boosting_submission.csv

