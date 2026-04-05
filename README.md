# Titanic

Kaggle Titanic survival prediction starter repository. The project is organized for data processing, model development, training, submission generation, and cluster execution with a single consistent Python entry workflow.

## Project Layout

```text
Titanic/
|-- artifacts/
|   |-- checkpoints/          # Serialized trained models
|   `-- logs/                 # Local and Slurm logs
|-- cluster/
|   `-- train_baseline.slurm  # Slurm template for cluster runs
|-- configs/
|   `-- baseline.json         # Baseline experiment config
|-- data/
|   |-- processed/            # Cleaned or feature-engineered datasets
|   `-- raw/                  # Raw Kaggle data
|       |-- gender_submission.csv
|       |-- test.csv
|       `-- train.csv
|-- models/
|   |-- __init__.py
|   |-- base.py               # Shared model interface
|   |-- factory.py            # Central model builder
|   `-- sklearn_baseline.py   # sklearn baseline implementation
|-- results/
|   |-- metrics/              # Validation metrics and summaries
|   `-- submissions/          # CSV files ready for Kaggle submission
|-- run/
|   |-- predict.py            # Prediction and submission entrypoint
|   `-- train.py              # Training entrypoint
|-- .gitignore
`-- requirements.txt
```

## Directory Conventions

- `data/raw/` stores original datasets and should stay unchanged.
- `data/processed/` stores cleaned tables, engineered features, or prepared splits.
- `models/` stores model code only, not trained weights.
- `artifacts/checkpoints/` stores trained model objects and checkpoints.
- `run/` stores command-line entrypoints so experiments follow a consistent interface.
- `results/submissions/` stores generated Kaggle submission files.
- `cluster/` stores Slurm scripts for future GPU or large-scale training runs.

## Default Workflow

1. Build processed datasets with `python run/prepare_data.py`.
2. Configure an experiment in `configs/baseline.json`. The default baseline uses a random forest on a curated feature set: core structured raw columns plus engineered features, excluding `PassengerId`, `Name`, `Ticket`, and `Cabin`.
3. Train a model with `python run/train.py` to run cross-validation and save a final checkpoint fit on all labeled rows.
4. Generate a submission with `python run/predict.py`.

## Quick Start

```bash
pip install -r requirements.txt
python run/prepare_data.py
python run/train.py --config configs/baseline.json
python run/predict.py --model-path artifacts/checkpoints/baseline.joblib
```

## Extension Notes

- Add new models under `models/` and register them through `models/factory.py`.
- Add new experiment configs under `configs/`.
- If you later switch to deep learning, keep the `run/train.py` interface and replace the implementation behind `models/`.

## Validation

- The default training workflow uses 5-fold stratified cross-validation configured in `configs/baseline.json`.
- Per-fold metrics plus the mean and standard deviation are written to `results/metrics/`.
- After cross-validation, the training script refits the model on the full training table before saving the checkpoint used for submission generation.
