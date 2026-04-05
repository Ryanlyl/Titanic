from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, stdev

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Titanic model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "baseline.json",
        help="Path to the JSON config file.",
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "train.csv",
        help="Path to the Titanic training CSV. Defaults to processed data.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "checkpoints" / "baseline.joblib",
        help="Where to store the trained model.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=PROJECT_ROOT / "results" / "metrics" / "baseline_metrics.json",
        help="Where to store validation metrics.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_validation_splitter(config: dict) -> StratifiedKFold:
    validation_config = config.get("validation", {})
    n_splits = validation_config.get("n_splits", 5)
    if n_splits < 2:
        raise ValueError("validation.n_splits must be at least 2 for cross-validation.")

    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=validation_config.get("shuffle", True),
        random_state=validation_config.get("random_state", 42),
    )


def run_cross_validation(
    features: pd.DataFrame,
    targets: pd.Series,
    config: dict,
) -> tuple[list[dict[str, float | int]], list[float]]:
    splitter = build_validation_splitter(config)

    fold_metrics: list[dict[str, float | int]] = []
    fold_accuracies: list[float] = []
    for fold_index, (train_idx, val_idx) in enumerate(
        splitter.split(features, targets),
        start=1,
    ):
        x_train = features.iloc[train_idx]
        x_val = features.iloc[val_idx]
        y_train = targets.iloc[train_idx]
        y_val = targets.iloc[val_idx]

        fold_model = build_model(config)
        fold_model.fit(x_train, y_train)
        val_predictions = fold_model.predict(x_val)
        accuracy = float(accuracy_score(y_val, val_predictions))
        fold_accuracies.append(accuracy)

        fold_metrics.append(
            {
                "fold": fold_index,
                "accuracy": round(accuracy, 6),
                "num_train_rows": int(len(x_train)),
                "num_validation_rows": int(len(x_val)),
            }
        )

    return fold_metrics, fold_accuracies


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if not args.train_path.exists():
        raise FileNotFoundError(
            "Training data not found at "
            f"{args.train_path}. Run `python run/prepare_data.py` first, "
            "or pass --train-path."
        )

    data = pd.read_csv(args.train_path)
    if "Survived" not in data.columns:
        raise ValueError("Training data must contain the 'Survived' column.")

    features = data.drop(columns=["Survived"])
    targets = data["Survived"]

    fold_metrics, fold_accuracies = run_cross_validation(features, targets, config)
    cv_mean_accuracy = mean(fold_accuracies)
    cv_std_accuracy = stdev(fold_accuracies) if len(fold_accuracies) > 1 else 0.0

    # Refit on the full labeled dataset so the saved checkpoint uses all
    # available supervision while keeping the cross-validation estimate above.
    final_model = build_model(config)
    final_model.fit(features, targets)

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    final_model.save(args.model_output)

    metrics = {
        "model_type": config.get("model_type", "sklearn"),
        "estimator_name": config.get("estimator_name", "logistic_regression"),
        "cv_accuracy_mean": round(float(cv_mean_accuracy), 6),
        "cv_accuracy_std": round(float(cv_std_accuracy), 6),
        "num_folds": int(len(fold_metrics)),
        "fold_metrics": fold_metrics,
        "num_full_training_rows": int(len(features)),
    }
    args.metrics_output.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for fold in fold_metrics:
        print(f"Fold {fold['fold']} accuracy: {float(fold['accuracy']):.4f}")
    print(f"CV accuracy mean: {cv_mean_accuracy:.4f}")
    print(f"CV accuracy std: {cv_std_accuracy:.4f}")
    print(f"Saved model to: {args.model_output}")
    print(f"Saved metrics to: {args.metrics_output}")


if __name__ == "__main__":
    main()
