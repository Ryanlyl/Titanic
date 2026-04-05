from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

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
        default=PROJECT_ROOT / "data" / "raw" / "train.csv",
        help="Path to the Titanic training CSV.",
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
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    data = pd.read_csv(args.train_path)
    if "Survived" not in data.columns:
        raise ValueError("Training data must contain the 'Survived' column.")

    features = data.drop(columns=["Survived"])
    targets = data["Survived"]

    validation_config = config.get("validation", {})
    x_train, x_val, y_train, y_val = train_test_split(
        features,
        targets,
        test_size=validation_config.get("test_size", 0.2),
        random_state=validation_config.get("random_state", 42),
        stratify=targets,
    )

    model = build_model(config)
    model.fit(x_train, y_train)

    val_predictions = model.predict(x_val)
    accuracy = accuracy_score(y_val, val_predictions)

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_output)

    metrics = {
        "model_type": config.get("model_type", "sklearn"),
        "estimator_name": config.get("estimator_name", "logistic_regression"),
        "validation_accuracy": round(float(accuracy), 6),
        "num_train_rows": int(len(x_train)),
        "num_validation_rows": int(len(x_val)),
    }
    args.metrics_output.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Saved model to: {args.model_output}")
    print(f"Saved metrics to: {args.metrics_output}")


if __name__ == "__main__":
    main()

