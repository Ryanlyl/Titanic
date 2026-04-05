from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from models.base import BaseTitanicModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Kaggle submission file.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "checkpoints" / "baseline.joblib",
        help="Path to a trained Titanic model.",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "test.csv",
        help="Path to the Titanic test CSV. Defaults to processed data.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "submissions" / "baseline_submission.csv",
        help="Where to store the Kaggle submission CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.test_path.exists():
        raise FileNotFoundError(
            "Test data not found at "
            f"{args.test_path}. Run `python run/prepare_data.py` first, "
            "or pass --test-path."
        )

    test_data = pd.read_csv(args.test_path)
    model = BaseTitanicModel.load(args.model_path)

    predictions = model.predict(test_data)
    submission = pd.DataFrame(
        {
            "PassengerId": test_data["PassengerId"],
            "Survived": predictions.astype(int),
        }
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output_path, index=False)
    print(f"Saved submission to: {args.output_path}")


if __name__ == "__main__":
    main()