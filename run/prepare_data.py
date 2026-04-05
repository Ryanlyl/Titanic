from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features import save_processed_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build processed Titanic datasets with engineered features."
    )
    parser.add_argument(
        "--raw-train-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "train.csv",
        help="Path to the raw Titanic training CSV.",
    )
    parser.add_argument(
        "--raw-test-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "test.csv",
        help="Path to the raw Titanic test CSV.",
    )
    parser.add_argument(
        "--processed-train-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "train.csv",
        help="Where to write the processed training CSV.",
    )
    parser.add_argument(
        "--processed-test-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "test.csv",
        help="Where to write the processed test CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_train, processed_test = save_processed_datasets(
        raw_train_path=args.raw_train_path,
        raw_test_path=args.raw_test_path,
        processed_train_path=args.processed_train_path,
        processed_test_path=args.processed_test_path,
    )

    print(
        f"Saved processed train data to: {args.processed_train_path} "
        f"({processed_train.shape[0]} rows, {processed_train.shape[1]} columns)"
    )
    print(
        f"Saved processed test data to: {args.processed_test_path} "
        f"({processed_test.shape[0]} rows, {processed_test.shape[1]} columns)"
    )


if __name__ == "__main__":
    main()