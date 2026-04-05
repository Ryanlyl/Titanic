from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENGINEERED_FEATURE_COLUMNS = [
    "Title",
    "TitleGroup",
    "NameLength",
    "FamilySize",
    "IsAlone",
    "CabinDeck",
    "CabinKnown",
    "TicketGroupSize",
    "FarePerPerson",
    "AgeMissing",
    "FareMissing",
]

_TITLE_NORMALIZATION = {
    "Mlle": "Miss",
    "Mme": "Mrs",
    "Ms": "Miss",
}
_RARE_TITLES = {
    "Capt",
    "Col",
    "Countess",
    "Don",
    "Dona",
    "Dr",
    "Jonkheer",
    "Lady",
    "Major",
    "Rev",
    "Sir",
}


def _extract_title(name_series: pd.Series) -> pd.Series:
    return (
        name_series.fillna("")
        .str.extract(r",\s*([^\.]+)\.", expand=False)
        .fillna("Unknown")
        .str.strip()
    )


def _group_title(title_series: pd.Series) -> pd.Series:
    grouped = title_series.replace(_TITLE_NORMALIZATION)
    return grouped.replace({title: "Rare" for title in _RARE_TITLES})


def _extract_cabin_deck(cabin_series: pd.Series) -> pd.Series:
    return cabin_series.fillna("U").astype(str).str[0].str.upper()


def _build_ticket_group_size(ticket_series: pd.Series) -> pd.Series:
    normalized = ticket_series.fillna("").astype(str).str.strip().copy()
    missing_mask = normalized.eq("")
    if missing_mask.any():
        normalized.loc[missing_mask] = [
            f"UNKNOWN_{index}" for index in normalized.index[missing_mask]
        ]

    counts = normalized.value_counts()
    return normalized.map(counts).astype(int)


def has_engineered_features(data: pd.DataFrame) -> bool:
    return set(ENGINEERED_FEATURE_COLUMNS).issubset(data.columns)


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"Age", "Cabin", "Fare", "Name", "Parch", "SibSp", "Ticket"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Cannot engineer features without columns: {missing}")

    engineered = data.copy()
    title = _extract_title(engineered["Name"])
    ticket_group_size = _build_ticket_group_size(engineered["Ticket"])

    engineered["Title"] = title
    engineered["TitleGroup"] = _group_title(title)
    engineered["NameLength"] = engineered["Name"].fillna("").str.len()
    engineered["FamilySize"] = (
        engineered["SibSp"].fillna(0) + engineered["Parch"].fillna(0) + 1
    )
    engineered["IsAlone"] = (engineered["FamilySize"] == 1).astype(int)
    engineered["CabinDeck"] = _extract_cabin_deck(engineered["Cabin"])
    engineered["CabinKnown"] = engineered["Cabin"].notna().astype(int)
    engineered["TicketGroupSize"] = ticket_group_size
    engineered["FarePerPerson"] = engineered["Fare"] / engineered["TicketGroupSize"]
    engineered["AgeMissing"] = engineered["Age"].isna().astype(int)
    engineered["FareMissing"] = engineered["Fare"].isna().astype(int)
    return engineered


def ensure_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    if has_engineered_features(data):
        return data.copy()
    return engineer_features(data)


def build_processed_datasets(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "Survived" not in train_data.columns:
        raise ValueError("Training data must contain the 'Survived' column.")

    processed_train = engineer_features(train_data)
    processed_test = engineer_features(test_data)
    return processed_train, processed_test


def save_processed_datasets(
    raw_train_path: str | Path,
    raw_test_path: str | Path,
    processed_train_path: str | Path,
    processed_test_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_train_path = Path(raw_train_path)
    raw_test_path = Path(raw_test_path)
    processed_train_path = Path(processed_train_path)
    processed_test_path = Path(processed_test_path)

    train_data = pd.read_csv(raw_train_path)
    test_data = pd.read_csv(raw_test_path)
    processed_train, processed_test = build_processed_datasets(train_data, test_data)

    processed_train_path.parent.mkdir(parents=True, exist_ok=True)
    processed_test_path.parent.mkdir(parents=True, exist_ok=True)
    processed_train.to_csv(processed_train_path, index=False)
    processed_test.to_csv(processed_test_path, index=False)
    return processed_train, processed_test


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