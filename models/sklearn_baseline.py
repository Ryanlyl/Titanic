from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .base import BaseTitanicModel

NUMERIC_FEATURES = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "IsAlone",
]
CATEGORICAL_FEATURES = [
    "Sex",
    "Embarked",
    "Title",
    "CabinDeck",
]


def _extract_title(name_series: pd.Series) -> pd.Series:
    return (
        name_series.fillna("")
        .str.extract(r",\s*([^\.]+)\.", expand=False)
        .fillna("Unknown")
    )


def _extract_cabin_deck(cabin_series: pd.Series) -> pd.Series:
    return cabin_series.fillna("U").astype(str).str[0]


class TitanicSklearnModel(BaseTitanicModel):
    """A simple sklearn baseline with lightweight feature engineering."""

    def __init__(
        self,
        estimator_name: str = "logistic_regression",
        estimator_params: dict[str, Any] | None = None,
    ) -> None:
        self.estimator_name = estimator_name
        self.estimator_params = estimator_params or {}
        self.pipeline = self._build_pipeline()

    def _build_estimator(self) -> Any:
        if self.estimator_name == "logistic_regression":
            default_params = {
                "max_iter": 1000,
            }
            default_params.update(self.estimator_params)
            return LogisticRegression(**default_params)

        if self.estimator_name == "random_forest":
            default_params = {
                "n_estimators": 300,
                "random_state": 42,
            }
            default_params.update(self.estimator_params)
            return RandomForestClassifier(**default_params)

        raise ValueError(f"Unsupported estimator_name: {self.estimator_name}")

    def _build_pipeline(self) -> Pipeline:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, NUMERIC_FEATURES),
                ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
            ]
        )

        return Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", self._build_estimator()),
            ]
        )

    @staticmethod
    def prepare_features(features: pd.DataFrame) -> pd.DataFrame:
        prepared = features.copy()
        prepared["Title"] = _extract_title(prepared["Name"])
        prepared["FamilySize"] = (
            prepared["SibSp"].fillna(0) + prepared["Parch"].fillna(0) + 1
        )
        prepared["IsAlone"] = (prepared["FamilySize"] == 1).astype(int)
        prepared["CabinDeck"] = _extract_cabin_deck(prepared["Cabin"])
        return prepared

    def fit(self, features: pd.DataFrame, targets: pd.Series) -> "TitanicSklearnModel":
        prepared = self.prepare_features(features)
        self.pipeline.fit(prepared, targets)
        return self

    def predict(self, features: pd.DataFrame) -> Any:
        prepared = self.prepare_features(features)
        return self.pipeline.predict(prepared)

