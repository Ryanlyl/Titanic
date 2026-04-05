from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import ensure_engineered_features, select_model_features

from .base import BaseTitanicModel


class TitanicSklearnModel(BaseTitanicModel):
    """A sklearn baseline built on a curated Titanic feature table."""

    def __init__(
        self,
        estimator_name: str = "logistic_regression",
        estimator_params: dict[str, Any] | None = None,
    ) -> None:
        self.estimator_name = estimator_name
        self.estimator_params = estimator_params or {}
        self.pipeline: Pipeline | None = None
        self.numeric_features_: list[str] = []
        self.categorical_features_: list[str] = []
        self.feature_columns_: list[str] = []

    def _build_estimator(self) -> Any:
        if self.estimator_name == "logistic_regression":
            default_params = {
                "max_iter": 1000,
            }
            default_params.update(self.estimator_params)
            return LogisticRegression(**default_params)

        if self.estimator_name == "random_forest":
            default_params = {
                "n_estimators": 400,
                "max_depth": 10,
                "min_samples_leaf": 2,
                "min_samples_split": 4,
                "n_jobs": 1,
                "random_state": 42,
            }
            default_params.update(self.estimator_params)
            return RandomForestClassifier(**default_params)

        if self.estimator_name == "gradient_boosting":
            default_params = {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "random_state": 42,
            }
            default_params.update(self.estimator_params)
            return GradientBoostingClassifier(**default_params)

        raise ValueError(f"Unsupported estimator_name: {self.estimator_name}")

    def _infer_feature_columns(self, features: pd.DataFrame) -> tuple[list[str], list[str]]:
        numeric_features = features.select_dtypes(include=["number", "bool"]).columns.tolist()
        categorical_features = [
            column for column in features.columns if column not in numeric_features
        ]
        return numeric_features, categorical_features

    def _build_pipeline(
        self,
        numeric_features: list[str],
        categorical_features: list[str],
    ) -> Pipeline:
        transformers: list[tuple[str, Any, list[str]]] = []

        if numeric_features:
            numeric_steps: list[tuple[str, Any]] = [
                ("imputer", SimpleImputer(strategy="median"))
            ]
            if self.estimator_name == "logistic_regression":
                numeric_steps.append(("scaler", StandardScaler()))

            numeric_pipeline = Pipeline(steps=numeric_steps)
            transformers.append(("numeric", numeric_pipeline, numeric_features))

        if categorical_features:
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]
            )
            transformers.append(
                ("categorical", categorical_pipeline, categorical_features)
            )

        if not transformers:
            raise ValueError("No feature columns available for training.")

        preprocessor = ColumnTransformer(transformers=transformers)

        return Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", self._build_estimator()),
            ]
        )

    @staticmethod
    def prepare_features(features: pd.DataFrame) -> pd.DataFrame:
        prepared = ensure_engineered_features(features)
        return select_model_features(prepared)

    def fit(self, features: pd.DataFrame, targets: pd.Series) -> "TitanicSklearnModel":
        prepared = self.prepare_features(features)
        self.numeric_features_, self.categorical_features_ = self._infer_feature_columns(
            prepared
        )
        self.feature_columns_ = prepared.columns.tolist()
        self.pipeline = self._build_pipeline(
            numeric_features=self.numeric_features_,
            categorical_features=self.categorical_features_,
        )
        self.pipeline.fit(prepared.loc[:, self.feature_columns_], targets)
        return self

    def predict(self, features: pd.DataFrame) -> Any:
        if self.pipeline is None:
            raise ValueError("Model has not been fitted yet.")

        prepared = self.prepare_features(features)
        missing_columns = [
            column for column in self.feature_columns_ if column not in prepared.columns
        ]
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"Prediction data is missing required feature columns: {missing}")

        # Keep the prediction table aligned with the exact train-time schema.
        return self.pipeline.predict(prepared.loc[:, self.feature_columns_])

