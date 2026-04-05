from __future__ import annotations

from typing import Any

from .sklearn_baseline import TitanicSklearnModel


def build_model(config: dict[str, Any]) -> TitanicSklearnModel:
    model_type = config.get("model_type", "sklearn")
    if model_type != "sklearn":
        raise ValueError(f"Unsupported model_type: {model_type}")

    estimator_name = config.get("estimator_name", "logistic_regression")
    estimator_params = config.get("estimator_params", {})
    return TitanicSklearnModel(
        estimator_name=estimator_name,
        estimator_params=estimator_params,
    )
