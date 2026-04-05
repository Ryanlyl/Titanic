from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


class BaseTitanicModel(ABC):
    """Common interface for all Titanic models in this repository."""

    @abstractmethod
    def fit(self, features: pd.DataFrame, targets: pd.Series) -> "BaseTitanicModel":
        """Train the model on a feature table and label vector."""

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> Any:
        """Predict survival labels for a feature table."""

    def save(self, path: str | Path) -> None:
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target_path)

    @classmethod
    def load(cls, path: str | Path) -> "BaseTitanicModel":
        return joblib.load(path)

