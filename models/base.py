from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn


class BaseTitanicModel(ABC):
    """Common interface for all Titanic models in this repository."""

    @abstractmethod
    def fit(self, features: pd.DataFrame, targets: pd.Series) -> "BaseTitanicModel":
        """Train the model on a feature table and label vector."""

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> Any:
        """Predict survival labels for a feature table."""

    @staticmethod
    def _metadata_path(path: Path) -> Path:
        return path.with_suffix(".meta.json")

    def _build_metadata(self) -> dict[str, Any]:
        return {
            "format": "titanic_model",
            "model_class": type(self).__name__,
            "estimator_name": getattr(self, "estimator_name", None),
            "library_versions": {
                "scikit_learn": sklearn.__version__,
                "pandas": pd.__version__,
                "joblib": joblib.__version__,
            },
        }

    def save(self, path: str | Path) -> None:
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target_path)
        metadata_path = self._metadata_path(target_path)
        metadata_path.write_text(
            json.dumps(self._build_metadata(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "BaseTitanicModel":
        source_path = Path(path)
        metadata_path = cls._metadata_path(source_path)
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            trained_version = metadata.get("library_versions", {}).get("scikit_learn")
            current_version = sklearn.__version__
            if trained_version and trained_version != current_version:
                raise RuntimeError(
                    f"Model checkpoint version mismatch for {source_path}: trained with "
                    f"scikit-learn {trained_version}, current environment has {current_version}. "
                    "Retrain the model in the current environment before running predict."
                )

        try:
            loaded_model = joblib.load(source_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model checkpoint at {source_path}. "
                f"Current environment uses scikit-learn {sklearn.__version__}. "
                "This usually means the checkpoint was trained with a different "
                "scikit-learn version. Retrain the model in this environment and try again."
            ) from exc

        if not isinstance(loaded_model, cls):
            raise TypeError(
                f"Expected checkpoint at {source_path} to contain {cls.__name__}, "
                f"but found {type(loaded_model).__name__}."
            )
        return loaded_model
