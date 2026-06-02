from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_ARTIFACT_DIR = REPO_ROOT / "outputs" / "models" / "xgb_svd_smote"


@dataclass
class ModelArtifacts:
    vectorizer: Any
    svd: Any
    label_encoder: Any
    model: Any
    metrics: dict[str, Any] | None
    artifact_dir: Path

    @property
    def loaded(self) -> bool:
        return all(
            item is not None
            for item in (self.vectorizer, self.svd, self.label_encoder, self.model)
        )

    def predict(self, sequence: str, top_k: int = 5) -> dict[str, Any]:
        features = self.vectorizer.transform([sequence])
        reduced = self.svd.transform(features)
        probabilities = self.model.predict_proba(reduced)[0]

        top_indices = np.argsort(probabilities)[::-1][:top_k]
        top_labels = self.label_encoder.inverse_transform(top_indices)
        top_predictions = [
            {
                "genus": str(label),
                "probability": float(probabilities[index]),
            }
            for index, label in zip(top_indices, top_labels)
        ]

        best = top_predictions[0]
        return {
            "predicted_genus": best["genus"],
            "confidence": best["probability"],
            "top_5": top_predictions,
            "sequence_length": len(sequence),
        }


def load_model_artifacts(artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> ModelArtifacts:
    artifact_dir = artifact_dir.resolve()
    vectorizer = joblib.load(artifact_dir / "vectorizer.joblib")
    svd = joblib.load(artifact_dir / "svd.joblib")
    label_encoder = joblib.load(artifact_dir / "label_encoder.joblib")
    model = load_xgboost_model(artifact_dir)
    metrics = load_metrics(artifact_dir / "metrics.json")

    return ModelArtifacts(
        vectorizer=vectorizer,
        svd=svd,
        label_encoder=label_encoder,
        model=model,
        metrics=metrics,
        artifact_dir=artifact_dir,
    )


def load_xgboost_model(artifact_dir: Path):
    json_path = artifact_dir / "xgboost_model.json"
    joblib_path = artifact_dir / "xgboost_model.joblib"

    if json_path.exists():
        import xgboost as xgb

        model = xgb.XGBClassifier()
        model.load_model(json_path)
        model.set_params(device="cpu")
        return model

    if joblib_path.exists():
        model = joblib.load(joblib_path)
        if hasattr(model, "set_params"):
            model.set_params(device="cpu")
        return model

    raise FileNotFoundError(
        f"Missing XGBoost model artifact. Expected {json_path} or {joblib_path}."
    )


def load_metrics(metrics_path: Path) -> dict[str, Any] | None:
    if not metrics_path.exists():
        return None
    return json.loads(metrics_path.read_text(encoding="utf-8"))
