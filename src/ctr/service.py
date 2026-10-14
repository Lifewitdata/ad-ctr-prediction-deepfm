"""FastAPI prediction service.

Loads the preprocessing objects and the model selected at training time
(``artifacts/model_choice.json``), then serves calibrated click probabilities.
Missing numeric features are imputed with training medians, unknown or absent
categories map to the out-of-vocabulary index - exactly the logic used in
training, because the same saved ``Preprocessor`` is applied here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pydantic
from fastapi import FastAPI, HTTPException
from pydantic import ConfigDict

from . import __version__, features
from .config import artifacts_dir as default_artifacts_dir
from .config import load_config


def request_schema() -> type[pydantic.BaseModel]:
    cfg = load_config()
    fields: dict = {c: (float | None, None) for c in cfg["features"]["numeric_cols"]}
    fields.update({c: (str | None, None) for c in cfg["features"]["categorical_cols"]})
    return pydantic.create_model(
        "ImpressionFeatures", __config__=ConfigDict(extra="forbid"), **fields
    )


ImpressionFeatures = request_schema()


def predict_proba(kind: str, model, preprocessor, frame: pd.DataFrame) -> np.ndarray:
    """Score a raw model over a feature frame through the single serving path.

    Notebook 03 and the API both call this, so offline evaluation and online
    scoring build model inputs identically by construction.
    """
    processed = preprocessor.transform(frame)
    if kind == "constant":
        proba = model.predict_proba(processed["Xcat"])
    elif kind == "lr_sparse":
        proba = model.predict_proba(features.to_lr_matrix(preprocessor, processed))
    elif kind == "lgbm_frame":
        proba = model.predict_proba(features.to_lgbm_frame(processed))
    elif kind == "deepfm":
        raw = model.predict(features.to_deepfm_inputs(processed), verbose=0).ravel()
        proba = np.column_stack([1 - raw, raw])
    else:
        raise ValueError(f"unknown input_kind: {kind}")
    return proba[:, 1]


class Bundle:
    """Preprocessing + model as saved by notebook 02."""

    def __init__(self, artifacts: Path) -> None:
        self.metadata = json.loads((artifacts / "model_choice.json").read_text(encoding="utf-8"))
        self.preprocessor = features.Preprocessor.load(artifacts / "preprocessor.joblib")
        self.kind = self.metadata["input_kind"]
        model_path = artifacts / self.metadata["model_file"]
        if self.kind == "deepfm":
            import keras

            from .deepfm import FMInteraction  # noqa: F401  (registers the custom layer)

            self.model = keras.models.load_model(model_path)
        else:
            import joblib

            self.model = joblib.load(model_path)
        self.model_name = self.metadata["model_name"]
        self.model_version = self.metadata["model_version"]

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return predict_proba(self.kind, self.model, self.preprocessor, frame)


def create_app(artifacts: str | Path | None = None) -> FastAPI:
    cfg = load_config()
    directory = Path(artifacts) if artifacts else default_artifacts_dir(cfg)
    app = FastAPI(title="Ad CTR Prediction API", version=__version__)
    bundle = Bundle(directory) if (directory / "model_choice.json").exists() else None
    app.state.bundle = bundle

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_loaded": app.state.bundle is not None}

    @app.get("/model-info")
    def model_info() -> dict:
        if app.state.bundle is None:
            raise HTTPException(status_code=503, detail="model artifacts not found; run notebooks 02/03")
        return {
            "model_name": app.state.bundle.model_name,
            "model_version": app.state.bundle.model_version,
            "metrics": app.state.bundle.metadata.get("val_metrics", {}),
        }

    @app.post("/predict")
    def predict(payload: ImpressionFeatures) -> dict:
        if app.state.bundle is None:
            raise HTTPException(status_code=503, detail="model artifacts not found; run notebooks 02/03")
        frame = pd.DataFrame([payload.model_dump()])
        probability = float(app.state.bundle.predict_proba(frame)[0])
        return {
            "click_probability": probability,
            "model_name": app.state.bundle.model_name,
            "model_version": app.state.bundle.model_version,
        }

    return app


app = create_app()
