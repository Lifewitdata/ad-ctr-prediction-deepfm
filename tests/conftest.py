"""Shared fixtures.

The suite runs on a small synthetic frame with the real Criteo schema
(``label`` + ``I1``-``I13`` + ``C1``-``C26``) and a learnable signal, so
neither the downloaded data nor the trained artifacts are required: the
service tests build their own artifacts in a temp directory from the same
``Preprocessor`` and model classes the notebooks use.
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from ctr import features  # noqa: E402
from ctr.config import load_config  # noqa: E402

RARE_VALUE = "ffff0000"  # appears 3 times: below min_category_count=10
SEEN_A, SEEN_B = "a1b2c3d4", "00ff00ff"


def make_frame(n: int = 1200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame()
    frame["I1"] = rng.normal(0, 1, n)
    frame["I2"] = np.abs(rng.lognormal(0, 2.5, n))  # heavy right skew -> log1p
    frame["I3"] = rng.normal(5, 3, n)
    for i in range(4, 14):
        frame[f"I{i}"] = rng.normal(0, 1, n).round(1)
    for i in range(1, 27):
        frame[f"C{i}"] = rng.choice(["a1b2c3d4", "00ff00ff", "12345678", "abcdef01"], n)
    frame["C1"] = rng.choice([SEEN_A, SEEN_B], n, p=[0.4, 0.6])
    frame["C2"] = rng.choice(["deadbeef", "cafebabe"], n, p=[0.5, 0.5])
    frame.loc[frame.index[:3], "C5"] = RARE_VALUE
    frame.loc[rng.random(n) < 0.10, "I1"] = np.nan
    frame.loc[rng.random(n) < 0.15, "I8"] = np.nan
    frame.loc[rng.random(n) < 0.05, "C3"] = np.nan
    logit = (
        1.4 * frame["I1"].fillna(0.0).to_numpy()
        + 1.6 * (frame["C1"] == SEEN_A).to_numpy()
        + 1.0 * (frame["C2"] == "deadbeef").to_numpy()
        - 1.9
    )
    frame["label"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.int8)
    return frame


@pytest.fixture(scope="session")
def cfg() -> dict:
    return load_config()


@pytest.fixture(scope="session")
def frame() -> pd.DataFrame:
    return make_frame()


@pytest.fixture(scope="session")
def train_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.iloc[:900].reset_index(drop=True)


@pytest.fixture(scope="session")
def holdout_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.iloc[900:].reset_index(drop=True)


@pytest.fixture(scope="session")
def preprocessor(cfg: dict, train_frame: pd.DataFrame) -> features.Preprocessor:
    """Fitted with the project's real feature config (23 frozen indicators)."""
    return features.Preprocessor.from_config(cfg["features"]).fit(train_frame)


def _write_choice(directory, kind: str, model_file: str, name: str) -> None:
    payload = {
        "model_name": name,
        "model_version": "pytest",
        "input_kind": kind,
        "model_file": model_file,
        "selection_rule": "fixture",
        "val_metrics": {"log_loss": 0.0},
    }
    (directory / "model_choice.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture(scope="session")
def bundle_dir(tmp_path_factory, preprocessor, train_frame):
    """A complete saved-artifact directory holding the selected model type."""
    import joblib

    from ctr import models

    directory = tmp_path_factory.mktemp("artifacts_lr")
    preprocessor.save(directory / "preprocessor.joblib")
    processed = preprocessor.transform(train_frame)
    x = features.to_lr_matrix(preprocessor, processed)
    y = train_frame["label"].to_numpy()
    joblib.dump(models.LogisticModel(C=1.0, max_iter=500).fit(x, y), directory / "logistic_regression.joblib")
    _write_choice(directory, "lr_sparse", "logistic_regression.joblib", "logistic_regression")
    return directory


@pytest.fixture(scope="session")
def deepfm_bundle_dir(tmp_path_factory, preprocessor, train_frame):
    """Saved DeepFM artifacts, exercising the keras serializer round trip."""
    from ctr import deepfm as ctr_deepfm

    directory = tmp_path_factory.mktemp("artifacts_deepfm")
    preprocessor.save(directory / "preprocessor.joblib")
    numeric_dim = preprocessor.transform(train_frame)["Xnum"].shape[1]
    wrapper = ctr_deepfm.DeepFMWrapper(
        numeric_dim=numeric_dim,
        vocab_sizes=features.vocab_sizes(preprocessor),
        embedding_dim=4,
        hidden_units=(8, 4),
        dropout=0.0,
        learning_rate=1e-3,
        batch_size=64,
        seed=42,
    )
    wrapper.save(directory / "deepfm.keras")
    _write_choice(directory, "deepfm", "deepfm.keras", "deepfm")
    return directory
