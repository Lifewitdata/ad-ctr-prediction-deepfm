"""Serving layer: artifact loading, the shared scoring path, and the FastAPI
endpoints. Artifacts are built from the fixture preprocessor, so the tests
run without the notebooks having executed."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ctr import features, service

from conftest import SEEN_A, make_frame


@pytest.fixture(scope="session")
def client(bundle_dir):
    return TestClient(service.create_app(bundle_dir))


@pytest.fixture(scope="session")
def deepfm_client(deepfm_bundle_dir):
    return TestClient(service.create_app(deepfm_bundle_dir))


def _record(frame: pd.DataFrame, i: int = 0) -> dict:
    row = frame.iloc[i]
    record = {}
    for col in frame.columns:
        if col == "label":
            continue
        value = row[col]
        record[col] = None if pd.isna(value) else (
            float(value) if col.startswith("I") else str(value)
        )
    return record


# --- Bundle / scoring path -------------------------------------------------

def test_bundle_metadata_matches_choice(bundle_dir):
    bundle = service.Bundle(bundle_dir)
    assert bundle.kind == "lr_sparse"
    assert bundle.model_name == "logistic_regression"
    assert bundle.model_version == "pytest"


def test_predict_proba_dispatch_all_kinds(bundle_dir, deepfm_bundle_dir, preprocessor, holdout_frame):
    frame = holdout_frame.iloc[:32]
    lr_bundle = service.Bundle(bundle_dir)
    p_lr = service.predict_proba("lr_sparse", lr_bundle.model, lr_bundle.preprocessor, frame)
    assert p_lr.shape == (32,)
    assert ((p_lr > 0) & (p_lr < 1)).all()

    dfm_bundle = service.Bundle(deepfm_bundle_dir)
    p_dfm = service.predict_proba("deepfm", dfm_bundle.model, dfm_bundle.preprocessor, frame)
    assert p_dfm.shape == (32,)
    assert ((p_dfm > 0) & (p_dfm < 1)).all()


def test_predict_proba_rejects_unknown_kind(preprocessor, holdout_frame):
    with pytest.raises(ValueError, match="unknown input_kind"):
        service.predict_proba("not-a-kind", None, preprocessor, holdout_frame.iloc[:5])


def test_bundle_agrees_with_module_scoring(bundle_dir, holdout_frame):
    bundle = service.Bundle(bundle_dir)
    direct = service.predict_proba(bundle.kind, bundle.model, bundle.preprocessor, holdout_frame.iloc[:16])
    via_bundle = bundle.predict_proba(holdout_frame.iloc[:16])
    assert np.array_equal(direct, via_bundle)


# --- HTTP endpoints ---------------------------------------------------------

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    payload = response.json()
    assert payload["model_name"] == "logistic_regression"
    assert payload["model_version"] == "pytest"


def test_predict_returns_probability(client, holdout_frame):
    response = client.post("/predict", json=_record(holdout_frame, 0))
    assert response.status_code == 200
    payload = response.json()
    assert 0.0 < payload["click_probability"] < 1.0
    assert payload["model_name"] == "logistic_regression"
    assert payload["model_version"] == "pytest"


def test_predict_matches_direct_scoring(client, bundle_dir, holdout_frame):
    record = _record(holdout_frame, 3)
    response = client.post("/predict", json=record)
    bundle = service.Bundle(bundle_dir)
    direct = float(bundle.predict_proba(pd.DataFrame([record]))[0])
    assert response.json()["click_probability"] == pytest.approx(direct, abs=1e-12)


def test_predict_is_deterministic(client, holdout_frame):
    record = _record(holdout_frame, 1)
    a = client.post("/predict", json=record).json()["click_probability"]
    b = client.post("/predict", json=record).json()["click_probability"]
    assert a == b


def test_predict_accepts_all_missing_record(client):
    """Everything null exercises imputation + OOV end to end."""
    response = client.post("/predict", json={})
    assert response.status_code == 200
    assert 0.0 < response.json()["click_probability"] < 1.0


def test_predict_accepts_unseen_category(client):
    response = client.post("/predict", json={"C1": "value-never-seen"})
    assert response.status_code == 200


def test_predict_allows_explicit_nulls(client, holdout_frame):
    record = _record(holdout_frame, 2)
    record["I1"] = None
    record["C1"] = None
    response = client.post("/predict", json=record)
    assert response.status_code == 200


def test_predict_rejects_unknown_fields(client, holdout_frame):
    record = _record(holdout_frame, 0)
    record["not_a_feature"] = 1.0
    response = client.post("/predict", json=record)
    assert response.status_code == 422


def test_predict_rejects_wrong_types(client):
    response = client.post("/predict", json={"I1": "not-a-number"})
    assert response.status_code == 422


def test_missing_artifacts_returns_503(tmp_path):
    app = service.create_app(tmp_path)  # empty directory
    client = TestClient(app)
    assert client.get("/health").json()["model_loaded"] is False
    assert client.get("/model-info").status_code == 503
    assert client.post("/predict", json={}).status_code == 503


def test_deepfm_bundle_serves_predictions(deepfm_client, holdout_frame):
    response = deepfm_client.post("/predict", json=_record(holdout_frame, 5))
    assert response.status_code == 200
    payload = response.json()
    assert 0.0 < payload["click_probability"] < 1.0
    assert payload["model_name"] == "deepfm"


def test_service_roundtrip_preserves_ranking_within_a_record_batch(client, bundle_dir, holdout_frame):
    """Scoring the same 20 rows record-by-record and in one call agree."""
    frame = holdout_frame.iloc[:20]
    batch = service.Bundle(bundle_dir).predict_proba(frame)
    singles = np.array(
        [client.post("/predict", json=_record(frame, i)).json()["click_probability"] for i in range(len(frame))]
    )
    assert np.allclose(batch, singles, atol=1e-9)
