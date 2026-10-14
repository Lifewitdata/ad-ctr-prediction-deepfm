"""Model wrappers: baseline behaviour, the FM interaction formula, and a
one-epoch DeepFM smoke test including the Keras serializer round trip."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ctr import models

from conftest import make_frame


def test_constant_model_predicts_train_ctr():
    y = np.array([0, 0, 1, 1, 1])
    model = models.ConstantModel().fit(None, y)
    proba = model.predict_proba(np.zeros((7, 2)))
    assert proba.shape == (7, 2)
    assert np.allclose(proba[:, 1], 0.6)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_logistic_model_learns_separable_signal():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(800, 3))
    y = (0.9 * x[:, 0] + 0.5 * x[:, 1] > 0.4).astype(int)
    model = models.LogisticModel(C=1.0, max_iter=500).fit(x, y)
    proba = model.predict_proba(x)
    assert proba.shape == (800, 2)
    assert ((proba >= 0) & (proba <= 1)).all()
    from ctr.evaluate import binary_metrics

    assert binary_metrics(y, proba[:, 1])["roc_auc"] > 0.85


def test_lgbm_model_fits_with_early_stopping():
    rng = np.random.default_rng(1)
    x = pd.DataFrame(
        {
            "C1": rng.integers(0, 4, 600),
            "C2": rng.integers(0, 3, 600),
            "I1": rng.normal(size=600),
        }
    )
    y = ((x["C1"] == 1) | (x["I1"] > 0.8)).astype(int)
    params = {"n_estimators": 60, "learning_rate": 0.1, "num_leaves": 7}
    model = models.LGBMModel(params, cat_columns=["C1", "C2"], seed=42)
    model.fit(x.iloc[:480], y.iloc[:480], eval_set=(x.iloc[480:], y.iloc[480:]))
    proba = model.predict_proba(x.iloc[480:])
    assert proba.shape == (120, 2)
    assert model.best_iteration_ is not None


def test_fm_interaction_matches_manual_formula():
    import keras

    from ctr.deepfm import FMInteraction

    rng = np.random.default_rng(2)
    embeddings = rng.normal(size=(3, 5, 4)).astype("float32")
    layer = FMInteraction()
    got = np.asarray(layer(keras.ops.convert_to_tensor(embeddings)))
    summed = embeddings.sum(axis=1)
    manual = 0.5 * (np.square(summed) - np.square(embeddings).sum(axis=1)).sum(axis=-1, keepdims=True)
    assert np.allclose(got, manual, atol=1e-5)
    assert got.shape == (3, 1)


def test_deepfm_build_and_predict_shapes():
    from ctr import deepfm as ctr_deepfm

    vocabulary = [5, 4, 3]
    model = ctr_deepfm.build_deepfm(numeric_dim=2, vocab_sizes=vocabulary, embedding_dim=3,
                                    hidden_units=(6, 4))
    inputs = {"num": np.zeros((7, 2), dtype="float32")}
    for i, size in enumerate(vocabulary):
        inputs[f"C{i + 1}"] = np.ones((7, 1), dtype="int32")
    outputs = np.asarray(model.predict(inputs, verbose=0))
    assert outputs.shape == (7, 1)
    assert ((outputs >= 0) & (outputs <= 1)).all()


@pytest.mark.slow
def test_deepfm_one_epoch_fit_and_prediction():
    from ctr import deepfm as ctr_deepfm

    frame = make_frame(n=240, seed=5)
    processed = _mini_processed(frame)
    inputs = {k: v[:180] for k, v in processed.items()}
    val = {k: v[180:] for k, v in processed.items()}
    wrapper = ctr_deepfm.DeepFMWrapper(
        numeric_dim=inputs["num"].shape[1],
        vocab_sizes=[6, 5, 5],
        embedding_dim=3,
        hidden_units=(8, 4),
        dropout=0.0,
        batch_size=64,
        seed=42,
    )
    wrapper.fit(inputs, frame["label"].to_numpy()[:180], validation_data=(val, frame["label"].to_numpy()[180:]),
                epochs=1, patience=1)
    assert wrapper.history_ is not None and len(wrapper.history_.history["loss"]) == 1
    proba = wrapper.predict_proba(val)
    assert proba.shape == (60, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def _mini_processed(frame):
    """Encode the fixture into DeepFM input tensors without a full Preprocessor."""
    rng_ok = frame[["I1", "I3"]].fillna(0.0).to_numpy(dtype="float32")
    inputs = {"num": rng_ok}
    for col, size in (("C1", 6), ("C2", 5), ("C3", 5)):
        codes = frame[col].astype("object").map(
            {v: i + 1 for i, v in enumerate(frame[col].dropna().unique()[: size - 1])}
        ).fillna(0).astype("int32")
        inputs[col] = codes.to_numpy().reshape(-1, 1)
    return inputs


def test_deepfm_wrapper_save_load_roundtrip(tmp_path):
    """The saved format must reload through the service's Bundle path."""
    import keras

    import ctr.deepfm  # noqa: F401  (registers the custom layer)
    from ctr import deepfm as ctr_deepfm

    wrapper = ctr_deepfm.DeepFMWrapper(
        numeric_dim=2, vocab_sizes=[5, 4], embedding_dim=3, hidden_units=(6, 4),
        dropout=0.0, seed=42,
    )
    path = tmp_path / "deepfm.keras"
    wrapper.save(path)
    loaded = keras.models.load_model(path)
    inputs = {"num": np.zeros((4, 2), dtype="float32"), "C1": np.ones((4, 1), dtype="int32"),
              "C2": np.ones((4, 1), dtype="int32")}
    a = np.asarray(wrapper.model_.predict(inputs, verbose=0))
    b = np.asarray(loaded.predict(inputs, verbose=0))
    assert np.allclose(a, b, atol=1e-6)
