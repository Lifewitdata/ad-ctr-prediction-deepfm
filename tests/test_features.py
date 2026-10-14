"""Preprocessing invariants: train-only fitting, imputation, OOV mapping,
missing indicators, the three model-input layouts, and persistence."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ctr import features

from conftest import RARE_VALUE, SEEN_A, SEEN_B, make_frame


def test_indicator_columns_match_frozen_config(cfg, preprocessor):
    expected = set(cfg["features"]["missing_indicator_columns"])
    assert set(preprocessor.missing_indicator_columns) == expected
    assert len(expected) == 23  # notebook 01 froze 23 of 24 gappy columns


def test_transform_shapes_and_dtypes(cfg, preprocessor, holdout_frame):
    processed = preprocessor.transform(holdout_frame)
    n_num = len(cfg["features"]["numeric_cols"])
    n_ind = len(cfg["features"]["missing_indicator_columns"])
    assert processed["Xnum"].shape == (len(holdout_frame), n_num + n_ind)
    assert processed["Xcat"].shape == (len(holdout_frame), 26)
    assert processed["Xnum"].dtypes.eq("float32").all()
    assert processed["Xcat"].dtypes.eq("int32").all()
    assert np.isfinite(processed["Xnum"].to_numpy()).all()
    assert (processed["Xcat"].to_numpy() >= 0).all()


def test_transform_adds_all_indicator_columns(preprocessor, holdout_frame):
    processed = preprocessor.transform(holdout_frame)
    for col in preprocessor.missing_indicator_columns:
        assert f"{col}_missing" in processed["Xnum"].columns
    # indicator fires exactly where the raw value is null
    mask = holdout_frame["I8"].isna()
    assert (processed["Xnum"]["I8_missing"].to_numpy() == mask.astype("float32")).all()


def test_label_passthrough(preprocessor, holdout_frame):
    processed = preprocessor.transform(holdout_frame)
    assert "y" in processed
    assert np.array_equal(processed["y"], holdout_frame["label"].to_numpy(dtype=np.int8))
    assert "y" not in preprocessor.transform(holdout_frame.drop(columns=["label"]))


def test_missing_numeric_is_imputed_with_train_median(preprocessor, train_frame, holdout_frame):
    """The frozen median - not the holdout median - fills missing values."""
    processed = preprocessor.transform(holdout_frame)
    numeric = preprocessor.numeric
    nan_rows = holdout_frame["I1"].isna()
    assert nan_rows.any()
    expected = (numeric.medians_["I1"] - numeric.means_["I1"]) / numeric.stds_["I1"]
    got = processed["Xnum"].loc[nan_rows, "I1"].to_numpy()
    assert np.allclose(got, expected, atol=1e-5)


def test_log1p_applied_only_to_skewed_train_columns(preprocessor):
    # I2 is strongly right-skewed in the fixture; I1 is roughly normal
    assert "I2" in preprocessor.numeric.log1p_columns
    assert "I1" not in preprocessor.numeric.log1p_columns
    assert all(preprocessor.numeric.skew_[c] is not None for c in preprocessor.numeric.columns)


def test_vocab_drops_rare_train_values(preprocessor, train_frame):
    """A value seen in train but below min_count still maps to OOV 0."""
    assert RARE_VALUE in set(train_frame["C5"])
    assert RARE_VALUE not in preprocessor.vocab.maps_["C5"]
    processed = preprocessor.transform(train_frame)
    rare_rows = train_frame["C5"] == RARE_VALUE
    assert (processed["Xcat"].loc[rare_rows, "C5"] == 0).all()


def test_vocab_keeps_frequent_train_values(preprocessor, train_frame):
    code_a = preprocessor.vocab.maps_["C1"][SEEN_A]
    code_b = preprocessor.vocab.maps_["C1"][SEEN_B]
    assert code_a >= 1 and code_b >= 1 and code_a != code_b
    processed = preprocessor.transform(train_frame)
    assert (processed["Xcat"].loc[train_frame["C1"] == SEEN_A, "C1"] == code_a).all()


def test_unseen_and_missing_categories_map_to_oov(preprocessor):
    row = make_frame(n=4, seed=99)
    row["C1"] = ["zzzzzzzz", "still-unseen", SEEN_A, np.nan]  # noqa: PD005
    row["C2"] = [np.nan, np.nan, np.nan, np.nan]
    processed = preprocessor.transform(row)
    assert (processed["Xcat"]["C1"].to_numpy()[:2] == features.OOV_INDEX).all()
    assert processed["Xcat"]["C1"].iloc[2] == preprocessor.vocab.maps_["C1"][SEEN_A]
    assert processed["Xcat"]["C1"].iloc[3] == features.OOV_INDEX
    assert (processed["Xcat"]["C2"] == features.OOV_INDEX).all()


def test_vocab_sizes_reserve_oov_slot(preprocessor):
    sizes = features.vocab_sizes(preprocessor)
    assert len(sizes) == 26
    for col, size in zip(preprocessor.vocab.columns, sizes):
        assert size == len(preprocessor.vocab.maps_[col]) + 1
        assert size >= 1


def test_unseen_fraction(preprocessor, holdout_frame):
    fraction = preprocessor.vocab.unseen_fraction(holdout_frame)
    for value in fraction.values():
        assert 0.0 <= value <= 1.0
    absent = holdout_frame.copy()
    absent["C11"] = np.nan
    assert np.isnan(preprocessor.vocab.unseen_fraction(absent)["C11"])
    all_unseen = holdout_frame.copy()
    all_unseen["C11"] = "not-in-vocab"
    assert preprocessor.vocab.unseen_fraction(all_unseen)["C11"] == 1.0


def test_fit_does_not_mutate_input_frame(preprocessor_factory, train_frame):
    before = train_frame.copy(deep=True)
    features.Preprocessor.from_config({"numeric_cols": ["I1"], "categorical_cols": ["C1"],
                                       "skew_threshold": 2.0, "min_category_count": 2,
                                       "add_missing_indicators": False}).fit(train_frame)
    pd.testing.assert_frame_equal(train_frame, before)


@pytest.fixture(scope="module")
def preprocessor_factory():
    return features.Preprocessor


def test_to_lr_matrix_layout(cfg, preprocessor, holdout_frame):
    processed = preprocessor.transform(holdout_frame)
    matrix = features.to_lr_matrix(preprocessor, processed)
    n_num = len(cfg["features"]["numeric_cols"]) + len(cfg["features"]["missing_indicator_columns"])
    assert matrix.shape == (len(holdout_frame), sum(features.vocab_sizes(preprocessor)) + n_num)
    # every row owns exactly 26 one-hot entries (one per category column) or fewer if
    # a category code lands on OOV... OOV still sets a 1 in column 0, so exactly 26
    row_counts = np.asarray(matrix[:, : sum(features.vocab_sizes(preprocessor))].sum(axis=1)).ravel()
    assert (row_counts == 26).all()


def test_to_lgbm_frame_layout(preprocessor, holdout_frame):
    frame = features.to_lgbm_frame(preprocessor.transform(holdout_frame))
    numeric = preprocessor.transform(holdout_frame)["Xnum"]
    assert list(frame.columns[:26]) == preprocessor.vocab.columns
    assert frame.shape == (len(holdout_frame), 26 + numeric.shape[1])
    assert frame[preprocessor.vocab.columns].dtypes.eq("int32").all()


def test_to_deepfm_inputs_layout(preprocessor, holdout_frame):
    processed = preprocessor.transform(holdout_frame)
    inputs = features.to_deepfm_inputs(processed)
    assert set(inputs) == {"num"} | set(preprocessor.vocab.columns)
    assert inputs["num"].shape == processed["Xnum"].shape
    for col in preprocessor.vocab.columns:
        assert inputs[col].shape == (len(holdout_frame), 1)
        assert inputs[col].dtype == np.int32


def test_missing_indicator_disabled_without_config(train_frame):
    prep = features.Preprocessor.from_config(
        {"numeric_cols": ["I1", "I8"], "categorical_cols": ["C1", "C3"],
         "skew_threshold": 2.0, "min_category_count": 2, "add_missing_indicators": False}
    ).fit(train_frame)
    processed = prep.transform(train_frame)
    assert list(processed["Xnum"].columns) == ["I1", "I8"]


def test_missing_indicator_fallback_flags_gappy_columns(train_frame):
    """Without a frozen list, fit() flags every column with any missing value."""
    prep = features.Preprocessor.from_config(
        {"numeric_cols": ["I1", "I8"], "categorical_cols": ["C1", "C3"],
         "skew_threshold": 2.0, "min_category_count": 2, "add_missing_indicators": True}
    ).fit(train_frame)
    assert set(prep.missing_indicator_columns) == {"I1", "I8", "C3"}


def test_persistence_roundtrip_identical_output(preprocessor, holdout_frame, tmp_path):
    path = tmp_path / "prep.joblib"
    preprocessor.save(path)
    loaded = features.Preprocessor.load(path)
    a = preprocessor.transform(holdout_frame)
    b = loaded.transform(holdout_frame)
    pd.testing.assert_frame_equal(a["Xnum"], b["Xnum"])
    pd.testing.assert_frame_equal(a["Xcat"], b["Xcat"])
    assert loaded.missing_indicator_columns == preprocessor.missing_indicator_columns
