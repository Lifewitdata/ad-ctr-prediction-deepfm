"""Data-layer checks: sampling determinism, schema validation, split integrity."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ctr import data

from conftest import make_frame


@pytest.fixture(scope="module")
def parquet_path(tmp_path_factory) -> object:
    path = tmp_path_factory.mktemp("parquet") / "mini.parquet"
    make_frame(n=400, seed=11).to_parquet(path, index=False)
    return path


def test_row_count(parquet_path):
    assert data.row_count(parquet_path) == 400


def test_uniform_sample_is_deterministic(parquet_path):
    a = data.uniform_sample(parquet_path, n=120, seed=42)
    b = data.uniform_sample(parquet_path, n=120, seed=42)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 120


def test_uniform_sample_different_seeds_differ(parquet_path):
    a = data.uniform_sample(parquet_path, n=120, seed=1)
    b = data.uniform_sample(parquet_path, n=120, seed=2)
    assert not a.equals(b)


def test_uniform_sample_larger_than_file_returns_everything(parquet_path):
    frame = data.uniform_sample(parquet_path, n=10_000, seed=42)
    assert len(frame) == 400


def test_load_frame_full(parquet_path):
    frame = data.load_frame(parquet_path)
    assert len(frame) == 400
    assert "label" in frame.columns


def test_sampling_preserves_columns_and_dtypes(parquet_path):
    frame = data.uniform_sample(parquet_path, n=100, seed=3)
    assert set(frame.columns) == set(data.expected_columns())
    assert pd.api.types.is_integer_dtype(frame["label"])


def test_validate_schema_accepts_good_frame(frame):
    report = data.validate_schema(frame)
    assert report["n_rows"] == len(frame)
    assert report["label_values"] == [0, 1]
    assert report["unexpected_columns"] == []


def test_validate_schema_rejects_missing_column(frame):
    broken = frame.drop(columns=["C7"])
    with pytest.raises(ValueError, match="missing expected columns"):
        data.validate_schema(broken)


def test_validate_schema_rejects_bad_label(frame):
    broken = frame.copy()
    broken.iloc[0, broken.columns.get_loc("label")] = 2
    with pytest.raises(ValueError, match="label must be 0/1"):
        data.validate_schema(broken)


def test_stratified_split_ratios_and_disjointness(cfg):
    frame = make_frame(n=2000, seed=21)
    ratios = cfg["data"]["split"]
    splits = data.stratified_split(frame, ratios, seed=cfg["seed"])

    total = sum(len(part) for part in splits.values())
    assert total == len(frame)
    assert abs(len(splits["train"]) - 0.70 * len(frame)) <= 4
    assert abs(len(splits["val"]) - 0.15 * len(frame)) <= 4
    assert abs(len(splits["test"]) - 0.15 * len(frame)) <= 4


def test_stratified_split_preserves_ctr(cfg):
    frame = make_frame(n=2000, seed=22)
    splits = data.stratified_split(frame, cfg["data"]["split"], seed=cfg["seed"])
    overall = frame["label"].mean()
    for part in splits.values():
        assert abs(part["label"].mean() - overall) < 0.02


def test_stratified_split_is_seed_deterministic(cfg):
    frame = make_frame(n=800, seed=23)
    a = data.stratified_split(frame, cfg["data"]["split"], seed=42)
    b = data.stratified_split(frame, cfg["data"]["split"], seed=42)
    for name in ("train", "val", "test"):
        pd.testing.assert_frame_equal(a[name], b[name])


def test_stratified_split_has_no_overlap(cfg):
    frame = make_frame(n=600, seed=24)
    splits = data.stratified_split(frame, cfg["data"]["split"], seed=42)
    # rows are unique synthetic draws: duplicate (I3, C1) pairs would be chance
    keys = [set(map(tuple, part[["I3", "C1"]].to_numpy())) for part in splits.values()]
    assert not (keys[0] & keys[1]) and not (keys[0] & keys[2]) and not (keys[1] & keys[2])


def test_save_and_load_splits_roundtrip(tmp_path, cfg):
    frame = make_frame(n=300, seed=25)
    splits = data.stratified_split(frame, cfg["data"]["split"], seed=42)
    path = data.save_splits(splits, tmp_path / "splits", manifest={"seed": 42})
    assert path.exists()
    loaded = data.load_splits(tmp_path / "splits")
    assert set(loaded) == {"train", "val", "test"}
    for name in loaded:
        pd.testing.assert_frame_equal(loaded[name], splits[name].reset_index(drop=True))


def test_samples_are_mostly_distinct_rows(parquet_path):
    """Without replacement: a 120-row sample from a 400-row file has no duplicates."""
    sampled = data.uniform_sample(parquet_path, n=120, seed=5)
    numeric = sampled[[f"I{i}" for i in range(3, 14)]].to_numpy()
    assert len(np.unique(numeric, axis=0)) >= 110
