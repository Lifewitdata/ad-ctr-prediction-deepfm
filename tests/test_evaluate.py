"""Evaluation utilities: hand-checked metric values, calibration tables,
latency benchmarking and segment reporting."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from ctr import evaluate


def test_binary_metrics_hand_computed():
    y = np.array([1, 0])
    p = np.array([0.8, 0.4])
    metrics = evaluate.binary_metrics(y, p)
    expected_log_loss = -(np.log(0.8) + np.log(0.6)) / 2
    assert metrics["log_loss"] == pytest.approx(expected_log_loss, abs=1e-9)
    assert metrics["brier"] == pytest.approx((0.04 + 0.16) / 2, abs=1e-9)
    assert metrics["roc_auc"] == 1.0
    assert metrics["n"] == 2
    assert metrics["base_rate"] == 0.5


def test_binary_metrics_constant_half():
    y = np.array([0, 0, 1, 1])
    p = np.full(4, 0.5)
    metrics = evaluate.binary_metrics(y, p)
    assert metrics["log_loss"] == pytest.approx(np.log(2), abs=1e-9)
    assert metrics["brier"] == pytest.approx(0.25, abs=1e-9)
    assert metrics["roc_auc"] == 0.5


def test_binary_metrics_perfect_separation():
    y = np.array([0] * 5 + [1] * 5)
    p = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.75, 0.8, 0.85, 0.9, 0.95])
    metrics = evaluate.binary_metrics(y, p)
    assert metrics["roc_auc"] == 1.0
    assert metrics["average_precision"] == 1.0
    assert metrics["log_loss"] < 0.25
    assert metrics["brier"] < 0.05


def test_binary_metrics_clips_extreme_probabilities():
    y = np.array([1, 0])
    metrics = evaluate.binary_metrics(y, np.array([1.0, 0.0]))
    assert np.isfinite(metrics["log_loss"])
    assert metrics["log_loss"] == pytest.approx(-np.log(1 - evaluate.EPS), abs=1e-9)


def test_reliability_table_partitions_all_rows():
    rng = np.random.default_rng(0)
    y = (rng.random(1000) < 0.3).astype(int)
    p = np.clip(y * 0.3 + rng.normal(0.3, 0.1, 1000), 0, 1)
    table = evaluate.reliability_table(y, p, n_bins=15)
    assert table["n"].sum() == 1000
    assert list(table.columns) == ["bin", "n", "observed_ctr", "mean_predicted", "ci_low", "ci_high"]
    # the Wilson interval is built on the rounded click count, so allow 0.5/n slack
    slack = 0.5 / table["n"]
    assert (table["ci_low"] - slack <= table["observed_ctr"]).all()
    assert (table["observed_ctr"] <= table["ci_high"] + slack).all()


def test_expected_calibration_error_zero_when_perfectly_calibrated():
    # two groups, each exactly at its predicted rate
    y = np.array([1] * 100 + [0] * 400 + [1] * 400 + [0] * 100)
    p = np.array([0.2] * 500 + [0.8] * 500)
    table = evaluate.reliability_table(y, p, n_bins=10)
    assert evaluate.expected_calibration_error(table) == pytest.approx(0.0, abs=1e-12)


def test_expected_calibration_error_positive_when_miscalibrated():
    # predictions say 0.25/0.75, but the observed rate is 1.0/0.0 in both bins
    y = np.array([1] * 500 + [0] * 500)
    p = np.array([0.25] * 500 + [0.75] * 500)
    table = evaluate.reliability_table(y, p, n_bins=10)
    assert evaluate.expected_calibration_error(table) == pytest.approx(0.75, abs=1e-9)


def test_latency_benchmark_shape_and_ordering():
    counter = {"calls": 0}

    def predict(x):
        counter["calls"] += 1
        time.sleep(0.001)

    x = np.zeros((300, 4))
    table = evaluate.latency_benchmark(predict, x, batch_sizes=(1, 64), reps=5, warmup=2)
    assert list(table["batch_size"]) == [1, 64]
    assert (table["p50_ms"] > 0).all()
    assert (table["p50_ms"] <= table["p95_ms"]).all()
    assert (table["rows_per_second"] > 0).all()
    # warm-up + timed reps for each batch size
    assert counter["calls"] == (2 + 5) * 2


def test_latency_benchmark_dict_inputs():
    def predict(payload):
        assert set(payload) == {"a", "b"}

    x = {"a": np.zeros((64, 2)), "b": np.zeros((64, 1))}
    table = evaluate.latency_benchmark(predict, x, batch_sizes=(16,), reps=3, warmup=1)
    assert table.iloc[0]["batch_size"] == 16


def test_segment_report_masks():
    y = np.array([0, 1, 0, 1, 1, 0])
    p = np.array([0.1, 0.9, 0.2, 0.8, 0.6, 0.3])
    segments = {"all": np.ones(6, dtype=bool), "half": np.array([1, 1, 1, 0, 0, 0], dtype=bool),
                "empty": np.zeros(6, dtype=bool)}
    table = evaluate.segment_report(y, p, segments)
    assert list(table["segment"]) == ["all", "half"]  # empty masks are dropped
    assert table.loc[table["segment"] == "all", "n"].iloc[0] == 6
    assert table.loc[table["segment"] == "half", "n"].iloc[0] == 3
