"""Offline evaluation: ranking metrics, calibration tables, prediction
latency and per-segment reports. All metrics assume the same held-out rows
for every model being compared."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from .eda import wilson_ci

EPS = 1e-7


def binary_metrics(y_true, p) -> dict:
    """Log loss (primary), ROC-AUC, average precision (the PR-AUC variant we
    report: sklearn's average_precision_score, i.e. the step-wise precision of
    the precision-recall curve), and the Brier score."""
    y_true = np.asarray(y_true).astype("float64")
    p = np.clip(np.asarray(p, dtype="float64"), EPS, 1 - EPS)
    return {
        "n": int(len(y_true)),
        "base_rate": float(y_true.mean()),
        "log_loss": float(log_loss(y_true, p, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y_true, p)),
        "average_precision": float(average_precision_score(y_true, p)),
        "brier": float(brier_score_loss(y_true, p)),
    }


def reliability_table(y_true, p, n_bins: int = 15) -> pd.DataFrame:
    """Quantile-binned calibration table: mean predicted vs observed CTR."""
    y_true = np.asarray(y_true).astype("float64")
    p = np.asarray(p, dtype="float64")
    bins = pd.qcut(p, q=n_bins, duplicates="drop")
    grouped = pd.DataFrame({"y": y_true, "p": p, "bin": bins}).groupby("bin", observed=True)
    table = grouped.agg(n=("y", "size"), observed_ctr=("y", "mean"), mean_predicted=("p", "mean"))
    table = table.reset_index()
    cis = [wilson_ci(int(round(row.observed_ctr * row.n)), int(row.n)) for row in table.itertuples()]
    table["ci_low"] = [c[0] for c in cis]
    table["ci_high"] = [c[1] for c in cis]
    return table


def expected_calibration_error(table: pd.DataFrame) -> float:
    """ECE = sum over bins of (n_bin / n) * |observed - predicted|."""
    n = table["n"].sum()
    return float(((table["n"] / n) * (table["observed_ctr"] - table["mean_predicted"]).abs()).sum())


def latency_benchmark(
    predict_fn,
    X,
    batch_sizes=(1, 32, 256, 1024),
    reps: int = 50,
    warmup: int = 5,
) -> pd.DataFrame:
    """Median / p95 wall-clock latency per batch size, single process, CPU.

    ``predict_fn`` must accept the same structure as the model inputs; X must
    contain at least max(batch_sizes) rows.
    """
    rows = []
    for batch in batch_sizes:
        sub = _take_rows(X, 0, batch)
        for _ in range(warmup):
            predict_fn(sub)
        times = []
        for _ in range(reps):
            start = time.perf_counter()
            predict_fn(sub)
            times.append(time.perf_counter() - start)
        times = np.asarray(times)
        p50 = float(np.percentile(times, 50))
        rows.append(
            {
                "batch_size": int(batch),
                "p50_ms": p50 * 1000,
                "p95_ms": float(np.percentile(times, 95)) * 1000,
                "rows_per_second": float(batch / p50),
            }
        )
    return pd.DataFrame(rows)


def segment_report(y_true, p, segments: dict[str, np.ndarray]) -> pd.DataFrame:
    """Metrics restricted to each boolean mask, for honest failure analysis."""
    y_true = np.asarray(y_true)
    p = np.asarray(p, dtype="float64")
    rows = []
    for name, mask in segments.items():
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() == 0:
            continue
        row = binary_metrics(y_true[mask], p[mask])
        row["segment"] = name
        rows.append(row)
    result = pd.DataFrame(rows)
    return result[["segment", "n", "base_rate", "log_loss", "roc_auc", "average_precision", "brier"]]


def _take_rows(X, i: int, j: int):
    if isinstance(X, dict):
        return {k: np.asarray(v)[i:j] for k, v in X.items()}
    if hasattr(X, "iloc"):
        return X.iloc[i:j]
    return X[i:j]
