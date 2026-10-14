"""Reusable EDA helpers: confidence intervals, smoothed CTR, drift and
categorical-support tables used by notebook 01."""
from __future__ import annotations

import numpy as np
import pandas as pd


def wilson_ci(clicks: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Assumes independent Bernoulli draws with a common CTR: a simplifying
    assumption for repeated ad impressions that is stated in the notebook.
    """
    if n == 0:
        return (np.nan, np.nan)
    p = clicks / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def smoothed_ctr(clicks: float, n: float, prior: float, strength: float = 50.0) -> float:
    """Empirical-Bayes smoothed CTR: (clicks + strength*prior) / (n + strength).

    ``strength`` is a pseudo-count toward the global prior; it keeps
    rare categories from topping naive CTR rankings by chance.
    """
    return (clicks + strength * prior) / (n + strength)


def missing_summary(frame: pd.DataFrame) -> pd.DataFrame:
    n = len(frame)
    rows = [
        {
            "column": c,
            "n_missing": int(frame[c].isna().sum()),
            "pct_missing": 100.0 * float(frame[c].isna().mean()),
        }
        for c in frame.columns
    ]
    return pd.DataFrame(rows).sort_values("n_missing", ascending=False, ignore_index=True)


def missing_patterns(frame: pd.DataFrame, columns: list[str], top: int = 12) -> pd.DataFrame:
    """Most frequent co-occurring missingness patterns among ``columns``.

    Row patterns are packed into two 64-bit keys (39 columns fit comfortably) so
    the count is a single ``value_counts`` on a vectorised hash rather than a
    per-row Python loop.
    """
    pattern = frame[columns].isna().to_numpy()
    n_rows, n_cols = pattern.shape
    left = pattern[:, :32].astype(np.uint64)
    right = pattern[:, 32:].astype(np.uint64)
    lw = np.uint64(1) << np.arange(left.shape[1], dtype=np.uint64) if left.shape[1] else None
    key_l = (left @ lw) if lw is not None else np.zeros(n_rows, dtype=np.uint64)
    key_r = (
        right @ (np.uint64(1) << np.arange(right.shape[1], dtype=np.uint64))
        if right.shape[1]
        else np.zeros(n_rows, dtype=np.uint64)
    )
    counts = pd.Series(list(zip(key_l.tolist(), key_r.tolist()))).value_counts().head(top)
    rows = []
    for (kl, kr), cnt in counts.items():
        bits = [(kl >> j) & 1 for j in range(min(n_cols, 32))]
        bits += [(kr >> (j - 32)) & 1 for j in range(32, n_cols)]
        missing_cols = [columns[j] for j, b in enumerate(bits) if b]
        rows.append(
            {
                "n_rows": int(cnt),
                "pct_rows": 100.0 * cnt / len(frame),
                "n_missing_in_pattern": len(missing_cols),
                "columns_missing": ", ".join(missing_cols) if missing_cols else "(none)",
            }
        )
    return pd.DataFrame(rows)


def near_constant_report(frame: pd.DataFrame, columns: list[str], threshold: float = 0.995) -> pd.DataFrame:
    rows = []
    for c in columns:
        vc = frame[c].value_counts(dropna=False)
        top_share = float(vc.iloc[0] / len(frame))
        if top_share >= threshold:
            rows.append(
                {
                    "column": c,
                    "top_value": str(vc.index[0]),
                    "top_value_share_pct": 100 * top_share,
                    "n_unique": int(frame[c].nunique(dropna=True)),
                }
            )
    return pd.DataFrame(rows)


def ctr_by_quantile_bins(frame: pd.DataFrame, column: str, q: int = 10, label: str = "label") -> pd.DataFrame:
    """CTR per q-quantile bin of a numeric column, with Wilson intervals."""
    data = frame[[column, label]].dropna()
    try:
        bins = pd.qcut(data[column], q=q, duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    grouped = data.groupby(bins, observed=True)[label].agg(["size", "sum"])
    grouped.columns = ["n", "clicks"]
    cis = [wilson_ci(int(k), int(n)) for k, n in zip(grouped["clicks"], grouped["n"])]
    grouped["ctr"] = grouped["clicks"] / grouped["n"]
    grouped["ci_low"] = [c[0] for c in cis]
    grouped["ci_high"] = [c[1] for c in cis]
    return grouped.reset_index().rename(columns={column: "bin"})


def rank_bins(series: pd.Series, q: int = 10) -> pd.Series:
    """Equal-count bin index 1..q over non-missing values (NTILE convention:
    larger bins first, tied values split by current row order).

    ``pd.qcut`` keeps equal values together and collapses tie-heavy columns to
    fewer than q bins; rank bins guarantee q groups, which matches what SQL
    ``NTILE`` does and keeps decile sizes comparable across columns.
    """
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    valid = series.notna()
    n = int(valid.sum())
    if n == 0:
        return out
    ranks = series[valid].rank(method="first").to_numpy()
    base, rem = divmod(n, q)
    edges = np.cumsum([base + 1] * rem + [base] * (q - rem))
    out[valid] = np.searchsorted(edges, ranks - 1, side="right") + 1
    return out


def ctr_by_rank_bins(frame: pd.DataFrame, column: str, q: int = 10, label: str = "label") -> pd.DataFrame:
    """CTR per equal-count bin of a numeric column, with Wilson intervals."""
    bins = rank_bins(frame[column], q=q)
    data = pd.DataFrame({"bin": bins, label: frame[label]}).dropna(subset=["bin"])
    grouped = data.groupby("bin")[label].agg(["size", "sum"])
    grouped.columns = ["n", "clicks"]
    cis = [wilson_ci(int(k), int(n)) for k, n in zip(grouped["clicks"], grouped["n"])]
    grouped["ctr"] = grouped["clicks"] / grouped["n"]
    grouped["ci_low"] = [c[0] for c in cis]
    grouped["ci_high"] = [c[1] for c in cis]
    return grouped.reset_index()


def category_ctr_table(
    frame: pd.DataFrame,
    column: str,
    label: str = "label",
    min_support: int | None = None,
    prior: float | None = None,
    strength: float = 50.0,
) -> pd.DataFrame:
    """Frequency, raw CTR and optionally support-filtered / smoothed CTR per category."""
    grouped = frame.groupby(column, dropna=True)[label].agg(["size", "sum"])
    grouped.columns = ["n", "clicks"]
    if min_support is not None:
        grouped = grouped[grouped["n"] >= min_support]
    grouped["ctr"] = grouped["clicks"] / grouped["n"]
    if prior is not None:
        grouped["ctr_smoothed"] = [
            smoothed_ctr(k, n, prior, strength) for k, n in zip(grouped["clicks"], grouped["n"])
        ]
    return grouped.sort_values("n", ascending=False)


def ks_table(train: pd.DataFrame, val: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Per-column two-sample Kolmogorov-Smirnov statistic between two partitions."""
    from scipy.stats import ks_2samp

    rows = []
    for c in columns:
        a = train[c].dropna()
        b = val[c].dropna()
        stat, p = ks_2samp(a, b)
        rows.append(
            {
                "column": c,
                "ks_stat": stat,
                "p_value": p,
                "mean_train": float(a.mean()),
                "mean_val": float(b.mean()),
                "missing_pct_train": 100 * float(train[c].isna().mean()),
                "missing_pct_val": 100 * float(val[c].isna().mean()),
            }
        )
    return pd.DataFrame(rows)


def category_overlap_table(train: pd.DataFrame, val: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Share of validation categories never seen in training (and vice versa)."""
    rows = []
    for c in columns:
        train_levels = set(train[c].dropna().unique())
        val_vals = val[c].dropna()
        unseen = (~val_vals.isin(train_levels)).mean() if len(val_vals) else np.nan
        rows.append(
            {
                "column": c,
                "train_levels": len(train_levels),
                "val_levels": int(val[c].nunique()),
                "val_rows_unseen_pct": 100 * float(unseen),
            }
        )
    return pd.DataFrame(rows)
