"""Data loading, schema validation, sampling and splitting.

The raw file is a community mirror of the Criteo Display Advertising Challenge
training data. Its schema is the original one: a binary ``Label`` column, 13
integer numeric columns ``I1``-``I13`` (with missing values) and 26 hashed
categorical identifier columns ``C1``-``C26`` (hex strings, stored as text).

Nothing in this project assigns real-world meaning to the I/C features: they
are anonymised and we treat the I columns as numeric inputs and the C columns
as nominal identifiers.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RAW_LABEL = "Label"
N_LABEL = "label"


def expected_columns(n_numeric: int = 13, n_categorical: int = 26) -> list[str]:
    cols = [N_LABEL]
    cols += [f"I{i}" for i in range(1, n_numeric + 1)]
    cols += [f"C{i}" for i in range(1, n_categorical + 1)]
    return cols


def iter_batches(path: str | Path, batch_rows: int = 131072):
    """Stream the parquet file in row batches (bounded memory)."""
    pf = pq.ParquetFile(path)
    yield from pf.iter_batches(batch_size=batch_rows)


def row_count(path: str | Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def uniform_sample(
    path: str | Path,
    n: int,
    seed: int,
    batch_rows: int = 131072,
) -> pd.DataFrame:
    """Simple random sample without replacement over row positions.

    Deterministic for a given ``seed``: numpy's Generator draws the positions,
    and the parquet file is streamed in batches, so the full table is never
    materialised in memory. If ``n`` >= the number of rows, everything is
    returned.
    """
    pf = pq.ParquetFile(path)
    total = pf.metadata.num_rows
    if n is None or n >= total:
        return normalize(pf.read().to_pandas())

    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(total, size=n, replace=False))

    pieces: list[pa.Table] = []
    offset = 0
    pos = 0
    for batch in pf.iter_batches(batch_size=batch_rows):
        m = batch.num_rows
        stop = int(np.searchsorted(selected, offset + m, side="left"))
        if stop > pos:
            local = (selected[pos:stop] - offset).astype(np.int64)
            pieces.append(pa.Table.from_batches([batch.take(pa.array(local))]))
        pos = stop
        offset += m
        if pos >= n:
            break

    frame = pa.concat_tables(pieces).to_pandas()
    return normalize(frame)


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename the raw ``Label`` column to the lowercase name used everywhere else."""
    if RAW_LABEL in frame.columns:
        frame = frame.rename(columns={RAW_LABEL: N_LABEL})
    return frame


def load_frame(
    path: str | Path,
    n: int | None = None,
    seed: int = 42,
    batch_rows: int = 131072,
) -> pd.DataFrame:
    frame = uniform_sample(path, n, seed, batch_rows) if n else pq.ParquetFile(path).read().to_pandas()
    return normalize(frame)


def validate_schema(
    frame: pd.DataFrame,
    n_numeric: int = 13,
    n_categorical: int = 26,
) -> dict:
    """Structural checks. Raises on problems that make the frame unusable."""
    expected = expected_columns(n_numeric, n_categorical)
    missing = [c for c in expected if c not in frame.columns]
    if missing:
        raise ValueError(f"missing expected columns: {missing[:5]}... ({len(missing)} total)")

    report: dict = {
        "n_rows": int(len(frame)),
        "n_columns": int(frame.shape[1]),
        "unexpected_columns": [c for c in frame.columns if c not in expected],
    }
    label_values = sorted(pd.unique(frame[N_LABEL].dropna()))
    report["label_values"] = [int(v) for v in label_values]
    if set(report["label_values"]) - {0, 1} or frame[N_LABEL].isna().any():
        raise ValueError(f"label must be 0/1 without missing values, found {report['label_values']}")

    numeric_cols = [f"I{i}" for i in range(1, n_numeric + 1)]
    numeric_bad = [c for c in numeric_cols if not pd.api.types.is_numeric_dtype(frame[c])]
    report["numeric_columns_not_numeric_dtype"] = numeric_bad
    if numeric_bad:
        raise ValueError(f"I columns must be numeric, found {numeric_bad}")

    categorical_cols = [f"C{i}" for i in range(1, n_categorical + 1)]
    report["categorical_dtypes"] = sorted({str(frame[c].dtype) for c in categorical_cols})
    return report


def stratified_split(
    frame: pd.DataFrame,
    ratios: dict,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Seeded stratified split over the label, used because no reliable
    timestamps are available. Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    labels = frame[N_LABEL].to_numpy()
    assignment = np.empty(len(frame), dtype=object)
    for cls in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == cls))
        n = len(idx)
        n_train = int(np.floor(ratios["train"] * n))
        n_val = int(np.floor(ratios["val"] * n))
        assignment[idx[:n_train]] = "train"
        assignment[idx[n_train:n_train + n_val]] = "val"
        assignment[idx[n_train + n_val:]] = "test"
    return {
        name: frame.loc[assignment == name].reset_index(drop=True)
        for name in ("train", "val", "test")
    }


def save_splits(splits: dict[str, pd.DataFrame], directory: str | Path, manifest: dict) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_parquet(directory / f"{name}.parquet", index=False)
    payload = dict(manifest)
    payload["splits"] = {
        name: {
            "rows": int(len(frame)),
            "ctr": float(frame[N_LABEL].mean()),
        }
        for name, frame in splits.items()
    }
    path = directory / "split_manifest.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_splits(directory: str | Path) -> dict[str, pd.DataFrame]:
    directory = Path(directory)
    return {
        name: normalize(pd.read_parquet(directory / f"{name}.parquet"))
        for name in ("train", "val", "test")
    }
