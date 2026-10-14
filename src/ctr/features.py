"""Preprocessing fitted on training data only, plus persistence.

Design rules (enforced here and demonstrated in notebook 01, section G):
- every statistic (medians, means, skewness, vocabularies) is computed on the
  training split only; validation and test frames are transformed with those
  frozen statistics;
- category vocabularies keep values with >= ``min_category_count`` training
  occurrences; everything else (rare, unseen, missing) maps to the shared
  out-of-vocabulary index 0;
- numeric missing values are imputed with the train median before the
  sign-preserving log1p transform (when enabled) and standardisation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

OOV_INDEX = 0


@dataclass
class NumericPrep:
    columns: list[str]
    skew_threshold: float = 2.0
    log1p_columns: list[str] = field(default_factory=list)
    medians_: dict[str, float] = field(default_factory=dict)
    means_: dict[str, float] = field(default_factory=dict)
    stds_: dict[str, float] = field(default_factory=dict)
    skew_: dict[str, float] = field(default_factory=dict)
    fitted_: bool = False

    def _impute(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame[self.columns].astype("float64").fillna(self.medians_)

    def _apply_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for c in self.log1p_columns:
            out[c] = np.sign(out[c]) * np.log1p(np.abs(out[c]))
        return out

    def fit(self, frame: pd.DataFrame) -> "NumericPrep":
        for c in self.columns:
            x = frame[c].astype("float64")
            skew = float(x.skew())
            self.skew_[c] = skew
            if np.isfinite(skew) and abs(skew) > self.skew_threshold:
                self.log1p_columns.append(c)
            self.medians_[c] = float(x.median())
        transformed = self._apply_transform(self._impute(frame))
        for c in self.columns:
            self.means_[c] = float(transformed[c].mean())
            std = float(transformed[c].std(ddof=0))
            self.stds_[c] = std if np.isfinite(std) and std > 0 else 1.0
        self.fitted_ = True
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        transformed = self._apply_transform(self._impute(frame))
        for c in self.columns:
            transformed[c] = (transformed[c] - self.means_[c]) / self.stds_[c]
        return transformed.astype("float32")


@dataclass
class CategoricalVocab:
    columns: list[str]
    min_count: int = 10
    maps_: dict[str, dict[str, int]] = field(default_factory=dict)
    sizes_: dict[str, int] = field(default_factory=dict)
    fitted_: bool = False

    def fit(self, frame: pd.DataFrame) -> "CategoricalVocab":
        for c in self.columns:
            counts = frame[c].value_counts(dropna=True)
            kept = counts[counts >= self.min_count]
            self.maps_[c] = {str(v): i + 1 for i, v in enumerate(kept.index)}
            self.sizes_[c] = len(kept) + 1  # index 0 reserved for OOV/missing
        self.fitted_ = True
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for c in self.columns:
            codes = frame[c].astype("object").map(self.maps_[c])
            out[c] = codes.fillna(OOV_INDEX).astype("int32")
        return pd.DataFrame(out, index=frame.index)

    def unseen_fraction(self, frame: pd.DataFrame) -> dict[str, float]:
        """Share of present values that are not in the training vocabulary."""
        result = {}
        for c in self.columns:
            present = frame[c].notna()
            if present.sum() == 0:
                result[c] = float("nan")
                continue
            unseen = present & ~frame[c].isin(set(self.maps_[c]))
            result[c] = float(unseen.sum() / present.sum())
        return result


@dataclass
class Preprocessor:
    numeric: NumericPrep
    vocab: CategoricalVocab
    add_missing_indicators: bool = False
    missing_indicator_columns: list[str] = field(default_factory=list)

    @classmethod
    def from_config(cls, features_cfg: dict, add_missing_indicators: bool | None = None) -> "Preprocessor":
        return cls(
            numeric=NumericPrep(
                columns=list(features_cfg["numeric_cols"]),
                skew_threshold=float(features_cfg["skew_threshold"]),
            ),
            vocab=CategoricalVocab(
                columns=list(features_cfg["categorical_cols"]),
                min_count=int(features_cfg["min_category_count"]),
            ),
            add_missing_indicators=bool(
                features_cfg["add_missing_indicators"]
                if add_missing_indicators is None
                else add_missing_indicators
            ),
            # notebook 01 froze the flagged list in config.yaml (23 of 24 gappy
            # columns); when the list is absent, fit() falls back to flagging all
            missing_indicator_columns=list(features_cfg.get("missing_indicator_columns", [])),
        )

    def fit(self, frame: pd.DataFrame) -> "Preprocessor":
        self.numeric.fit(frame)
        self.vocab.fit(frame)
        if self.add_missing_indicators and not self.missing_indicator_columns:
            # fallback when the config names no columns: flag every gappy column
            # (categorical missingness otherwise shares OOV index 0)
            self.missing_indicator_columns = [
                c
                for c in list(self.numeric.columns) + list(self.vocab.columns)
                if frame[c].isna().any()
            ]
        return self

    def transform(self, frame: pd.DataFrame, label_col: str = "label") -> dict:
        x_num = self.numeric.transform(frame)
        for c in self.missing_indicator_columns:
            x_num[f"{c}_missing"] = frame[c].isna().astype("float32")
        x_cat = self.vocab.transform(frame)
        out = {"Xnum": x_num, "Xcat": x_cat}
        if label_col in frame.columns:
            out["y"] = frame[label_col].to_numpy(dtype=np.int8)
        return out

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "Preprocessor":
        return joblib.load(path)


def vocab_sizes(preprocessor: Preprocessor) -> list[int]:
    return [preprocessor.vocab.sizes_[c] for c in preprocessor.vocab.columns]


def to_lr_matrix(preprocessor: Preprocessor, processed: dict) -> sp.csr_matrix:
    """Sparse matrix for linear models: one-hot categories + dense numerics."""
    x_cat, x_num = processed["Xcat"], processed["Xnum"]
    n = len(x_cat)
    columns = list(x_cat.columns)
    offsets = np.cumsum([0] + [preprocessor.vocab.sizes_[c] for c in columns])
    rows = np.tile(np.arange(n), len(columns))
    cols = np.concatenate(
        [x_cat[c].to_numpy(dtype=np.int64) + offsets[i] for i, c in enumerate(columns)]
    )
    one_hot = sp.csr_matrix(
        (np.ones(len(cols), dtype=np.float32), (rows, cols)),
        shape=(n, int(offsets[-1])),
    )
    return sp.hstack([one_hot, sp.csr_matrix(x_num.to_numpy(dtype=np.float32))]).tocsr()


def to_lgbm_frame(processed: dict) -> pd.DataFrame:
    return pd.concat([processed["Xcat"], processed["Xnum"]], axis=1)


def to_deepfm_inputs(processed: dict) -> dict[str, np.ndarray]:
    """Keras functional models take one input tensor per feature field."""
    x_cat, x_num = processed["Xcat"], processed["Xnum"]
    inputs = {"num": x_num.to_numpy(dtype=np.float32)}
    for c in x_cat.columns:
        inputs[c] = x_cat[c].to_numpy(dtype=np.int32).reshape(-1, 1)
    return inputs
