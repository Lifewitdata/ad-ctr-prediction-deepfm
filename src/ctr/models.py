"""Baseline models with one common interface: fit(X, y, eval_set) and
predict_proba(X) -> (n, 2). DeepFM lives in ``deepfm.py`` and follows the
same interface through its wrapper."""
from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.linear_model import LogisticRegression


class ConstantModel:
    """Predicts the training-set CTR for every impression.

    This is the reference point that any real model must beat on log loss.
    """

    def __init__(self) -> None:
        self.p_: float | None = None

    def fit(self, X, y, **kwargs) -> "ConstantModel":
        self.p_ = float(np.mean(y))
        return self

    def predict_proba(self, X) -> np.ndarray:
        n = _n_rows(X)
        p = np.full(n, self.p_)
        return np.column_stack([1 - p, p])


class LogisticModel:
    def __init__(self, C: float = 1.0, max_iter: int = 1000, seed: int = 42) -> None:
        self.C = C
        self.max_iter = max_iter
        self.seed = seed
        self.model_: LogisticRegression | None = None

    def fit(self, X, y, **kwargs) -> "LogisticModel":
        self.model_ = LogisticRegression(
            C=self.C, max_iter=self.max_iter, random_state=self.seed, solver="lbfgs"
        ).fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model_.predict_proba(X)


class LGBMModel:
    def __init__(self, params: dict, cat_columns: list[str], seed: int = 42) -> None:
        self.params = dict(params)
        self.cat_columns = list(cat_columns)
        self.seed = seed
        self.model_: LGBMClassifier | None = None
        self.best_iteration_: int | None = None

    def fit(self, X, y, eval_set=None, **kwargs) -> "LGBMModel":
        params = dict(self.params)
        early_rounds = int(params.pop("early_stopping_rounds", 50))
        self.model_ = LGBMClassifier(**params, random_state=self.seed, verbosity=-1)
        fit_kwargs: dict = {}
        if eval_set is not None:
            eval_x, eval_y = eval_set
            fit_kwargs["eval_X"] = eval_x
            fit_kwargs["eval_y"] = eval_y
            fit_kwargs["callbacks"] = [early_stopping(early_rounds, verbose=False), log_evaluation(0)]
        self.model_.fit(X, y, categorical_feature=self.cat_columns, **fit_kwargs)
        self.best_iteration_ = getattr(self.model_, "best_iteration_", None)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model_.predict_proba(X)


def _n_rows(X) -> int:
    if isinstance(X, dict):
        return len(next(iter(X.values())))
    return len(X)
