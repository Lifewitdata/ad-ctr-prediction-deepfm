"""SQL layer: every statement in sql/ must parse and run against a frame with
the real schema via DuckDB, including parameter binding."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ctr import sql_runner

from conftest import make_frame

SQL_DIR = Path(__file__).resolve().parents[1] / "sql"


@pytest.fixture(scope="module")
def connection(tmp_path_factory):
    path = tmp_path_factory.mktemp("sql") / "mini.parquet"
    make_frame(n=500, seed=31).to_parquet(path, index=False)
    return sql_runner.connect(path)


def test_all_sql_files_present():
    files = sorted(p.name for p in SQL_DIR.glob("*.sql"))
    assert files == [
        "01_overview.sql",
        "02_missing_values.sql",
        "03_category_frequency_ctr.sql",
        "04_numeric_bins_ctr.sql",
        "05_min_support.sql",
    ]


def test_overview_counts_match_frame(connection):
    result = sql_runner.run_file(connection, SQL_DIR / "01_overview.sql").iloc[0]
    assert result["impressions"] == 500
    assert 0 < result["clicks"] <= 500
    assert result["ctr"] == pytest.approx(result["clicks"] / result["impressions"], abs=1e-9)


def test_missing_values_covers_all_gappy_columns(connection):
    result = sql_runner.run_file(connection, SQL_DIR / "02_missing_values.sql")
    assert len(result) == 39  # 13 numeric + 26 categorical
    by_feature = dict(zip(result["feature"], result["n_missing"]))
    assert by_feature["I1"] > 0 and by_feature["I8"] > 0 and by_feature["C3"] > 0
    assert by_feature["I3"] == 0


def test_category_frequency_ctr_uses_support_filter(connection):
    result = sql_runner.run_file(connection, SQL_DIR / "03_category_frequency_ctr.sql",
                                 params={"min_support": 25})
    assert set(result["feature"]) <= {f"C{i}" for i in range(1, 27)}
    assert (result["n"] >= 25).all()
    assert ((result["ctr"] >= 0) & (result["ctr"] <= 1)).all()
    assert len(result) <= 100  # LIMIT in the statement


def test_numeric_bins_ctr_returns_deciles(connection):
    result = sql_runner.run_file(connection, SQL_DIR / "04_numeric_bins_ctr.sql")
    assert len(result) == 20  # two features x 10 bins
    assert set(result["feature"]) == {"I1", "I2"}
    assert ((result["ctr"] >= 0) & (result["ctr"] <= 1)).all()


def test_min_support_share_parameterised(connection):
    loose = sql_runner.run_file(connection, SQL_DIR / "05_min_support.sql", params={"min_support": 5})
    strict = sql_runner.run_file(connection, SQL_DIR / "05_min_support.sql", params={"min_support": 30})
    assert len(loose) == 26
    # raising the bar can only move more impressions into the "rare" bucket
    # (NULL, i.e. zero rare mass, is treated as 0)
    merged = loose.merge(strict, on="feature", suffixes=("_loose", "_strict"))
    pct_strict = merged["pct_impressions_rare_strict"].fillna(0.0)
    pct_loose = merged["pct_impressions_rare_loose"].fillna(0.0)
    assert (pct_strict >= pct_loose - 1e-9).all()


def test_run_dir_executes_every_file(connection):
    outputs = sql_runner.run_dir(connection, SQL_DIR, params={"min_support": 10})
    assert set(outputs) == {"01_overview", "02_missing_values", "03_category_frequency_ctr",
                            "04_numeric_bins_ctr", "05_min_support"}
    assert all(isinstance(frame, pd.DataFrame) for frame in outputs.values())


def test_unused_parameters_are_not_bound(connection):
    """run_file only binds $names appearing in the statement."""
    result = sql_runner.run_file(connection, SQL_DIR / "01_overview.sql",
                                 params={"min_support": 10, "irrelevant": "x"})
    assert len(result) == 1
