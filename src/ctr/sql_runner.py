"""Execute the sql/*.sql analyses with DuckDB against a parquet file.

Every SQL file is a single statement (possibly with CTEs) that selects from
the ``impressions`` view; numeric parameters are passed as ``$name``
placeholders so the same SQL is validated and reused by the notebooks and
tests.
"""
from __future__ import annotations

import re
from pathlib import Path

import duckdb
import pandas as pd

_PARAM_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")


def connect(parquet_path) -> duckdb.DuckDBPyConnection:
    path = str(parquet_path).replace("\\", "/").replace("'", "''")
    con = duckdb.connect(database=":memory:")
    con.execute(f"CREATE OR REPLACE VIEW impressions AS SELECT * FROM read_parquet('{path}')")
    return con


def run_file(con: duckdb.DuckDBPyConnection, sql_path, params: dict | None = None) -> pd.DataFrame:
    """Run one SQL file. Only parameters actually referenced with $name in the
    statement are passed, so a shared parameter dict can be reused across the
    whole sql/ directory."""
    sql = Path(sql_path).read_text(encoding="utf-8")
    used = set(_PARAM_RE.findall(sql))
    payload = {k: v for k, v in (params or {}).items() if k in used}
    return con.execute(sql, payload).fetchdf()


def run_dir(con: duckdb.DuckDBPyConnection, sql_dir, params: dict | None = None) -> dict[str, pd.DataFrame]:
    return {
        path.stem: run_file(con, path, params)
        for path in sorted(Path(sql_dir).glob("*.sql"))
    }
