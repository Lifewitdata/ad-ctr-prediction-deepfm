"""Execute a generated notebook against the project venv kernel.

Usage:
    venv/Scripts/python.exe scripts/run_notebook.py notebooks/01_ctr_in_depth_eda.ipynb

Runs the notebook in place (outputs are saved into the .ipynb) with the
working directory set to the project root, using the local kernelspec in
jupyter/kernels (found via JUPYTER_PATH, so nothing is installed user-wide).
Exits non-zero on the first failing cell and prints the traceback tail.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("JUPYTER_PATH", str(PROJECT_ROOT / "jupyter"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import nbformat  # noqa: E402
from nbclient import NotebookClient  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    path = (PROJECT_ROOT / sys.argv[1]).resolve()
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=1800,
        kernel_name="ad-ctr-deepfm",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
        allow_errors=False,
    )
    try:
        client.execute()
    except Exception as exc:  # noqa: BLE001 - report the failing cell clearly
        print(f"EXECUTION FAILED: {type(exc).__name__}: {exc}")
        nbformat.write(nb, path)
        return 1
    nbformat.write(nb, path)
    print(f"OK: executed {path.name} ({sum(1 for c in nb.cells if c.cell_type == 'code')} code cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
