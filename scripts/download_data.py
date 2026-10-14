"""Download the raw Criteo pool used by this project.

Source: the community mirror ``seanv507/Criteo`` on Hugging Face, a re-hosted
copy of the Kaggle "Criteo Display Advertising Challenge" training file
(45,840,617 rows in total: ``Label`` + ``I1``-``I13`` + ``C1``-``C26``,
original raw format). This script pulls the mirror's 4,584,062-row partition
in parquet form; the notebooks then sample from it deterministically.

Using the original Kaggle release instead: create a Kaggle account, accept
the competition rules on https://www.kaggle.com/c/criteo-display-ad-challenge,
run ``kaggle competitions download -c criteo-display-ad-challenge`` and convert
train.txt to parquet (see README, "Using the original Kaggle file").
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

DEFAULT_URL = (
    "https://huggingface.co/api/datasets/seanv507/Criteo/parquet/default/validation/0.parquet"
)
DEFAULT_SHA256 = "14cc566914522fb2039e86abf7b342cede7fb8a27e6ef388dac72d855b8b9866"
DEFAULT_ROWS = 4_584_062
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "data" / "raw" / "criteo_pool.parquet"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as fh:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            progress = f"{100 * done / total:5.1f}%" if total else f"{done} bytes"
            print(f"\r  downloading: {progress}", end="", flush=True)
    print()
    tmp.replace(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the raw Criteo pool")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.out.exists() and not args.force:
        print(f"already present: {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"downloading {args.url}")
        download(args.url, args.out)

    digest = sha256_of(args.out)
    print("sha256:", digest)
    if args.url == DEFAULT_URL:
        if digest != DEFAULT_SHA256:
            print("WARNING: checksum differs from the recorded value; verify the source")
            return 1
        print("checksum matches the recorded value")
        import pyarrow.parquet as pq

        rows = pq.ParquetFile(args.out).metadata.num_rows
        print(f"rows: {rows}")
        if rows != DEFAULT_ROWS:
            print("WARNING: row count differs from the recorded value")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
