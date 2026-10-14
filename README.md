# Ad Click-Through Rate Prediction — Machine Learning + DeepFM (Criteo)

An end-to-end CTR prediction project on the real Criteo Display Advertising Challenge
schema: reproducible EDA on a deterministic sample, train-fitted preprocessing, four
model families trained and compared on validation (constant baseline, logistic
regression, LightGBM, DeepFM), a **sealed test split scored exactly once**, calibration
and latency analysis, and a FastAPI service that serves the same scoring path the
notebooks evaluated.

This is an **independent portfolio project**. It uses a public research dataset and
open-source tools; it is not affiliated with, and does not claim to reproduce, any
publisher's production systems or data.

## Results at a glance (sealed test split, 15,002 rows)

| model | test log loss (primary) | ROC-AUC | avg. precision | Brier |
|---|---|---|---|---|
| **logistic regression** (selected) | **0.47893** | **0.76556** | **0.54726** | **0.15600** |
| DeepFM (k=4) | 0.48268 | 0.76139 | 0.54399 | 0.15711 |
| LightGBM (7 leaves) | 0.48496 | 0.75712 | 0.53945 | 0.15780 |
| constant baseline | 0.56972 | 0.50000 | 0.25683 | 0.19087 |

- The validation-based selection **survived the sealed split**: logistic regression
  was best on validation (0.48142) and stayed best on test (val → test delta −0.0025).
  No model was re-selected or re-tuned after the test split was opened.
- Calibration: ECE 0.0081 for the selected model (0.0107 LightGBM, 0.0131 DeepFM);
  14 of 15 reliability bins contain their predicted level inside Wilson 95% CIs.
- Serving: ~70 ms p50 per 1,024-row batch (~14k rows/s, CPU-only, single process) —
  DeepFM costs ~4–5× that latency while scoring worse.
- Weakest segments: rows with a missing input (98.4% of traffic), rows with unseen
  category values (79.5%), and the rare complete-input population (248 rows, ECE 0.065).

Full tables and figures: `reports/nb03_summary.json`, `reports/tables/test_*.csv`,
`reports/figures/nb03_*.png`. Every number above is measured in
`notebooks/03_ctr_model_evaluation.ipynb` on the same held-out rows for all models.

## Business framing

Ad-supported publishing monetises through impression-level auctions, where the
**predicted click-through rate (pCTR) is a price ingredient** — expected revenue per
impression scales as `eCPM ≈ pCTR × bid` — as well as a ranking key. Better pCTR
estimate accuracy therefore affects both which ads compete for a slot and what they pay.
Two honest caveats shape everything here: **CTR is not revenue** (revenue depends on
click value, which this dataset does not contain), and offline probability quality does
not translate directly into online lift (§ *What this does not claim* below).

## Dataset

- Source file: community mirror of the Kaggle **Criteo Display Advertising Challenge
  training set** (`seanv507/Criteo` on Hugging Face), a 4,584,062-row partition of the
  full 45,840,617-row release. Same original schema: `Label` (0/1), 13 integer numeric
  columns `I1`–`I13` (with missing values), 26 hashed categorical columns `C1`–`C26`
  (8-character hex strings).
- **No meanings are invented for the anonymous features.** `I*` are treated as numeric
  inputs, `C*` as nominal identifiers, and nothing more.
- The project works on a deterministic **100,000-row simple random sample** (seeded,
  streamed in batches, documented in `config.yaml` and notebook 01): ~458× the raw row
  count of a typical tutorial dataset while staying PC-friendly.
- Download + integrity check: `make data` (or `python scripts/download_data.py`).
  The script records and verifies the sha256
  (`14cc5669…b8b9866`) and row count of the mirror partition.
- Using the original Kaggle file instead: accept the competition rules at
  `kaggle.com/c/criteo-display-ad-challenge`, download `train.txt`, convert it to
  parquet with the same column names, and point `data.raw_path` in `config.yaml` at it.

## Repository layout

```
config.yaml                  single source of truth: seed, paths, feature/model settings
requirements.txt             pinned environment (Python 3.11, CPU)
Makefile                     setup / data / eda / train / eval / notebooks / test / serve
src/ctr/
  data.py                    streaming sample, schema validation, stratified split
  features.py                Preprocessor: train-fitted medians/log1p/z-scores, vocabularies
  models.py                  constant / logistic / LightGBM wrappers (common interface)
  deepfm.py                  DeepFM in Keras 3 (linear + FMInteraction + deep MLP)
  evaluate.py                log loss, ROC-AUC, PR-AUC, Brier, reliability/ECE, latency, segments
  eda.py                     Wilson CIs, profiling helpers used by notebook 01
  sql_runner.py              DuckDB execution of the sql/ files
  service.py                 FastAPI app + shared scoring path (used by nb03 and the API)
sql/                         01_overview … 05_min_support (DuckDB, parameterised)
scripts/
  download_data.py           fetch + checksum the raw pool
  build_nb0{1,2,3}_*.py      generate and execute each notebook end to end
  run_notebook.py            nbclient runner (project-local kernelspec)
notebooks/                   01 EDA · 02 training · 03 evaluation (executed, outputs stored)
tests/                       pytest suite (75 tests; no data download or artifacts needed)
reports/                     figures, tables, nb03_summary.json, eda_decisions.json
artifacts/                   preprocessor + trained models + model_choice.json (generated)
```

## Setup and reproduction

```bash
# Python 3.11 environment (uv recommended; plain venv works too)
uv venv --python 3.11 venv
uv pip install --python venv/Scripts/python.exe -r requirements.txt   # or: source venv/bin/activate && pip install -r requirements.txt

make data        # download + verify the raw Criteo pool
make eda         # notebook 01: EDA, decisions frozen to config.yaml
make train       # notebook 02: train + tune 4 models, select on validation
make eval        # notebook 03: score the sealed test split once
make test        # pytest suite
make serve       # FastAPI at http://127.0.0.1:8000
```

Each `build_nb*.py` script writes the notebook and executes it with `nbclient` using a
project-local kernelspec, so notebooks in the repo always carry real, current outputs.
The split is 70/15/15 stratified by label (69,999 / 14,999 / 15,002 rows, seed 42);
the test split is only read in notebook 03.

## Notebook guide

1. **01 — EDA (`notebooks/01_ctr_in_depth_eda.ipynb`)**. Structure/integrity, class
   imbalance, missingness with Wilson CIs and an a-priori flagging rule (disjoint CIs
   *and* |CTR gap| ≥ 0.5 pp → 23 of 24 gappy columns flagged; `C6` fails and stays
   unflagged), numeric distributions and skew, categorical long tail (34,092 distinct
   values in the widest train column; vocabularies with `min_count = 10` + shared OOV
   index 0), feature
   interactions, leakage/splitting review, train-vs-val partition comparison, and a
   decision table. The same queries are also written in DuckDB SQL (`sql/*.sql`) and
   cross-checked against the pandas results in the notebook.
2. **02 — Training (`notebooks/02_ctr_model_training.ipynb`)**. Constant baseline, L2
   logistic regression on sparse one-hot, LightGBM with native categoricals, and DeepFM
   (Keras 3: linear + FM term + deep MLP over shared embeddings). Bounded tuning sweeps
   (bracketed grids: LR `C ∈ {0.01…3.0}`, LightGBM `num_leaves ∈ {3…127}`, DeepFM
   `k ∈ {2,4,8,16}`), early stopping on validation log loss, artifacts saved to
   `artifacts/` for serving. **No assumption was made that DeepFM would win** — the
   validation table chose logistic regression, and all sweeps preferred small capacity
   with strong regularisation.
3. **03 — Evaluation (`notebooks/03_ctr_model_evaluation.ipynb`)**. Integrity checks,
   headline metrics on the sealed split, reliability curves + ECE, ROC/PR curves,
   p50/p95 prediction latency at several batch sizes (hardware and method stated), and
   operational segments (missing inputs, unseen categories; missing-slice Wilson CIs
   re-tested on test). Ends with why offline gains are not CTR or revenue gains.

## Serving the model

```bash
make serve   # or: uvicorn ctr.service:app --app-dir src --port 8000

curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
     -d '{"I1": 3.2, "C1": "a1b2c3d4"}'
# {"click_probability": 0.10120602697134018, "model_name": "logistic_regression", "model_version": "nb02-20261014"}
```

Endpoints: `GET /health`, `GET /model-info`, `POST /predict` (all 39 features optional;
missing numerics are median-imputed, absent/unseen categories map to the OOV index —
forbidden extra fields return 422). Notebook 03 and the API call the **same**
`ctr.service.predict_proba` function, so offline evaluation and online scoring cannot
drift apart by construction.

## Tests and CI

`tests/` runs on a small synthetic frame with the real schema — no download, no trained
artifacts required (the service tests build their own bundle in a temp directory):

- `test_data.py` — sampling determinism, schema validation, split ratios/stratification.
- `test_features.py` — train-only fitting, median imputation, log1p selection, OOV
  mapping, indicator columns vs the frozen config, all three model-input layouts,
  persistence round trip.
- `test_models.py` — metric-adjacent model behaviour, the FM interaction formula against
  a manual computation, and a one-epoch DeepFM smoke test including Keras serialization.
- `test_evaluate.py` — hand-computed metric values, calibration table partitioning,
  ECE on perfectly/ badly calibrated constructions, latency harness, segment reporting.
- `test_sql.py` — every statement in `sql/` parses and runs via DuckDB against the
  real schema, including `$min_support` parameter binding.
- `test_service.py` — bundle loading, the shared scoring path, and the HTTP endpoints
  (parity between single-record and batch scoring, null handling, unseen categories,
  422/503 behaviour).

GitHub Actions runs the full suite on Python 3.11 (`.github/workflows/ci.yml`).

## What this does not claim

- **No timestamps in the data pool → random split, not temporal.** The evaluation
  measures probability fidelity on i.i.d. traffic from the same period; real traffic
  drifts by hour, campaign and season, and a production system would validate on future
  windows and monitor calibration continuously.
- **Logged impressions are auction winners.** Training data inherits the incumbent
  policy's selection bias, and new predictions would alter which impressions get logged
  next (feedback loop) — neither is visible offline.
- **CTR is not revenue, and offline deltas are not online lift.** The 0.006 test
  log-loss gap between models is real but small; whether it moves revenue requires an
  online A/B test with revenue and CTR-by-position endpoints.
- **Calibration is measured on 15,002 rows.** Good ECE here does not guarantee
  calibration later; deployed calibration needs rolling re-checks.
- **Latency numbers are indicative.** Single process, CPU, no serving optimisations
  (ONNX, batching layer, thread pinning); a real ad server benchmarks on its own
  hardware with its own tail targets.
- **DeepFM simplification, stated.** The 36 numeric inputs (13 numerics + 23 missing
  indicators) enter as one projected field plus raw MLP inputs rather than 36 separate
  embedding fields — documented in `src/ctr/deepfm.py` and notebook 02.

## Environment

Python 3.11; pandas 3.0.6, numpy 2.4.6, scikit-learn 1.9.1, LightGBM 4.7.0,
TensorFlow 2.21.0 / Keras 3.15.1 (CPU), DuckDB 1.5.6, FastAPI 0.143.1 (full pins in
`requirements.txt`). All random seeds fixed (42); LightGBM runs with
`deterministic=True`, TensorFlow with op-level determinism enabled.
