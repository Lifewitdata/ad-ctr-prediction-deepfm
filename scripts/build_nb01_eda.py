"""Generate notebooks/01_ctr_in_depth_eda.ipynb and (by default) execute it.

The notebook is written as real cells with real outputs: run

    venv/Scripts/python.exe scripts/build_nb01_eda.py

to regenerate and execute it in place (add --no-exec to only rewrite the file).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# ---------------------------------------------------------------------------
# Title + orientation
# ---------------------------------------------------------------------------
md(r"""
# Ad Click-Through Rate Prediction - In-Depth EDA (Notebook 01 of 3)

**Project.** CTR prediction on the Criteo Display Advertising Challenge dataset - a
public, anonymised benchmark from display advertising - built as an independent
portfolio project. Everything here runs on openly available data; nothing describes
or depends on any specific publisher's production systems.

**The three notebooks.**
1. `01_ctr_in_depth_eda.ipynb` (this one) - structure, integrity, class imbalance,
   missing values, numeric and categorical behaviour, interactions, leakage-safe
   splitting, train/validation comparison, SQL cross-checks, decision table.
2. `02_ctr_model_training.ipynb` - constant baseline, logistic regression, gradient
   boosted trees and DeepFM trained on the same split.
3. `03_ctr_model_evaluation.ipynb` - held-out evaluation: log loss (primary), ROC-AUC,
   PR-AUC, calibration, latency, missing/unseen-input behaviour.

**Business context in one paragraph.** An *impression* is one ad served; a *click* is a
user action on it; CTR = clicks / impressions. Ad-supported publishers use click
predictions to rank candidate ads for each impression, to price inventory and to decide
when not to show an ad. CTR is **not** conversions and **not** revenue: a click can
bounce, a conversion can happen without a click, and revenue depends on bids,
viewability and attribution windows. A 1% better predicted CTR does not mean 1% more
revenue - later notebooks keep that gap explicit.

**Section map** (spec section -> this notebook): sampling -> *S*; structure/integrity
-> *A*; leakage-aware partitions -> *G*; class imbalance -> *B*; missing values -> *C*;
numeric -> *D*; categorical -> *E*; interactions -> *F*; train-vs-val -> *H*; SQL
cross-checks -> later section; conclusions -> *I*.

**How to run.** From the project root:

```bash
python scripts/download_data.py   # fetch the raw pool once (~209 MB) if missing
make eda                          # builds + executes this notebook in place
```

All configuration (sample size, seed, split ratios, transform thresholds) lives in
`config.yaml`. Every output in this file was produced by executing the notebook
end-to-end from top to bottom.
""")

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
code(r"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

T_START = time.time()

# Project bootstrap: find the root (folder containing config.yaml) from the CWD.
_here = Path.cwd()
PROJECT_ROOT = next(p for p in [_here, *_here.parents] if (p / "config.yaml").exists())
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib.pyplot as plt
import seaborn as sns

from ctr import data as ctr_data
from ctr import eda as ctr_eda
from ctr import sql_runner
from ctr.config import data_file, load_config

sns.set_theme(style="whitegrid", rc={"figure.dpi": 110, "axes.titlesize": 10, "axes.labelsize": 9, "font.size": 9})

cfg = load_config()
SEED = int(cfg["seed"])
NUM_COLS = list(cfg["features"]["numeric_cols"])
CAT_COLS = list(cfg["features"]["categorical_cols"])
FEATURE_COLS = NUM_COLS + CAT_COLS

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=110, bbox_inches="tight")
    return path


def ctr_row(y):
    n = len(y)
    clicks = int(np.asarray(y).sum())
    lo, hi = ctr_eda.wilson_ci(clicks, n)
    return {"n": n, "clicks": clicks, "ctr": clicks / n, "ci_low": lo, "ci_high": hi,
            "ci_95_pct": "[{:.2f}%, {:.2f}%]".format(100 * lo, 100 * hi)}


pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 170)
pd.set_option("display.max_rows", 80)

print("project root:", PROJECT_ROOT)
print("pandas", pd.__version__, "| numpy", np.__version__)

raw_path = data_file(cfg, "raw_path")
sample_path = data_file(cfg, "sample_path")
splits_dir = data_file(cfg, "splits_dir")
if not raw_path.exists():
    raise FileNotFoundError(
        "raw data not found at " + str(raw_path) + "\n"
        "Run:  python scripts/download_data.py   (then re-run this notebook)."
    )
""")

# ---------------------------------------------------------------------------
# S. Sampling
# ---------------------------------------------------------------------------
md(r"""
## S. Sampling - how the development dataset was built

**Q:** Which data exactly do we analyse, and how was this sample drawn?
**Why it matters:** every later number (CTR, missingness, split sizes) is conditioned
on this choice. If the sample is not a fair draw from the pool, all downstream
statistics inherit the bias.

**The raw pool.** `data/raw/criteo_pool.parquet` is a community mirror of the Criteo
Display Advertising Challenge *training* data (the original distribution is a
tab-separated file with no header; 45,840,617 labelled impressions in total). The
mirror file used here is a 4,584,062-row partition of that data, already converted to
parquet. Schema as verified in section A: `Label` (0/1), `I1`-`I13` (anonymised
numeric features, with missing entries) and `C1`-`C26` (8-character hex hashes of
anonymised categorical features). Two provenance caveats we carry through the whole
project:

1. the mirror's own split construction is unverified, and
2. there is **no timestamp column**, so real traffic order cannot be reconstructed.

We therefore treat rows as exchangeable draws and never claim temporal validity.

**Sampling method.** Simple random sample *without replacement* over row positions:
`numpy.random.default_rng(42).choice(total_rows, size=sample_rows, replace=False)`.
The parquet file is streamed in 131,072-row batches and only the selected positions
are materialised, so the full table never sits in memory. `sample_rows` (100,000 by
default) and the seed are configuration values - to scale up, set
`data.sample_rows` in `config.yaml` (for example 1000000, or `null` for all rows) and
rebuild.
""")

code(r"""
t0 = time.time()
pool_rows = ctr_data.row_count(raw_path)
if sample_path.exists():
    sample = pd.read_parquet(sample_path)
    origin = "loaded cached sample"
else:
    sample = ctr_data.uniform_sample(
        raw_path, n=int(cfg["data"]["sample_rows"]), seed=SEED, batch_rows=int(cfg["data"]["batch_rows"])
    )
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(sample_path, index=False)
    origin = "drew fresh sample"
sample = ctr_data.normalize(sample)
print("{origin}: {n:,} of {pool:,} pool rows in {secs:.1f}s (seed={seed})".format(
    origin=origin, n=len(sample), pool=pool_rows, secs=time.time() - t0, seed=SEED))
print("sample file:", sample_path)

# Pool-level facts, computed with the same SQL files that are later reused on the
# sample: DuckDB queries the parquet directly, no Python loop.
con_pool = sql_runner.connect(raw_path)
pool_overview = sql_runner.run_file(con_pool, PROJECT_ROOT / "sql" / "01_overview.sql")
pool_missing = sql_runner.run_file(con_pool, PROJECT_ROOT / "sql" / "02_missing_values.sql")
print()
print(pool_overview.to_string(index=False))

# Representativeness: sample vs pool CTR (Wilson interval) and missingness.
samp_ctr = ctr_row(sample["label"])
print("pool CTR {p:.5f} | sample CTR {s:.5f} {ci}".format(
    p=pool_overview["ctr"][0], s=samp_ctr["ctr"], ci=samp_ctr["ci_95_pct"]))

pool_null_pct = 100 * pool_missing["n_missing"] / pool_rows
samp_null_pct = 100 * sample.isna().mean().reindex(pool_missing["feature"]).values
rep = pd.DataFrame({
    "feature": pool_missing["feature"],
    "pool_missing_pct": pool_null_pct.round(3),
    "sample_missing_pct": samp_null_pct.round(3),
})
rep["abs_diff_pp"] = (rep["pool_missing_pct"] - rep["sample_missing_pct"]).abs().round(3)
rep = rep[rep["pool_missing_pct"] > 0].sort_values("abs_diff_pp", ascending=False)
print("columns with missingness: {n}; largest sample-vs-pool gap: {g:.3f} pp".format(
    n=len(rep), g=rep["abs_diff_pp"].max()))
print(rep.head(8).to_string(index=False))

assert abs(samp_ctr["ctr"] - float(pool_overview["ctr"][0])) < 0.006, "sample CTR far outside 4-sigma of the pool"
assert rep["abs_diff_pp"].max() < 0.6, "sample missingness differs from the pool beyond sampling noise"
""")

md(r"""
**What we see.** The sample CTR sits well inside +-4 standard errors of the pool CTR and
every missingness share is within a fraction of a percentage point of its pool value -
consistent with a fair uniform draw (subject to the mirror caveat above). The sampling
cost is seconds for 100,000 rows because only selected positions are materialised.

**Implication.** We proceed with the 100,000-row development sample. Every later
statistic is computed on its train/validation/test splits; scaling to 1M rows is a
config change plus a data rebuild, not a code change.
""")

# ---------------------------------------------------------------------------
# A. Structure and integrity
# ---------------------------------------------------------------------------
md(r"""
## A. Structure and integrity

**Q:** Does the file match the documented schema, and what state is it in?
**Why it matters:** model code written against the wrong schema, or an ill-defined
label, silently produces nonsense metrics. These checks run before any analysis.

Checks: column set and dtypes, label domain and missingness, memory footprint, full-row
duplicates (reported, deliberately **not** deleted - see below), and constant or
near-constant columns.
""")

code(r"""
schema = ctr_data.validate_schema(sample)
print("validate_schema:", json.dumps(schema, indent=2))
print()

struct = pd.DataFrame({
    "dtype": sample.dtypes.astype(str),
    "n_missing": sample.isna().sum(),
    "pct_missing": (100 * sample.isna().mean()).round(3),
})
struct["role"] = ["label" if c == "label" else ("numeric" if c in NUM_COLS else "categorical")
                  for c in struct.index]
print(struct.groupby(["role", "dtype"], as_index=False).agg(
    columns=("dtype", "size"), total_missing=("n_missing", "sum")).to_string(index=False))

mem_bytes = int(sample.memory_usage(deep=True).sum())
print()
print("sample memory (deep): {mb:.1f} MB for {r:,} rows x {c} columns".format(
    mb=mem_bytes / 1e6, r=len(sample), c=sample.shape[1]))

label_counts = sample["label"].value_counts().sort_index()
print("label domain:", list(label_counts.index), "| counts:", label_counts.to_dict())
print()

dups_incl_label = int(sample.duplicated().sum())
h_feat = pd.util.hash_pandas_object(sample[FEATURE_COLS], index=False)
feat_dup_rows = int(h_feat.duplicated().sum())
grp = pd.DataFrame({"h": h_feat, "label": sample["label"].to_numpy()}).groupby("h")["label"]
mixed_labels = int((grp.nunique() > 1).sum())
print("full-row duplicates (features + label identical): {:,} rows ({:.3f}%)".format(
    dups_incl_label, 100 * dups_incl_label / len(sample)))
print("rows sharing an identical feature vector (64-bit hash, collisions negligible): {:,} ({:.3f}%)".format(
    feat_dup_rows, 100 * feat_dup_rows / len(sample)))
print("feature-identical groups whose labels disagree: {:,}".format(mixed_labels))
print("duplicates are REPORTED, not deleted (see interpretation below)")

nc = ctr_eda.near_constant_report(sample, FEATURE_COLS, threshold=0.995)
print()
print("near-constant columns (top value >= 99.5% of rows): {}".format(len(nc)))
print(nc.to_string(index=False) if len(nc) else "none")
""")

md(r"""
**What we see.** The file matches the documented Criteo schema exactly: one 0/1 label
without gaps, 13 numeric columns (12 float64 for the columns with gaps, 1 int64), 26
string columns, no unexpected columns. The 100k-row sample fits in about 51 MB. And the
duplicate check comes back empty: **no** full-row duplicates and **no** repeated feature
vectors at all. That is itself informative - across 39 mostly high-entropy anonymous
features, fingerprints essentially never collide at this scale, so there is nothing to
deduplicate and no duplicated-label confusion to resolve.

**Decisions.**
- Duplicate policy: reported, never auto-deleted. Here the question is moot (zero
  repeats); had repeats existed they would have been kept - repeated contexts are
  legitimate traffic - with cross-partition overlap quantified in section G.
- **No near-constant column was silently dropped.** If any column were effectively
  constant it would be harmless anyway (trees ignore it, L2 shrinks it); removing
  features is a modelling decision that belongs to validation, not to cleaning.
""")

# ---------------------------------------------------------------------------
# G. Splits and leakage-safe design
# ---------------------------------------------------------------------------
md(r"""
## G. Partitions and leakage-safe design

*(Placed before all target-driven analysis on purpose: the partitions must exist before
the label is allowed to influence anything.)*

**Q:** How do we partition the data, and how do we stop the label from leaking?
**Why it matters:** the classic CTR modelling failures are leakage failures - target
encoding computed over all rows, imputation or scaling fitted on everything, features
chosen after peeking at the test set. Leakage inflates offline metrics and the damage
in production is worse than a bad report: an auction model scored on rows whose labels
it has already seen bids for the wrong inventory.

**Split design (given that no timestamps exist).**

- 70 / 15 / 15 train / validation / test, stratified on the label, seed 42 - reproducible
  and label-balanced by construction.
- Roles: **train** fits everything, including every statistic (medians, means,
  vocabularies); **validation** is for model choice, early stopping and any threshold
  tuning; **test** is locked away until notebook 03.
- A time-based split would be more realistic for advertising (tomorrow's traffic is not
  yesterday's), but there is no timestamp column and inventing an order would be
  fabrication. This is a stated limitation: a random split measures generalisation to
  *exchangeable* traffic, not to future traffic.

**Rules enforced downstream** (implemented in `src/ctr/features.py`, demonstrated in the
cells below):

1. every fitted statistic is computed on the train split only and frozen before
   val/test are transformed;
2. no target encoding anywhere in the pipeline (the demo below shows why);
3. the test split is only ever transformed and scored - never used to pick features,
   tune models or fit statistics.
""")

code(r"""
manifest_path = splits_dir / "split_manifest.json"
if manifest_path.exists():
    splits = ctr_data.load_splits(splits_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print("loaded existing splits from", splits_dir)
else:
    splits = ctr_data.stratified_split(sample, ratios=cfg["data"]["split"], seed=SEED)
    manifest = {
        "seed": SEED,
        "ratios": cfg["data"]["split"],
        "method": "stratified on label; per-class seeded permutation of row positions",
        "source": str(sample_path),
        "rows_in_sample": int(len(sample)),
    }
    ctr_data.save_splits(splits, splits_dir, manifest)
    print("created splits in", splits_dir)
train, val, test = splits["train"], splits["val"], splits["test"]

split_summary = pd.DataFrame({name: ctr_row(part["label"]) for name, part in splits.items()}).T
split_summary["ci_95_pct"] = split_summary["ci_95_pct"]
print(split_summary.to_string())
print("total rows:", sum(len(p) for p in splits.values()), "| sample rows:", len(sample))

assert sum(len(p) for p in splits.values()) == len(sample)
ctrs = [split_summary.loc[name, "ctr"] for name in ("train", "val", "test")]
assert max(ctrs) - min(ctrs) < 0.01, "stratification failed: base rates differ across partitions"
manifest
""")

md(r"""
### G.1 Why target encoding is banned here (with numbers)

A common way to feed a high-cardinality category into a model is to replace it with the
average CTR of that category. Computed over the rows you train on, each row's encoding
contains its own label - which is exactly what the model is asked to predict. The cells
below quantify the inflation on `C1` and then check a second leakage surface: identical
feature vectors appearing on both sides of a partition boundary.
""")

code(r"""
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

col = "C1"
prior = float(train["label"].mean())

# (a) in-sample: encode each row with CTRs computed on the whole train split,
#     including the row's own label - the classic leak.
enc_insample = train.groupby(col)["label"].transform("mean")
auc_insample = roc_auc_score(train["label"], enc_insample)

# (b) out-of-fold: each row encoded from the other four folds only.
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
enc_oof = pd.Series(np.nan, index=train.index)
for fit_idx, apply_idx in kf.split(train):
    stats = train.iloc[fit_idx].groupby(col)["label"].mean()
    enc_oof.loc[train.index[apply_idx]] = train.iloc[apply_idx][col].map(stats).to_numpy()
enc_oof = enc_oof.fillna(prior)
auc_oof = roc_auc_score(train["label"], enc_oof)

# (d) honest: train-only statistics applied to validation rows.
enc_val_honest = val[col].map(train.groupby(col)["label"].mean()).fillna(prior)
auc_val_honest = roc_auc_score(val["label"], enc_val_honest)

# (c) full-data statistics (train+val+test) applied to validation rows: val labels leak in.
full_stats = sample.groupby(col)["label"].mean()
enc_val_full = val[col].map(full_stats).fillna(sample["label"].mean())
auc_val_full = roc_auc_score(val["label"], enc_val_full)

demo1 = pd.DataFrame({
    "encoding_of_C1": ["train rows, in-sample stats (a)", "train rows, out-of-fold stats (b)",
                       "val rows, train-only stats (d)", "val rows, full-data stats (c)"],
    "scored_on": ["train", "train", "val", "val"],
    "auc": np.round([auc_insample, auc_oof, auc_val_honest, auc_val_full], 4),
})
print(demo1.to_string(index=False))

support = train[col].value_counts()
print()
print("{c} levels in train: {n:,}; levels with support <= 2 (one row's label becomes the encoding): {rare:,}".format(
    c=col, n=int(support.size), rare=int((support <= 2).sum())))
assert auc_insample > auc_oof, "expected in-sample encoding to look stronger than out-of-fold"
""")

code(r"""
# Second leakage surface: identical feature vectors across partitions.
train_hash = pd.Index(pd.util.hash_pandas_object(train[FEATURE_COLS], index=False))
val_hash = pd.Index(pd.util.hash_pandas_object(val[FEATURE_COLS], index=False))
test_hash = pd.Index(pd.util.hash_pandas_object(test[FEATURE_COLS], index=False))

val_in_train = 100 * val_hash.isin(train_hash).mean()
test_in_seen = 100 * (test_hash.isin(train_hash) | test_hash.isin(val_hash)).mean()
print("val rows whose full feature vector also appears in train: {:.3f}%".format(val_in_train))
print("test rows whose feature vector appears in train or val:   {:.3f}%".format(test_in_seen))
print()
print("Both numbers being 0.000% means no identical context straddles a partition")
print("boundary in this sample. The check stays in the notebook because it is cheap and")
print("must be re-verified whenever the sample size or pool changes.")
""")

md(r"""
**What we see.** `C1` on its own is a weak feature (AUC near 0.50), which makes the demo
clean: in-sample encoding still forges AUC 0.519 out of an honest out-of-fold value of
0.500, and scoring validation rows with statistics that include validation labels lifts
the apparent AUC from 0.497 to 0.517. The mechanism is visible in the support counts -
274 of the 443 `C1` levels appear at most twice in the training split, so for those rows
the encoding *is* the row's own label wearing a category costume. Separately, the
feature-vector overlap check comes back at 0.000%: no identical context straddles a
partition boundary in this sample.

**Decisions carried forward.**

- No target encoding anywhere. Categories enter the models as one-hot (logistic
  regression), native categorical splits (LightGBM) or embedding tables (DeepFM);
  vocabularies keep training-only counts and everything rare/unseen/missing maps to one
  shared OOV slot.
- Statistics are fitted once on train and frozen before touching val/test.
- The duplicate-overlap check is a standard assertion now (zero here), re-run whenever
  the sample size or pool changes.
- The test split stays sealed until notebook 03.
""")

# ---------------------------------------------------------------------------
# B. Class imbalance
# ---------------------------------------------------------------------------
md(r"""
## B. Class imbalance

**Q:** How imbalanced is the label, and what does that mean for training and evaluation?
**Why it matters:** CTR data is heavily imbalanced, and the most common failure is
reporting accuracy - which a constant model wins. We fix the evaluation frame here,
before any model exists.
""")

code(r"""
train_ctr, val_ctr, test_ctr = (ctr_row(p["label"]) for p in (train, val, test))
balance = pd.DataFrame({"train": train_ctr, "val": val_ctr, "test": test_ctr}).T
print(balance.to_string())

from sklearn.metrics import log_loss

p0 = train_ctr["ctr"]
const_logloss = log_loss(train["label"], np.full(len(train), p0), labels=[0, 1])
majority_acc = max(p0, 1 - p0)
print()
print("constant predictor p = train CTR {p:.4f}: log loss {ll:.4f}  [= binary entropy of the base rate]".format(p=p0, ll=const_logloss))
print("always-predict-0 accuracy: {a:.4f}  <- the number that flatters a useless model".format(a=1 - p0))
assert abs(const_logloss - (-(p0 * np.log(p0) + (1 - p0) * np.log(1 - p0)))) < 1e-9

fig, ax = plt.subplots(figsize=(5, 3))
counts = train["label"].value_counts().sort_index()
bars = ax.bar(["not clicked (0)", "clicked (1)"], counts.values, color=["#7f8c8d", "#2471a3"])
for b, v in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width() / 2, v, "{:,}\n({:.1%})".format(v, v / len(train)),
            ha="center", va="bottom")
ax.set_title("Train label balance (n = {:,})".format(len(train)))
ax.set_ylabel("impressions")
ax.set_ylim(0, counts.max() * 1.18)
save_fig(fig, "nb01_class_balance.png")
plt.show()
""")

md(r"""
**What we see.** Roughly three in four impressions do not click. A model that never
predicts a click would score about 74% accuracy while ranking nothing. The constant
predictor's log loss (~0.570) is the entropic floor of "no information" - any model that
cannot beat it on held-out data is worthless regardless of accuracy.

**Implications.**
- Log loss (binary cross-entropy) is the primary metric: it is a *proper scoring rule*,
  so it is minimised only by predicting true probabilities. Accuracy and even AUC can
  look fine while probabilities are unusable for auction decisions.
- We do **not** resample or reweight classes for the primary models. Class weights and
  negative down-sampling change what the predicted probability means (it stops being
  P(click)), so anyone using them must re-calibrate before serving. We train with the
  natural imbalance and measure calibration explicitly in notebook 03 - the damage would
  be visible there.
- The imbalance also caps naive metrics: with PR-AUC (average precision), the no-skill
  level equals the base rate, so it is quoted alongside its baseline everywhere.
""")

# ---------------------------------------------------------------------------
# C. Missing values
# ---------------------------------------------------------------------------
md(r"""
## C. Missing values

**Q:** Where are the gaps, do they co-occur in patterns, and does "missing" carry signal?
**Why it matters:** Criteo's missingness is structural, not sprinkles of noise - several
columns vanish together for whole slices of traffic. The fill strategy changes both the
model inputs and the meaning of the result. One thing we will not do is *claim
causation*: with anonymised features we can only say that the missing slice behaves
differently, never why.
""")

code(r"""
miss = ctr_eda.missing_summary(train)
miss = miss[miss["n_missing"] > 0].reset_index(drop=True)
print("columns with missing values in train: {} of {}".format(len(miss), train.shape[1]))
print(miss.to_string(index=False))
miss.to_csv(TABLE_DIR / "missing_summary.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 3.6))
ax.bar(miss["column"], miss["pct_missing"], color="#b03a2e")
ax.set_ylabel("% missing (train)")
ax.set_title("Missing share per column (train split)")
plt.xticks(rotation=90)
save_fig(fig, "nb01_missing_per_column.png")
plt.show()
""")

code(r"""
pat = ctr_eda.missing_patterns(train, FEATURE_COLS, top=10)
print("Top missingness patterns in the train split:")
print(pat[["n_rows", "pct_rows", "n_missing_in_pattern", "columns_missing"]].to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 3))
top = pat.head(8).iloc[::-1].reset_index(drop=True)
ax.barh(["#{:d}".format(i + 1) for i in range(len(top))], top["pct_rows"], color="#2471a3")
ax.set_xlabel("% of train rows")
ax.set_title("Top-8 missingness patterns (ranked as in the table above)")
save_fig(fig, "nb01_missing_patterns.png")
plt.show()
""")

code(r"""
rows = []
for c in miss["column"]:
    m = train[c].isna()
    a, b = ctr_row(train.loc[m, "label"]), ctr_row(train.loc[~m, "label"])
    overlap = not (a["ci_low"] > b["ci_high"] or b["ci_low"] > a["ci_high"])
    rows.append({
        "column": c,
        "n_missing": a["n"], "n_present": b["n"],
        "ctr_missing": a["ctr"], "ctr_present": b["ctr"],
        "gap_pp": 100 * (a["ctr"] - b["ctr"]),
        "ci_low_missing": a["ci_low"], "ci_high_missing": a["ci_high"],
        "ci_low_present": b["ci_low"], "ci_high_present": b["ci_high"],
        "cis_overlap": overlap,
    })
miss_ctr = pd.DataFrame(rows).sort_values("gap_pp", key=lambda s: s.abs(), ascending=False)
miss_ctr["flagged"] = (~miss_ctr["cis_overlap"]) & (miss_ctr["gap_pp"].abs() >= 0.5)
print("Missing vs present CTR, train split (gap = CTR_missing - CTR_present):")
print(miss_ctr[["column", "n_missing", "n_present", "ctr_missing", "ctr_present",
                "gap_pp", "cis_overlap", "flagged"]].round(4).to_string(index=False))
miss_ctr.to_csv(TABLE_DIR / "missing_vs_present_ctr.csv", index=False)

top8 = miss_ctr.head(8)
x = np.arange(len(top8))
fig, ax = plt.subplots(figsize=(9, 3.6))
w = 0.38
err_m = np.vstack([top8["ctr_missing"] - top8["ci_low_missing"], top8["ci_high_missing"] - top8["ctr_missing"]])
err_p = np.vstack([top8["ctr_present"] - top8["ci_low_present"], top8["ci_high_present"] - top8["ctr_present"]])
ax.bar(x - w / 2, 100 * top8["ctr_missing"], w, yerr=100 * err_m, capsize=3,
       label="value missing", color="#b03a2e")
ax.bar(x + w / 2, 100 * top8["ctr_present"], w, yerr=100 * err_p, capsize=3,
       label="value present", color="#2471a3")
ax.set_xticks(x)
ax.set_xticklabels(top8["column"])
ax.set_ylabel("CTR (%)")
ax.set_title("CTR when a value is missing vs present (train, Wilson 95% CI); top-8 by |gap|")
ax.legend()
save_fig(fig, "nb01_ctr_by_missingness.png")
plt.show()

flagged = miss_ctr.loc[miss_ctr["flagged"], "column"].tolist()
decision_ind = len(flagged) > 0
print()
print("Decision rule (fixed before looking): flag a column when the Wilson CIs of missing")
print("vs present CTR do not overlap AND |gap| >= 0.5 pp (multiple-testing guard).")
print("flagged columns: {} -> {}".format(len(flagged), flagged))
print("=> recommended add_missing_indicators:", decision_ind)
""")

md(r"""
**What we see.** Missingness is far from uniform: a handful of columns are missing for
very large slices of traffic, and the top patterns show that gaps travel in groups -
`I1` and `I10` are missing on exactly the same rows, as are `C3 / C4 / C12 / C16 / C21 /
C24` (identical missing counts and identical CTRs in the table above) - so several
"missing slices" are really one slice. Whether that slice clicks differently is what the
missing-vs-present table quantifies; each gap comes with its sample size and Wilson
interval, and the flagging rule (non-overlapping CIs plus a 0.5 pp minimum) guards
against calling tiny differences on huge samples meaningful. One flagged column (`I8`)
has only 35 missing rows in train - an indicator for it is nearly free but nearly
constant; the substantive flags are the other 22.

**Plan (decided here, applied in notebook 02).**
- Numeric gaps -> train-median imputation. Medians are computed on the train split only
  and frozen; rows are never dropped (dropping rows for the ~76%-missing columns would
  delete most of the data).
- Categorical gaps -> the shared OOV index 0, which already means "unknown" for rare and
  unseen values. Missing is not invented as a new semantic category.
- Missing indicators: the a-priori rule flags 23 of the 24 gappy columns (`C6` is the
  one failure - its Wilson CIs overlap), so `config.yaml` sets
  `add_missing_indicators: true` and records those 23 columns in
  `missing_indicator_columns`; each contributes one 0/1 input in notebook 02. Notebook 02
  can ablate this against imputation-only if the models show it is carrying no weight.
- No causal claims: the table describes co-behaviour of a missing slice, and the
  missing flag may proxy for traffic types whose meaning is anonymised away.
""")

# ---------------------------------------------------------------------------
# D. Numeric columns
# ---------------------------------------------------------------------------
md(r"""
## D. Numeric columns (I1-I13)

**Q:** What shape are the numeric distributions - skew, zeros, ranges - and do they
separate clickers from non-clickers?
**Why it matters:** transform and imputation choices depend on distribution shape.
The I columns are anonymised: we describe their distributions numerically and never
assign them meanings like "price" or "age".
""")

code(r"""
def numeric_profile(frame, cols):
    out = []
    for c in cols:
        x = frame[c].astype("float64")
        n = int(x.notna().sum())
        row = {"column": c, "non_null": n, "missing_pct": 100 * (1 - n / len(frame)),
               "min": x.min(), "mean": x.mean(), "std": x.std(),
               "max": x.max(), "zero_pct": 100 * (x == 0).mean(), "neg_pct": 100 * (x < 0).mean(),
               "skew_raw": x.skew()}
        row.update({"p{:.0f}".format(p * 100): x.quantile(p) for p in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)})
        out.append(row)
    cols_order = ["column", "non_null", "missing_pct", "min", "p1", "p5", "p25", "p50",
                  "p75", "p95", "p99", "max", "mean", "std", "zero_pct", "neg_pct", "skew_raw"]
    return pd.DataFrame(out)[cols_order]


prof = numeric_profile(train, NUM_COLS)
print("Numeric descriptives (train split):")
print(prof.round(3).to_string(index=False))
prof.to_csv(TABLE_DIR / "numeric_descriptives.csv", index=False)
""")

code(r"""
fig, axes = plt.subplots(4, 4, figsize=(11, 8))
for ax, c in zip(axes.ravel(), NUM_COLS):
    x = train[c].dropna()
    ax.hist(x, bins=60, color="#2471a3")
    ax.set_yscale("log")
    ax.set_title("{}  (missing {:d}%)".format(c, int(round(100 * train[c].isna().mean()))), fontsize=8)
for ax in axes.ravel()[len(NUM_COLS):]:
    ax.axis("off")
fig.suptitle("Numeric distributions, train split (counts on log scale)", y=1.0)
fig.tight_layout()
save_fig(fig, "nb01_numeric_histograms.png")
plt.show()
""")

code(r"""
thr = float(cfg["features"]["skew_threshold"])
sk = prof.set_index("column")[["skew_raw", "zero_pct", "neg_pct"]].sort_values(
    "skew_raw", key=lambda s: s.abs(), ascending=False)
log_cols = [c for c in NUM_COLS if abs(train[c].astype("float64").skew()) > thr]
print("columns selected for sign-preserving log1p (|skew| > {:.1f}): {} of 13 -> {}".format(
    thr, len(log_cols), log_cols))
print("range check: {} of 13 columns contain negative values (plain log would be invalid); "
      "skewness is computed on train only, never on val/test".format(int((prof["neg_pct"] > 0).sum())))

chk = []
for c in log_cols:
    x = train[c].astype("float64")
    xf = x.fillna(x.median())
    xl = np.sign(xf) * np.log1p(np.abs(xf))
    chk.append({"column": c, "skew_raw": x.skew(), "skew_after_log1p": xl.skew()})
chk = pd.DataFrame(chk)
print()
print("Skew before/after the fitted transform (train only):")
print(chk.round(3).to_string(index=False))
assert np.isfinite(chk["skew_after_log1p"]).all()

fig, ax = plt.subplots(figsize=(9, 3.2))
colors = ["#b03a2e" if abs(v) > thr else "#7f8c8d" for v in sk["skew_raw"]]
ax.bar(sk.index, sk["skew_raw"], color=colors)
ax.axhline(thr, ls="--", c="k", lw=0.8)
ax.axhline(-thr, ls="--", c="k", lw=0.8)
ax.set_ylabel("skew (raw)")
ax.set_title("Raw skew per numeric column (red = |skew| > {:.1f}, gets log1p)".format(thr))
plt.xticks(rotation=90)
save_fig(fig, "nb01_numeric_skew.png")
plt.show()
""")

code(r"""
rows = []
for c in NUM_COLS:
    a = train.loc[train["label"] == 1, c].dropna()
    b = train.loc[train["label"] == 0, c].dropna()
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    rows.append({"column": c, "mean_clicked": a.mean(), "mean_not": b.mean(),
                 "median_clicked": a.median(), "median_not": b.median(),
                 "cohens_d": (a.mean() - b.mean()) / pooled if pooled else np.nan})
sep = pd.DataFrame(rows).sort_values("cohens_d", key=lambda s: s.abs(), ascending=False)
print("Clicked vs not - effect sizes (train; |d| < 0.2 is small, > 0.5 medium):")
print(sep.round(3).to_string(index=False))
sep.to_csv(TABLE_DIR / "numeric_clicked_vs_not.csv", index=False)
print()
print("With n ~ 70k, p-values are uninformative (everything is 'significant'); effect")
print("sizes and intervals are the honest summary. These are raw-scale comparisons; the")
print("fitted log1p transform reshapes magnitudes, not signs.")
""")

code(r"""
deciles = {c: ctr_eda.ctr_by_rank_bins(train, c, q=10).set_index("bin")["ctr"] for c in NUM_COLS}
dec_mat = pd.DataFrame(deciles).T
dec_mat.columns = ["d{}".format(i) for i in range(1, 11)]
print("CTR (%) per equal-count decile (train). Each decile holds about {:,} rows.".format(int(len(train) / 10)))
print((100 * dec_mat).round(2).to_string())
dec_mat.to_csv(TABLE_DIR / "numeric_ctr_by_decile.csv")

gaps = dec_mat["d10"] - dec_mat["d1"]
print()
print("top-decile minus bottom-decile CTR (percentage points):")
print((100 * gaps.sort_values(key=lambda s: s.abs(), ascending=False)).round(2).to_string())

chosen = list(gaps.abs().sort_values(ascending=False).head(2).index)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharex=True)
for ax, c in zip(axes, chosen):
    t = ctr_eda.ctr_by_rank_bins(train, c, q=10)
    ax.errorbar(t["bin"], 100 * t["ctr"],
                yerr=[100 * (t["ctr"] - t["ci_low"]), 100 * (t["ci_high"] - t["ctr"])],
                marker="o", capsize=3, color="#2471a3")
    ax.set_title("{}: CTR by decile".format(c))
    ax.set_xlabel("decile of equal-count bins")
axes[0].set_ylabel("CTR (%)")
fig.suptitle("Two columns chosen by a stated rule: largest |d10 - d1| gap (exploratory choice)", y=1.04)
save_fig(fig, "nb01_ctr_by_decile.png")
plt.show()
print("bins are equal-count rank bins (the NTILE convention); tied values may split across bins.")
""")

md(r"""
**What we see.** The numeric columns are long right tails with mass at zero (see the
zero shares above), which is exactly where the |skew| threshold bites: 12 of 13 columns
are flagged, and the one column with negative values (`I2`) is why the transform must be
sign-preserving rather than a plain log. The before/after table shows `log1p` collapsing
|skew| from 2.5-58 down to mostly under 1.7; `I12` improves from 14.3 to 6.6 but stays
heavy - the goal is compression of the tail, not normality. Clicked and not-clicked rows
differ with small-to-moderate effect sizes (max |Cohen's d| = 0.35 for `I10`), and the
decile matrix shows real structure: `I6` swings from 54.9% to 14.6% CTR across deciles
(a 40 pp spread), `I5` falls 35 pp, `I11` rises 31 pp. Univariate signal is genuine but
uneven - some columns (`I2`, `I9`) barely move across deciles.

**Decisions.**
- Sign-preserving log1p for columns above the skew threshold, fitted (i.e. the threshold
  applied) on train only; the range check above motivates the sign-preserving form.
- Standardisation to z-scores with train-only means/stds.
- **No outlier removal.** The features are anonymised, so "extreme" has no semantic
  meaning; extreme ranges may be legitimate traffic. L2-regularised linear models and
  tree splits tolerate tails, and dropping rows would corrupt the impression
  distribution. If evaluation shows tail sensitivity, that becomes a modelling
  experiment - not reflexive cleaning.
- Imputation: train medians, applied to every partition (keeps all rows).
""")

# ---------------------------------------------------------------------------
# E. Categorical columns
# ---------------------------------------------------------------------------
md(r"""
## E. Categorical columns (C1-C26)

**Q:** How large are the vocabularies, how heavy is the long tail, and which categories
look predictive?
**Why it matters:** vocabulary design rules the model's memory and the fate of rare
values. Ranking categories by raw CTR without a support filter is the classic EDA
trap - the leaderboard fills with values seen a handful of times. Category IDs are
anonymised hex hashes and are displayed exactly as stored; no meaning is guessed.
""")

code(r"""
rows = []
for c in CAT_COLS:
    s = train[c]
    vc = s.value_counts(dropna=True)
    rows.append({
        "column": c, "n_levels": int(vc.size),
        "top1_share_pct": 100 * float(vc.iloc[0]) / s.notna().sum(),
        "levels_ge_10": int((vc >= 10).sum()),
        "rows_in_levels_lt_10_pct": 100 * float(vc[vc < 10].sum()) / len(train),
        "rows_in_levels_lt_50_pct": 100 * float(vc[vc < 50].sum()) / len(train),
        "max_level_n": int(vc.max()), "missing_pct": 100 * s.isna().mean(),
    })
cat_prof = pd.DataFrame(rows).sort_values("n_levels", ascending=False).reset_index(drop=True)
print("Categorical profile (train split):")
print(cat_prof.round(2).to_string(index=False))
cat_prof.to_csv(TABLE_DIR / "categorical_profile.csv", index=False)
""")

code(r"""
fig, ax = plt.subplots(figsize=(7, 3.6))
for c in cat_prof["column"].head(4):
    vc = train[c].value_counts()
    ax.plot(np.arange(1, len(vc) + 1), vc.to_numpy(), lw=1.2, label="{} ({:,} levels)".format(c, len(vc)))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("category rank (log)")
ax.set_ylabel("impressions (log)")
ax.set_title("Rank-frequency curves: 4 widest vocabularies (train)")
ax.legend(fontsize=8)
save_fig(fig, "nb01_category_longtail.png")
plt.show()
""")

code(r"""
def with_ci(tbl):
    ci = [ctr_eda.wilson_ci(int(k), int(n)) for k, n in zip(tbl["clicks"], tbl["n"])]
    out = tbl.copy()
    out["ci_low"] = [c[0] for c in ci]
    out["ci_high"] = [c[1] for c in ci]
    return out


min_count = int(cfg["features"]["min_category_count"])
t = ctr_eda.category_ctr_table(train, "C1", min_support=min_count, prior=prior, strength=50)
oov_pct = 100 * (1 - t["n"].sum() / len(train))
print("C1: {:,} levels with >= {} train rows; rows outside the vocabulary (OOV): {:.2f}%".format(
    len(t), min_count, oov_pct))

top_raw = with_ci(t.sort_values("ctr", ascending=False).head(10))
top_sm = with_ci(t.sort_values("ctr_smoothed", ascending=False).head(10))
print()
print("Top-10 C1 by RAW CTR (support >= {}):".format(min_count))
print(top_raw[["n", "clicks", "ctr", "ci_low", "ci_high"]].round(4).to_string())
print()
print("Top-10 C1 by SMOOTHED CTR (empirical-Bayes toward prior {:.4f}, strength 50):".format(prior))
print(top_sm[["n", "clicks", "ctr", "ctr_smoothed"]].round(4).to_string())
print()
print("median support: raw top-10 = {} rows | smoothed top-10 = {} rows".format(
    int(top_raw["n"].median()), int(top_sm["n"].median())))
assert top_raw["n"].median() < top_sm["n"].median(), "expected smoothed ranking to prefer better-supported categories"
""")

code(r"""
labels = ["<10 (OOV slot)", "10-49", "50-199", "200+"]
band_of = pd.cut(train["C1"].value_counts(), bins=[-1, 9, 49, 199, np.inf], labels=labels)
rows = []
for lab in labels:
    levels = band_of[band_of == lab].index
    m = train["C1"].isin(levels)
    r = ctr_row(train.loc[m, "label"])
    r["band"] = lab
    r["levels"] = len(levels)
    rows.append(r)
band_tbl = pd.DataFrame(rows)[["band", "levels", "n", "ctr", "ci_low", "ci_high"]]
print("C1 support bands (train):")
print(band_tbl.round(4).to_string(index=False))

vocab = {c: int(cat_prof.set_index("column").loc[c, "levels_ge_10"]) + 1 for c in CAT_COLS}
width = sum(vocab.values())
print()
print("vocabulary widths: one-hot (LR) prefix = {:,} columns + {} numerics".format(width, len(NUM_COLS)))
print("embedding tables (DeepFM, k=8 float32): {:,} params = {:.1f} MB".format(width * 8, width * 8 * 4 / 1e6))
print("LightGBM native categoricals: 26 int32 columns/row = {} bytes/row (codes, not one-hot)".format(26 * 4))
""")

md(r"""
**What we see.** The vocabularies span four orders of magnitude - from `C3` with 34,092
distinct values in train down to near-binary columns with 3-11 - and for the widest
columns most traffic sits in levels below the support threshold (57.5% of train rows for
`C3`, 51.7% of row-value pairs outside the >=10 vocabulary pool-wide per the SQL check),
while for `C1` only 1.05% of rows fall outside. The raw-CTR leaderboard confirms the
classic trap: the top raw values are levels with 10-41 impressions carrying 33-50% CTR
and intervals spanning 0.15-0.75; the smoothed ranking (empirical-Bayes toward the
global prior, strength 50) shifts the top-10's median support from 13 to 41 rows. For
`C1` the support bands are nearly indistinguishable (OOV band 26.5% CTR vs 25.7% for
200+ levels, overlapping intervals) - the "unknown" mass behaves like the known mass for
this column, which is a useful baseline expectation for the OOV slot in the models, not
a law; other columns' bands can be checked the same way.

**Decisions.**
- `min_category_count = 10` on train: levels with at least 10 train rows get an index;
  everything else shares OOV index 0. This is a judgment call balancing vocabulary
  breadth against per-level reliability - visible in the support bands.
- LR sees one-hot counts; LightGBM sees integer codes declared as native categoricals;
  DeepFM learns 8-dimensional embeddings. Notebook 02 measures what each option buys.
- The OOV and missing slot is a *single* bucket: the model can only learn "unknown", not
  which flavour of unknown - a simplification stated for transparency (missing
  indicators, enabled per section C, partially recover the distinction).
""")

# ---------------------------------------------------------------------------
# F. Interactions
# ---------------------------------------------------------------------------
md(r"""
## F. A first look at interactions

**Q:** Do features interact - does one feature's effect on CTR depend on another?
**Why it matters:** the FM and deep components of DeepFM exist to learn interactions.
EDA should suggest whether there is headroom beyond an additive model. Two honesty
rules apply: (1) we deliberately limit this to one numeric-numeric pair and one
numeric-categorical pair chosen by stated rules, because scanning hundreds of
combinations guarantees chance "discoveries" (multiple comparisons); (2) everything
here is a *hypothesis* to test on validation data, not a fact - cells have limited
support and apparent patterns may be noise.
""")

code(r"""
corr = train[NUM_COLS].corr()
fig, ax = plt.subplots(figsize=(6.2, 5.2))
sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f",
            annot_kws={"size": 6}, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Pearson correlation, numeric columns (train, pairwise-complete)")
save_fig(fig, "nb01_correlation_heatmap.png")
plt.show()

# keep one row per unordered pair (drop the mirror entry)
pairs = corr.where(~np.eye(len(NUM_COLS), dtype=bool)).stack()
pairs = pairs[pairs.index.get_level_values(0) < pairs.index.get_level_values(1)]
pairs = pairs.sort_values(key=lambda s: s.abs(), ascending=False)
print("strongest |r| pairs (each unordered pair once):")
print(pairs.head(6).round(3).to_string())
strong = pairs[pairs.abs() >= 0.5]
print()
print("pairs with |r| >= 0.5: {} -> {}".format(
    len(strong), ["{}~{} r={:.2f}".format(a, b, r) for (a, b), r in strong.items()]))
""")

code(r"""
(p1, p2) = pairs.index[0]
cell = pd.DataFrame({
    "a": ctr_eda.rank_bins(train[p1], 5),
    "b": ctr_eda.rank_bins(train[p2], 5),
    "y": train["label"].to_numpy(),
}).dropna()
piv = cell.pivot_table(index="a", columns="b", values="y", aggfunc="mean")
cnt = cell.pivot_table(index="a", columns="b", values="y", aggfunc="size")

fig, ax = plt.subplots(figsize=(5.4, 4.2))
sns.heatmap(100 * piv, annot=True, fmt=".1f", cmap="viridis", ax=ax,
            cbar_kws={"label": "CTR (%)"})
ax.set_title("CTR (%) by quintile: {} (rows) x {} (cols)\nchosen as the strongest |r| pair".format(p1, p2))
ax.set_xlabel("{} quintile".format(p2))
ax.set_ylabel("{} quintile".format(p1))
save_fig(fig, "nb01_ctr_pair_heatmap.png")
plt.show()
print("cell row counts: min {} / max {}".format(int(cnt.min().min()), int(cnt.max().max())))
""")

code(r"""
num_for_heat = chosen[0]
top_levels = train["C1"].value_counts().head(8).index
sub = train[train["C1"].isin(top_levels)].copy()
sub["q"] = ctr_eda.rank_bins(sub[num_for_heat], 5)
piv = sub.pivot_table(index="C1", columns="q", values="label", aggfunc="mean").reindex(top_levels)
cnt = sub.pivot_table(index="C1", columns="q", values="label", aggfunc="size").reindex(top_levels)

fig, ax = plt.subplots(figsize=(7.2, 4.4))
sns.heatmap(100 * piv, annot=True, fmt=".1f", cmap="magma", ax=ax,
            cbar_kws={"label": "CTR (%)"})
ax.set_title("CTR (%) by {} quintile x top-8 C1 levels (train)".format(num_for_heat))
ax.set_xlabel("{} quintile (computed within shown rows)".format(num_for_heat))
ax.set_ylabel("C1 (anonymised id, top-8 by frequency)")
save_fig(fig, "nb01_ctr_numcat_heatmap.png")
plt.show()
print("cell row counts: min {} / max {}".format(int(cnt.min().min()), int(cnt.max().max())))
""")

md(r"""
**What we see.** The numeric block is mostly well-behaved but not orthogonal: the
strongest unique pair is `I11`-`I7` at |r| = 0.67, followed by `I4`-`I13` (0.62) and
`I4`-`I8` (0.51); everything else is far below. All columns are kept - correlated inputs
are handled (regularisation shrinks the linear twins, trees pick one) - but the strongest
pair gets the interaction heatmap because correlated twins are also the natural
candidates for genuine joint effects. One honesty flag sits under both heatmaps: the
strongest pair's grid includes a corner supported by ~16 rows and the categorical grid
goes down to ~141; those are exactly the cells where chance patterns hide, so nothing
downstream depends on any single cell - the models get to test these hypotheses on
validation data instead.

**What this means for modelling.** This is the motivation for the models in notebook 02:
tree ensembles split greedily on interactions, the FM component learns every pairwise
embedding dot-product (data-efficiently, even for rarely co-occurring pairs), and the
deep component adds higher-order structure. Whether they actually deliver on this
dataset is an empirical question answered on validation data - EDA raises the
hypothesis, it does not settle it.
""")

# ---------------------------------------------------------------------------
# H. Train vs validation
# ---------------------------------------------------------------------------
md(r"""
## H. Train vs validation: partition comparison

**Q:** Is the validation split a fair stand-in for the training distribution?
**Why it matters:** if validation differs systematically - missingness, category mixes,
base rate - then model selection on it silently optimises the wrong thing. With a
random split the differences should look like sampling noise, and this section verifies
that rather than assuming it. This is explicitly *not* a temporal-drift analysis: no
timestamps exist, so drift is out of scope by data availability, not by choice.
""")

code(r"""
ks = ctr_eda.ks_table(train, val, NUM_COLS)
top_ks = ks.loc[ks["ks_stat"].idxmax()]
print("max KS across 13 numeric columns: {:.4f} ({})".format(top_ks["ks_stat"], top_ks["column"]))
print(ks.round(4).to_string(index=False))
print()
print("missingness train vs val: max |diff| = {:.3f} pp".format(
    (ks["missing_pct_train"] - ks["missing_pct_val"]).abs().max()))
ks.to_csv(TABLE_DIR / "ks_train_val.csv", index=False)
assert ks["ks_stat"].max() < 0.02, "numeric distributions differ between train and val beyond sampling noise"

ov = ctr_eda.category_overlap_table(train, val, CAT_COLS)
worst = ov.loc[ov["val_rows_unseen_pct"].idxmax()]
print()
print("unseen category levels: worst column {} with {:.2f}% of val rows".format(
    worst["column"], worst["val_rows_unseen_pct"]))
print(ov.round(3).to_string(index=False))
ov.to_csv(TABLE_DIR / "category_overlap.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 3.2))
ax.bar(ov["column"], ov["val_rows_unseen_pct"], color="#2471a3")
ax.set_ylabel("% of val rows unseen in train")
ax.set_title("Validation rows with a category level not present in the train vocabulary")
plt.xticks(rotation=90)
save_fig(fig, "nb01_val_unseen_categories.png")
plt.show()
""")

md(r"""
**What we see.** Numeric partitions match closely: the largest KS statistic is 0.0148
(`I3`, nominal p = 0.028 - the one result out of 13 comparisons that wobbles slightly,
which is what a 5% test expects to produce at this seed), everything else is at
noise level, missingness matches within 0.4 pp, and the base rates agree to within
0.003 pp. The one number to carry into notebook 03 is the unseen-category share: the
widest vocabularies route large slices of validation traffic to OOV (`C3` 44.4%, `C12`
40.6%, `C21` 37.4%, `C16` 32.6% of val rows), while mid and narrow columns stay in the
0-12% range. Those are exactly the levels that were rare (<10 rows) in train, so this is
the vocab decision from section E showing up as a measurable serving condition.

**Limitation, stated plainly.** These checks validate a *random* partition of
exchangeable rows. They say nothing about how the model would face tomorrow's traffic -
no timestamp exists to build or test a temporal split, so temporal generalisation
remains untested by construction.
""")

# ---------------------------------------------------------------------------
# SQL cross-checks
# ---------------------------------------------------------------------------
md(r"""
## SQL cross-checks (DuckDB)

**Q:** Do the Python numbers survive an independent engine?
**Why it matters:** in practice these numbers live in a warehouse, not a notebook.
The repo ships five SQL files under `sql/` that DuckDB runs directly against the
parquet file; the same statements would run on BigQuery or Snowflake with only the
table reference changed. Here we re-run the shipped files and *assert* agreement with
the pandas computations - two engines agreeing exactly is a strong correctness signal.

**Conventions to watch:** SQL `GROUP BY` treats `NULL` as a group value (pandas mirrors
use `dropna=False` where that matters); `NTILE` assigns tied rows to bins in
implementation-defined order, so per-bin CTR can wobble slightly while totals agree
exactly - that is a documented property, not an error.
""")

code(r"""
con = sql_runner.connect(sample_path)
sql_res = sql_runner.run_dir(con, PROJECT_ROOT / "sql",
                             params={"min_support": int(cfg["features"]["min_category_count"])})
print("ran:", ", ".join(sorted(sql_res)))

ovl = sql_res["01_overview"]
assert int(ovl["impressions"][0]) == len(sample)
assert int(ovl["clicks"][0]) == int(sample["label"].sum())
assert abs(float(ovl["ctr"][0]) - float(sample["label"].mean())) < 1e-12
print("impressions / clicks / CTR agree exactly: {:,} / {:,} / {:.9f}".format(
    len(sample), int(sample["label"].sum()), float(sample["label"].mean())))

m = sql_res["02_missing_values"]
assert (m["n_missing"].astype(int).to_numpy() == sample.isna().sum().reindex(m["feature"]).astype(int).to_numpy()).all()
print("missing counts agree for all {} columns".format(len(m)))

cf = sql_res["03_category_frequency_ctr"]
c1_sql = cf[(cf["feature"] == "C1") & cf["category_id"].notna()].sort_values("n", ascending=False).head(10)
py_c1 = sample.groupby("C1")["label"].agg(["size", "sum"]).sort_values("size", ascending=False).head(10)
cmp = c1_sql.set_index("category_id").join(py_c1, how="left")
assert (cmp["n"].astype(int) == cmp["size"].astype(int)).all()
assert (cmp["clicks"].astype(int) == cmp["sum"].astype(int)).all()
c1_null = cf[(cf["feature"] == "C1") & cf["category_id"].isna()]
assert (len(c1_null) == 0) or (int(c1_null["n"].iloc[0]) == int(sample["C1"].isna().sum()))
print("C1 top-10: n and clicks agree exactly; rows in SQL's NULL group for C1: {}".format(
    int(sample["C1"].isna().sum())))
print()
print(c1_sql.round(4).to_string(index=False))

ms = sql_res["05_min_support"].set_index("feature")
min_count = int(cfg["features"]["min_category_count"])
for c in CAT_COLS:
    vc_all = sample[c].value_counts(dropna=False)
    assert int(ms.loc[c, "distinct_values"]) == len(vc_all)
    assert int(ms.loc[c, "values_meeting_support"]) == int((vc_all >= min_count).sum())
print()
print("min-support stats agree for all 26 categorical columns (NULL treated as a value, as SQL does)")
print(ms.head(5).to_string())
""")

code(r"""
bn = sql_res["04_numeric_bins_ctr"]
for feat in ["I1", "I2"]:
    b = bn[bn["feature"] == feat].sort_values("decile").reset_index(drop=True)
    py = ctr_eda.ctr_by_rank_bins(sample, feat, q=10)
    assert len(b) == len(py) == 10
    assert int(b["n"].sum()) == int(sample[feat].notna().sum())
    assert int(b["clicks"].sum()) == int(sample.loc[sample[feat].notna(), "label"].sum())
    assert abs(float(b["min_value"].iloc[0]) - float(sample[feat].min())) < 1e-9
    assert abs(float(b["max_value"].iloc[-1]) - float(sample[feat].max())) < 1e-9
    assert np.isclose((b["n"] * b["avg_value"]).sum() / b["n"].sum(), sample[feat].mean(), rtol=1e-9)
    max_ctr_diff = float(np.abs(b["ctr"].to_numpy() - py["ctr"].to_numpy()).max())
    print("{}: totals, extremes and weighted mean agree; per-bin CTR max |SQL - pandas| = {:.5f} (NTILE tie order)".format(
        feat, max_ctr_diff))
print()
print("I1 decile detail (SQL):")
print(bn[bn["feature"] == "I1"].round(4).to_string(index=False))
""")

md(r"""
**What we see.** Row counts, click counts, overall CTR, all 39 missing-value counts, the
top-10 `C1` category frequencies, and the min-support statistics agree *exactly* between
pandas and the shipped SQL. The decile cross-check agrees on every order-independent
quantity (totals, extremes, weighted mean) and differs by well under a percentage point
per bin purely because `NTILE` breaks ties in its own order - documented above.

**Why this matters beyond correctness.** The `03_category_frequency_ctr` result also
shows the NULL label convention in action: a missing category forms its own group in
SQL. That is exactly the OOV traffic the models must handle gracefully, and notebook 03
revisits it as a segment metric.
""")

# ---------------------------------------------------------------------------
# I. Conclusions
# ---------------------------------------------------------------------------
md(r"""
## I. Conclusions and decisions

**Q:** What did we learn, and what does it decide for the rest of the project?
**Why it matters:** analysis that does not end in decisions is decoration. Each row
below pairs a finding with its evidence, its limitation, and the action it triggers.
Observations (things the data shows) and hypotheses (things to test on validation) are
kept separate.
""")

code(r"""
decision_rows = [
    {"type": "observation",
     "finding": "Schema matches the documented Criteo format; label clean 0/1",
     "evidence": "validate_schema OK; {} columns; no label gaps".format(sample.shape[1]),
     "limitation": "mirror provenance and original split construction unverified",
     "action": "proceed with this pool"},
    {"type": "observation",
     "finding": "Heavy class imbalance",
     "evidence": "train CTR {:.4f}; constant log loss {:.4f}; always-0 accuracy {:.3f}".format(p0, const_logloss, 1 - p0),
     "limitation": "-",
     "action": "log loss primary; no resampling; calibration measured in nb03"},
    {"type": "observation",
     "finding": "Structural missing values concentrated in specific columns",
     "evidence": "{} columns gappy; max {:.1f}%".format(len(miss), miss["pct_missing"].max()),
     "limitation": "anonymous features - the causes are unknowable",
     "action": "train-median imputation (numeric), OOV index 0 (categorical)"},
    {"type": "observation",
     "finding": "Missing-vs-present CTR gaps, flagged by an a-priori rule",
     "evidence": "{} columns flagged of {} gappy".format(len(flagged), len(miss)),
     "limitation": "multiple comparisons; missingness may proxy for traffic mix",
     "action": "features.add_missing_indicators = {} for the {} flagged columns (listed in config.yaml)".format(str(decision_ind).lower(), len(flagged))},
    {"type": "observation",
     "finding": "Heavy right skew; negative ranges in several columns",
     "evidence": "{} of 13 columns |skew| > {}; {} columns contain negatives".format(len(log_cols), thr, int((prof["neg_pct"] > 0).sum())),
     "limitation": "skew measured on train only (by design)",
     "action": "sign-preserving log1p + train-fitted z-scores; no outlier removal"},
    {"type": "observation",
     "finding": "Long-tail categorical vocabularies",
     "evidence": "widest {}: {:,} levels; {}% of train rows fall outside the >=10-support vocab (C1)".format(
         cat_prof.iloc[0]["column"], int(cat_prof.iloc[0]["n_levels"]), round(oov_pct, 2)),
     "limitation": "min_count=10 is a judgment call, revisited via model metrics",
     "action": "vocab keeps train levels with >=10 rows; rest -> shared OOV 0"},
    {"type": "observation",
     "finding": "Modest univariate signal",
     "evidence": "largest decile gap {:.2f} pp ({}); max |numeric-numeric r| = {:.2f}".format(
         100 * gaps.abs().max(), gaps.abs().idxmax(), float(pairs.abs().max())),
     "limitation": "univariate view; effect sizes small",
     "action": "expect limited single-feature power; lean on interactions + strong regularisation"},
    {"type": "observation",
     "finding": "Leakage surfaces quantified and closed",
     "evidence": "in-sample C1 encoding AUC {:.3f} vs out-of-fold {:.3f}; train-vs-val encoding gap shown".format(auc_insample, auc_oof),
     "limitation": "no timestamps - random split is the only honest option",
     "action": "no target encoding; train-fitted statistics; test split sealed until nb03"},
    {"type": "observation",
     "finding": "Partitions are distributionally consistent",
     "evidence": "max KS {:.4f}; base-rate spread {:.2f} pp; max unseen share {:.2f}% of val rows".format(
         ks["ks_stat"].max(), 100 * (max(ctrs) - min(ctrs)), ov["val_rows_unseen_pct"].max()),
     "limitation": "random split != future traffic (no timestamps to do better)",
     "action": "proceed to training (nb02); report unseen-input segments in nb03"},
    {"type": "hypothesis",
     "finding": "Interactions beyond pairwise may add signal (FM / deep components)",
     "evidence": "heatmaps show structure but cell-level noise is real",
     "limitation": "EDA cannot separate signal from chance at this support",
     "action": "test on validation: DeepFM vs LR vs GBDT (nb02)"},
    {"type": "hypothesis",
     "finding": "Missingness itself is partially recoverable information",
     "evidence": "CTR gaps for missing slices, above",
     "limitation": "may be redundant with imputed values for tree models",
     "action": "if indicators enabled, ablate against imputation-only on validation"},
]
dec_table = pd.DataFrame(decision_rows)
dec_table.to_csv(TABLE_DIR / "eda_decision_table.csv", index=False)
print(dec_table[["type", "finding", "action"]].to_string(index=False))

decisions = {
    "generated": datetime.now().isoformat(timespec="seconds"),
    "seed": SEED,
    "sample_rows": int(len(sample)),
    "split_rows": {k: int(len(p)) for k, p in splits.items()},
    "split_ctrs": {k: round(float(p["label"].mean()), 6) for k, p in splits.items()},
    "skew_log1p_columns": log_cols,
    "min_category_count": min_count,
    "add_missing_indicators_recommended": bool(decision_ind),
    "flagged_missing_columns": flagged,
}
(FIG_DIR.parent / "eda_decisions.json").write_text(json.dumps(decisions, indent=2), encoding="utf-8")
print()
print(json.dumps(decisions, indent=2))
""")

md(r"""
**Carried into notebook 02.**
- Split files and manifest on disk (`data/processed/splits/`); test sealed.
- Preprocessing contract: train-fitted medians, sign-preserving log1p on the flagged
  columns, z-scores, vocab with `min_category_count = 10`, OOV index 0; missing-indicator
  decision recorded in `reports/eda_decisions.json` and mirrored into `config.yaml`.
- Baseline to beat: constant log loss (the entropic floor), then logistic regression,
  LightGBM and DeepFM compared on validation with log loss primary.

**Limitations of this EDA, stated once more.** Anonymised features (no semantics, no
causal readings); mirror provenance unverified; no timestamps (random split only, no
temporal validation); duplicate feature vectors across partitions were checked and found
absent rather than assumed away; CTR as the label says nothing about revenue - clicks are
an intermediate outcome, and notebook 03's closing section keeps that boundary explicit.
""")

code(r"""
elapsed = time.time() - T_START
figs = sorted((FIG_DIR).glob("nb01_*.png"))
tbls = sorted(TABLE_DIR.glob("*.csv"))
print("EDA complete in {:.0f}s".format(elapsed))
print("figures saved ({}):".format(len(figs)))
for p in figs:
    print(" -", p.relative_to(PROJECT_ROOT))
print("tables saved ({}):".format(len(tbls)))
for p in tbls:
    print(" -", p.relative_to(PROJECT_ROOT))
print("decisions:", (FIG_DIR.parent / "eda_decisions.json").relative_to(PROJECT_ROOT))
""")

# ---------------------------------------------------------------------------
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {
    "kernelspec": {
        "display_name": "Python (ad-ctr-deepfm)",
        "language": "python",
        "name": "ad-ctr-deepfm",
    },
    "language_info": {"name": "python", "version": "3.11"},
}

out = ROOT / "notebooks" / "01_ctr_in_depth_eda.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print("wrote", out)

if "--no-exec" not in sys.argv:
    result = subprocess.call([sys.executable, str(ROOT / "scripts" / "run_notebook.py"), str(out.relative_to(ROOT))])
    raise SystemExit(result)
