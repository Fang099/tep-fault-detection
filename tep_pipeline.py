"""Tennessee Eastman Process: fault detection and diagnosis pipeline.

Trains several families of models for two tasks, picks the best on validation, optionally runs the one-time
final test, and exports JSON files for the project website.

  1. Detection (binary): normal (0) vs fault active (1). Fault-free and pre-injection rows are normal.
  2. Diagnosis (20 classes): which fault is active, using post-injection rows only.

Data: Kaggle `afrniomelo/tep-csv`. One row = one 3-minute time step of one simulation run. Inputs are the 52
process variables only (41 xmeas_* + 11 xmv_*); identifiers and labels are never inputs.
  - Training: 500 fault-free runs + 20 faults x 500 runs, 500 samples each. Fault injected after sample 20.
  - Testing: 960 samples per run. Fault injected after sample 160.

Protocol
  1. Split the training files by simulation number (80% / 20%): faulty run k of every fault type starts as a copy of
     fault-free run k, so all runs with the same number go to the same side.
  2. Train every variant on the training runs; choose thresholds and the best variant on validation.
  3. Retrain the chosen variants on all training runs, keep the validation thresholds, evaluate once on test.

Model families (--families picks which to train; the majority baseline is always included)
  logreg  Logistic regression: class weighting none/balanced (detection only), penalty none/L1/L2, strength C.
          scikit-learn's default LogisticRegression() is already L2 with C=1; "no penalty" here is unregularized.
  pcalr   PCA + logistic regression: the first k principal components as inputs.
  pcamon  PCA process monitoring (detection only): Hotelling's T^2 and SPE, fitted on fault-free data, no labels.
  tree    Decision tree: maximum depth.
  rf      Random forest (bagged trees): maximum depth x number of trees.
  gb      Gradient boosting (scikit-learn's histogram version): tree depth x boosting rounds.
  ada     AdaBoost with small decision trees: tree depth x boosting rounds.
  xgb     XGBoost: tree depth x boosting rounds (skipped if xgboost is not installed).
Forests and boosted models are trained once per depth at the largest size; smaller forests (the first k trees)
and fewer boosting rounds (the first k rounds) are evaluated from that same fit, so the website can show every
size without retraining.

Inputs: every family uses the 52 instantaneous readings. Families listed in --history-families are also trained
with recent history added: a trailing 30-minute (10-sample) mean and standard deviation of each reading, computed
within each run from past samples only (156 inputs). A window sweep (--hist-windows) then retrains the best
30-minute settings of each of those families with other window lengths.

Exports per model: score histograms (with the exact thresholds as bin edges), detection delays at several
false-alarm rates plus a chance reference from fault-free runs, run-level bootstrap intervals, calibration
(probability models) and score traces for one validation run per fault (run viewer). Results exported by an
older version are re-exported from the saved models without retraining.

Usage
  python tep_pipeline.py --data-dir /path/to/csvs --runs-per-fault 100
  python tep_pipeline.py --data-dir /path/to/csvs --runs-per-fault 100 --families tree,rf
  python tep_pipeline.py --data-dir /path/to/csvs --runs-per-fault 0 --final-test   # all runs + test

Results are written to <out-dir>/outputs_<N>runs (or outputs_full): models_v3/ and web_data/ (website files).
Finished models are skipped on re-runs, so an interrupted run can simply be restarted.
"""
import argparse
import inspect
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 200)

ALL_FAMILIES = ["logreg", "pcalr", "pcamon", "tree", "rf", "gb", "ada", "xgb"]
HISTORY_CAPABLE = ["logreg", "tree", "rf", "gb", "ada", "xgb"]

# ---- Command-line settings -----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="TEP fault detection and diagnosis pipeline")
parser.add_argument("--data-dir", default="dataset",
                    help="folder with the 4 TEP CSVs; the Parquet cache is saved here too")
parser.add_argument("--work-dir", default=None,
                    help="optional fast local folder to copy the Parquet cache into (e.g. /content/dataset on Colab)")
parser.add_argument("--out-dir", default=None, help="where results go (default: --data-dir)")
parser.add_argument("--runs-per-fault", type=int, default=100,
                    help="use the first N runs of each fault type for train/val; 0 = all 500")
parser.add_argument("--families", default=",".join(ALL_FAMILIES),
                    help=f"comma-separated model families to train (default: all of {','.join(ALL_FAMILIES)})")
parser.add_argument("--history-families", default="logreg,rf,xgb",
                    help="families also trained with 30-minute rolling mean/std inputs (empty string = none)")
parser.add_argument("--det-stride", type=int, default=2, help="detection trains on every k-th time step")
parser.add_argument("--diag-stride", type=int, default=5, help="diagnosis trains on every k-th time step")
parser.add_argument("--selection-far", type=float, default=0.05,
                    help="false-alarm rate on validation normal rows used to set thresholds and pick the best model")
parser.add_argument("--test-runs-per-fault", type=int, default=100,
                    help="faulty test runs used per fault (the full faulty test file is ~10M rows)")
parser.add_argument("--hist-windows", default="2,5,20,40",
                    help="extra history windows (in 3-minute samples) for the window sweep; '' to skip it")
parser.add_argument("--final-test", action="store_true", help="retrain the chosen models and evaluate on test")
parser.add_argument("--force-retrain", action="store_true",
                    help="retrain the families listed in --families even if already exported (others are kept as they are)")
args = parser.parse_args()

# ---- Paths ---------------------------------------------------------------------------------------
DRIVE_DIR = Path(args.data_dir)                              # source CSVs + persistent Parquet cache
DATA_DIR = Path(args.work_dir) if args.work_dir else DRIVE_DIR
RUNS_PER_FAULT = args.runs_per_fault or None                 # None = all runs
OUT_DIR = Path(args.out_dir or DRIVE_DIR) / ("outputs_full" if RUNS_PER_FAULT is None
                                              else f"outputs_{RUNS_PER_FAULT}runs")
MODEL_DIR = OUT_DIR / "models_v3"     # v3: split by simulation number (older models used a leaky split); saved compressed (forests are large)
WEB_DIR = OUT_DIR / "web_data"                               # everything the website needs
for d in (DATA_DIR, MODEL_DIR, WEB_DIR / "detection", WEB_DIR / "diagnosis"):
    d.mkdir(parents=True, exist_ok=True)

# ---- Experiment settings -------------------------------------------------------------------------
SEED = 42
FAULT_ONSET = {"train": 20, "test": 160}   # last pre-injection sample in each faulty run
SAMPLE_MINUTES = 3
ID_COLS = ["faultNumber", "simulationRun", "sample"]

# Regularization strengths (website slider values; "none" = no penalty). The penalty per training row scales like
# 1/(C x rows), so with all 500 runs the interesting range moves to smaller C than with 100 runs. At 100 runs, lasso
# kept every sensor from C = 0.01 up and matched no penalty, while C = 0.1 and 1 took 20-25 minutes each; the grid
# therefore stops at 0.01 and is finer where sensors start to drop out.
C_GRID = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 1e-2]
CLASS_WEIGHTS = [None, "balanced"]                     # detection only
SELECTION_FAR = args.selection_far
FAR_LEVELS = [0.01, 0.02, 0.05, 0.10, 0.20]            # detection delays are precomputed at these alarm rates
CONSECUTIVE = 3           # alarms (or correct diagnoses) in a row needed to count as detected

# Neighbouring 3-minute samples are nearly identical, so training on every k-th one cuts time with little loss.
# Validation and test always use every row.
DET_STRIDE = args.det_stride
DIAG_STRIDE = args.diag_stride
TEST_RUNS_PER_FAULT = args.test_runs_per_fault
FORCE_RETRAIN = args.force_retrain

# ---- Tree-based model grids ------------------------------------------------------------------------
# Each entry is a website slider. Forests/boosting are fitted once per depth at the largest size; the smaller
# sizes are read off that same fit.
TREE_DEPTHS = [2, 4, 6, 8, 10, 12, 15, 20]
RF_DEPTHS = [4, 8, 12, 16, 20]
RF_TREES = [1, 5, 10, 25, 50, 100]
RF_SAMPLE = 0.25          # each tree is grown on a random 25% of the training rows (keeps forests fast)
GB_DEPTHS = [2, 4, 6, 8]
GB_ROUNDS = [10, 25, 50, 100, 200]
ADA_DEPTHS = [1, 2, 3]
ADA_ROUNDS = [10, 25, 50, 100]
ADA_SAMPLE = 0.25         # AdaBoost rounds can't run in parallel, so it trains on a random 25% of the rows
XGB_DEPTHS = [2, 4, 6, 8]
XGB_ROUNDS = [10, 25, 50, 100, 200]
LEARNING_RATE = 0.1       # gradient boosting and XGBoost
# L2 penalty on leaf values for gradient boosting. scikit-learn's default is 0, which let multiclass diagnosis
# diverge at depth 6-8 (a leaf whose rows already had near-zero probability for a fault got a near-zero curvature
# term, so its update for that fault exploded). XGBoost's default is 1, so this matches it.
GB_L2 = 1.0

# ---- PCA, history, uncertainty ---------------------------------------------------------------------
PCA_COMPONENTS = [2, 5, 10, 15, 20, 30, 40]       # PCA + logistic regression
PCAMON_COMPONENTS = [5, 10, 15, 20, 25, 30]       # PCA monitoring
PCAMON_ALPHA = 0.01                               # classical control limits are computed at 99% for comparison
HIST_WINDOW = 10                                  # 10 samples = 30 minutes of history (full grid)
# Window sweep: the best 30-minute settings of each history family are retrained with these other windows
# (2, 5, 20, 40 samples = 6 min, 15 min, 1 h, 2 h), so the website can show how much history helps.
HIST_LR_PENALTIES = ["none", "l2"]                # lasso is slow on 156 inputs, so history LR uses none/ridge
HIST_XGB_DEPTHS = [4, 8]                          # XGBoost with history: fewer depths to keep run time down
BOOTSTRAP = 200                                   # run-level bootstrap resamples for 95% intervals
EXPORT_VERSION = 3                                # bump when the exported files gain new content

SWEEP_WINDOWS = [int(w) for w in args.hist_windows.split(",") if w.strip()] if args.hist_windows else []
SWEEP_WINDOWS = [w for w in SWEEP_WINDOWS if w != HIST_WINDOW]

FAMILIES = [f.strip() for f in args.families.split(",") if f.strip()]
HISTORY = [f.strip() for f in args.history_families.split(",") if f.strip()]
assert not set(HISTORY) - set(HISTORY_CAPABLE), f"--history-families must be among {HISTORY_CAPABLE}"
unknown = set(FAMILIES) - set(ALL_FAMILIES)
assert not unknown, f"unknown families: {unknown}; choose from {ALL_FAMILIES}"
if "xgb" in FAMILIES and XGBClassifier is None:
    print("xgboost is not installed: skipping XGBoost (pip install xgboost to include it)")
    FAMILIES.remove("xgb")

print(f"Data: {DRIVE_DIR}  Results: {OUT_DIR}  Runs per fault: {RUNS_PER_FAULT or 'all'}  Families: {','.join(FAMILIES)}")

# ====================================================================================================
# ## 1. Load data
# The first run parses each CSV once and saves a float32 Parquet copy to Drive. Later sessions copy the Parquet
# file to local disk and load it in seconds.

def load_tep(name: str) -> pd.DataFrame:
    local_pq, drive_pq = DATA_DIR / f"{name}.parquet", DRIVE_DIR / f"{name}.parquet"
    if not local_pq.exists() and drive_pq.exists() and DATA_DIR != DRIVE_DIR:
        shutil.copy(drive_pq, local_pq)
    if local_pq.exists():
        return pd.read_parquet(local_pq)

    # Parse straight into float32/int32: the faulty test file has ~10M rows, too big for memory as float64.
    header = pd.read_csv(DRIVE_DIR / f"{name}.csv", nrows=0).columns
    dtypes = {c: "float32" for c in header if c.startswith(("xmeas_", "xmv_"))}
    df = pd.read_csv(DRIVE_DIR / f"{name}.csv", dtype=dtypes)     # ids may be written as 1.0, so convert after
    float_cols = df.select_dtypes("float64").columns.difference(ID_COLS)
    df[float_cols] = df[float_cols].astype("float32")
    df[ID_COLS] = df[ID_COLS].astype("int32")
    try:
        df.to_parquet(local_pq, index=False)
        if DATA_DIR != DRIVE_DIR:
            shutil.copy(local_pq, drive_pq)
    except ImportError:
        print("pyarrow not installed; skipping parquet cache")
    return df


def prepare(df_normal: pd.DataFrame, df_faulty: pd.DataFrame, onset: int) -> pd.DataFrame:
    """Stack normal + faulty runs, add a unique run id and the binary detection target."""
    df = pd.concat([df_normal, df_faulty], ignore_index=True)
    # Run numbers restart for every fault, so (faultNumber, simulationRun) identifies a run.
    df["run_id"] = df["faultNumber"] * 1000 + df["simulationRun"]
    # Faulty runs are normal until the fault is injected after `onset`.
    df["target"] = ((df["faultNumber"] > 0) & (df["sample"] > onset)).astype("int8")
    return df.sort_values(["run_id", "sample"], ignore_index=True)


train_all = prepare(load_tep("TEP_FaultFree_Training"), load_tep("TEP_Faulty_Training"), FAULT_ONSET["train"])
if RUNS_PER_FAULT is not None:
    train_all = train_all[train_all["simulationRun"] <= RUNS_PER_FAULT].reset_index(drop=True)
FEATURES = [c for c in train_all.columns if c.startswith(("xmeas_", "xmv_"))]
assert len(FEATURES) == 52

print("Shape:", train_all.shape, " Runs:", train_all["run_id"].nunique())

# Data-quality facts for the website, measured on the data actually used.
QUALITY = {
    "rows": int(len(train_all)), "runs": int(train_all["run_id"].nunique()),
    "missing_values": int(train_all[FEATURES].isna().sum().sum()),
    "duplicate_rows": int(train_all.duplicated(subset=FEATURES).sum()),
    "constant_columns": [c for c in FEATURES if train_all[c].nunique() <= 1],
    "class_counts": {"normal": int((train_all["target"] == 0).sum()), "faulty": int((train_all["target"] == 1).sum())},
}
print(train_all["target"].value_counts().rename("rows"))

# ====================================================================================================
# ## 2. Train / validation split by simulation number
# In this dataset, faulty run k of every fault type is the fault-free run k with the fault switched on: up to the
# fault onset (and for a while after it, until the fault shows) the 52 readings are identical, row for row.
# Splitting by (faultNumber, simulationRun) would therefore put exact copies of validation rows into training.
# Instead, 20% of the simulation numbers go to validation for every fault type at once, so all copies of a
# simulation stay on the same side. Each fault type still keeps 80% of its runs for training.

sim_numbers = np.sort(train_all["simulationRun"].unique())
_, val_sims = train_test_split(sim_numbers, test_size=0.2, random_state=SEED)
is_val = train_all["simulationRun"].isin(val_sims)
train_df = train_all[~is_val].reset_index(drop=True)
val_df = train_all[is_val].reset_index(drop=True)
del train_all
assert set(train_df["run_id"]).isdisjoint(val_df["run_id"])
assert set(train_df["simulationRun"]).isdisjoint(val_df["simulationRun"])


def row_hashes(df):
    return pd.util.hash_pandas_object(df[FEATURES], index=False).to_numpy()


# Leakage check: how many validation rows also appear, reading for reading, in training? Should be 0.
QUALITY["val_rows_copied_from_train"] = int(np.isin(row_hashes(val_df), row_hashes(train_df)).sum())
print("Validation rows identical to a training row:", QUALITY["val_rows_copied_from_train"])

y_val = val_df["target"].to_numpy()
faults_val = val_df["faultNumber"].to_numpy()
val_post = y_val == 1                        # diagnosis is evaluated on post-injection rows only
FAULTS = np.arange(1, 21)

# Training rows: detection uses every DET_STRIDE-th time step, diagnosis every DIAG_STRIDE-th post-fault step.
det_rows = (train_df["sample"] % DET_STRIDE == 0).to_numpy()
diag_rows = ((train_df["target"] == 1) & (train_df["sample"] % DIAG_STRIDE == 0)).to_numpy()
keep_rows = det_rows | diag_rows             # only these rows are kept in memory
y_det = train_df.loc[det_rows, "target"].to_numpy()
y_diag = train_df.loc[diag_rows, "faultNumber"].to_numpy()

# ====================================================================================================
# ## 2b. Inputs: instantaneous readings, optionally with recent history
# History columns are a trailing mean and standard deviation over the last HIST_WINDOW samples of the same run
# (the current sample included, nothing after it), so a prediction only uses information available at that time.
# Scalers are fitted on training runs only.

# Input sets: "inst" (the 52 readings), "hist" (plus a 30-minute history, the full grid) and "hist{w}" for the
# window sweep (e.g. "hist40" = 2 hours).

def window_of(fs):
    return None if fs == "inst" else HIST_WINDOW if fs == "hist" else int(fs[4:])


def fs_for_window(w):
    return "hist" if w == HIST_WINDOW else f"hist{w}"


def input_names(fs):
    w = window_of(fs)
    return FEATURES if w is None else FEATURES + [f"{f}_mean{w}" for f in FEATURES] + [f"{f}_std{w}" for f in FEATURES]


INPUT_NAMES = {fs: input_names(fs) for fs in ["inst", "hist"] + [fs_for_window(w) for w in SWEEP_WINDOWS]}


def window_text(w):
    m = w * SAMPLE_MINUTES
    return f"{m}-min" if m < 60 else f"{m // 60}-hour" if m % 60 == 0 else f"{m / 60:g}-hour"


def history_block(df, window, chunk_rows=200_000):
    """Trailing rolling mean and std (ddof=1) of every reading over the last `window` samples of the same run,
    the current sample included. Matches pandas rolling(window, min_periods=1).mean() and
    rolling(window, min_periods=2).std().fillna(0). Computed with cumulative sums, whole runs at a time, in chunks,
    so the full dataset fits in memory. df must be sorted by run and sample (it is)."""
    x = df[FEATURES].to_numpy(np.float32)
    run = df["run_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, run[1:] != run[:-1]])          # first row of each run
    bounds = np.r_[starts, len(run)]
    mean, std = np.empty_like(x), np.empty_like(x)
    a = 0
    while a < len(run):
        b = bounds[np.searchsorted(bounds, a + chunk_rows, side="left")] if a + chunk_rows < len(run) else len(run)
        b = max(b, bounds[np.searchsorted(bounds, a, side="right")])   # at least one whole run
        xc = x[a:b].astype(np.float64)
        xc -= xc.mean(axis=0)                                           # centring keeps the sums small and exact
        c1 = np.vstack([np.zeros((1, xc.shape[1])), np.cumsum(xc, axis=0)])
        c2 = np.vstack([np.zeros((1, xc.shape[1])), np.cumsum(xc * xc, axis=0)])
        idx = np.arange(b - a)
        run_start = starts[np.searchsorted(starts, np.arange(a, b), side="right") - 1] - a
        lo = np.maximum(idx - window + 1, run_start)
        n = (idx - lo + 1)[:, None].astype(np.float64)
        s1, s2 = c1[idx + 1] - c1[lo], c2[idx + 1] - c2[lo]
        m = s1 / n
        var = np.where(n > 1, np.maximum(s2 - n * m * m, 0.0) / np.maximum(n - 1, 1), 0.0)
        mean[a:b] = (m + x[a:b].astype(np.float64).mean(axis=0)).astype(np.float32)
        std[a:b] = np.sqrt(var).astype(np.float32)
        a = b
    return mean, std


def raw_inputs(df, fs):
    x = df[FEATURES].to_numpy(np.float32)
    return x if fs == "inst" else np.hstack([x, *history_block(df, window_of(fs))])


def build_inputs(fs):
    """Standardized training (kept rows only) and validation inputs for one input set; scaler fitted on training."""
    t0 = time.time()
    raw = raw_inputs(train_df, fs)
    scalers[fs] = StandardScaler().fit(raw)
    X_train[fs] = scalers[fs].transform(raw[keep_rows]).astype(np.float32)
    del raw
    X_val[fs] = scalers[fs].transform(raw_inputs(val_df, fs)).astype(np.float32)
    print(f"Inputs '{fs}': {X_train[fs].shape[1]} columns ({time.time() - t0:.0f}s)", flush=True)


FEATURE_SETS = ["inst"] + (["hist"] if HISTORY else [])
scalers, X_train, X_val = {}, {}, {}
for fs in FEATURE_SETS:
    build_inputs(fs)
det_in_keep, diag_in_keep = det_rows[keep_rows], diag_rows[keep_rows]
normal_in_keep = (train_df.loc[keep_rows, "faultNumber"] == 0).to_numpy()   # fault-free training runs (PCA monitoring)


def task_data(task, fs):
    """(training inputs, training labels, validation inputs) for one task and input set."""
    if task == "detection":
        return X_train[fs][det_in_keep], y_det, X_val[fs]
    return X_train[fs][diag_in_keep], y_diag, X_val[fs][val_post]


print(f"Train: {train_df.shape}  Val: {val_df.shape}")
print(f"Detection training rows: {int(det_rows.sum())}   Diagnosis training rows: {int(diag_rows.sum())}")

# Validation runs, for the bootstrap and the run viewer.
run_codes, run_uniques = pd.factorize(val_df["run_id"].to_numpy())
run_fault = val_df.groupby("run_id", sort=False)["faultNumber"].first().reindex(run_uniques).to_numpy()
TRACE_RUNS = [int(val_df.loc[val_df["faultNumber"] == f, "run_id"].min()) for f in range(0, 21)
              if (val_df["faultNumber"] == f).any()]

# ====================================================================================================
# ## 3. Models
# ### Logistic regression
# `make_lr` builds one logistic-regression variant. It handles both the new scikit-learn API (1.8+, where
# `l1_ratio` selects the penalty) and the old one (`penalty="l1"`). Using the wrong one silently gives the
# wrong penalty, so the version is detected automatically.

NEW_LR_API = inspect.signature(LogisticRegression).parameters["penalty"].default == "deprecated"


def make_lr(penalty: str, C: float | None, class_weight=None, multiclass=False) -> LogisticRegression:
    kw = dict(class_weight=class_weight, max_iter=1000, random_state=SEED)
    if penalty == "l1":
        # liblinear is fast for binary L1; multinomial L1 needs saga.
        kw.update(solver="saga", tol=1e-3, max_iter=300) if multiclass else kw.update(solver="liblinear")
    if penalty == "none":
        return LogisticRegression(**kw, **({"C": np.inf} if NEW_LR_API else {"penalty": None}))
    if NEW_LR_API:
        kw["l1_ratio"] = 1.0 if penalty == "l1" else 0.0
    else:
        kw["penalty"] = penalty
    return LogisticRegression(C=C, **kw)


# Sanity check: a strongly regularized lasso must set some coefficients to exactly zero.
if "logreg" in FAMILIES:
    _X, _y, _ = task_data("detection", "inst")
    _idx = np.random.default_rng(SEED).choice(len(_X), size=min(20000, len(_X)), replace=False)
    assert (make_lr("l1", 1e-4).fit(_X[_idx], _y[_idx]).coef_ == 0).any(), \
        "L1 penalty was not applied; check the scikit-learn version"
    print("scikit-learn API:", "new (l1_ratio)" if NEW_LR_API else "old (penalty=)", "- lasso check passed")

PENALTY_NAMES = {"none": "No penalty", "l1": "Lasso (L1)", "l2": "Ridge (L2)", "baseline": "Majority class"}
HIST_TAG = {fs: "" if fs == "inst" else "_" + fs for fs in INPUT_NAMES}                    # e.g. "_hist", "_hist40"
HIST_LABEL = {fs: "" if fs == "inst" else f", {window_text(window_of(fs))} history" for fs in INPUT_NAMES}


def model_id(task, penalty, C=None, class_weight=None, fs="inst"):
    parts = [task, penalty]
    if task == "detection" and penalty != "baseline":
        parts.append("balanced" if class_weight == "balanced" else "unweighted")
    if C is not None:
        parts.append(f"C{C:g}")
    return "_".join(parts) + HIST_TAG[fs]


def model_label(penalty, C=None, class_weight=None, fs="inst"):
    label = PENALTY_NAMES[penalty]
    if class_weight == "balanced":
        label += ", balanced"
    if C is not None:
        label += f", C={C:g}"
    return label + HIST_LABEL[fs]

# ====================================================================================================
# ## 4. Evaluation helpers
# **Detection** metrics (fault = positive class): precision, recall, F1, PR-AUC, confusion matrix, plus normal
# recall (1 - false alarm rate) and ROC-AUC, since faults are ~92% of rows and the majority baseline already
# scores high on fault-class metrics.
#
# **Diagnosis** metrics: macro-F1, per-fault recall, confusion matrix.
#
# **Speed:** for each faulty run, minutes from fault injection until `CONSECUTIVE` alarms in a row (detection),
# or until `CONSECUTIVE` correct diagnoses in a row (diagnosis). Requiring a short streak stops one lucky or
# random prediction from counting. The same rule applied to fault-free runs, as if a fault had started at the
# usual time, gives a chance reference: how often normal operation alone triggers a "detection".
#
# **Uncertainty:** a run-level bootstrap. Validation runs are resampled with replacement within each fault type
# (whole runs, because rows inside a run are strongly correlated) and the metrics recomputed 200 times; the
# 2.5th and 97.5th percentiles give a 95% interval. The threshold stays fixed at its validation value.

def threshold_at_far(y, scores, far):
    """Lowest score threshold that flags at most `far` of the normal rows as faulty (flag = score >= threshold).

    If many rows share the same score (e.g. a lasso model with every coefficient at zero), the plain quantile
    could flag far more than `far`, so the threshold is nudged just above the tied value."""
    normal = scores[y == 0]
    t = np.quantile(normal, 1 - far)
    if (normal >= t).mean() > far + 1e-9:
        t = np.nextafter(t, np.inf)
    return float(t)


def detection_metrics(y, scores, threshold):
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "PR_AUC": average_precision_score(y, scores),
        "ROC_AUC": roc_auc_score(y, scores) if np.ptp(scores) > 0 else 0.5,
        "normal_recall": tn / (tn + fp),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def diagnosis_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=FAULTS)
    recall = recall_score(y_true, y_pred, labels=FAULTS, average=None, zero_division=0)
    return {
        "macro_F1": f1_score(y_true, y_pred, labels=FAULTS, average="macro", zero_division=0),
        "accuracy": float((y_true == y_pred).mean()),
    }, cm, recall


def streak_delays(run_ids, faults, samples, flags, onset, k=CONSECUTIVE):
    """Per fault type: median minutes until k consecutive flags after `onset`, and share of runs where it happens.
    Fault-free runs (fault 0) are included, measured as if a fault had started at `onset`. Rows must be sorted
    by run and sample (they are)."""
    flags = flags & (samples > onset)
    cs = np.concatenate([[0], np.cumsum(flags)])
    idx = np.arange(len(flags))
    start = np.maximum(idx - k + 1, 0)
    streak = (idx - k + 1 >= 0) & (run_ids[start] == run_ids) & (cs[idx + 1] - cs[start] == k)
    per_run = pd.DataFrame({
        "run": run_ids, "fault": faults,
        "minutes": np.where(streak, (samples - onset) * SAMPLE_MINUTES, np.nan),
    }).groupby("run").agg(fault=("fault", "first"), minutes=("minutes", "min"))
    out = per_run.groupby("fault")["minutes"].agg(median_minutes="median", detected="count", runs="size")
    out["detected_share"] = out["detected"] / out["runs"]
    return out[["median_minutes", "detected_share"]]


def delays_to_json(table):
    """Split a streak_delays table into per-fault entries and the fault-free chance reference."""
    row = lambda r: [None if pd.isna(r.median_minutes) else float(r.median_minutes), round(float(r.detected_share), 4)]
    by_fault = {int(f): row(r) for f, r in table.iterrows() if f > 0}
    reference = row(table.loc[0]) if 0 in table.index else None
    return by_fault, reference


def bootstrap_detection(scores, threshold):
    """95% intervals for recall, false alarm rate, precision and F1 at a fixed threshold (run-level bootstrap)."""
    alarm, pos = scores >= threshold, y_val == 1
    R = len(run_uniques)
    counts = np.stack([np.bincount(run_codes, weights=(alarm & pos).astype(float), minlength=R),
                       np.bincount(run_codes, weights=(~alarm & pos).astype(float), minlength=R),
                       np.bincount(run_codes, weights=(alarm & ~pos).astype(float), minlength=R),
                       np.bincount(run_codes, weights=(~alarm & ~pos).astype(float), minlength=R)], axis=1)
    rng = np.random.default_rng(SEED)            # same resamples for every model, so comparisons are paired
    totals = np.zeros((BOOTSTRAP, 4))
    for f in np.unique(run_fault):
        idx = np.flatnonzero(run_fault == f)
        totals += counts[idx][rng.integers(0, len(idx), (BOOTSTRAP, len(idx)))].sum(axis=1)
    tp, fn, fp, tn = totals.T
    with np.errstate(divide="ignore", invalid="ignore"):
        recall, far = tp / (tp + fn), fp / (fp + tn)
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    ci = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]
    return {"recall": ci(recall), "far": ci(far), "precision": ci(precision), "F1": ci(f1), "resamples": BOOTSTRAP}


def bootstrap_diagnosis(pred):
    """95% intervals for macro-F1 and accuracy (run-level bootstrap within each fault type)."""
    codes = run_codes[val_post]
    per_run = np.zeros((len(run_uniques), len(FAULTS)))
    np.add.at(per_run, (codes, pred.astype(int) - 1), 1)
    rng = np.random.default_rng(SEED)
    cms = np.zeros((BOOTSTRAP, len(FAULTS), len(FAULTS)))
    for f in FAULTS:
        idx = np.flatnonzero(run_fault == f)
        cms[:, f - 1, :] = per_run[idx][rng.integers(0, len(idx), (BOOTSTRAP, len(idx)))].sum(axis=1)
    tp = np.diagonal(cms, axis1=1, axis2=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(cms.sum(1) > 0, tp / cms.sum(1), 0.0)
        rec = np.where(cms.sum(2) > 0, tp / cms.sum(2), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    macro, acc = f1.mean(axis=1), tp.sum(1) / cms.sum((1, 2))
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"macro_F1": ci(macro), "accuracy": ci(acc), "resamples": BOOTSTRAP}


def calibration(scores):
    """Reliability data for probability scores: 10 equal-width bins of predicted probability."""
    b = np.minimum((scores * 10).astype(int), 9)
    bins = []
    for k in range(10):
        m = b == k
        if m.any():
            bins.append([float(scores[m].mean()), float(y_val[m].mean()), int(m.sum())])
    ece = sum(abs(mp - fr) * n for mp, fr, n in bins) / len(scores)
    return {"bins": bins, "brier": float(np.mean((scores - y_val) ** 2)), "ece": float(ece)}


def clean(obj):
    """Make numpy values JSON-serializable and round floats."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if not np.isfinite(obj) else round(float(obj), 6)
    return obj

# ====================================================================================================
# ## 5. Website exports
# GitHub Pages is static, so the site can't run models. Each model gets one JSON file.
#
# **Detection files** hold histograms of the model's scores for normal rows and for each fault, with
# quantile-spaced bin edges plus the exact thresholds for each false-alarm level. Row counts in bin `i` are
# scores in `[edge[i], edge[i+1])`, so for a slider threshold `t = edge[i]`, rows flagged as faulty = sum of bins
# `i..end`. From that the browser computes the confusion matrix and every metric for any threshold.
# Detection delays are precomputed at several false-alarm rates because they can't be rebuilt from histograms.
# A separate small file per model holds score traces for one validation run per fault (the run viewer), stored
# as each score's percentile among normal validation rows so every model type shares one scale.
#
# **Diagnosis files** hold the confusion matrix, per-fault recall and diagnosis delays.
#
# Both say which inputs the model relies on: logistic regression stores its coefficients (on standardized
# inputs); tree models store feature importances (how much each input's splits reduce error).

def score_histograms(y, scores, faults, extra_edges=(), n_bins=200):
    edges = np.unique(np.concatenate([np.quantile(scores, np.linspace(0, 1, n_bins + 1)), list(extra_edges)]))
    edges = edges[(edges >= scores.min()) & (edges <= scores.max())] if len(extra_edges) else edges
    if len(edges) < 2:                                  # constant scores (majority baseline)
        edges = np.array([scores.min(), scores.min() + 1e-6])
    return {
        "bin_edges": edges,
        "normal_counts": np.histogram(scores[y == 0], edges)[0],
        "fault_counts": {int(f): np.histogram(scores[(y == 1) & (faults == f)], edges)[0]
                         for f in np.unique(faults[y == 1])},
    }


def coefficients(model, names=FEATURES):
    if not hasattr(model, "coef_"):
        return None
    coef = model.coef_
    return {"features": names, "values": coef[0] if coef.shape[0] == 1 else coef,
            "nonzero": int((coef != 0).any(axis=0).sum())}


def importances(values, kind, names=FEATURES):
    """Feature importances normalized to sum to 1 (None if the model doesn't provide them)."""
    if values is None:
        return None
    values = np.asarray(values, dtype=float)
    total = values.sum()
    values = values / total if total > 0 else values
    return {"features": names, "values": values, "nonzero": int((values > 0).sum()), "kind": kind}


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(payload), separators=(",", ":")))


def write_traces(mid, scores):
    """Score of every sample in one validation run per fault, as a percentile (0-1000) among normal rows."""
    normal = np.sort(scores[y_val == 0])
    run_ids = val_df["run_id"].to_numpy()
    out = []
    for r in TRACE_RUNS:
        m = run_ids == r
        rank = np.searchsorted(normal, scores[m], side="left") / len(normal)
        out.append({"run": r, "fault": int(r // 1000), "rank": np.round(rank * 1000).astype(int)})
    write_json(WEB_DIR / "detection" / "traces" / f"{mid}.json",
               {"onset": FAULT_ONSET["train"], "sample_minutes": SAMPLE_MINUTES, "runs": out})

# ====================================================================================================
# ### Model families
# Every family produces "checkpoints": dicts with the model's settings, its validation output, which inputs it
# relies on, and how long it took. Detection output is the fault score of every validation row; diagnosis output
# is the predicted fault of every post-injection validation row. Forests and boosted models fit once at the
# largest size and read off the smaller sizes.

FAMILY_INFO = {
    "baseline": {"label": "Majority baseline", "params": []},
    "logreg": {"label": "Logistic regression", "params": []},   # the website has dedicated controls for these
    "pcalr": {"label": "PCA + logistic regression",
              "params": [{"key": "k", "label": "Principal components kept", "values": PCA_COMPONENTS}]},
    "pcamon": {"label": "PCA monitoring (T² / SPE)", "tasks": ["detection"],
               "params": [{"key": "stat", "label": "Statistic", "values": ["T2", "SPE"], "type": "choice"},
                          {"key": "k", "label": "Principal components kept", "values": PCAMON_COMPONENTS}]},
    "tree": {"label": "Decision tree", "params": [{"key": "depth", "label": "Maximum depth", "values": TREE_DEPTHS}]},
    "rf": {"label": "Random forest", "params": [{"key": "depth", "label": "Maximum depth", "values": RF_DEPTHS},
                                                {"key": "trees", "label": "Number of trees", "values": RF_TREES}]},
    "gb": {"label": "Gradient boosting", "params": [{"key": "depth", "label": "Tree depth", "values": GB_DEPTHS},
                                                    {"key": "rounds", "label": "Boosting rounds", "values": GB_ROUNDS}]},
    "ada": {"label": "AdaBoost", "params": [{"key": "depth", "label": "Depth of each small tree", "values": ADA_DEPTHS},
                                            {"key": "rounds", "label": "Boosting rounds", "values": ADA_ROUNDS}]},
    "xgb": {"label": "XGBoost", "params": [{"key": "depth", "label": "Tree depth", "values": XGB_DEPTHS},
                                           {"key": "rounds", "label": "Boosting rounds", "values": XGB_ROUNDS}]},
}
PARAM_TAGS = {"depth": "d", "trees": "t", "rounds": "r", "k": "k"}


def unit_id(task, family, params, fs):
    if family == "pcamon":
        return f"{task}_pcamon_{params['stat'].lower()}_k{params['k']}"
    return f"{task}_{family}_" + "_".join(f"{PARAM_TAGS[k]}{v}" for k, v in params.items()) + HIST_TAG[fs]


def unit_label(family, params, fs):
    p = params
    return {
        "pcalr": lambda: f"PCA + logistic regression, {p['k']} components",
        "pcamon": lambda: f"PCA monitoring, {'T²' if p['stat'] == 'T2' else 'SPE'}, {p['k']} components",
        "tree": lambda: f"Decision tree, depth {p['depth']}",
        "rf": lambda: f"Random forest, depth {p['depth']}, {p['trees']} tree{'s' if p['trees'] != 1 else ''}",
        "gb": lambda: f"Gradient boosting, depth {p['depth']}, {p['rounds']} rounds",
        "ada": lambda: f"AdaBoost, depth-{p['depth']} trees, {p['rounds']} rounds",
        "xgb": lambda: f"XGBoost, depth {p['depth']}, {p['rounds']} rounds",
    }[family]() + HIST_LABEL[fs]


def family_units(family, fs):
    """The separate fits a family needs (one per depth, one per component count, or one for everything)."""
    return {"pcalr": PCA_COMPONENTS, "pcamon": [None], "tree": TREE_DEPTHS, "rf": RF_DEPTHS, "gb": GB_DEPTHS,
            "ada": ADA_DEPTHS, "xgb": HIST_XGB_DEPTHS if fs == "hist" else XGB_DEPTHS}[family]


def checkpoint_params(family, unit):
    if family == "pcalr":
        return [{"k": unit}]
    if family == "pcamon":
        return [{"stat": s, "k": k} for s in ("T2", "SPE") for k in PCAMON_COMPONENTS]
    if family == "tree":
        return [{"depth": unit}]
    key, values = {"rf": ("trees", RF_TREES), "gb": ("rounds", GB_ROUNDS),
                   "ada": ("rounds", ADA_ROUNDS), "xgb": ("rounds", XGB_ROUNDS)}[family]
    return [{"depth": unit, key: v} for v in values]


def subsample(X, y, fraction, seed=SEED):
    idx = np.sort(np.random.default_rng(seed).choice(len(X), size=int(len(X) * fraction), replace=False))
    return X[idx], y[idx]


def task_output(task, proba, classes):
    """Detection: probability of a fault. Diagnosis: predicted fault number."""
    return proba[:, 1].astype("float64") if task == "detection" else classes[np.argmax(proba, axis=1)]


def build_tree_model(task, family, params):
    """A model with these exact settings (also used to retrain the chosen model for the final test)."""
    d = params.get("depth")
    if family == "tree":
        return DecisionTreeClassifier(max_depth=d, random_state=SEED)
    if family == "rf":
        return RandomForestClassifier(n_estimators=params.get("trees", max(RF_TREES)), max_depth=d,
                                      max_samples=RF_SAMPLE, n_jobs=-1, random_state=SEED)
    if family == "gb":
        return HistGradientBoostingClassifier(max_iter=params.get("rounds", max(GB_ROUNDS)), max_depth=d,
                                              learning_rate=LEARNING_RATE, l2_regularization=GB_L2,
                                              early_stopping=False, random_state=SEED)
    if family == "ada":
        return AdaBoostClassifier(DecisionTreeClassifier(max_depth=d), n_estimators=params.get("rounds", max(ADA_ROUNDS)),
                                  random_state=SEED)
    if family == "xgb":
        return XGBClassifier(n_estimators=params.get("rounds", max(XGB_ROUNDS)), max_depth=d, learning_rate=LEARNING_RATE,
                             tree_method="hist", n_jobs=-1, random_state=SEED,
                             eval_metric="logloss" if task == "detection" else "mlogloss")
    raise ValueError(family)


def fit_tree_model(task, family, depth, X, y):
    if family == "ada":
        X, y = subsample(X, y, ADA_SAMPLE)
    model = build_tree_model(task, family, {"depth": depth})
    if family == "xgb":
        # XGBoost needs class labels 0..K-1, so faults 1..20 are shifted down and back up afterwards.
        model.fit(X, np.searchsorted(np.unique(y), y))
        model.tep_classes_ = np.unique(y)
    else:
        model.fit(X, y)
    return model


def tree_checkpoints(task, family, depth, model, Xv, secs, names):
    """Validation output at every checkpoint size, read off one fitted model."""
    cp = lambda params, out, explain, s: {"params": params, "output": out, "explain": explain, "secs": s}
    if family == "tree":
        return [cp({"depth": depth}, task_output(task, model.predict_proba(Xv), model.classes_),
                   importances(model.feature_importances_, "impurity", names), secs)]
    if family == "rf":
        out, total, imp = [], None, np.zeros(len(names))
        for k, est in enumerate(model.estimators_, start=1):
            p = est.predict_proba(Xv)
            total = p if total is None else total + p
            imp += est.feature_importances_
            if k in RF_TREES:
                out.append(cp({"depth": depth, "trees": k}, task_output(task, total / k, model.classes_),
                              importances(imp / k, "impurity", names), secs * k / len(model.estimators_)))
        return out
    if family == "gb":
        # scikit-learn's histogram boosting has no built-in feature importances.
        return [cp({"depth": depth, "rounds": k}, task_output(task, p, model.classes_), None, secs * k / max(GB_ROUNDS))
                for k, p in enumerate(model.staged_predict_proba(Xv), start=1) if k in GB_ROUNDS]
    if family == "ada":
        n_fit = len(model.estimators_)       # AdaBoost stops early if a round fits perfectly or is no better than chance
        staged = list(model.staged_predict_proba(Xv))
        imp, wsum, cum = np.zeros(len(names)), 0.0, []
        for k in range(n_fit):
            imp = imp + model.estimator_weights_[k] * model.estimators_[k].feature_importances_
            wsum += model.estimator_weights_[k]
            cum.append(imp / wsum if wsum > 0 else imp)
        return [cp({"depth": depth, "rounds": r}, task_output(task, staged[min(r, n_fit) - 1], model.classes_),
                   importances(cum[min(r, n_fit) - 1], "weighted impurity", names), secs * min(r, n_fit) / max(ADA_ROUNDS))
                for r in ADA_ROUNDS]
    if family == "xgb":
        gain = model.get_booster().get_score(importance_type="gain")
        imp = importances([gain.get(f"f{j}", 0.0) for j in range(len(names))], "gain", names)
        classes = getattr(model, "tep_classes_", FAULTS if task == "diagnosis" else np.array([0, 1]))
        return [cp({"depth": depth, "rounds": r}, task_output(task, model.predict_proba(Xv, iteration_range=(0, r)),
                                                               classes), imp, secs * r / max(XGB_ROUNDS))
                for r in XGB_ROUNDS]
    raise ValueError(family)


def fit_pcalr(task, k, X, y):
    pca = PCA(n_components=k, random_state=SEED).fit(X)
    lr = make_lr("none", None, "balanced" if task == "detection" else None, multiclass=(task == "diagnosis"))
    lr.fit(pca.transform(X), y)
    return {"pca": pca, "lr": lr}


def pcalr_checkpoint(task, k, model, Xv, secs):
    pca, lr = model["pca"], model["lr"]
    proba = lr.predict_proba(pca.transform(Xv))
    # The model is linear in the original (standardized) sensors: z = w . (P x) = (P^T w) . x
    coef = lr.coef_ @ pca.components_
    explain = {"features": FEATURES, "values": coef[0] if coef.shape[0] == 1 else coef, "nonzero": len(FEATURES)}
    extra = {"explained_variance": float(pca.explained_variance_ratio_.sum())}
    return [{"params": {"k": k}, "output": task_output(task, proba, lr.classes_), "explain": explain, "secs": secs,
             "extra": extra}]


def pca_monitor_fit(X_normal):
    """PCA of normal operation. The readings are re-standardized with the fault-free rows' own mean and standard
    deviation (the shared scaler is fitted on all training rows, most of them faulty), as in classical monitoring,
    so the model never uses anything from faulty data."""
    scaler = StandardScaler().fit(X_normal)
    pca = PCA(random_state=SEED).fit(scaler.transform(X_normal))
    pca.tep_scaler_ = scaler
    return pca


def pca_monitor_scores(pca, X, k):
    """Hotelling's T^2 (distance inside the k-dimensional model plane, scaled by each component's variance) and
    SPE (squared distance from the plane, i.e. what the model can't explain)."""
    P, lam = pca.components_[:k], pca.explained_variance_[:k]
    if hasattr(pca, "tep_scaler_"):
        X = pca.tep_scaler_.transform(X)
    Xc = X - pca.mean_
    T = Xc @ P.T
    t2 = np.sum(T ** 2 / lam, axis=1)
    spe = np.sum((Xc - T @ P) ** 2, axis=1)
    return t2.astype("float64"), spe.astype("float64")


def pca_monitor_limits(pca, k, n):
    """Classical control limits at 1 - PCAMON_ALPHA: F-distribution for T^2, Jackson-Mudholkar for SPE."""
    a = 1 - PCAMON_ALPHA
    t2_lim = k * (n - 1) * (n + 1) / (n * (n - k)) * stats.f.ppf(a, k, n - k)
    rest = pca.explained_variance_[k:]
    th1, th2, th3 = (rest ** 1).sum(), (rest ** 2).sum(), (rest ** 3).sum()
    h0 = 1 - 2 * th1 * th3 / (3 * th2 ** 2)
    c = stats.norm.ppf(a)
    spe_lim = th1 * (c * np.sqrt(2 * th2 * h0 ** 2) / th1 + 1 + th2 * h0 * (h0 - 1) / th1 ** 2) ** (1 / h0)
    return float(t2_lim), float(spe_lim)


def pcamon_checkpoints(Xv, X_normal, secs_fit):
    t0 = time.time()
    pca = pca_monitor_fit(X_normal)
    secs = secs_fit + time.time() - t0
    out = []
    for k in PCAMON_COMPONENTS:
        t2, spe = pca_monitor_scores(pca, Xv, k)
        t2_lim, spe_lim = pca_monitor_limits(pca, k, len(X_normal))
        for stat, s, lim in (("T2", t2, t2_lim), ("SPE", spe, spe_lim)):
            normal = s[y_val == 0]
            extra = {"score_kind": "stat", "explained_variance": float(pca.explained_variance_ratio_[:k].sum()),
                     "classical_limit": lim, "classical_far": float((normal >= lim).mean()),
                     "classical_recall": float((s[y_val == 1] >= lim).mean()), "train_normal_rows": int(len(X_normal))}
            out.append({"params": {"stat": stat, "k": k}, "output": s, "explain": None, "secs": secs, "extra": extra})
    return pca, out


def describe(summary):
    """Fill in family/params for summaries written before families existed (older logistic-regression files)."""
    if "family" not in summary:
        summary["family"] = "baseline" if summary.get("penalty") == "baseline" else "logreg"
        summary["params"] = {} if summary["family"] == "baseline" else {
            "penalty": summary["penalty"], "C": summary.get("C"), "class_weight": summary.get("class_weight", "none")}
    summary.setdefault("features", "inst")
    return summary

# ====================================================================================================
# ## 6. Evaluate and export one model
# Shared by every family, so all models are measured exactly the same way.

def export_detection(mid, label, family, params, scores, fit_seconds, explain, fs="inst", baseline=False, extra=None):
    # The baseline always predicts "fault"; every other model is set to 5% false alarms on validation.
    threshold = 0.5 if baseline else threshold_at_far(y_val, scores, SELECTION_FAR)
    metrics = detection_metrics(y_val, scores, threshold)
    delays, thresholds = {}, [threshold]
    for far in FAR_LEVELS:
        t = threshold_at_far(y_val, scores, far)
        thresholds.append(t)
        by_fault, reference = delays_to_json(streak_delays(
            val_df["run_id"].to_numpy(), faults_val, val_df["sample"].to_numpy(), scores >= t, FAULT_ONSET["train"]))
        delays[str(far)] = {"threshold": t, "by_fault": by_fault, "reference": reference}
    extra = dict(extra or {})
    score_kind = extra.pop("score_kind", "prob")
    summary = {"id": mid, "family": family, "label": label, "params": params, "features": fs,
               "n_inputs": len(INPUT_NAMES[fs]), "score_kind": score_kind, "fit_seconds": round(fit_seconds, 1),
               "val": metrics, "ci": None if baseline else bootstrap_detection(scores, threshold),
               "coef_nonzero": (explain or {}).get("nonzero"), "export_version": EXPORT_VERSION, **extra}
    payload = {"summary": summary, "val": {**score_histograms(y_val, scores, faults_val, thresholds), "delays": delays}}
    if score_kind == "prob" and not baseline:
        payload["calibration"] = calibration(scores)
    payload["coefficients" if family in ("logreg", "baseline", "pcalr") else "importances"] = explain
    write_json(WEB_DIR / "detection" / f"{mid}.json", payload)
    write_traces(mid, scores)
    print(f"{mid:48s} recall@5%FAR={metrics['recall']:.3f}  ({fit_seconds:.0f}s)", flush=True)
    return clean(summary)


def export_diagnosis(mid, label, family, params, pred, fit_seconds, explain, fs="inst", extra=None):
    true = faults_val[val_post]
    metrics, cm, recall = diagnosis_metrics(true, pred)
    by_fault, _ = delays_to_json(streak_delays(val_df["run_id"].to_numpy()[val_post], true,
                                               val_df["sample"].to_numpy()[val_post], pred == true, FAULT_ONSET["train"]))
    summary = {"id": mid, "family": family, "label": label, "params": params, "features": fs,
               "n_inputs": len(INPUT_NAMES[fs]), "fit_seconds": round(fit_seconds, 1), "val": metrics,
               "ci": None if family == "baseline" else bootstrap_diagnosis(pred),
               "coef_nonzero": (explain or {}).get("nonzero"), "export_version": EXPORT_VERSION, **(extra or {})}
    payload = {"summary": summary, "val": {"labels": FAULTS, "confusion_matrix": cm, "recall": recall, "delays": by_fault}}
    payload["coefficients" if family in ("logreg", "baseline", "pcalr") else "importances"] = explain
    write_json(WEB_DIR / "diagnosis" / f"{mid}.json", payload)
    print(f"{mid:48s} macro-F1={metrics['macro_F1']:.3f}  ({fit_seconds:.0f}s)", flush=True)
    return clean(summary)


def existing_summary(task, mid):
    """Summary of an exported model (any version), or None."""
    path = WEB_DIR / task / f"{mid}.json"
    return describe(json.loads(path.read_text())["summary"]) if path.exists() else None


def up_to_date(summary):
    return summary is not None and summary.get("export_version", 1) >= EXPORT_VERSION


def forced(family):
    return FORCE_RETRAIN and family in FAMILIES

# ====================================================================================================
# ## 7. Train and evaluate every model
# For each model: reuse its export if current; otherwise reload the saved model and re-export (no retraining);
# otherwise train it if its family was requested. Families not requested keep whatever was exported before.

def run_linear(task, fs="inst"):
    """Majority baseline plus logistic-regression variants for one task and input set."""
    X, y, Xv = task_data(task, fs)
    weights = CLASS_WEIGHTS if task == "detection" else [None]
    penalties = HIST_LR_PENALTIES if fs == "hist" else ["none", "l1", "l2"]
    configs = [("baseline", None, None)] if fs == "inst" else []
    for cw in weights:
        configs += [(p, C, cw) for p in penalties for C in ([None] if p == "none" else C_GRID)]
    train_lr = ("logreg" in FAMILIES) and (fs == "inst" or "logreg" in HISTORY)
    summaries = []
    for penalty, C, cw in configs:
        mid = model_id(task, penalty, C, cw, fs)
        family = "baseline" if penalty == "baseline" else "logreg"
        old = existing_summary(task, mid)
        if up_to_date(old) and not forced(family):
            summaries.append(old)
            continue
        path = MODEL_DIR / f"{mid}.joblib"
        if path.exists() and not forced(family) and old is not None:
            model, secs = joblib.load(path), old.get("fit_seconds", 0.0)      # re-export from the saved model
        elif family == "baseline" or train_lr:
            model = (DummyClassifier(strategy="most_frequent") if family == "baseline"
                     else make_lr(penalty, C, cw, multiclass=(task == "diagnosis")))
            t0 = time.time()
            model.fit(X, y)
            secs = time.time() - t0
            joblib.dump(model, path, compress=3)
        else:
            if old is not None:
                summaries.append(old)                 # keep the older export as it is
            continue
        params = {} if family == "baseline" else {"penalty": penalty, "C": C, "class_weight": cw or "none"}
        extra = {"penalty": penalty, "C": C, **({"class_weight": cw or "none"} if task == "detection" else {})}
        label, explain = model_label(penalty, C, cw, fs), coefficients(model, INPUT_NAMES[fs])
        if task == "detection":
            s = export_detection(mid, label, family, params, model.predict_proba(Xv)[:, 1].astype("float64"), secs,
                                 explain, fs, baseline=(family == "baseline"), extra=extra)
        else:
            s = export_diagnosis(mid, label, family, params, model.predict(Xv), secs, explain, fs, extra=extra)
        summaries.append(describe(s))
    return summaries


def run_family(task, family, fs="inst"):
    if family == "pcamon" and task != "detection":
        return []
    X, y, Xv = task_data(task, fs)
    names = INPUT_NAMES[fs]
    train_this = family in FAMILIES and (fs == "inst" or family in HISTORY)
    summaries = []
    for unit in family_units(family, fs):
        cps = checkpoint_params(family, unit)
        ids = [unit_id(task, family, p, fs) for p in cps]
        olds = [existing_summary(task, mid) for mid in ids]
        if all(up_to_date(o) for o in olds) and not forced(family):
            summaries += olds
            continue
        if family == "pcamon":
            # Cheap to recompute, so it is never stored: recompute when requested, otherwise keep older exports.
            if not train_this:
                summaries += [o for o in olds if o is not None]
                continue
            _, checkpoints = pcamon_checkpoints(Xv, X_train["inst"][normal_in_keep], 0.0)
        else:
            name = f"{task}_pcalr_k{unit}" if family == "pcalr" else f"{task}_{family}_d{unit}{HIST_TAG[fs]}"
            model_path = MODEL_DIR / f"{name}.joblib"
            if model_path.exists() and not forced(family):
                model = joblib.load(model_path)            # re-export from the saved model, no retraining
                secs = max((o.get("fit_seconds", 0.0) for o in olds if o is not None), default=0.0)
            elif train_this:
                t0 = time.time()
                model = fit_pcalr(task, unit, X, y) if family == "pcalr" else fit_tree_model(task, family, unit, X, y)
                secs = time.time() - t0
                joblib.dump(model, model_path, compress=3)
            else:
                summaries += [o for o in olds if o is not None]
                continue
            checkpoints = (pcalr_checkpoint(task, unit, model, Xv, secs) if family == "pcalr"
                           else tree_checkpoints(task, family, unit, model, Xv, secs, names))
        for c in checkpoints:
            mid, label = unit_id(task, family, c["params"], fs), unit_label(family, c["params"], fs)
            if task == "detection":
                summaries.append(export_detection(mid, label, family, c["params"], c["output"], c["secs"], c["explain"],
                                                  fs, extra=c.get("extra")))
            else:
                summaries.append(export_diagnosis(mid, label, family, c["params"], c["output"], c["secs"], c["explain"],
                                                  fs, extra=c.get("extra")))
    return summaries


def run_task(task):
    summaries = run_linear(task, "inst")
    if "hist" in FEATURE_SETS:
        summaries += run_linear(task, "hist")
    for family in ALL_FAMILIES:
        if family == "logreg":
            continue
        for fs in (["inst", "hist"] if family in HISTORY and "hist" in FEATURE_SETS else ["inst"]):
            t0 = time.time()
            summaries += run_family(task, family, fs)
            if family in FAMILIES:
                print(f"--- {task} {FAMILY_INFO[family]['label']}{HIST_LABEL[fs]}: {time.time() - t0:.0f}s", flush=True)
    return summaries


def summary_table(summaries, metric_keys):
    rows = [{"id": s["id"], "family": s["family"], "inputs": s.get("features", "inst"), "used": s.get("coef_nonzero"),
             "selectable": s.get("selectable", True), **{k: s["val"][k] for k in metric_keys}} for s in summaries]
    return pd.DataFrame(rows).set_index("id")


det_summaries = run_task("detection")
diag_summaries = run_task("diagnosis")

# ### History window sweep
# The best 30-minute settings of each history family (chosen on validation, as everywhere else) are retrained
# with the other windows. One window's inputs are built at a time and freed afterwards to keep memory down.

def fit_with_params(task, family, params, X, y):
    """A single model with exactly these settings."""
    if family == "logreg":
        cw = None if params.get("class_weight", "none") == "none" else params["class_weight"]
        return make_lr(params["penalty"], params["C"], cw, multiclass=(task == "diagnosis")).fit(X, y)
    if family == "pcalr":
        return fit_pcalr(task, params["k"], X, y)
    model = build_tree_model(task, family, params)
    if family == "ada":
        X, y = subsample(X, y, ADA_SAMPLE)
    if family == "xgb":
        model.fit(X, np.searchsorted(np.unique(y), y))
        model.tep_classes_ = np.unique(y)
    else:
        model.fit(X, y)
    return model


def sweep_jobs():
    jobs = []
    for task, summaries, key in (("detection", det_summaries, "recall"), ("diagnosis", diag_summaries, "macro_F1")):
        for family in HISTORY:
            cands = [s for s in summaries if s["family"] == family and s.get("features") == "hist"]
            if not cands:
                continue
            best = max(cands, key=lambda s: s["val"][key])
            for w in SWEEP_WINDOWS:
                fs = fs_for_window(w)
                p = best["params"]
                if family == "logreg":
                    mid = model_id(task, p["penalty"], p["C"], None if p.get("class_weight", "none") == "none"
                                   else p["class_weight"], fs)
                    label = model_label(p["penalty"], p["C"], p.get("class_weight") if p.get("class_weight") != "none" else None, fs)
                else:
                    mid, label = unit_id(task, family, p, fs), unit_label(family, p, fs)
                jobs.append({"task": task, "family": family, "params": p, "fs": fs, "id": mid, "label": label, "base": best})
    return jobs


def final_scores(task, summary, model, X):
    family, p = summary["family"], summary["params"]
    if family == "pcamon":
        t2, spe = pca_monitor_scores(model, X, p["k"])
        return t2 if p["stat"] == "T2" else spe
    if family == "pcalr":
        proba = model["lr"].predict_proba(model["pca"].transform(X))
        classes = model["lr"].classes_
    else:
        proba = model.predict_proba(X)
        classes = getattr(model, "tep_classes_", getattr(model, "classes_", None))
    return task_output(task, proba, classes)


def final_explain(summary, model, names):
    family = summary["family"]
    if family in ("logreg", "baseline"):
        return coefficients(model, names)
    if family == "pcalr":
        coef = model["lr"].coef_ @ model["pca"].components_
        return {"features": FEATURES, "values": coef[0] if coef.shape[0] == 1 else coef, "nonzero": len(FEATURES)}
    if family in ("tree", "rf", "ada") or family == "xgb":
        if family == "xgb":
            gain = model.get_booster().get_score(importance_type="gain")
            return importances([gain.get(f"f{j}", 0.0) for j in range(len(names))], "gain", names)
        return importances(model.feature_importances_, "impurity", names)
    return None


def run_sweep():
    jobs = sweep_jobs()
    if not jobs:
        return
    print(f"\nHistory window sweep: {len(jobs)} models", flush=True)
    for w in SWEEP_WINDOWS:
        fs = fs_for_window(w)
        todo = []
        for j in (j for j in jobs if j["fs"] == fs):
            old = existing_summary(j["task"], j["id"])
            if old is not None:
                old["selectable"] = w <= FAULT_ONSET["train"]
            if up_to_date(old) and not forced(j["family"]):
                (det_summaries if j["task"] == "detection" else diag_summaries).append(old)
            else:
                todo.append((j, old))
        if not todo:
            continue
        build_inputs(fs)
        for j, old in todo:
            task, family, p = j["task"], j["family"], j["params"]
            X, y, Xv = task_data(task, fs)
            path = MODEL_DIR / f"{j['id']}.joblib"
            if path.exists() and not forced(family) and old is not None:
                model, secs = joblib.load(path), old.get("fit_seconds", 0.0)
            elif family in FAMILIES:
                t0 = time.time()
                model = fit_with_params(task, family, p, X, y)
                secs = time.time() - t0
                joblib.dump(model, path, compress=3)
            else:
                if old is not None:
                    (det_summaries if task == "detection" else diag_summaries).append(old)
                continue
            explain = final_explain({"family": family}, model, INPUT_NAMES[fs])
            # Training faults start after sample 20 (1 hour). A longer window is still partly empty when a fault
            # starts in training and validation, but full in the test files (fault after 8 hours), so validation
            # flatters it. Such windows are shown on the website but not used to pick the best model.
            extra = {"sweep_of": j["base"]["id"], "hist_window": w, "selectable": w <= FAULT_ONSET["train"]}
            if family == "logreg":
                extra.update({"penalty": p["penalty"], "C": p["C"],
                              **({"class_weight": p.get("class_weight", "none")} if task == "detection" else {})})
            if task == "detection":
                s = export_detection(j["id"], j["label"], family, p, final_scores(task, {"family": family, "params": p}, model, Xv),
                                     secs, explain, fs, extra=extra)
                det_summaries.append(describe(s))
            else:
                s = export_diagnosis(j["id"], j["label"], family, p, final_scores(task, {"family": family, "params": p}, model, Xv),
                                     secs, explain, fs, extra=extra)
                diag_summaries.append(describe(s))
        for d in (X_train, X_val, scalers):
            d.pop(fs, None)

run_sweep()

det_table = summary_table(det_summaries, ["recall", "precision", "F1", "ROC_AUC", "normal_recall"])

# ### Choosing the best detection model
# All models are compared at the same operating point (5% false alarms on validation normal rows), fixed before
# looking at the results, so the best model is the one that catches the most faults there.
candidates = det_table[(det_table["family"] != "baseline") & det_table["selectable"]]
best_det_id = candidates["recall"].idxmax()
best_det = next(s for s in det_summaries if s["id"] == best_det_id)
print("\nBest detection model per family and input set (recall at 5% false alarms):")
grp = candidates.groupby(["family", "inputs"])["recall"].idxmax()
print(candidates.loc[grp].sort_values("recall", ascending=False).round(4))
print("Best detection model:", best_det["label"])

diag_table = summary_table(diag_summaries, ["macro_F1", "accuracy"])
diag_candidates = diag_table[(diag_table["family"] != "baseline") & diag_table["selectable"]]
best_diag_id = diag_candidates["macro_F1"].idxmax()
best_diag = next(s for s in diag_summaries if s["id"] == best_diag_id)
print("\nBest diagnosis model per family and input set (macro-F1):")
grp = diag_candidates.groupby(["family", "inputs"])["macro_F1"].idxmax()
print(diag_candidates.loc[grp].sort_values("macro_F1", ascending=False).round(4))
print("Best diagnosis model:", best_diag["label"])

# ====================================================================================================
# ## 8. Data summary and manifest
# data_summary.json: quality checks, per-variable statistics, correlations and PCA variance on fault-free training
# runs. data_traces.json: every sensor for one run of each fault type (every 2nd sample), for the data explorer.

def export_data_summary():
    normal = train_df[train_df["faultNumber"] == 0]
    X_norm = StandardScaler().fit_transform(normal[FEATURES].to_numpy(np.float32))    # standardized on normal rows
    pca = PCA(random_state=SEED).fit(X_norm)
    stats_tbl = normal[FEATURES].agg(["mean", "std", "min", "max"]).T
    write_json(WEB_DIR / "data_summary.json", {
        "quality": QUALITY,
        "variables": {f: {k: float(stats_tbl.loc[f, k]) for k in ("mean", "std", "min", "max")} for f in FEATURES},
        "correlation": np.round(np.corrcoef(normal[FEATURES].to_numpy(np.float64), rowvar=False), 3),
        "pca_variance": pca.explained_variance_ratio_,
        "features": FEATURES,
    })
    runs_out = []
    for f in range(0, 21):
        # The lowest-numbered run of this fault type, from whichever split holds it.
        options = [(df_["simulationRun"][df_["faultNumber"] == f].min(), df_) for df_ in (train_df, val_df)
                   if (df_["faultNumber"] == f).any()]
        if not options:
            continue
        r, src = min(options, key=lambda o: o[0])
        run = src[(src["faultNumber"] == f) & (src["simulationRun"] == r) & (src["sample"] % 2 == 0)].sort_values("sample")
        runs_out.append({"fault": f, "run": int(r), "sample": run["sample"].to_numpy(),
                         "values": {c: np.round(run[c].to_numpy(np.float64), 4) for c in FEATURES}})
    write_json(WEB_DIR / "data_traces.json", {"onset": FAULT_ONSET["train"], "sample_minutes": SAMPLE_MINUTES,
                                              "runs": runs_out})


def best_by_family(summaries, key):
    out = {}
    for s in summaries:
        f = s["family"]
        if f != "baseline" and s.get("selectable", True) and (f not in out or s["val"][key] > out[f][1]):
            out[f] = (s["id"], s["val"][key])
    return {f: v[0] for f, v in out.items()}


def write_manifest(test_results=None):
    present = [f for f in ["baseline"] + ALL_FAMILIES
               if any(s["family"] == f for s in det_summaries + diag_summaries)]
    write_json(WEB_DIR / "manifest.json", {
        "dataset": {
            "features": FEATURES, "runs_per_fault": RUNS_PER_FAULT or 500,
            "train_runs": int(train_df["run_id"].nunique()), "val_runs": int(val_df["run_id"].nunique()),
            "val_rows": int(len(val_df)), "val_fault_share": float(y_val.mean()),
            "fault_onset": FAULT_ONSET, "sample_minutes": SAMPLE_MINUTES,
        },
        "settings": {"split": "simulation_number", "C_grid": C_GRID, "class_weights": [c or "none" for c in CLASS_WEIGHTS],
                     "selection_far": SELECTION_FAR, "far_levels": FAR_LEVELS, "consecutive": CONSECUTIVE,
                     "det_stride": DET_STRIDE, "diag_stride": DIAG_STRIDE, "rf_sample": RF_SAMPLE,
                     "ada_sample": ADA_SAMPLE, "learning_rate": LEARNING_RATE, "gb_l2": GB_L2,
                     "hist_window": HIST_WINDOW, "hist_windows": sorted([HIST_WINDOW] + SWEEP_WINDOWS), "bootstrap": BOOTSTRAP, "pcamon_alpha": PCAMON_ALPHA,
                     "export_version": EXPORT_VERSION, "inputs": {k: len(v) for k, v in INPUT_NAMES.items()},
                     "trace_runs": TRACE_RUNS},
        "families": [{"key": f, **FAMILY_INFO[f]} for f in present],
        "detection": {"models": det_summaries, "best": best_det_id,
                      "best_by_family": best_by_family(det_summaries, "recall"),
                      "selection_rule": f"highest validation recall at {SELECTION_FAR:.0%} false alarms"},
        "diagnosis": {"models": diag_summaries, "best": best_diag_id,
                      "best_by_family": best_by_family(diag_summaries, "macro_F1"),
                      "selection_rule": "highest validation macro-F1"},
        "test": test_results,
    })


export_data_summary()
manifest_path = WEB_DIR / "manifest.json"
previous_test = json.loads(manifest_path.read_text()).get("test") if manifest_path.exists() else None
write_manifest(previous_test)
print("Wrote", manifest_path)

# ====================================================================================================
# ## 9. Final test evaluation (run once, after model selection)
# 1. Refit the scalers and retrain the chosen detection and diagnosis models on all training runs (train + val).
# 2. Keep the **validation** threshold for detection; nothing is tuned on test.
# 3. Evaluate once on the test files (fault injected after sample 160) and export the results.
#
# The majority baselines are evaluated too, for comparison.

def fit_final(task, summary, X, y, X_normal=None):
    family, p = summary["family"], summary["params"]
    if family == "baseline":
        model = DummyClassifier(strategy="most_frequent").fit(X, y)
    elif family == "logreg":
        cw = None if p.get("class_weight", "none") == "none" else p["class_weight"]
        model = make_lr(p["penalty"], p["C"], cw, multiclass=(task == "diagnosis")).fit(X, y)
    elif family == "pcalr":
        model = fit_pcalr(task, p["k"], X, y)
    elif family == "pcamon":
        model = pca_monitor_fit(X_normal)
    else:
        model = build_tree_model(task, family, p)
        if family == "ada":
            X, y = subsample(X, y, ADA_SAMPLE)
        if family == "xgb":
            model.fit(X, np.searchsorted(np.unique(y), y))
            model.tep_classes_ = np.unique(y)
        else:
            model.fit(X, y)
    joblib.dump(model, MODEL_DIR / f"final_{summary['id']}.joblib", compress=3)
    return model


if args.final_test:
    full_df = pd.concat([train_df, val_df]).sort_values(["run_id", "sample"], ignore_index=True)
    test_normal = load_tep("TEP_FaultFree_Testing")
    test_faulty = load_tep("TEP_Faulty_Testing")
    test_faulty = test_faulty[test_faulty["simulationRun"] <= TEST_RUNS_PER_FAULT]
    test_df = prepare(test_normal, test_faulty, FAULT_ONSET["test"])
    del test_normal, test_faulty
    y_test = test_df["target"].to_numpy()
    faults_test = test_df["faultNumber"].to_numpy()
    runs_test, samples_test = test_df["run_id"].to_numpy(), test_df["sample"].to_numpy()
    test_post = y_test == 1
    test_copies = int(np.isin(row_hashes(test_df), row_hashes(full_df)).sum())
    print("Test rows identical to a training row:", test_copies)
    test_results = {"rows": int(len(test_df)), "fault_share": float(y_test.mean()), "rows_copied_from_train": test_copies,
                    "runs_per_fault": TEST_RUNS_PER_FAULT, "detection": {}, "diagnosis": {}}

    chosen = [("detection", best_det), ("detection", next(s for s in det_summaries if s["family"] == "baseline")),
              ("diagnosis", best_diag), ("diagnosis", next(s for s in diag_summaries if s["family"] == "baseline"))]
    # The chosen models exactly as validated (trained on the training runs only), on test with their validation
    # threshold. This is the model the threshold was set for, so its test false alarm rate shows how well the 5%
    # carries over; the retrained models below are the headline result.
    test_results["as_validated"] = {"detection": {}, "diagnosis": {}}
    for task, summary in chosen:
        if summary["family"] == "baseline":
            continue
        fs = summary.get("features", "inst")
        if fs not in X_train:
            build_inputs(fs)
        if summary["family"] == "pcamon":
            model = pca_monitor_fit(X_train["inst"][normal_in_keep])
        else:
            X, y, _ = task_data(task, fs)
            model = fit_with_params(task, summary["family"], summary["params"], X, y)
            del X
        X_test_v = scalers[fs].transform(raw_inputs(test_df, fs)).astype(np.float32)
        if task == "detection":
            m = detection_metrics(y_test, final_scores(task, summary, model, X_test_v), summary["val"]["threshold"])
        else:
            m, _, _ = diagnosis_metrics(faults_test[test_post], final_scores(task, summary, model, X_test_v[test_post]))
        test_results["as_validated"][task][summary["id"]] = m
        print(f"TEST (as validated) {summary['label']:45s}", {k: round(v, 4) for k, v in m.items() if not isinstance(v, int)})
        del model, X_test_v

    # Free the training/validation inputs first: on the full dataset memory is the limit here.
    X_train.clear(); X_val.clear()
    import gc; gc.collect()
    final_inputs = {}
    for fs in {s.get("features", "inst") for _, s in chosen}:
        raw_full = raw_inputs(full_df, fs)
        sc = StandardScaler(copy=False).fit(raw_full)               # transform in place (float32 stays float32)
        raw_test = raw_inputs(test_df, fs)
        final_inputs[fs] = (sc.transform(raw_full), sc.transform(raw_test))
        del raw_full, raw_test

    for task, summary in chosen:
        fs = summary.get("features", "inst")
        X_all, X_test = final_inputs[fs]
        if task == "detection":
            rows = (full_df["sample"] % DET_STRIDE == 0).to_numpy()
            X_fit, y_fit = X_all[rows], full_df.loc[rows, "target"].to_numpy()
        else:
            rows = ((full_df["target"] == 1) & (full_df["sample"] % DIAG_STRIDE == 0)).to_numpy()
            X_fit, y_fit = X_all[rows], full_df.loc[rows, "faultNumber"].to_numpy()
        X_normal = X_all[(full_df["faultNumber"] == 0).to_numpy()]
        model = fit_final(task, summary, X_fit, y_fit, X_normal)
        explain = final_explain(summary, model, INPUT_NAMES[fs])
        key = "coefficients" if summary["family"] in ("logreg", "baseline", "pcalr") else "importances"
        if task == "detection":
            scores = final_scores(task, summary, model, X_test)
            threshold = summary["val"]["threshold"]                    # chosen on validation
            metrics = detection_metrics(y_test, scores, threshold)
            by_fault, reference = delays_to_json(streak_delays(runs_test, faults_test, samples_test,
                                                               scores >= threshold, FAULT_ONSET["test"]))
            test_results["detection"][summary["id"]] = metrics
            write_json(WEB_DIR / "detection" / f"test_{summary['id']}.json", {
                "summary": {**summary, "test": metrics}, key: explain,
                "test": {**score_histograms(y_test, scores, faults_test, [threshold]),
                         "delays": {"validation_threshold": {"threshold": threshold, "by_fault": by_fault,
                                                             "reference": reference}}}})
        else:
            pred = final_scores(task, summary, model, X_test[test_post])
            true = faults_test[test_post]
            metrics, cm, recall = diagnosis_metrics(true, pred)
            by_fault, _ = delays_to_json(streak_delays(runs_test[test_post], true, samples_test[test_post],
                                                       pred == true, FAULT_ONSET["test"]))
            test_results["diagnosis"][summary["id"]] = metrics
            write_json(WEB_DIR / "diagnosis" / f"test_{summary['id']}.json", {
                "summary": {**summary, "test": metrics}, key: explain,
                "test": {"labels": FAULTS, "confusion_matrix": cm, "recall": recall, "delays": by_fault}})
        print(f"TEST {summary['label']:45s}", {k: round(v, 4) for k, v in metrics.items() if not isinstance(v, int)})

    write_manifest(test_results)
    print("Manifest updated with test results.")
