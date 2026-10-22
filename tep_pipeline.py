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
  1. Split the training files by simulation run (80% / 20% of runs per fault type, stratified).
  2. Train every variant on the training runs; choose thresholds and the best variant on validation.
  3. Retrain the chosen variants on all training runs, keep the validation thresholds, evaluate once on test.

Model families (--families picks which to train; the majority baseline is always included)
  logreg  Logistic regression: class weighting none/balanced (detection only), penalty none/L1/L2, strength C.
          scikit-learn's default LogisticRegression() is already L2 with C=1; "no penalty" here is unregularized.
  tree    Decision tree: maximum depth.
  rf      Random forest (bagged trees): maximum depth x number of trees.
  gb      Gradient boosting (scikit-learn's histogram version): tree depth x boosting rounds.
  ada     AdaBoost with small decision trees: tree depth x boosting rounds.
  xgb     XGBoost: tree depth x boosting rounds (skipped if xgboost is not installed).
Forests and boosted models are trained once per depth at the largest size; smaller forests (the first k trees)
and fewer boosting rounds (the first k rounds) are evaluated from that same fit, so the website can show every
size without retraining.

Usage
  python tep_pipeline.py --data-dir /path/to/csvs --runs-per-fault 100
  python tep_pipeline.py --data-dir /path/to/csvs --runs-per-fault 100 --families tree,rf
  python tep_pipeline.py --data-dir /path/to/csvs --runs-per-fault 0 --final-test   # all runs + test

Results are written to <out-dir>/outputs_<N>runs (or outputs_full): models/ and web_data/ (website files).
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

ALL_FAMILIES = ["logreg", "tree", "rf", "gb", "ada", "xgb"]

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
parser.add_argument("--det-stride", type=int, default=2, help="detection trains on every k-th time step")
parser.add_argument("--diag-stride", type=int, default=5, help="diagnosis trains on every k-th time step")
parser.add_argument("--selection-far", type=float, default=0.05,
                    help="false-alarm rate on validation normal rows used to set thresholds and pick the best model")
parser.add_argument("--test-runs-per-fault", type=int, default=100,
                    help="faulty test runs used per fault (the full faulty test file is ~10M rows)")
parser.add_argument("--final-test", action="store_true", help="retrain the chosen models and evaluate on test")
parser.add_argument("--force-retrain", action="store_true", help="retrain models even if already exported")
args = parser.parse_args()

# ---- Paths ---------------------------------------------------------------------------------------
DRIVE_DIR = Path(args.data_dir)                              # source CSVs + persistent Parquet cache
DATA_DIR = Path(args.work_dir) if args.work_dir else DRIVE_DIR
RUNS_PER_FAULT = args.runs_per_fault or None                 # None = all runs
OUT_DIR = Path(args.out_dir or DRIVE_DIR) / ("outputs_full" if RUNS_PER_FAULT is None
                                              else f"outputs_{RUNS_PER_FAULT}runs")
MODEL_DIR = OUT_DIR / "models"
WEB_DIR = OUT_DIR / "web_data"                               # everything the website needs
for d in (DATA_DIR, MODEL_DIR, WEB_DIR / "detection", WEB_DIR / "diagnosis"):
    d.mkdir(parents=True, exist_ok=True)

# ---- Experiment settings -------------------------------------------------------------------------
SEED = 42
FAULT_ONSET = {"train": 20, "test": 160}   # last pre-injection sample in each faulty run
SAMPLE_MINUTES = 3
ID_COLS = ["faultNumber", "simulationRun", "sample"]

C_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]   # website slider values; "none" = no penalty
CLASS_WEIGHTS = [None, "balanced"]                     # detection only
SELECTION_FAR = args.selection_far
FAR_LEVELS = [0.01, 0.05, 0.10]                        # detection delays are precomputed at these alarm rates
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

FAMILIES = [f.strip() for f in args.families.split(",") if f.strip()]
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

    df = pd.read_csv(DRIVE_DIR / f"{name}.csv")
    float_cols = df.select_dtypes("float64").columns
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
print(train_all["target"].value_counts().rename("rows"))

# ====================================================================================================
# ## 2. Train / validation split by run
# Stratified on `faultNumber` (0 = fault-free), so each fault type keeps 400 train / 100 validation runs.

runs = train_all[["run_id", "faultNumber"]].drop_duplicates()
train_run_ids, val_run_ids = train_test_split(
    runs["run_id"], test_size=0.2, stratify=runs["faultNumber"], random_state=SEED
)
is_val = train_all["run_id"].isin(val_run_ids)
train_df = train_all[~is_val].reset_index(drop=True)
val_df = train_all[is_val].reset_index(drop=True)
del train_all
assert set(train_df["run_id"]).isdisjoint(val_df["run_id"])

scaler = StandardScaler().fit(train_df[FEATURES].to_numpy())   # fit on training runs only
X_val = scaler.transform(val_df[FEATURES].to_numpy())
y_val = val_df["target"].to_numpy()
faults_val = val_df["faultNumber"].to_numpy()

det_rows = (train_df["sample"] % DET_STRIDE == 0).to_numpy()
X_det = scaler.transform(train_df.loc[det_rows, FEATURES].to_numpy())
y_det = train_df.loc[det_rows, "target"].to_numpy()

diag_rows = ((train_df["target"] == 1) & (train_df["sample"] % DIAG_STRIDE == 0)).to_numpy()
X_diag = scaler.transform(train_df.loc[diag_rows, FEATURES].to_numpy())
y_diag = train_df.loc[diag_rows, "faultNumber"].to_numpy()

val_post = y_val == 1                        # diagnosis is evaluated on post-injection rows only
FAULTS = np.arange(1, 21)

print(f"Train: {train_df.shape}  Val: {val_df.shape}")
print(f"Detection training rows: {X_det.shape}   Diagnosis training rows: {X_diag.shape}")

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
    _idx = np.random.default_rng(SEED).choice(len(X_det), size=min(20000, len(X_det)), replace=False)
    assert (make_lr("l1", 1e-4).fit(X_det[_idx], y_det[_idx]).coef_ == 0).any(), \
        "L1 penalty was not applied; check the scikit-learn version"
    print("scikit-learn API:", "new (l1_ratio)" if NEW_LR_API else "old (penalty=)", "- lasso check passed")

PENALTY_NAMES = {"none": "No penalty", "l1": "Lasso (L1)", "l2": "Ridge (L2)", "baseline": "Majority class"}


def model_id(task, penalty, C=None, class_weight=None):
    parts = [task, penalty]
    if task == "detection" and penalty != "baseline":
        parts.append("balanced" if class_weight == "balanced" else "unweighted")
    if C is not None:
        parts.append(f"C{C:g}")
    return "_".join(parts)


def model_label(penalty, C=None, class_weight=None):
    label = PENALTY_NAMES[penalty]
    if class_weight == "balanced":
        label += ", balanced"
    if C is not None:
        label += f", C={C:g}"
    return label

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
# random prediction from counting.

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
    """Per fault type: median minutes until k consecutive post-onset flags, and share of runs where it happens.

    Rows must be sorted by run and sample (they are)."""
    flags = flags & (samples > onset)
    cs = np.concatenate([[0], np.cumsum(flags)])
    idx = np.arange(len(flags))
    start = np.maximum(idx - k + 1, 0)
    streak = (idx - k + 1 >= 0) & (run_ids[start] == run_ids) & (cs[idx + 1] - cs[start] == k)
    per_run = pd.DataFrame({
        "run": run_ids, "fault": faults,
        "minutes": np.where(streak, (samples - onset) * SAMPLE_MINUTES, np.nan),
    }).groupby("run").agg(fault=("fault", "first"), minutes=("minutes", "min"))
    per_run = per_run[per_run["fault"] > 0]
    out = per_run.groupby("fault")["minutes"].agg(median_minutes="median", detected="count", runs="size")
    out["detected_share"] = out["detected"] / out["runs"]
    return out[["median_minutes", "detected_share"]]


def delays_to_json(table):
    return {int(f): [None if pd.isna(r.median_minutes) else float(r.median_minutes), round(float(r.detected_share), 4)]
            for f, r in table.iterrows()}


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
# quantile-spaced bin edges. Row counts in bin `i` are scores in `[edge[i], edge[i+1])`, so for a slider
# threshold `t = edge[i]`, rows flagged as faulty = sum of bins `i..end`. From that the browser computes the
# confusion matrix, precision, recall, F1, normal recall and per-fault recall for any threshold.
# Detection delays are precomputed at a few false-alarm rates because they can't be rebuilt from histograms.
#
# **Diagnosis files** hold the confusion matrix, per-fault recall and diagnosis delays.
#
# Both say which of the 52 variables the model relies on: logistic regression stores its coefficients (on
# standardized inputs); tree models store feature importances (how much each variable's splits reduce error).
#
# `manifest.json` lists every model, its settings, validation metrics, the chosen best model and test results.

def score_histograms(y, scores, faults, n_bins=200):
    edges = np.unique(np.quantile(scores, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:                                  # constant scores (majority baseline)
        edges = np.array([edges[0], edges[0] + 1e-6])
    return {
        "bin_edges": edges,
        "normal_counts": np.histogram(scores[y == 0], edges)[0],
        "fault_counts": {int(f): np.histogram(scores[(y == 1) & (faults == f)], edges)[0]
                         for f in np.unique(faults[y == 1])},
    }


def coefficients(model):
    if not hasattr(model, "coef_"):
        return None
    coef = model.coef_
    return {"features": FEATURES, "values": coef[0] if coef.shape[0] == 1 else coef,
            "nonzero": int((coef != 0).any(axis=0).sum())}


def importances(values, kind):
    """Feature importances normalized to sum to 1 (None if the model doesn't provide them)."""
    if values is None:
        return None
    values = np.asarray(values, dtype=float)
    total = values.sum()
    values = values / total if total > 0 else values
    return {"features": FEATURES, "values": values, "nonzero": int((values > 0).sum()), "kind": kind}


def write_json(path, payload):
    path.write_text(json.dumps(clean(payload), separators=(",", ":")))

# ====================================================================================================
# ### Tree-based models
# Every family below produces a list of "checkpoints": (params, validation output, importances, fit seconds).
# Detection output is the fault probability for every validation row; diagnosis output is the predicted fault
# for every post-injection validation row. Forests and boosted models fit once at the largest size and read off
# the smaller sizes, so the fit time of a checkpoint is the share of the full fit it needed.

FAMILY_INFO = {
    "baseline": {"label": "Majority baseline", "params": []},
    "logreg": {"label": "Logistic regression", "params": []},   # the website has dedicated controls for these
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
PARAM_TAGS = {"depth": "d", "trees": "t", "rounds": "r"}


def tree_id(task, family, params):
    return f"{task}_{family}_" + "_".join(f"{PARAM_TAGS[k]}{v}" for k, v in params.items())


def tree_label(family, params):
    p = params
    return {
        "tree": lambda: f"Decision tree, depth {p['depth']}",
        "rf": lambda: f"Random forest, depth {p['depth']}, {p['trees']} tree{'s' if p['trees'] != 1 else ''}",
        "gb": lambda: f"Gradient boosting, depth {p['depth']}, {p['rounds']} rounds",
        "ada": lambda: f"AdaBoost, depth-{p['depth']} trees, {p['rounds']} rounds",
        "xgb": lambda: f"XGBoost, depth {p['depth']}, {p['rounds']} rounds",
    }[family]()


def subsample(X, y, fraction, seed=SEED):
    idx = np.random.default_rng(seed).choice(len(X), size=int(len(X) * fraction), replace=False)
    return X[np.sort(idx)], y[np.sort(idx)]


def task_output(task, proba, classes):
    """Detection: probability of a fault. Diagnosis: predicted fault number."""
    return proba[:, 1].astype("float64") if task == "detection" else classes[np.argmax(proba, axis=1)]


def fit_family(task, family, depth, X, y, Xv):
    """Fit one family at one depth and return checkpoints [(params, output, importances, fit_seconds)]."""
    t0 = time.time()
    if family == "tree":
        m = DecisionTreeClassifier(max_depth=depth, random_state=SEED).fit(X, y)
        secs = time.time() - t0
        return m, [({"depth": depth}, task_output(task, m.predict_proba(Xv), m.classes_),
                    importances(m.feature_importances_, "impurity"), secs)]

    if family == "rf":
        m = RandomForestClassifier(n_estimators=max(RF_TREES), max_depth=depth, max_samples=RF_SAMPLE,
                                   n_jobs=-1, random_state=SEED).fit(X, y)
        secs = time.time() - t0
        out, total, imp = [], None, np.zeros(len(FEATURES))
        for k, est in enumerate(m.estimators_, start=1):
            p = est.predict_proba(Xv)
            total = p if total is None else total + p
            imp += est.feature_importances_
            if k in RF_TREES:
                out.append(({"depth": depth, "trees": k}, task_output(task, total / k, m.classes_),
                            importances(imp / k, "impurity"), secs * k / len(m.estimators_)))
        return m, out

    if family == "gb":
        m = HistGradientBoostingClassifier(max_iter=max(GB_ROUNDS), max_depth=depth, learning_rate=LEARNING_RATE,
                                           early_stopping=False, random_state=SEED).fit(X, y)
        secs = time.time() - t0
        out = []
        for k, p in enumerate(m.staged_predict_proba(Xv), start=1):
            if k in GB_ROUNDS:
                # scikit-learn's histogram boosting has no built-in feature importances.
                out.append(({"depth": depth, "rounds": k}, task_output(task, p, m.classes_), None, secs * k / max(GB_ROUNDS)))
        return m, out

    if family == "ada":
        Xs, ys = subsample(X, y, ADA_SAMPLE)
        m = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth), n_estimators=max(ADA_ROUNDS),
                               random_state=SEED).fit(Xs, ys)
        secs = time.time() - t0
        n_fit = len(m.estimators_)          # AdaBoost stops early if a round fits perfectly or gets no better than chance
        staged = list(m.staged_predict_proba(Xv))
        out, imp, wsum = [], np.zeros(len(FEATURES)), 0.0
        weights = m.estimator_weights_
        cum_imp = []
        for k in range(n_fit):
            imp = imp + weights[k] * m.estimators_[k].feature_importances_
            wsum += weights[k]
            cum_imp.append(imp / wsum if wsum > 0 else imp)
        for r in ADA_ROUNDS:
            k = min(r, n_fit)
            out.append(({"depth": depth, "rounds": r}, task_output(task, staged[k - 1], m.classes_),
                        importances(cum_imp[k - 1], "weighted impurity"), secs * k / max(ADA_ROUNDS)))
        return m, out

    if family == "xgb":
        # XGBoost needs class labels 0..K-1, so faults 1..20 are shifted down by one and back up afterwards.
        classes = np.unique(y)
        y_enc = np.searchsorted(classes, y)
        m = XGBClassifier(n_estimators=max(XGB_ROUNDS), max_depth=depth, learning_rate=LEARNING_RATE,
                          tree_method="hist", n_jobs=-1, random_state=SEED,
                          eval_metric="logloss" if task == "detection" else "mlogloss").fit(X, y_enc)
        secs = time.time() - t0
        gain = m.get_booster().get_score(importance_type="gain")
        imp = importances([gain.get(f"f{j}", 0.0) for j in range(len(FEATURES))], "gain")
        out = []
        for r in XGB_ROUNDS:
            p = m.predict_proba(Xv, iteration_range=(0, r))
            out.append(({"depth": depth, "rounds": r}, task_output(task, p, classes), imp, secs * r / max(XGB_ROUNDS)))
        return m, out

    raise ValueError(family)


def family_depths(family):
    return {"tree": TREE_DEPTHS, "rf": RF_DEPTHS, "gb": GB_DEPTHS, "ada": ADA_DEPTHS, "xgb": XGB_DEPTHS}[family]


def checkpoint_params(family, depth):
    if family == "tree":
        return [{"depth": depth}]
    key, values = {"rf": ("trees", RF_TREES), "gb": ("rounds", GB_ROUNDS),
                   "ada": ("rounds", ADA_ROUNDS), "xgb": ("rounds", XGB_ROUNDS)}[family]
    return [{"depth": depth, key: v} for v in values]


def build_tree_model(task, family, params):
    """A model with these exact settings, for the final test (retrained at that size)."""
    d = params["depth"]
    if family == "tree":
        return DecisionTreeClassifier(max_depth=d, random_state=SEED)
    if family == "rf":
        return RandomForestClassifier(n_estimators=params["trees"], max_depth=d, max_samples=RF_SAMPLE,
                                      n_jobs=-1, random_state=SEED)
    if family == "gb":
        return HistGradientBoostingClassifier(max_iter=params["rounds"], max_depth=d, learning_rate=LEARNING_RATE,
                                              early_stopping=False, random_state=SEED)
    if family == "ada":
        return AdaBoostClassifier(DecisionTreeClassifier(max_depth=d), n_estimators=params["rounds"], random_state=SEED)
    if family == "xgb":
        return XGBClassifier(n_estimators=params["rounds"], max_depth=d, learning_rate=LEARNING_RATE,
                             tree_method="hist", n_jobs=-1, random_state=SEED,
                             eval_metric="logloss" if task == "detection" else "mlogloss")
    raise ValueError(family)


def describe(summary):
    """Fill in family/params for summaries written before families existed (older logistic-regression files)."""
    if "family" not in summary:
        summary["family"] = "baseline" if summary.get("penalty") == "baseline" else "logreg"
        summary["params"] = {} if summary["family"] == "baseline" else {
            "penalty": summary["penalty"], "C": summary.get("C"), "class_weight": summary.get("class_weight", "none")}
    return summary

# ====================================================================================================
# ## 6. Evaluate and export one model
# Shared by every family, so all models are measured exactly the same way.

def export_detection(mid, label, family, params, scores, fit_seconds, explain, baseline=False, extra=None):
    # The baseline always predicts "fault"; every other model is set to 5% false alarms on validation.
    threshold = 0.5 if baseline else threshold_at_far(y_val, scores, SELECTION_FAR)
    metrics = detection_metrics(y_val, scores, threshold)
    delays = {}
    for far in FAR_LEVELS:
        t = threshold_at_far(y_val, scores, far)
        delays[str(far)] = {"threshold": t, "by_fault": delays_to_json(streak_delays(
            val_df["run_id"].to_numpy(), faults_val, val_df["sample"].to_numpy(), scores >= t, FAULT_ONSET["train"]))}
    summary = {"id": mid, "family": family, "label": label, "params": params, "fit_seconds": round(fit_seconds, 1),
               "val": metrics, "coef_nonzero": (explain or {}).get("nonzero"), **(extra or {})}
    payload = {"summary": summary, "val": {**score_histograms(y_val, scores, faults_val), "delays": delays}}
    payload["coefficients" if family in ("logreg", "baseline") else "importances"] = explain
    write_json(WEB_DIR / "detection" / f"{mid}.json", payload)
    print(f"{mid:45s} recall@5%FAR={metrics['recall']:.3f}  ({fit_seconds:.0f}s)", flush=True)
    return clean(summary)


def export_diagnosis(mid, label, family, params, pred, fit_seconds, explain, extra=None):
    true = faults_val[val_post]
    metrics, cm, recall = diagnosis_metrics(true, pred)
    delays = delays_to_json(streak_delays(val_df["run_id"].to_numpy()[val_post], true,
                                          val_df["sample"].to_numpy()[val_post], pred == true, FAULT_ONSET["train"]))
    summary = {"id": mid, "family": family, "label": label, "params": params, "fit_seconds": round(fit_seconds, 1),
               "val": metrics, "coef_nonzero": (explain or {}).get("nonzero"), **(extra or {})}
    payload = {"summary": summary, "val": {"labels": FAULTS, "confusion_matrix": cm, "recall": recall, "delays": delays}}
    payload["coefficients" if family in ("logreg", "baseline") else "importances"] = explain
    write_json(WEB_DIR / "diagnosis" / f"{mid}.json", payload)
    print(f"{mid:45s} macro-F1={metrics['macro_F1']:.3f}  ({fit_seconds:.0f}s)", flush=True)
    return clean(summary)


def load_existing(task, mid):
    path = WEB_DIR / task / f"{mid}.json"
    if path.exists() and not FORCE_RETRAIN:
        return describe(json.loads(path.read_text())["summary"])
    return None

# ====================================================================================================
# ## 7. Train and evaluate every model
# Baseline and logistic regression first, then the tree-based families. Each task uses its own training rows:
# detection all rows, diagnosis post-injection rows only.

TASK_DATA = {
    "detection": (X_det, y_det, X_val),
    "diagnosis": (X_diag, y_diag, X_val[val_post]),
}


def run_linear(task):
    """Majority baseline plus every logistic-regression variant for one task."""
    X, y, Xv = TASK_DATA[task]
    configs = [("baseline", None, None)]
    weights = CLASS_WEIGHTS if task == "detection" else [None]
    if "logreg" in FAMILIES:
        for cw in weights:
            configs.append(("none", None, cw))
            configs += [(p, C, cw) for p in ("l1", "l2") for C in C_GRID]
    summaries = []
    for penalty, C, cw in configs:
        mid = model_id(task, penalty, C, cw)
        existing = load_existing(task, mid)
        if existing:
            summaries.append(existing)
            continue
        model = (DummyClassifier(strategy="most_frequent") if penalty == "baseline"
                 else make_lr(penalty, C, cw, multiclass=(task == "diagnosis")))
        t0 = time.time()
        model.fit(X, y)
        secs = time.time() - t0
        joblib.dump(model, MODEL_DIR / f"{mid}.joblib")
        family = "baseline" if penalty == "baseline" else "logreg"
        params = {} if family == "baseline" else {"penalty": penalty, "C": C, "class_weight": cw or "none"}
        extra = {"penalty": penalty, "C": C, **({"class_weight": cw or "none"} if task == "detection" else {})}
        label = model_label(penalty, C, cw)
        if task == "detection":
            s = export_detection(mid, label, family, params, model.predict_proba(Xv)[:, 1].astype("float64"), secs,
                                 coefficients(model), baseline=(penalty == "baseline"), extra=extra)
        else:
            s = export_diagnosis(mid, label, family, params, model.predict(Xv), secs, coefficients(model), extra=extra)
        summaries.append(describe(s))
    return summaries


def run_trees(task, family):
    X, y, Xv = TASK_DATA[task]
    summaries = []
    for depth in family_depths(family):
        ids = [tree_id(task, family, p) for p in checkpoint_params(family, depth)]
        existing = [load_existing(task, mid) for mid in ids]
        if all(existing):
            summaries += existing
            continue
        model, checkpoints = fit_family(task, family, depth, X, y, Xv)
        joblib.dump(model, MODEL_DIR / f"{task}_{family}_d{depth}.joblib")
        for params, output, explain, secs in checkpoints:
            mid, label = tree_id(task, family, params), tree_label(family, params)
            if task == "detection":
                summaries.append(export_detection(mid, label, family, params, output, secs, explain))
            else:
                summaries.append(export_diagnosis(mid, label, family, params, output, secs, explain))
    return summaries


def run_task(task):
    summaries = run_linear(task)
    for family in FAMILIES:
        if family != "logreg":
            t0 = time.time()
            summaries += run_trees(task, family)
            print(f"--- {task} {FAMILY_INFO[family]['label']}: {time.time() - t0:.0f}s", flush=True)
    return summaries


def summary_table(summaries, metric_keys):
    rows = [{"id": s["id"], "family": s["family"], "sensors": s.get("coef_nonzero"),
             **{k: s["val"][k] for k in metric_keys}} for s in summaries]
    return pd.DataFrame(rows).set_index("id")


det_summaries = run_task("detection")
det_table = summary_table(det_summaries, ["recall", "precision", "F1", "ROC_AUC", "normal_recall"])

# ### Choosing the best detection model
# All models are compared at the same operating point (5% false alarms on validation normal rows), so the best
# model is the one that catches the most faults there: highest validation recall.
candidates = det_table[det_table["family"] != "baseline"]
best_det_id = candidates["recall"].idxmax()
best_det = next(s for s in det_summaries if s["id"] == best_det_id)
print("\nBest detection model per family (recall at 5% false alarms):")
print(candidates.loc[candidates.groupby("family")["recall"].idxmax()].sort_values("recall", ascending=False).round(4))
print("Best detection model:", best_det["label"])

diag_summaries = run_task("diagnosis")
diag_table = summary_table(diag_summaries, ["macro_F1", "accuracy"])

# The best diagnosis model has the highest validation macro-F1.
diag_candidates = diag_table[diag_table["family"] != "baseline"]
best_diag_id = diag_candidates["macro_F1"].idxmax()
best_diag = next(s for s in diag_summaries if s["id"] == best_diag_id)
print("\nBest diagnosis model per family (macro-F1):")
print(diag_candidates.loc[diag_candidates.groupby("family")["macro_F1"].idxmax()].sort_values("macro_F1", ascending=False).round(4))
print("Best diagnosis model:", best_diag["label"])

# ====================================================================================================
# ## 8. Manifest
# The website reads this first. The `test` section is filled in by section 9.

def best_by_family(summaries, key):
    out = {}
    for s in summaries:
        f = s["family"]
        if f != "baseline" and (f not in out or s["val"][key] > out[f][1]):
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
        "settings": {"C_grid": C_GRID, "class_weights": [c or "none" for c in CLASS_WEIGHTS],
                     "selection_far": SELECTION_FAR, "far_levels": FAR_LEVELS, "consecutive": CONSECUTIVE,
                     "det_stride": DET_STRIDE, "diag_stride": DIAG_STRIDE, "rf_sample": RF_SAMPLE,
                     "ada_sample": ADA_SAMPLE, "learning_rate": LEARNING_RATE},
        "families": [{"key": f, **FAMILY_INFO[f]} for f in present],
        "detection": {"models": det_summaries, "best": best_det_id,
                      "best_by_family": best_by_family(det_summaries, "recall"),
                      "selection_rule": f"highest validation recall at {SELECTION_FAR:.0%} false alarms"},
        "diagnosis": {"models": diag_summaries, "best": best_diag_id,
                      "best_by_family": best_by_family(diag_summaries, "macro_F1"),
                      "selection_rule": "highest validation macro-F1"},
        "test": test_results,
    })


manifest_path = WEB_DIR / "manifest.json"
previous_test = json.loads(manifest_path.read_text()).get("test") if manifest_path.exists() else None
write_manifest(previous_test)
print("Wrote", manifest_path)

# ====================================================================================================
# ## 9. Final test evaluation (run once, after model selection)
# 1. Refit the scaler and retrain the chosen detection and diagnosis models on all training runs (train + val).
# 2. Keep the **validation** threshold for detection; nothing is tuned on test.
# 3. Evaluate once on the test files (fault injected after sample 160) and export the results.
#
# The majority baselines are evaluated too, for comparison.

def build_final(task, summary):
    family, p = summary["family"], summary["params"]
    if family == "baseline":
        return DummyClassifier(strategy="most_frequent")
    if family == "logreg":
        cw = None if p.get("class_weight", "none") == "none" else p["class_weight"]
        return make_lr(p["penalty"], p["C"], cw, multiclass=(task == "diagnosis"))
    return build_tree_model(task, family, p)


def fit_final(task, summary, X, y):
    model = build_final(task, summary)
    if summary["family"] == "ada":
        X, y = subsample(X, y, ADA_SAMPLE)
    classes = np.unique(y)
    if summary["family"] == "xgb":
        model.fit(X, np.searchsorted(classes, y))
    else:
        model.fit(X, y)
    joblib.dump(model, MODEL_DIR / f"final_{summary['id']}.joblib")
    return model, classes


def final_explain(summary, model):
    if summary["family"] in ("logreg", "baseline"):
        return coefficients(model)
    if hasattr(model, "feature_importances_") and summary["family"] != "gb":
        return importances(model.feature_importances_, "impurity" if summary["family"] != "xgb" else "gain")
    return None


if args.final_test:
    full_df = pd.concat([train_df, val_df], ignore_index=True)
    final_scaler = StandardScaler().fit(full_df[FEATURES].to_numpy())

    test_normal = load_tep("TEP_FaultFree_Testing")
    test_faulty = load_tep("TEP_Faulty_Testing")
    test_faulty = test_faulty[test_faulty["simulationRun"] <= TEST_RUNS_PER_FAULT]
    test_df = prepare(test_normal, test_faulty, FAULT_ONSET["test"])
    del test_normal, test_faulty
    X_test = final_scaler.transform(test_df[FEATURES].to_numpy())
    y_test = test_df["target"].to_numpy()
    faults_test = test_df["faultNumber"].to_numpy()
    runs_test, samples_test = test_df["run_id"].to_numpy(), test_df["sample"].to_numpy()
    test_post = y_test == 1
    test_results = {"rows": int(len(test_df)), "fault_share": float(y_test.mean()),
                    "runs_per_fault": TEST_RUNS_PER_FAULT, "detection": {}, "diagnosis": {}}

    # ---- Detection ----
    rows = (full_df["sample"] % DET_STRIDE == 0).to_numpy()
    X_full = final_scaler.transform(full_df.loc[rows, FEATURES].to_numpy())
    y_full = full_df.loc[rows, "target"].to_numpy()
    for summary in (best_det, next(s for s in det_summaries if s["family"] == "baseline")):
        model, classes = fit_final("detection", summary, X_full, y_full)
        scores = model.predict_proba(X_test)[:, 1].astype("float64")
        threshold = summary["val"]["threshold"]                     # chosen on validation
        metrics = detection_metrics(y_test, scores, threshold)
        delays = delays_to_json(streak_delays(runs_test, faults_test, samples_test,
                                              scores >= threshold, FAULT_ONSET["test"]))
        test_results["detection"][summary["id"]] = metrics
        explain = final_explain(summary, model)
        payload = {"summary": {**summary, "test": metrics},
                   "test": {**score_histograms(y_test, scores, faults_test),
                            "delays": {"validation_threshold": {"threshold": threshold, "by_fault": delays}}}}
        payload["coefficients" if summary["family"] in ("logreg", "baseline") else "importances"] = explain
        write_json(WEB_DIR / "detection" / f"test_{summary['id']}.json", payload)
        print(f"TEST {summary['label']:40s}", {k: round(v, 4) for k, v in metrics.items()})

    # ---- Diagnosis ----
    rows = ((full_df["target"] == 1) & (full_df["sample"] % DIAG_STRIDE == 0)).to_numpy()
    X_full = final_scaler.transform(full_df.loc[rows, FEATURES].to_numpy())
    y_full = full_df.loc[rows, "faultNumber"].to_numpy()
    for summary in (best_diag, next(s for s in diag_summaries if s["family"] == "baseline")):
        model, classes = fit_final("diagnosis", summary, X_full, y_full)
        pred = model.predict(X_test[test_post])
        if summary["family"] == "xgb":
            pred = classes[pred.astype(int)]
        true = faults_test[test_post]
        metrics, cm, recall = diagnosis_metrics(true, pred)
        delays = delays_to_json(streak_delays(runs_test[test_post], true, samples_test[test_post],
                                              pred == true, FAULT_ONSET["test"]))
        test_results["diagnosis"][summary["id"]] = metrics
        explain = final_explain(summary, model)
        payload = {"summary": {**summary, "test": metrics},
                   "test": {"labels": FAULTS, "confusion_matrix": cm, "recall": recall, "delays": delays}}
        payload["coefficients" if summary["family"] in ("logreg", "baseline") else "importances"] = explain
        write_json(WEB_DIR / "diagnosis" / f"test_{summary['id']}.json", payload)
        print(f"TEST {summary['label']:40s}", {k: round(v, 4) for k, v in metrics.items()})

    write_manifest(test_results)
    print("Manifest updated with test results.")
