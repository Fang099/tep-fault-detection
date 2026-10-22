"""Tennessee Eastman Process: fault detection and diagnosis pipeline.

Trains a grid of logistic-regression variants for two tasks, picks the best on validation, optionally runs
the one-time final test, and exports JSON files for the project website.

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

Logistic-regression variants: class weighting none/balanced (detection only), penalty none/L1 (lasso)/L2
(ridge), and regularization strength C (smaller C = stronger). scikit-learn's default LogisticRegression()
is already L2 with C=1; the "no penalty" models here are truly unregularized.

Usage
  python tep_pipeline.py --data-dir /path/to/csvs --runs-per-fault 100
  python tep_pipeline.py --data-dir /path/to/csvs --runs-per-fault 0 --final-test   # all runs + test

Results are written to <out-dir>/outputs_<N>runs (or outputs_full): models/ and web_data/ (website files).
Finished grid models are skipped on re-runs, so an interrupted run can simply be restarted.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 100)

# ---- Command-line settings -----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="TEP fault detection and diagnosis pipeline")
parser.add_argument("--data-dir", default="dataset",
                    help="folder with the 4 TEP CSVs; the Parquet cache is saved here too")
parser.add_argument("--work-dir", default=None,
                    help="optional fast local folder to copy the Parquet cache into (e.g. /content/dataset on Colab)")
parser.add_argument("--out-dir", default=None, help="where results go (default: --data-dir)")
parser.add_argument("--runs-per-fault", type=int, default=100,
                    help="use the first N runs of each fault type for train/val; 0 = all 500")
parser.add_argument("--det-stride", type=int, default=2, help="detection trains on every k-th time step")
parser.add_argument("--diag-stride", type=int, default=5, help="diagnosis trains on every k-th time step")
parser.add_argument("--selection-far", type=float, default=0.05,
                    help="false-alarm rate on validation normal rows used to set thresholds and pick the best model")
parser.add_argument("--test-runs-per-fault", type=int, default=100,
                    help="faulty test runs used per fault (the full faulty test file is ~10M rows)")
parser.add_argument("--final-test", action="store_true", help="retrain the chosen models and evaluate on test")
parser.add_argument("--force-retrain", action="store_true", help="retrain grid models even if already exported")
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

print(f"Data: {DRIVE_DIR}  Results: {OUT_DIR}  Runs per fault: {RUNS_PER_FAULT or 'all'}")

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
# Both include the model's coefficients (on standardized inputs), so the site can show which of the 52
# variables each model relies on and how lasso drives coefficients to exactly zero.
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


def write_json(path, payload):
    path.write_text(json.dumps(clean(payload), separators=(",", ":")))

# ====================================================================================================
# ## 6. Detection: train and evaluate every variant

def detection_grid():
    configs = [("baseline", None, None)]
    for cw in CLASS_WEIGHTS:
        configs.append(("none", None, cw))
        configs += [(p, C, cw) for p in ("l1", "l2") for C in C_GRID]
    return configs


def run_detection(penalty, C, class_weight):
    mid = model_id("detection", penalty, C, class_weight)
    path = WEB_DIR / "detection" / f"{mid}.json"
    if path.exists() and not FORCE_RETRAIN:
        return json.loads(path.read_text())["summary"]

    model = (DummyClassifier(strategy="most_frequent") if penalty == "baseline"
             else make_lr(penalty, C, class_weight))
    t0 = time.time()
    model.fit(X_det, y_det)
    fit_seconds = time.time() - t0
    joblib.dump(model, MODEL_DIR / f"{mid}.joblib")

    scores = model.predict_proba(X_val)[:, 1].astype("float64")
    # The baseline always predicts "fault"; every other model is set to 5% false alarms on validation.
    threshold = 0.5 if penalty == "baseline" else threshold_at_far(y_val, scores, SELECTION_FAR)
    metrics = detection_metrics(y_val, scores, threshold)
    delays = {}
    for far in FAR_LEVELS:
        t = threshold_at_far(y_val, scores, far)
        delays[str(far)] = {"threshold": t, "by_fault": delays_to_json(streak_delays(
            val_df["run_id"].to_numpy(), faults_val, val_df["sample"].to_numpy(),
            scores >= t, FAULT_ONSET["train"]))}

    summary = {"id": mid, "label": model_label(penalty, C, class_weight), "penalty": penalty, "C": C,
               "class_weight": class_weight or "none", "fit_seconds": round(fit_seconds, 1),
               "val": metrics, "coef_nonzero": (coefficients(model) or {}).get("nonzero")}
    write_json(path, {"summary": summary, "coefficients": coefficients(model),
                      "val": {**score_histograms(y_val, scores, faults_val), "delays": delays}})
    print(f"{mid:45s} recall@5%FAR={metrics['recall']:.3f}  ({fit_seconds:.0f}s)")
    return clean(summary)


det_summaries = [run_detection(*cfg) for cfg in detection_grid()]

det_table = pd.DataFrame([{**{k: s[k] for k in ("id", "penalty", "C", "class_weight", "coef_nonzero")}, **s["val"]}
                          for s in det_summaries]).set_index("id")
print(det_table.drop(columns=["TN", "FP", "FN", "TP"]).round(4))

# ====================================================================================================
# ### Choosing the best detection model
# All models are compared at the same operating point (5% false alarms on validation normal rows), so the best
# model is the one that catches the most faults there: highest validation recall. Ties go to the earlier,
# simpler model in the grid.

candidates = det_table[det_table["penalty"] != "baseline"]
best_det_id = candidates["recall"].idxmax()
best_det = next(s for s in det_summaries if s["id"] == best_det_id)
print("Best detection model:", best_det["label"])
print(pd.Series(best_det["val"]).round(4))

best_det_file = json.loads((WEB_DIR / "detection" / f"{best_det_id}.json").read_text())
print("\nPer-fault detection delay at 5% FAR (median minutes, share of runs detected):")
print(pd.DataFrame(best_det_file["val"]["delays"]["0.05"]["by_fault"], index=["median_min", "detected_share"]).T)

# ====================================================================================================
# ## 7. Diagnosis: train and evaluate every variant

def diagnosis_grid():
    return [("baseline", None), ("none", None)] + [(p, C) for p in ("l1", "l2") for C in C_GRID]


def run_diagnosis(penalty, C):
    mid = model_id("diagnosis", penalty, C)
    path = WEB_DIR / "diagnosis" / f"{mid}.json"
    if path.exists() and not FORCE_RETRAIN:
        return json.loads(path.read_text())["summary"]

    model = (DummyClassifier(strategy="most_frequent") if penalty == "baseline"
             else make_lr(penalty, C, multiclass=True))
    t0 = time.time()
    model.fit(X_diag, y_diag)
    fit_seconds = time.time() - t0
    joblib.dump(model, MODEL_DIR / f"{mid}.joblib")

    pred = model.predict(X_val[val_post])
    true = faults_val[val_post]
    metrics, cm, recall = diagnosis_metrics(true, pred)
    delays = delays_to_json(streak_delays(
        val_df["run_id"].to_numpy()[val_post], true, val_df["sample"].to_numpy()[val_post],
        pred == true, FAULT_ONSET["train"]))

    summary = {"id": mid, "label": model_label(penalty, C), "penalty": penalty, "C": C,
               "fit_seconds": round(fit_seconds, 1), "val": metrics,
               "coef_nonzero": (coefficients(model) or {}).get("nonzero")}
    write_json(path, {"summary": summary, "coefficients": coefficients(model),
                      "val": {"labels": FAULTS, "confusion_matrix": cm, "recall": recall, "delays": delays}})
    print(f"{mid:35s} macro-F1={metrics['macro_F1']:.3f}  ({fit_seconds:.0f}s)")
    return clean(summary)


diag_summaries = [run_diagnosis(*cfg) for cfg in diagnosis_grid()]

diag_table = pd.DataFrame([{**{k: s[k] for k in ("id", "penalty", "C", "coef_nonzero")}, **s["val"]}
                           for s in diag_summaries]).set_index("id")
print(diag_table.round(4))

# The best diagnosis model has the highest validation macro-F1.
best_diag_id = diag_table[diag_table["penalty"] != "baseline"]["macro_F1"].idxmax()
best_diag = next(s for s in diag_summaries if s["id"] == best_diag_id)
print("\nBest diagnosis model:", best_diag["label"])

# ====================================================================================================
# ## 8. Manifest
# The website reads this first. The `test` section is filled in by section 9.

def write_manifest(test_results=None):
    write_json(WEB_DIR / "manifest.json", {
        "dataset": {
            "features": FEATURES, "runs_per_fault": RUNS_PER_FAULT or 500,
            "train_runs": int(train_df["run_id"].nunique()), "val_runs": int(val_df["run_id"].nunique()),
            "val_rows": int(len(val_df)), "val_fault_share": float(y_val.mean()),
            "fault_onset": FAULT_ONSET, "sample_minutes": SAMPLE_MINUTES,
        },
        "settings": {"C_grid": C_GRID, "class_weights": [c or "none" for c in CLASS_WEIGHTS],
                     "selection_far": SELECTION_FAR, "far_levels": FAR_LEVELS, "consecutive": CONSECUTIVE,
                     "det_stride": DET_STRIDE, "diag_stride": DIAG_STRIDE},
        "detection": {"models": det_summaries, "best": best_det_id,
                      "selection_rule": f"highest validation recall at {SELECTION_FAR:.0%} false alarms"},
        "diagnosis": {"models": diag_summaries, "best": best_diag_id,
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
    for summary in (best_det, next(s for s in det_summaries if s["penalty"] == "baseline")):
        cw = None if summary["class_weight"] == "none" else summary["class_weight"]
        model = (DummyClassifier(strategy="most_frequent") if summary["penalty"] == "baseline"
                 else make_lr(summary["penalty"], summary["C"], cw))
        model.fit(X_full, y_full)
        joblib.dump(model, MODEL_DIR / f"final_{summary['id']}.joblib")
        scores = model.predict_proba(X_test)[:, 1].astype("float64")
        threshold = summary["val"]["threshold"]                     # chosen on validation
        metrics = detection_metrics(y_test, scores, threshold)
        delays = delays_to_json(streak_delays(runs_test, faults_test, samples_test,
                                              scores >= threshold, FAULT_ONSET["test"]))
        test_results["detection"][summary["id"]] = metrics
        write_json(WEB_DIR / "detection" / f"test_{summary['id']}.json", {
            "summary": {**summary, "test": metrics}, "coefficients": coefficients(model),
            "test": {**score_histograms(y_test, scores, faults_test),
                     "delays": {"validation_threshold": {"threshold": threshold, "by_fault": delays}}}})
        print(f"TEST {summary['label']:40s}", {k: round(v, 4) for k, v in metrics.items()})

    # ---- Diagnosis ----
    rows = ((full_df["target"] == 1) & (full_df["sample"] % DIAG_STRIDE == 0)).to_numpy()
    X_full = final_scaler.transform(full_df.loc[rows, FEATURES].to_numpy())
    y_full = full_df.loc[rows, "faultNumber"].to_numpy()
    for summary in (best_diag, next(s for s in diag_summaries if s["penalty"] == "baseline")):
        model = (DummyClassifier(strategy="most_frequent") if summary["penalty"] == "baseline"
                 else make_lr(summary["penalty"], summary["C"], multiclass=True))
        model.fit(X_full, y_full)
        joblib.dump(model, MODEL_DIR / f"final_{summary['id']}.joblib")
        pred = model.predict(X_test[test_post])
        true = faults_test[test_post]
        metrics, cm, recall = diagnosis_metrics(true, pred)
        delays = delays_to_json(streak_delays(runs_test[test_post], true, samples_test[test_post],
                                              pred == true, FAULT_ONSET["test"]))
        test_results["diagnosis"][summary["id"]] = metrics
        write_json(WEB_DIR / "diagnosis" / f"test_{summary['id']}.json", {
            "summary": {**summary, "test": metrics}, "coefficients": coefficients(model),
            "test": {"labels": FAULTS, "confusion_matrix": cm, "recall": recall, "delays": delays}})
        print(f"TEST {summary['label']:40s}", {k: round(v, 4) for k, v in metrics.items()})

    write_manifest(test_results)
    print("Manifest updated with test results.")