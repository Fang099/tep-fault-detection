"""Write the published results as readable CSV tables, and the exact train/validation split.

Reads only the website's result files in docs/data (written by tep_pipeline.py), so it needs no dataset and no
training. Run from the repository root:

    python scripts/make_tables.py

Outputs
  results/detection_validation.csv   every detection model: settings and validation scores at 5% false alarms
  results/diagnosis_validation.csv   every diagnosis model: settings and validation scores
  results/final_test.csv             the best models (and the majority baseline) on the test files
  results/per_fault.csv              per-fault results of the best models, validation and test
  results/end_to_end.csv             detection and diagnosis together (21 answers)
  results/learning_curve.csv         the best models retrained on fewer runs / sparser samples
  results/split_demo.csv             split by run vs split at random (extra experiment)
  splits/train_simulation_runs.txt, splits/validation_simulation_runs.txt
"""
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "docs" / "data"
OUT = ROOT / "results"
SPLITS = ROOT / "splits"
FAR = 0.05


def load(*parts):
    return json.loads(DATA.joinpath(*parts).read_text())


def write(path, rows):
    path.parent.mkdir(exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"{path.relative_to(ROOT)}: {len(rows)} rows")


def r(x, d=6):
    return None if x is None else round(float(x), d)


def settings(m):
    return "; ".join(f"{k}={v}" for k, v in (m.get("params") or {}).items())


def model_rows(manifest, task):
    best = manifest[task]["best"]
    rows = []
    for m in manifest[task]["models"]:
        v, ci = m["val"], m.get("ci") or {}
        row = {"id": m["id"], "family": m.get("family", "logreg"), "label": m["label"], "settings": settings(m),
               "inputs": m.get("features", "inst"), "n_inputs": m.get("n_inputs"),
               "eligible_for_best": m.get("selectable", True) is not False and m.get("family") != "svm",
               "chosen_as_best": m["id"] == best}
        if task == "detection":
            row.update({"threshold": r(v.get("threshold")), "recall": r(v["recall"]), "false_alarm_rate": r(1 - v["normal_recall"]),
                        "precision": r(v["precision"]), "F1": r(v["F1"]), "ROC_AUC": r(v.get("ROC_AUC")), "PR_AUC": r(v.get("PR_AUC")),
                        "recall_ci_low": r((ci.get("recall") or [None, None])[0]), "recall_ci_high": r((ci.get("recall") or [None, None])[1]),
                        "TN": v["TN"], "FP": v["FP"], "FN": v["FN"], "TP": v["TP"]})
        else:
            row.update({"macro_F1": r(v["macro_F1"]), "accuracy": r(v["accuracy"]),
                        "macro_F1_ci_low": r((ci.get("macro_F1") or [None, None])[0]), "macro_F1_ci_high": r((ci.get("macro_F1") or [None, None])[1])})
        row.update({"inputs_used": m.get("coef_nonzero"), "fit_seconds": m.get("fit_seconds")})
        rows.append(row)
    if task == "detection":   # models at the 5% false alarm rate first, best recall first; the baseline (100% alarms) last
        return sorted(rows, key=lambda x: (x["false_alarm_rate"] > 2 * FAR, -x["recall"]))
    return sorted(rows, key=lambda x: -x["macro_F1"])


def per_fault_detection(split):
    """Share of each fault's rows caught at the given split's stored threshold (exact: thresholds are bin edges)."""
    edges = np.array(split["bin_edges"])
    k = int(np.argmax(edges >= split["threshold"] - 1e-12)) if "threshold" in split else None
    return {int(f): float(np.sum(c[k:]) / np.sum(c)) for f, c in split["fault_counts"].items()}, k


def main():
    manifest = load("manifest.json")
    write(OUT / "detection_validation.csv", model_rows(manifest, "detection"))
    write(OUT / "diagnosis_validation.csv", model_rows(manifest, "diagnosis"))

    # Final test
    t, rows = manifest.get("test"), []
    if t:
        for task, key in [("detection", "recall"), ("diagnosis", "macro_F1")]:
            val = {m["id"]: m["val"] for m in manifest[task]["models"]}
            for mid, res in t[task].items():
                for version, vals in [("retrained on train+validation", res), ("as validated", t.get("as_validated", {}).get(task, {}).get(mid))]:
                    if not vals:
                        continue
                    row = {"task": task, "model": mid, "version": version}
                    if task == "detection":
                        row.update({"test_recall": r(vals["recall"]), "test_false_alarm_rate": r(1 - vals["normal_recall"]),
                                    "test_ROC_AUC": r(vals.get("ROC_AUC")), "test_PR_AUC": r(vals.get("PR_AUC")), "test_precision": r(vals.get("precision")),
                                    "val_recall": r(val[mid]["recall"]), "val_false_alarm_rate": r(1 - val[mid]["normal_recall"]), "val_ROC_AUC": r(val[mid].get("ROC_AUC")),
                                    "threshold_from_validation": r(val[mid].get("threshold"))})
                    else:
                        row.update({"test_macro_F1": r(vals["macro_F1"]), "test_accuracy": r(vals["accuracy"]),
                                    "val_macro_F1": r(val[mid]["macro_F1"]), "val_accuracy": r(val[mid]["accuracy"])})
                    rows.append(row)
        # Like for like: test recall at the same 5% false alarm rate, read from the stored test scores.
        best = manifest["detection"]["best"]
        tf = DATA / "detection" / f"test_{best}.json"
        if tf.exists():
            s = json.loads(tf.read_text())["test"]
            n = np.array(s["normal_counts"])
            f = np.sum([np.array(c) for c in s["fault_counts"].values()], axis=0)
            sn, sf = np.cumsum(n[::-1])[::-1] / n.sum(), np.cumsum(f[::-1])[::-1] / f.sum()
            i = int(np.argmax(sn <= FAR + 1e-12))
            rows.append({"task": "detection", "model": best, "version": "retrained, threshold set on test at 5% false alarms (like for like, not used for any choice)",
                         "test_recall": r(sf[i]), "test_false_alarm_rate": r(sn[i])})
        write(OUT / "final_test.csv", rows)

    # Per fault, best models
    rows = []
    bd, bg = manifest["detection"]["best"], manifest["diagnosis"]["best"]
    vd = load("detection", f"{bd}.json")["val"]
    thr = next(m for m in manifest["detection"]["models"] if m["id"] == bd)["val"]["threshold"]
    vd["threshold"] = thr
    det_val, _ = per_fault_detection(vd)
    det_test = {}
    tf = DATA / "detection" / f"test_{bd}.json"
    if tf.exists():
        ts = json.loads(tf.read_text())["test"]
        ts["threshold"] = thr
        det_test, _ = per_fault_detection(ts)
    dg = load("diagnosis", f"{bg}.json")["val"]
    cm, labels = np.array(dg["confusion_matrix"]), dg["labels"]
    for i, f in enumerate(labels):
        rowcm = cm[i].copy()
        right = rowcm[i] / rowcm.sum()
        rowcm[i] = 0
        j = int(np.argmax(rowcm))
        rows.append({"fault": f, "detection_recall_val": r(det_val.get(int(f))), "detection_recall_test": r(det_test.get(int(f))),
                     "diagnosis_recall_val": r(right), "most_common_wrong_name_val": labels[j], "share_given_that_name": r(rowcm[j] / cm[i].sum())})
    write(OUT / "per_fault.csv", rows)

    # End to end
    e = load("end_to_end.json")
    rows = []
    for split in ("val", "test"):
        for name, v in e.get(split, {}).items():
            rows.append({"split": split, "system": name, **{k: r(v[k]) for k in ("macro_F1", "accuracy", "false_alarm_rate", "faults_caught", "right_fault") if k in v}})
    write(OUT / "end_to_end.csv", rows)

    # Learning curve
    lc = load("learning_curve.json")
    rows = []
    for task in ("detection", "diagnosis"):
        for axis, label in (("runs", "share_of_training_runs"), ("stride", "every_kth_sample")):
            for k, v in lc[task][axis].items():
                rows.append({"task": task, "model": lc[task]["model"], "varied": label, "value": k, "training_rows": v.get("rows"),
                             "metric": "recall at 5% false alarms" if task == "detection" else "macro_F1", "score": r(v["metric"]),
                             "ci_low": r(v["ci"][0]), "ci_high": r(v["ci"][1])})
    write(OUT / "learning_curve.csv", rows)

    # Split demonstration
    sd = load("split_demo.json")
    rows = []
    for task in ("detection", "diagnosis"):
        for mk, label in sd["models"].items():
            for split in ("run", "random"):
                v = sd[task][mk][split]
                rows.append({"task": task, "model": label, "validation_split": "by run (unseen runs)" if split == "run" else "random rows",
                             **{k: r(x) for k, x in v.items() if k not in ("val_rows", "fit_seconds")}, "validation_rows": v["val_rows"],
                             "validation_rows_identical_to_a_training_row": sd["identical_val_rows"][split]})
    write(OUT / "split_demo.csv", rows)

    # The exact split, as tep_pipeline.py makes it (section 2): 20% of the simulation numbers 1-500, seed 42.
    sims = np.arange(1, 501)
    _, val = train_test_split(sims, test_size=0.2, random_state=42)
    val = np.sort(val)
    train = np.setdiff1d(sims, val)
    SPLITS.mkdir(exist_ok=True)
    head = ("# Simulation numbers (column simulationRun) in the {} set. The same numbers are used for the fault-free run and\n"
            "# all 20 faulty runs, so each line stands for 21 runs. Made exactly as in tep_pipeline.py:\n"
            "# train_test_split(simulation numbers 1-500, test_size=0.2, random_state=42). Test files are a separate download.\n")
    (SPLITS / "train_simulation_runs.txt").write_text(head.format("training") + "\n".join(map(str, train)) + "\n")
    (SPLITS / "validation_simulation_runs.txt").write_text(head.format("validation") + "\n".join(map(str, val)) + "\n")
    print(f"splits/: {len(train)} training and {len(val)} validation simulation numbers")


if __name__ == "__main__":
    main()
