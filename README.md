# Tennessee Eastman Process: Fault Detection and Diagnosis

How effective are different machine learning models at detecting and diagnosing faults in the Tennessee
Eastman Process (TEP) from its 52 process measurements and their recent history, and how quickly can faults be
detected?

Website: https://fang099.github.io/tep-fault-detection/

- **Detection:** is a fault active right now? (normal vs faulty)
- **Diagnosis:** which of the 20 fault types is active?

The project website (in `docs/`) lets you compare models, move the decision threshold and inspect each fault type.

## Data

Kaggle dataset [`afrniomelo/tep-csv`](https://www.kaggle.com/datasets/afrniomelo/tep-csv) (Rieth et al., 2017;
originally published on Harvard Dataverse). It is not stored in this repo (about 15 million rows); downloading it
needs a free Kaggle account, and its licence is stated on the dataset page. The code only reads the CSV files from a
folder: no Kaggle key is needed, and none should be committed. Download these four files into one folder:

- `TEP_FaultFree_Training.csv`, `TEP_Faulty_Training.csv`
- `TEP_FaultFree_Testing.csv`, `TEP_Faulty_Testing.csv`

One row is one 3-minute time step of one simulation run. The models use only the 52 process variables
(`xmeas_1`–`xmeas_41`, `xmv_1`–`xmv_11`).

## Method

- **Labels:** fault-free rows and rows before the fault is injected (after sample 20 in training, sample 160 in
  testing) are normal; rows after injection are faulty.
- **Split:** the training files are split by simulation number, 80% train / 20% validation, at random with seed 42
  (8,400 training runs and 2,100 validation runs when all the data is used). Faulty run *k* of every
  fault type starts as an exact copy of fault-free run *k* (identical readings until the fault shows), so all runs
  with the same simulation number go to the same side. The pipeline checks that no validation row is identical to a
  training row. The test files are used once, at the end.
- **Models:** majority-class baselines; logistic regression with no penalty, lasso (L1) and ridge (L2) at several
  strengths `C`, with and without class weighting (detection); decision trees (depth); random forests (depth, number
  of trees); gradient boosting, AdaBoost and XGBoost (tree depth, boosting rounds). Forests and boosted models are
  trained once per depth; smaller forests and fewer rounds are read off the same fit. PCA + logistic regression
  (number of components) and PCA process monitoring (Hotelling's T² and SPE, fitted on fault-free runs only).
  An RBF-kernel SVM (training-subset size, C) is trained on small random subsets only, since an exact kernel SVM
  can't handle millions of rows; it is shown for comparison and never chosen as the best model.
  Every model trains on every time step of every training run (random forests and AdaBoost: a random 25% per
  tree/model).
- **Temporal features (ablation):** selected families are also trained with a trailing 30-minute rolling mean and
  standard deviation of every reading, computed within each run from past samples only. A window sweep then
  retrains the best 30-minute settings of each of these families with 6-minute, 15-minute, 1-hour and 2-hour
  windows, to show how much history helps and how much later longer windows react.
- **Detection and diagnosis together:** the best detection and diagnosis models in a chain, and one model with
  21 answers (no fault, faults 1–20) trained with the best diagnosis settings, both at 5% false alarms.
- **How much data matters:** the best models retrained on 5–100% of the training runs and on 1-in-20 to every
  sample, scored on the same validation runs.
- **Uncertainty:** 95% intervals from a run-level bootstrap (whole validation runs resampled within each fault type).
- **Thresholds and model choice:** each detection model's threshold is set so 5% of normal validation rows raise
  an alarm; the best model catches the most faults at that point. The best diagnosis model has the highest
  validation macro-F1.
- **Final test:** the chosen models are retrained on all training runs and evaluated once on all test runs,
  keeping the validation thresholds; the exact validated models are evaluated too, as a check on the threshold.
- **Metrics:** precision, recall, F1, PR-AUC, ROC-AUC and confusion matrices for detection; macro-F1 and
  per-fault recall for diagnosis; median time from fault injection to detection (k alarms in a row) and to
  correct diagnosis (k correct in a row), for k = 1, 2, 3, 5, 10 (3 is the headline).

## Reproducing the results

**Google Colab:** open `run_colab.ipynb` from this repo in Colab (File → Open notebook → GitHub), set
`DATA_DIR` to the Drive folder holding the CSVs, and run the cells in order.

**Any computer** (recommended for the full dataset, which needs 16 GB of RAM):

```bash
pip install -r requirements.txt
python tep_pipeline.py --data-dir path/to/csvs --runs-per-fault 100      # subset, faster
python tep_pipeline.py --data-dir path/to/csvs --families tree,rf          # only some model families
python tep_pipeline.py --data-dir path/to/csvs --history-families ""     # no history inputs
python tep_pipeline.py --data-dir path/to/csvs --hist-windows ""         # skip the history window sweep
python tep_pipeline.py --data-dir path/to/csvs --runs-per-fault 0        # all 500 runs per fault
python tep_pipeline.py --data-dir path/to/csvs --runs-per-fault 0 --final-test
```

Run `python tep_pipeline.py --help` for all options. Results go to `outputs_<N>runs/` (or `outputs_full/`)
inside the data folder: trained models in `models_v3/`, website data in `web_data/`. Finished models are skipped
on re-runs, so an interrupted run can be restarted.

## Known limitations

The full list is on the website's Results page. The most important ones:

- **The first hour of a run.** Every training fault starts after exactly one hour, so the models learn that the
  start of a run is always fault-free. Almost half of the normal validation rows come from that first hour, so
  the 5% false alarm rate set on validation is an average of nearly 0% there and more afterwards; on the test
  files, with the threshold unchanged, it is 7.4%. At an equal 5% false alarm rate the test recall matches
  validation.
- **5% false alarms** is a yardstick for comparing models, not an operating point for a plant.
- **Simulation only**, with 20 known fault types; nothing here shows how a new kind of fault or a real plant
  would be handled.
- Package versions of the published run were not recorded; `requirements.txt` gives minimum versions.

## Repository layout

| Path | Contents |
|---|---|
| `tep_pipeline.py` | Data loading, training, evaluation and website export |
| `run_colab.ipynb` | Short notebook that runs the pipeline on Colab |
| `docs/` | Project website (GitHub Pages): plain HTML, CSS and JavaScript |
| `docs/data/` | Exported results the website reads (copied from `web_data/`) |
| `requirements.txt` | Python packages the pipeline needs |
| `LICENSE` | MIT licence for the code in this repository (not for the dataset) |

To preview the website locally: `cd docs && python -m http.server`, then open http://localhost:8000.
