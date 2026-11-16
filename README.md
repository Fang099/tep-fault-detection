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

**Any computer** (needed for the full dataset, which takes more memory than a free Colab session provides).
With the default options, these two commands reproduce everything on the website:

```bash
pip install -r requirements.txt
python tep_pipeline.py --data-dir path/to/csvs                # 1. train and validate every model on all the data
python tep_pipeline.py --data-dir path/to/csvs --final-test   # 2. once, at the end: retrain the chosen models, score test
```

Training every model on all the data took more than a day on a desktop PC. For a quicker check:

```bash
python tep_pipeline.py --data-dir path/to/csvs --runs-per-fault 100      # 100 runs per fault instead of 500
python tep_pipeline.py --data-dir path/to/csvs --families logreg,xgb     # only some model families
python tep_pipeline.py --data-dir path/to/csvs --hist-windows ""         # skip the history window sweep
```

**Google Colab:** open `run_colab.ipynb` from this repo in Colab (File → Open notebook → GitHub), set
`DATA_DIR` to the Drive folder holding the CSVs, and run the cells in order. It starts with 100 runs per fault,
which fits in a free session.

Run `python tep_pipeline.py --help` for all options. Results go to `outputs_<N>runs/` (or `outputs_full/`)
inside the data folder: trained models in `models_v3/`, website data in `web_data/`. Finished models are skipped
on re-runs, so an interrupted run can be restarted.

**Rebuilding the website from a new run:** copy the contents of `web_data/` into `docs/data/` (replacing what is
there), then preview locally with `cd docs && python -m http.server` and open http://localhost:8000. Pushing
`docs/` to the `main` branch publishes it on GitHub Pages (Settings → Pages → branch `main`, folder `/docs`).
The site is plain HTML, CSS and JavaScript with no build step.

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

## Results and the exact split

Every number on the website comes from the files in `docs/data/` (JSON, written by the pipeline). The same
results as readable tables, made from those files by `python scripts/make_tables.py`:

| File | Contents |
|---|---|
| `results/detection_validation.csv` | All 238 detection models: settings, threshold, recall at 5% false alarms, ROC-AUC, PR-AUC, 95% interval, confusion counts |
| `results/diagnosis_validation.csv` | All 178 diagnosis models: settings, macro-F1, accuracy, 95% interval |
| `results/final_test.csv` | The best models on the test files: as validated and retrained, plus a like-for-like row at 5% false alarms |
| `results/per_fault.csv` | Each fault: detection recall (validation and test) and diagnosis recall, with its most common wrong name |
| `results/end_to_end.csv` | Detection and diagnosis together (21 answers): the two-model chain and one 21-label model |
| `results/learning_curve.csv` | The best models retrained on fewer runs and on sparser samples |
| `results/split_demo.csv` | Extra experiment: validating on unseen runs vs on random rows |
| `splits/train_simulation_runs.txt`, `splits/validation_simulation_runs.txt` | The exact split: the 400 training and 100 validation simulation numbers (each covers the fault-free run and all 20 faulty runs with that number) |

`eligible_for_best = False` marks models shown for comparison but never allowed to be chosen (history windows longer
than one hour, and the RBF SVM trained on a subset).

## Repository layout

| Path | Contents |
|---|---|
| `tep_pipeline.py` | Data loading, labels, split, preprocessing, training, evaluation, final test and website export |
| `run_colab.ipynb` | Short notebook that runs the pipeline on Colab |
| `scripts/make_tables.py` | Writes `results/` and `splits/` from the published results (no data or training needed) |
| `results/`, `splits/` | Readable results tables and the exact train/validation split (see above) |
| `exploration/` | My first exploration notebook, kept as it was, with a note on what changed afterwards |
| `docs/` | Project website (GitHub Pages): plain HTML, CSS and JavaScript |
| `docs/data/` | Exported results the website reads (copied from `web_data/`) |
| `requirements.txt` | Python packages the pipeline needs |

To preview the website locally: `cd docs && python -m http.server`, then open http://localhost:8000.

## AI use and contributions

This project was built with the help of Claude, an AI assistant by Anthropic. The same statement is on the
website ([AI use and contributions](https://fang099.github.io/tep-fault-detection/index.html#ai-use)).

**What I did**

- Chose the dataset (Tennessee Eastman Process, Rieth et al. 2017) and the research question: detecting and
  diagnosing faults in a simulated chemical plant.
- Explored the data and tried simple ML models on my own first, to understand the data and the problem before
  bringing in AI.
- Worked out that the data has to be split by simulation run, not by row.
- Decided which model families to compare and which experiments to run, among them training on all the data, the
  learning curve on both axes, the single 21-label model as an alternative to the two-stage chain, PCA monitoring
  with history inputs, the RBF SVM on a subset, the random-split demonstration and which sensors matter.
- Ran all training, validation and the final test on my own and the lab's computers, and handled the practical
  problems along the way (memory limits, interrupted runs, choosing which slow models to skip).
- Reviewed the code and the explanations, asked for corrections where something was unclear or wrong, and tested
  the website page by page.

**What Claude did**

- Wrote most of the code: the training and evaluation pipeline (`tep_pipeline.py`) and the website.
- Designed most of the website's layout.
- Wrote most of the explanatory text on the site and the documentation of the pipeline.
- Proposed parts of the method that I then accepted, such as choosing thresholds at a fixed 5% false alarm rate on
  validation only, and bootstrap intervals over runs.
- Reviewed the code for bugs before the full training run.

**What AI did not do**

- Every number on the site was computed by the pipeline from the dataset, on runs I carried out. No result was
  written or estimated by the AI.
- The test files were used once, after all choices were made on validation.

## References

**Data and background**

- Downs, J. J. and Vogel, E. F. (1993). A plant-wide industrial process control problem. *Computers & Chemical
  Engineering*, 17(3), 245–255.
- Rieth, C. A., Amsel, B. D., Tran, R. and Cook, M. B. (2017). Additional Tennessee Eastman Process simulation data
  for anomaly detection evaluation. Harvard Dataverse. Distributed as CSV on Kaggle:
  [afrniomelo/tep-csv](https://www.kaggle.com/datasets/afrniomelo/tep-csv).
- Chiang, L. H., Russell, E. L. and Braatz, R. D. (2001). *Fault Detection and Diagnosis in Industrial Systems*.
  Springer.
- Jackson, J. E. and Mudholkar, G. S. (1979). Control procedures for residuals associated with principal component
  analysis. *Technometrics*, 21(3), 341–349. (The SPE limit used by PCA monitoring.)

**Software** (used as libraries; no outside code was copied into this repository)

- Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12,
  2825–2830.
- Chen, T. and Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of KDD 2016*, 785–794.
- NumPy (Harris et al., 2020, *Nature* 585, 357–362), pandas, SciPy and joblib.
- The website uses no JavaScript libraries; its charts are drawn by its own code. Fonts: IBM Plex Sans and
  IBM Plex Mono (SIL Open Font License), loaded from Google Fonts.
