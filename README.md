# Tennessee Eastman Process: Fault Detection and Diagnosis

How effective are different machine learning models at detecting and diagnosing faults in the Tennessee
Eastman Process (TEP) from instantaneous process measurements, and how quickly can faults be detected?

- **Detection:** is a fault active right now? (normal vs faulty)
- **Diagnosis:** which of the 20 fault types is active?

The project website (in `docs/`) lets you compare models, move the decision threshold and inspect each fault type.

## Data

Kaggle dataset [`afrniomelo/tep-csv`](https://www.kaggle.com/datasets/afrniomelo/tep-csv). It is not stored in
this repo. Download these four files into one folder:

- `TEP_FaultFree_Training.csv`, `TEP_Faulty_Training.csv`
- `TEP_FaultFree_Testing.csv`, `TEP_Faulty_Testing.csv`

One row is one 3-minute time step of one simulation run. The models use only the 52 process variables
(`xmeas_1`–`xmeas_41`, `xmv_1`–`xmv_11`).

## Method

- **Labels:** fault-free rows and rows before the fault is injected (after sample 20 in training, sample 160 in
  testing) are normal; rows after injection are faulty.
- **Split:** the training files are split by simulation number, 80% train / 20% validation. Faulty run *k* of every
  fault type starts as an exact copy of fault-free run *k* (identical readings until the fault shows), so all runs
  with the same simulation number go to the same side. The pipeline checks that no validation row is identical to a
  training row. The test files are used once, at the end.
- **Models:** majority-class baselines; logistic regression with no penalty, lasso (L1) and ridge (L2) at several
  strengths `C`, with and without class weighting (detection); decision trees (depth); random forests (depth, number
  of trees); gradient boosting, AdaBoost and XGBoost (tree depth, boosting rounds). Forests and boosted models are
  trained once per depth; smaller forests and fewer rounds are read off the same fit. PCA + logistic regression
  (number of components) and PCA process monitoring (Hotelling's T² and SPE, fitted on fault-free runs only).
- **Temporal features (ablation):** selected families are also trained with a trailing 30-minute rolling mean and
  standard deviation of every reading, computed within each run from past samples only.
- **Uncertainty:** 95% intervals from a run-level bootstrap (whole validation runs resampled within each fault type).
- **Thresholds and model choice:** each detection model's threshold is set so 5% of normal validation rows raise
  an alarm; the best model catches the most faults at that point. The best diagnosis model has the highest
  validation macro-F1.
- **Final test:** the chosen models are retrained on all training runs and evaluated once on the test files,
  keeping the validation thresholds.
- **Metrics:** precision, recall, F1, PR-AUC, ROC-AUC and confusion matrices for detection; macro-F1 and
  per-fault recall for diagnosis; median time from fault injection to detection (3 alarms in a row) and to
  correct diagnosis (3 correct in a row).

## Reproducing the results

**Google Colab:** open `run_colab.ipynb` from this repo in Colab (File → Open notebook → GitHub), set
`DATA_DIR` to the Drive folder holding the CSVs, and run the cells in order.

**Any computer:**

```bash
pip install -r requirements.txt
python tep_pipeline.py --data-dir path/to/csvs --runs-per-fault 100      # subset, faster
python tep_pipeline.py --data-dir path/to/csvs --families tree,rf          # only some model families
python tep_pipeline.py --data-dir path/to/csvs --history-families ""     # no history inputs
python tep_pipeline.py --data-dir path/to/csvs --runs-per-fault 0        # all 500 runs per fault
python tep_pipeline.py --data-dir path/to/csvs --runs-per-fault 0 --final-test
```

Run `python tep_pipeline.py --help` for all options. Results go to `outputs_<N>runs/` (or `outputs_full/`)
inside the data folder: trained models in `models/`, website data in `web_data/`. Finished models are skipped
on re-runs, so an interrupted run can be restarted.

## Repository layout

| Path | Contents |
|---|---|
| `tep_pipeline.py` | Data loading, training, evaluation and website export |
| `run_colab.ipynb` | Short notebook that runs the pipeline on Colab |
| `docs/` | Project website (GitHub Pages): plain HTML, CSS and JavaScript |
| `docs/data/` | Exported results the website reads (copied from `web_data/`) |

To preview the website locally: `cd docs && python -m http.server`, then open http://localhost:8000.
