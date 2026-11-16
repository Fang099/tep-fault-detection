# Early exploration

`early_trial.ipynb` is one of my first notebooks for this project, from before the full pipeline
(`tep_pipeline.py`) existed. It is kept as it was, as a record of how the project started. It is not part of
the published results and is not needed to reproduce them.

What it does:

- Loads the two training files and checks their shape, the runs per fault, the samples per run, missing values and
  data types.
- Labels detection rows (fault-free = 0; faulty runs = 1 from the fault onset).
- Splits the data into training and validation **by simulation run rather than by row**, and checks that no
  run is on both sides.
- Standardizes the readings (fitted on the training rows only).
- Compares the majority-class baseline with logistic regression, unweighted and with balanced class weights,
  at several thresholds.

What it showed, and what changed in the final pipeline:

- **Accuracy and F1 are misleading here.** About 92% of the validation rows are faulty, so always answering
  "fault" scores F1 0.956, and the unweighted logistic regression does practically the same (it flags all
  88,000 normal rows as faulty). This is why the final project fixes the false alarm rate (5% of normal rows) and
  reports how many faulty rows are caught at that rate, instead of ranking models by accuracy or F1.
- **The threshold matters more than the 0.5 default.** With balanced class weights, moving the threshold from
  0.1 to 0.8 trades faults caught (99% → 36%) against normal rows correctly left alone (0% → 100%). The final
  pipeline chooses every threshold the same way, on validation, at 5% false alarms.
- **Splitting by simulation number, not by (fault, run) pair.** This notebook splits the fault-free runs and the
  faulty runs separately. Faulty run *k* of every fault starts as an exact copy of fault-free run *k*, so that split
  can put copies of the same readings on both sides. The final pipeline holds out the same 20% of simulation
  numbers for all 21 kinds of run (see `splits/`).
- **Label boundary.** Here a faulty run counts as faulty from sample 20; the fault is switched on after sample 20,
  so the final pipeline labels samples 21 onward.

The notebook expects the CSVs in a `dataset/` folder next to it.
