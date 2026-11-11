// "Summary" panel on every page: the main insights of that page, computed from the data so they stay correct.
import { loadManifest, loadModel, loadTestModel, getJSON, pct, num, int, esc, fmtMinutes, familyLabel, familyOf, isHist, detectionView, bestModel } from "./site.js";

const median = a => { const s = [...a].sort((x, y) => x - y), n = s.length; return n ? (n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2) : null; };
const ok = x => x.selectable !== false && familyOf(x) !== "baseline";
const best = (list, key) => list.reduce((a, b) => (!a || key(b) > key(a) ? b : a), null);
const rec = m => m.val.recall, f1 = m => m.val.macro_F1;
const short = label => label.split(",")[0];

function core(m) {
  const subset = m.settings?.subset_families || [];
  const main = t => m[t].models.filter(x => ok(x) && !subset.includes(familyOf(x)));
  const det = m.detection.models.find(x => x.id === m.detection.best), dia = m.diagnosis.models.find(x => x.id === m.diagnosis.best);
  return {
    det, dia,
    detInst: best(main("detection").filter(x => !isHist(x.features)), rec),
    diaInst: best(main("diagnosis").filter(x => !isHist(x.features)), f1),
    lrDet: best(m.detection.models.filter(x => ok(x) && familyOf(x) === "logreg"), rec),
    lrDia: best(m.diagnosis.models.filter(x => ok(x) && familyOf(x) === "logreg"), f1),
  };
}

const SUMMARIES = {
  async overview(m) {
    const [s, split] = await Promise.all([getJSON("data/data_summary.json"), getJSON("data/split_demo.json").catch(() => null)]);
    const q = s.quality, c = core(m);
    let cum = 0, k90 = 0;
    for (const v of s.pca_variance) { cum += v; k90++; if (cum >= 0.9) break; }
    const C = s.correlation, F = s.features;
    let strong = 0;
    for (let i = 0; i < F.length; i++) for (let j = i + 1; j < F.length; j++) if (Math.abs(C[i][j]) >= 0.9) strong++;
    return [
      `<b>The data:</b> ${F.length} sensors, 20 fault types and ${int(q.runs)} simulated runs (${int(q.rows)} rows) for training and validation, with no missing values.`,
      `<b>Some sensors are near-duplicates:</b> ${strong} pairs correlate at 0.9 or more. Even so, it takes ${k90} of the ${F.length} directions found by PCA to cover 90% of normal variation.`,
      `<b>Repeated rows are expected:</b> ${pct(q.duplicate_rows / q.rows)} of rows repeat, because each faulty run copies its fault-free twin until the fault starts.`,
      `<b>Fair evaluation:</b> data is split by run, never by row, so 0 validation rows copy a training row${split ? `; a random split would have overstated a deep tree's ROC-AUC (${num(split.detection.tree.random.ROC_AUC, 3)} against ${num(split.detection.tree.run.ROC_AUC, 3)})` : ""}.`,
      `<b>Headline:</b> the best model catches ${pct(rec(c.det))} of faulty readings at 5% false alarms, and names the fault with macro-F1 ${num(f1(c.dia), 3)}.`,
    ];
  },

  async detection(m) {
    const c = core(m), file = await loadModel("detection", c.det.id), v = detectionView(file.val);
    const at = far => v.at(v.indexForFar(far)).recall;
    const per = v.perFault(v.indexForFar(0.05)), hard = v.faults.map((f, i) => [f, per[i]]).sort((a, b) => a[1] - b[1]).slice(0, 3);
    const d = (file.val.delays["0.05"] || Object.values(file.val.delays)[0]).by_fault;
    const times = Object.values(d).filter(x => x[0] != null && x[1] >= 0.5).map(x => x[0]);
    return [
      `<b>Best model</b> (${esc(c.det.label)}): it catches ${pct(rec(c.det))} of faulty readings while raising a false alarm on 5% of normal ones.`,
      `<b>The threshold is a trade-off:</b> the same model catches ${pct(at(0.01))} at 1% false alarms and ${pct(at(0.1))} at 10%.`,
      `<b>History matters:</b> adding recent averages and spreads lifts the best result from ${pct(rec(c.detInst))} (current readings only) to ${pct(rec(c.det))}; windows longer than an hour are not trusted, since validation flatters them.`,
      `<b>Hardest faults to catch:</b> ${hard.map(([f, r]) => `fault ${f} (${pct(r, 0)})`).join(", ")}: they barely change the readings.`,
      times.length ? `<b>Speed:</b> half of the faults are flagged within ${fmtMinutes(median(times))} of starting (3 alarms in a row).` : "",
    ];
  },

  async diagnosis(m) {
    const c = core(m), file = await loadModel("diagnosis", c.dia.id), g = file.val;
    const order = g.labels.map((f, i) => i).sort((a, b) => g.recall[a] - g.recall[b]);
    const worst = order[0], row = g.confusion_matrix[worst], tot = row.reduce((a, b) => a + b, 0) || 1;
    const mix = row.map((n, j) => [g.labels[j], n / tot]).filter(([f]) => f !== g.labels[worst]).sort((a, b) => b[1] - a[1]).slice(0, 2);
    const easy = g.recall.filter(r => r >= 0.95).length;
    return [
      `<b>Best model</b> (${esc(c.dia.label)}): macro-F1 ${num(f1(c.dia), 3)}, with ${pct(c.dia.val.accuracy)} of faulty readings named correctly.`,
      `<b>${easy > 10 ? "Most" : "Many"} faults are easy:</b> ${easy} of 20 are named correctly at least 95% of the time.`,
      `<b>A few are hard:</b> ${order.slice(0, 3).map(i => `fault ${g.labels[i]} (${pct(g.recall[i], 0)})`).join(", ")}. Fault ${g.labels[worst]} is mostly confused with ${mix.map(([f, s]) => `fault ${f} (${pct(s, 0)})`).join(" and ")}.`,
      `<b>History matters even more here:</b> macro-F1 rises from ${num(f1(c.diaInst), 3)} with current readings only to ${num(f1(c.dia), 3)}.`,
      c.lrDia ? `<b>Flexible models win:</b> the best logistic regression reaches ${num(f1(c.lrDia), 3)}.` : "",
    ];
  },

  async regularization(m) {
    const lr = m.detection.models.filter(x => familyOf(x) === "logreg" && x.features === "inst");
    const none = lr.filter(x => x.penalty === "none"), pen = lr.filter(x => x.penalty === "l1" || x.penalty === "l2");
    const bNone = best(none, rec), bPen = best(pen, rec);
    const lasso = lr.filter(x => x.penalty === "l1" && x.coef_nonzero != null);
    const near = bPen && lasso.filter(x => rec(x) >= rec(bPen) - 0.02).sort((a, b) => a.coef_nonzero - b.coef_nonzero)[0];
    const weak = best(lasso.filter(x => x.coef_nonzero <= 10), rec);
    return [
      bNone && bPen ? `<b>Penalties barely change the score:</b> the best penalized logistic regression catches ${pct(rec(bPen))} against ${pct(rec(bNone))} with no penalty (current readings only).` : "",
      near ? (near.coef_nonzero > 26
        ? `<b>Few sensors can be dropped:</b> lasso can switch sensors off, but staying within 2 points of the best still needs ${near.coef_nonzero} of 52 (${pct(rec(near))}).`
        : `<b>A few sensors are enough:</b> lasso keeps only ${near.coef_nonzero} of 52 sensors and stays within 2 points of the best (${pct(rec(near))}).`) : "",
      weak ? `<b>Too strong hurts:</b> with ${weak.coef_nonzero} sensors or fewer, the best lasso model catches only ${pct(rec(weak))}.` : "",
      `<b>Why:</b> with millions of training rows for 52 weights there is no overfitting to fix; the model's limit is its straight boundary, which a penalty can't bend.`,
      `<b>Where it still helps:</b> correlated sensors make unpenalized coefficients unstable; ridge steadies them and lasso picks a small, readable set.`,
    ];
  },

  async compare(m) {
    const c = core(m), subset = m.settings?.subset_families || [];
    const top = (t, key) => m[t].models.filter(x => ok(x) && !subset.includes(familyOf(x))).sort((a, b) => key(b) - key(a)).slice(0, 3);
    const d3 = top("detection", rec), g3 = top("diagnosis", f1);
    const ci = c.det.ci?.recall, ci2 = d3[1]?.ci?.recall;
    return [
      `<b>${int(m.detection.models.length + m.diagnosis.models.length)} models</b> were trained and compared on the same validation runs, detection at the same 5% false-alarm rate.`,
      `<b>Top detection models:</b> ${d3.map(x => `${esc(x.label)} (${pct(rec(x))})`).join("; ")}.`,
      `<b>Top diagnosis models:</b> ${g3.map(x => `${esc(x.label)} (${num(f1(x), 3)})`).join("; ")}.`,
      ci && ci2 ? `<b>The lead is real:</b> the best detection model's 95% interval (${pct(ci[0])}–${pct(ci[1])}) ${ci[0] > ci2[1] ? "does not overlap" : "overlaps"} the runner-up's (${pct(ci2[0])}–${pct(ci2[1])}).` : "",
      (() => {
        const top10 = m.detection.models.filter(x => ok(x) && !subset.includes(familyOf(x))).sort((a, b) => rec(b) - rec(a)).slice(0, 10);
        const fam = familyOf(top10[0]), n = top10.filter(x => familyOf(x) === fam).length;
        return `<b>${n >= 6 ? "One family dominates the top" : "The top is mixed"}:</b> ${n} of the 10 best detection models are ${esc(familyLabel(m, fam))}${n >= 6 ? ", with different history windows and sizes" : ""}.`;
      })(),
    ];
  },

  async results(m) {
    const c = core(m), t = m.test;
    const dt = t.detection[c.det.id], gt = t.diagnosis[c.dia.id];
    const [e2e, lc, tf, vf] = await Promise.all([getJSON("data/end_to_end.json").catch(() => null), getJSON("data/learning_curve.json").catch(() => null),
      loadTestModel("detection", c.det.id).catch(() => null), loadModel("detection", c.det.id).catch(() => null)]);
    let like = null, miss = "";
    if (tf?.test) { const tv = detectionView(tf.test); like = tv.at(tv.indexForFar(0.05)); }
    if (vf?.val) {
      const vv = detectionView(vf.val), ti = vv.indexForThreshold(c.det.val.threshold), r = vv.perFault(ti), tot = vv.faultTotals();
      const rows = vv.faults.map((f, k) => [f, tot[k] * (1 - r[k])]).sort((a, b) => b[1] - a[1]), all = rows.reduce((a, x) => a + x[1], 0);
      miss = `<b>Errors are concentrated:</b> faults ${rows.slice(0, 3).map(x => x[0]).join(", ")} account for ${pct(rows.slice(0, 3).reduce((a, x) => a + x[1], 0) / all, 0)} of all the faulty rows the best model misses.`;
    }
    const on = m.dataset.fault_onset?.train ?? 20, v = c.det.val;
    const valFirst = (m.dataset.val_runs * on) / (v.TN + v.FP), testFirst = (t.runs_per_fault * 21 * on) / (dt.TN + dt.FP);
    const lcd = lc?.detection?.runs, shares = lcd ? Object.keys(lcd).sort((a, b) => +a - +b) : [];   // keys as written ("0.05", "1.0")
    const svm = bestModel(m, "detection", "svm");
    return [
      `<b>The final test confirms validation:</b> at the same 5% false alarms the best model catches ${pct(like?.recall ?? dt.recall)} of faulty readings on unseen test runs (validation ${pct(rec(c.det))}), and it diagnoses with macro-F1 ${num(gt.macro_F1, 3)} (validation ${num(f1(c.dia), 3)}).`,
      `<b>But the threshold does not carry over exactly:</b> kept from validation, it gives ${pct(1 - dt.normal_recall)} false alarms on test instead of 5%. The model has learned that the first hour of a run is always fault-free, and ${pct(valFirst, 0)} of the normal validation rows come from that hour against ${pct(testFirst, 0)} on test.`,
      miss,
      e2e?.test?.single && e2e?.test?.chain ? `<b>One model or two:</b> a single model with 21 answers (macro-F1 ${num(e2e.test.single.macro_F1, 3)}) slightly beats detecting first and then diagnosing (${num(e2e.test.chain.macro_F1, 3)}).` : "",
      shares.length > 1 ? `<b>More data helps only a little:</b> with ${pct(+shares[0], 0)} of the runs, detection already reaches ${pct(lcd[shares[0]].metric)}, against ${pct(lcd[shares[shares.length - 1]].metric)} with all of them.` : "",
      svm ? `<b>Kernel methods don't scale here:</b> an RBF SVM, limited to ${int(svm.params.rows)} training rows, reaches ${pct(rec(svm))}.` : "",
    ];
  },

  async reproduce() {
    return [
      `<b>One script does everything:</b> <code>tep_pipeline.py</code> trains every model, chooses thresholds and models on validation, runs the final test once and writes the website's data.`,
      `<b>Resumable:</b> finished models are skipped, so an interrupted run continues where it stopped.`,
      `<b>Hardware:</b> the full data needs 16 GB of RAM; training every model took more than a day of computing on a desktop PC. A smaller number of runs per fault gives a quick test.`,
      `<b>Same split every time:</b> simulation numbers are divided 80% / 20% with a fixed seed (42), so a rerun trains and validates on the same runs.`,
      `<b>Two ways to run it:</b> on your own computer from a command prompt, or in Google Colab with the included notebook.`,
    ];
  },
};

export async function pageSummary(page) {
  const fn = SUMMARIES[page];
  if (!fn) return null;
  const m = await loadManifest();
  return (await fn(m)).filter(Boolean);
}
export const hasSummary = page => page in SUMMARIES;
