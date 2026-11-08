import {
  renderChrome, showError, loadManifest, loadModel, loadTestModel, bestModel, dataNote, detectionView, findModel as findAny,
  familyOf, familyLabel, modelLink, COLORS, pct, num, int, tile, esc,
  metricsAtFar, loadAllDetection, farControl, OFFICIAL_FAR, isHist, windowText, windowOf, getJSON, fmtMinutes,
  sensorWeights, inputInfo,
} from "../site.js";
import { barChart, lineChart, heatmap, legend } from "../charts.js";

renderChrome();
const $ = id => document.getElementById(id);
const findModel = (manifest, task, sel) => findAny(manifest, task, { family: "logreg", ...sel });

main().catch(showError);

async function main() {
  const manifest = await loadManifest();
  $("data-note").innerHTML = dataNote(manifest);
  const levels = manifest.settings?.far_levels || [0.01, 0.02, 0.05, 0.1, 0.2];
  farControl($("fam-far"), levels, fam.far, far => { fam.far = far; renderFamilies(manifest); });
  renderFamilies(manifest);
  renderFindings(manifest);
  renderSensorSummary(manifest).catch(showError);
  renderE2E(manifest).catch(showError);
  renderDataCurve(manifest);
  await renderTest(manifest);
}

// ---------- Findings (computed from the manifest so they stay true when results change) ----------
function renderFindings(manifest) {
  const inst = m => (m.features || "inst") === "inst";
  const det = manifest.detection.models.filter(inst), dia = manifest.diagnosis.models.filter(inst);
  const best = bestModel(manifest, "detection"), bestD = bestModel(manifest, "diagnosis");
  const lrDet = bestModel(manifest, "detection", "logreg"), lrDia = bestModel(manifest, "diagnosis", "logreg");
  const base = det.find(m => m.penalty === "baseline");
  const noneU = findModel(manifest, "detection", { penalty: "none", weighting: "none" });
  const noneB = findModel(manifest, "detection", { penalty: "none", weighting: "balanced" });
  const bestOf = (models, f) => (models.length ? models.reduce((a, b) => (f(b) > f(a) ? b : a)) : null);
  const regDet = bestOf(det.filter(m => m.penalty === "l1" || m.penalty === "l2"), m => m.val.recall);
  const regDia = bestOf(dia.filter(m => m.penalty === "l1" || m.penalty === "l2"), m => m.val.macro_F1);
  const noneDia = dia.find(m => m.penalty === "none");
  const lassoSmall = det.filter(m => m.penalty === "l1" && m.coef_nonzero > 0).sort((a, b) => a.coef_nonzero - b.coef_nonzero)[0];

  const items = [
    (() => {
      // Like-for-like: best non-linear vs best logistic regression, with the same inputs.
      const all = manifest.detection.models, allD = manifest.diagnosis.models;
      const pick = (ms, fs, lin, key) => {
        const c = ms.filter(m => m.selectable !== false && (fs === "hist" ? isHist(m.features) : (m.features || "inst") === fs) && familyOf(m) !== "baseline" && (familyOf(m) === "logreg") === lin);
        return c.length ? c.reduce((a, b) => (key(b) > key(a) ? b : a)) : null;
      };
      const rec = m => m.val.recall, f1 = m => m.val.macro_F1;
      const nlI = pick(all, "inst", false, rec), lrI = pick(all, "inst", true, rec);
      if (!nlI || !lrI || rec(nlI) <= rec(lrI)) return "";
      const nlH = pick(all, "hist", false, rec), lrH = pick(all, "hist", true, rec);
      const dI = [pick(allD, "inst", false, f1), pick(allD, "inst", true, f1)];
      let t = `<b>Non-linear models beat logistic regression on single snapshots.</b> At 5% false alarms the best of them,
        ${esc(nlI.label)}, catches ${pct(rec(nlI))} of faulty rows against ${pct(rec(lrI))} for the best logistic regression.`;
      if (dI[0] && dI[1]) t += ` For diagnosis: macro-F1 ${num(f1(dI[0]), 3)} against ${num(f1(dI[1]), 3)}.`;
      if (nlH && lrH) t += ` With history inputs the gap shrinks to ${pct(rec(nlH))} against ${pct(rec(lrH))}: much of what the
        trees gain from non-linear splits, a linear model can get from rolling means and standard deviations.`;
      return t;
    })(),
    (() => {
      const pm = bestModel(manifest, "detection", "pcamon");
      if (!pm || !lrDet) return "";
      const lrInst = det.filter(m => familyOf(m) === "logreg").reduce((x, y) => (y.val.recall > x.val.recall ? y : x));
      return `<b>${pm.val.recall > lrInst.val.recall ? "The label-free textbook method is a strong baseline." : "The label-free textbook method trails the supervised models."}</b> PCA monitoring (${esc(pm.label)}) never sees a fault label,
        yet catches ${pct(pm.val.recall)} of faulty rows at 5% false alarms, ${pm.val.recall > lrInst.val.recall ? "more than the" : "against"}
        ${pct(lrInst.val.recall)} ${pm.val.recall > lrInst.val.recall ? "" : "for the"} best logistic regression on the same snapshots. Supervised non-linear models do better
        because they learn what each fault looks like. See <a href="${modelLink("detection.html", pm)}">PCA monitoring</a>.`;
    })(),
    `<b>Accuracy and F1 are misleading here.</b> The majority baseline scores F1 ${num(base.val.F1, 3)} by alarming on every row,
     including every normal one. Compared at 5% false alarms, it catches nothing.`,
    `<b>The best detection model (${esc(best.label)}) catches ${pct(best.val.recall)} of faulty rows at 5% false alarms</b>
     (ROC-AUC ${num(best.val.ROC_AUC, 3)}). The remaining ${pct(1 - best.val.recall, 0)} of faulty rows look normal to it, mostly from the faults that barely change the readings.`,
    noneU && noneB ? `<b>Class weighting matters only a little.</b> Without a penalty, balanced weights catch ${pct(noneB.val.recall)} against
     ${pct(noneU.val.recall)} unweighted. At a fixed false-alarm rate, weighting mostly moves the threshold rather than changing the ranking of rows.` : "",
    (() => {
      if (!regDet || !regDia || !noneDia || !(noneU || noneB)) return "";      // not all trained yet
      const noneBest = [noneU, noneB].filter(Boolean).reduce((a, b) => (b.val.recall > a.val.recall ? b : a));
      const gain = Math.max(regDet.val.recall - noneBest.val.recall, regDia.val.macro_F1 - noneDia.val.macro_F1);
      return gain <= 0.005
        ? `<b>Regularization does not improve accuracy.</b> The best regularized detection model reaches ${pct(regDet.val.recall)}
           (${esc(regDet.label)}) against ${pct(noneBest.val.recall)} without a penalty; for diagnosis, macro-F1 ${num(regDia.val.macro_F1, 3)}
           against ${num(noneDia.val.macro_F1, 3)}. With 52 inputs and hundreds of thousands of rows, there is little overfitting to prevent.`
        : `<b>Mild regularization helps slightly.</b> Detection: ${pct(regDet.val.recall)} (${esc(regDet.label)}) against
           ${pct(noneBest.val.recall)} without a penalty. Diagnosis: macro-F1 ${num(regDia.val.macro_F1, 3)} against ${num(noneDia.val.macro_F1, 3)}.`;
    })(),
    lassoSmall ? `<b>Lasso shows how few sensors carry the signal.</b> ${esc(lassoSmall.label)} uses only ${lassoSmall.coef_nonzero} of 52 sensors
     and still catches ${pct(lassoSmall.val.recall)} of faulty rows at 5% false alarms. See <a href="regularization.html">Regularization</a>.` : "",
    `<b>Diagnosis is far above chance but uneven.</b> The best diagnosis model (${esc(bestD.label)}) reaches macro-F1
     ${num(bestD.val.macro_F1, 3)} against ${num(dia.find(m => m.penalty === "baseline").val.macro_F1, 3)} for the baseline;
     some faults are named almost perfectly while others are barely above chance. See <a href="diagnosis.html">Diagnosis</a>.`,
    (() => {
      const all = manifest.detection.models, h = all.filter(m => isHist(m.features) && m.selectable !== false);
      if (!h.length) return "";
      const bi = all.filter(m => inst(m) && familyOf(m) !== "baseline").reduce((a, b) => (b.val.recall > a.val.recall ? b : a));
      const bh = h.reduce((a, b) => (b.val.recall > a.val.recall ? b : a));
      const d = bh.val.recall - bi.val.recall;
      return d > 0.005
        ? `<b>Recent history helps.</b> Adding rolling means and standard deviations lifts the best detection model from
           ${pct(bi.val.recall)} (${esc(bi.label)}) to ${pct(bh.val.recall)} (${esc(bh.label)}) at 5% false alarms: drifts and changes in
           variability that a single snapshot can't show become visible. See <a href="detection.html#sensors">Detection</a>.`
        : `<b>History inputs don't help detection here.</b> The best model with rolling statistics reaches ${pct(bh.val.recall)}
           against ${pct(bi.val.recall)} from single snapshots.`;
    })(),
    (() => {
      // Window sweep: which window wins for each family, and does the longest window lose ground?
      const sweeps = manifest.detection.models.filter(m => m.sweep_of);
      if (!sweeps.length) return "";
      const fams = [...new Set(sweeps.map(familyOf))];
      const parts = fams.map(f => {
        const chain = manifest.detection.models.filter(m => familyOf(m) === f && (m.sweep_of || m.id === sweeps.find(x => familyOf(x) === f).sweep_of));
        const b = chain.filter(m => m.selectable !== false).reduce((a, c) => (c.val.recall > a.val.recall ? c : a));
        return `${esc(familyLabel(manifest, f))}: ${windowText(windowOf(b.features))} (${pct(b.val.recall)})`;
      });
      return `<b>The best amount of history depends on the model.</b> Best window at 5% false alarms (up to 1 hour, the longest window
        validation can judge fairly) — ${parts.join("; ")}.
        Longer windows catch slow, subtle faults better but react later to sudden ones. See <a href="detection.html#window">How much history helps</a>.`;
    })(),
    (() => {
      // RBF SVM: trained on a small subset only, so compared with the best model rather than chosen.
      const sv = bestModel(manifest, "detection", "svm"), sg = bestModel(manifest, "diagnosis", "svm");
      if (!sv && !sg) return "";
      const rows = Math.max(...[sv, sg].filter(Boolean).map(m => m.params.rows));
      return `<b>A kernel SVM can't use all the data.</b> An exact RBF support vector machine slows down with the square of the training rows,
        so it was trained on random subsets of up to ${int(rows)} rows (out of millions). Its best settings reach
        ${sv ? `${pct(sv.val.recall)} at 5% false alarms (${esc(sv.label)})` : ""}${sv && sg ? " and " : ""}${sg ? `macro-F1 ${num(sg.val.macro_F1, 3)} (${esc(sg.label)})` : ""},
        against ${pct(best.val.recall)} and ${num(bestD.val.macro_F1, 3)} for the best models trained on every row. It is shown for comparison, never picked as the best model.`;
    })(),
    `<b>Detection speed varies enormously by fault:</b> from minutes for sudden faults to many hours, or never, for subtle ones.
     See <a href="detection.html#speed">Detection</a>.`,
  ];
  // The five headline findings first; the rest under "More findings".
  const KEY = [3, 7, 8, 0, 2], ordered = [...KEY.map(i => items[i]), ...items.filter((_, i) => !KEY.includes(i))].filter(Boolean);
  const li = t => `<li>${t}</li>`;
  $("findings-list").innerHTML = ordered.slice(0, KEY.length).map(li).join("");
  $("findings-more").innerHTML = ordered.length > KEY.length ? `<details class="learn"><summary><span class="learn-tag">More</span>${ordered.length - KEY.length} more findings</summary>
    <div class="learn-body"><ul class="tight">${ordered.slice(KEY.length).map(li).join("")}</ul></div></details>` : "";
}

// ---------- Best model of each family ----------
const fam = { far: OFFICIAL_FAR, all: null };
// 95% bootstrap interval as a small "±" marker with the range in its tooltip (keeps table rows one line high).
const ciCell = (ci, fmt) => (ci && ci[1] > ci[0] ? ` <span class="ci-mark" tabindex="0" title="95% interval over validation runs: ${fmt(ci[0])} – ${fmt(ci[1])}">±</span>` : "");

function renderFamilies(manifest) {
  const official = Math.abs(fam.far - OFFICIAL_FAR) < 1e-9, farTxt = `${+(100 * fam.far).toFixed(1)}%`;
  if (!official && !fam.all) {
    $("fam-status").textContent = "Loading every model's scores…";
    loadAllDetection(manifest, (k, n) => { $("fam-status").textContent = `Loading every model's scores… ${k} of ${n}`; })
      .then(all => { fam.all = all; $("fam-status").textContent = ""; renderFamilies(manifest); }).catch(showError);
    return;
  }
  // Detection: at 5% use the selection stored in the manifest; at other rates re-rank every model of the family.
  const recallOf = m => (official ? m.val.recall : metricsAtFar(m, fam.all[m.id], fam.far).recall);
  const bestDetAt = key => {
    if (official) return bestModel(manifest, "detection", key);
    const ms = manifest.detection.models.filter(m => familyOf(m) === key);
    return ms.length ? ms.reduce((a, b) => (recallOf(b) > recallOf(a) ? b : a)) : null;
  };
  const fams = manifest.families.filter(f => f.key !== "baseline");
  const rows = fams.map(f => ({ f, det: bestDetAt(f.key), dia: bestModel(manifest, "diagnosis", f.key) }));
  rows.forEach(r => { r.rec = r.det ? recallOf(r.det) : null; });
  const top = rows.reduce((a, b) => ((b.rec ?? -1) > (a.rec ?? -1) ? b : a));
  const official5 = rows.reduce((a, b) => ((b.det?.val.recall ?? -1) > (a.det?.val.recall ?? -1) ? b : a));
  $("fam-status").innerHTML = official
    ? "Detection ranked at the official 5% false alarm rate." + (rows.some(r => r.det?.ci) ? " Hover over ± for a 95% bootstrap interval over validation runs." : "")
    : `Detection re-ranked at ${farTxt} false alarms: within each family the best settings are picked again at this rate. ${
        top.f.key === official5.f.key ? `<b>${esc(top.f.label)} stays on top.</b>` : `<b>${esc(top.f.label)} moves ahead of ${esc(official5.f.label)}</b>, which leads at 5%.`}`;
  $("families-table").innerHTML = `<thead><tr><th>Model</th><th>Best detection settings</th><th class="n">Caught at ${farTxt} false alarms</th>
      <th class="n">ROC-AUC</th><th>Best diagnosis settings</th><th class="n">Macro-F1</th></tr></thead><tbody>${
    rows.map(({ f, det, dia }) => `<tr class="${det?.id === manifest.detection.best || dia?.id === manifest.diagnosis.best ? "highlight" : ""}">
      <td><b>${esc(f.label)}</b></td>
      <td>${det ? `<a href="${modelLink("detection.html", det)}">${esc(det.label)}</a>` : "–"}</td>
      <td class="n">${det ? pct(recallOf(det)) + (official ? ciCell(det.ci?.recall, x => pct(x)) : "") : "–"}</td><td class="n">${det ? num(det.val.ROC_AUC, 3) : "–"}</td>
      <td>${dia ? `<a href="${modelLink("diagnosis.html", dia)}">${esc(dia.label)}</a>` : "–"}</td>
      <td class="n">${dia ? num(dia.val.macro_F1, 3) + ciCell(dia.ci?.macro_F1, x => num(x, 3)) : "–"}</td></tr>`).join("")}</tbody>`;
  if (rows.some(r => r.f.key === "svm" && (r.det || r.dia)))
    $("fam-status").innerHTML += ` The RBF SVM is trained on a small random subset of the rows (an exact kernel SVM can't handle millions), so it is shown
      for comparison but never chosen as the best model.`;
  legend($("legend-families"), [{ label: `Detection: faults caught at ${farTxt} false alarms`, color: COLORS.fault },
                                { label: "Diagnosis: macro-F1", color: COLORS.data }]);
  barChart($("chart-families"), {
    categories: rows.map(r => r.f.label),
    series: [{ name: "det", color: COLORS.fault, values: rows.map(r => r.rec) },
             { name: "dia", color: COLORS.data, values: rows.map(r => r.dia?.val.macro_F1 ?? null) }],
    x: { domain: [0, 1], fmt: x => num(x, 1), label: "Validation score (higher is better)" },
    labelWidth: 150, valueFmt: (x, k, s) => (s.name === "det" ? pct(x) : num(x, 3)),
    tip: k => `<b>${esc(rows[k].f.label)}</b><br>Detection: ${rows[k].det ? pct(rows[k].rec) : "–"}<br>Diagnosis: ${rows[k].dia ? num(rows[k].dia.val.macro_F1, 3) : "–"}`,
  });
}

// ---------- Final test ----------
// Summary of how the test compares with validation, computed from the manifest.
function testCallout(manifest, test) {
  const bd = manifest.detection.best, bg = manifest.diagnosis.best;
  const dv = manifest.detection.models.find(m => m.id === bd)?.val, gv = manifest.diagnosis.models.find(m => m.id === bg)?.val;
  const dt = test.detection[bd], gt = test.diagnosis[bg], da = test.as_validated?.detection?.[bd];
  if (!dv || !gv || !dt || !gt) return "";
  const far = r => 1 - r.normal_recall;
  const close = Math.abs(dt.recall - dv.recall) < 0.02 && Math.abs(gt.macro_F1 - gv.macro_F1) < 0.02;
  return `<p class="callout"><b>${close ? "The test confirms the validation results." : "The test differs from validation."}</b>
    Detection catches ${pct(dt.recall)} of faulty rows on test against ${pct(dv.recall)} on validation; diagnosis reaches macro-F1
    ${num(gt.macro_F1, 3)} against ${num(gv.macro_F1, 3)}. ${da ? `The false alarm rate drifts from the 5% set on validation to
    ${pct(far(da))} on test for the exact validated model (${pct(far(dt))} after retraining). The threshold sits in the far tail of
    the normal scores, where small differences between runs matter, and test runs are longer (48 hours against 25), so they reach
    later stretches of operation that no training run shows. A plant would recheck the threshold on its own normal data.` : ""} ${da ? `Retraining on the validation runs as well changes little (${pct(dt.recall)} against
    ${pct(da.recall)}), as the learning curve below predicts.` : ""}</p>`;
}

async function renderTest(manifest) {
  const body = $("test-body");
  const test = manifest.test;
  if (!test) {
    body.innerHTML = `<div class="empty" style="min-height:120px">The final test has not been run yet.<br>
      It runs once, after all models have been chosen on validation data.</div>`;
    return;
  }
  const detIds = Object.keys(test.detection), diaIds = Object.keys(test.diagnosis);
  const label = (task, id) => esc(manifest[task].models.find(m => m.id === id)?.label || id);
  const av = test.as_validated || {};
  const detRow = (name, r, sub) => `<tr class="${sub ? "sub" : ""}"><td>${name}</td><td class="n">${pct(r.recall)}</td>
        <td class="n">${pct(1 - r.normal_recall)}</td><td class="n">${pct(r.precision)}</td><td class="n">${num(r.F1, 3)}</td>
        <td class="n">${num(r.ROC_AUC, 3)}</td><td class="n">${num(r.PR_AUC, 3)}</td></tr>`;
  const diaRow = (name, r, sub) => `<tr class="${sub ? "sub" : ""}"><td>${name}</td><td class="n">${num(r.macro_F1, 3)}</td><td class="n">${pct(r.accuracy)}</td></tr>`;
  body.innerHTML = `
    <p class="intro">The chosen models, retrained on the training and validation runs together, and evaluated once on ${int(test.rows)} test rows
      (${test.runs_per_fault} faulty runs per fault type plus the fault-free test runs; ${pct(test.fault_share, 0)} faulty).
      Detection uses the threshold chosen on validation.${test.rows_copied_from_train != null
        ? ` Test rows identical to a training row: <b>${int(test.rows_copied_from_train)}</b>${test.rows_copied_from_train === 0 ? " (no overlap)" : ""}.` : ""}</p>
    <h3>Detection</h3>
    <div class="table-wrap"><table><thead><tr><th>Model</th><th class="n">Caught</th><th class="n">False alarms</th><th class="n">Precision</th>
      <th class="n">F1</th><th class="n">ROC-AUC</th><th class="n">PR-AUC</th></tr></thead><tbody>${
      detIds.map(id => detRow(label("detection", id), test.detection[id])).join("")}${
      Object.entries(av.detection || {}).map(([id, r]) => detRow(`${label("detection", id)} <span class="note">as validated</span>`, r, true)).join("")}</tbody></table></div>
    ${av.detection && Object.keys(av.detection).length ? `<p class="note">"As validated" is the same model trained on the training runs only, the exact model the
      threshold was set for. Its test false alarm rate shows how well the 5% chosen on validation carries over to new runs; the
      retrained model (all training runs) is the headline result.</p>` : ""}
    <h3>Diagnosis</h3>
    <div class="table-wrap"><table><thead><tr><th>Model</th><th class="n">Macro-F1</th><th class="n">Accuracy</th></tr></thead><tbody>${
      diaIds.map(id => diaRow(label("diagnosis", id), test.diagnosis[id])).join("")}${
      Object.entries(av.diagnosis || {}).map(([id, r]) => diaRow(`${label("diagnosis", id)} <span class="note">as validated</span>`, r, true)).join("")}</tbody></table></div>
    ${testCallout(manifest, test)}
    <details class="learn"><summary><span class="learn-tag">Chart</span>Validation against test, fault by fault (best detection model)</summary>
      <div class="learn-body"><div class="plot"><div id="legend-test"></div><div id="chart-test"></div></div></div></details>`;

  const best = bestModel(manifest, "detection");
  const [valFile, testFile] = await Promise.all([loadModel("detection", best.id), loadTestModel("detection", best.id)]);
  const vv = detectionView(valFile.val), tv = detectionView(testFile.test);
  const valRec = vv.perFault(vv.indexForThreshold(best.val.threshold));
  const testRec = tv.perFault(tv.indexForThreshold(best.val.threshold));
  legend($("legend-test"), [{ label: "Validation", color: COLORS.data }, { label: "Test", color: COLORS.b }]);
  barChart($("chart-test"), {
    categories: vv.faults.map(f => `Fault ${f}`),
    series: [{ name: "val", color: COLORS.data, values: valRec }, { name: "test", color: COLORS.b, values: testRec }],
    x: { domain: [0, 1], fmt: x => pct(x, 0), label: "Share of the fault's rows caught at the validation threshold" },
    labelWidth: 70, valueFmt: x => pct(x, 0),
  });
}

// ---------- Which sensors matter (from the importances already stored with every model) ----------
const KINDS = [["now", "Current reading"], ["mean", "Rolling mean"], ["std", "Rolling standard deviation"]];
const sens = { task: "detection", cache: {} };

// Per-sensor shares (summing to 1) plus the split by input kind, from one model file; null if it has no weights.
function sensorShares(file, features) {
  const w = sensorWeights(file);
  if (!w) return null;
  const names = (file.coefficients || file.importances).features;
  const mag = w.values.map(Math.abs), total = mag.reduce((a, b) => a + b, 0) || 1;
  const bySensor = Object.fromEntries(features.map(f => [f, 0])), byKind = { now: 0, mean: 0, std: 0 };
  const perSensorKind = Object.fromEntries(features.map(f => [f, { now: 0, mean: 0, std: 0 }]));
  names.forEach((n, j) => {
    const m = /^(.*)_(mean|std)\d*$/.exec(n), base = m ? m[1] : n, kind = m ? m[2] : "now";
    const v = mag[j] / total;
    if (!(base in bySensor)) return;
    bySensor[base] += v; byKind[kind] += v; perSensorKind[base][kind] += v;
  });
  return { bySensor, byKind, perSensorKind, kind: w.kind, method: w.method, hist: names.length > 52 };
}

async function loadSensorTask(manifest, task) {
  if (sens.cache[task]) return sens.cache[task];
  const features = manifest.dataset.features;
  const ids = Object.entries(manifest[task].best_by_family || {});
  const files = await Promise.all(ids.map(([, id]) => loadModel(task, id).catch(() => null)));
  const models = ids.map(([fam, id], k) => {
    const s = files[k] && sensorShares(files[k], features);
    return s && { fam, id, model: manifest[task].models.find(m => m.id === id), ...s };
  }).filter(Boolean);
  return (sens.cache[task] = { features, models, best: models.find(m => m.id === manifest[task].best) });
}

async function renderSensorSummary(manifest) {
  const seg = $("sens-task");
  seg.addEventListener("click", ev => {
    const b = ev.target.closest("button"); if (!b) return;
    sens.task = b.dataset.v;
    seg.querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b));
    drawSensorSummary(manifest).catch(showError);
  });
  await drawSensorSummary(manifest);
}

async function drawSensorSummary(manifest) {
  const body = $("sens-body"), task = sens.task;
  const d = await loadSensorTask(manifest, task);
  if (!d.best) { body.innerHTML = `<div class="empty" style="min-height:100px">The best model has no stored sensor weights.</div>`; return; }
  const b = d.best, f = d.features;
  const ranked = [...f].sort((x, y) => b.bySensor[y] - b.bySensor[x]);
  const top = ranked.slice(0, 15);
  const topShare = n => ranked.slice(0, n).reduce((a, s) => a + b.bySensor[s], 0);
  const others = d.models.filter(m => m.id !== b.id);
  const famName = k => familyLabel(manifest, k);
  const short = k => ({ logreg: "Log. reg.", pcalr: "PCA+LR", tree: "Tree", rf: "Forest", ada: "AdaBoost", xgb: "XGBoost", gb: "Grad. boost", svm: "SVM" }[k] || k);
  // Agreement: how many of the best model's top 10 sensors are in each other model's top 10.
  const top10 = new Set(ranked.slice(0, 10));
  const overlap = m => [...f].sort((x, y) => m.bySensor[y] - m.bySensor[x]).slice(0, 10).filter(s => top10.has(s)).length;
  const ov = others.map(m => [m, overlap(m)]);
  const k = b.byKind, kindTop = KINDS.reduce((a, c) => (k[c[0]] > k[a[0]] ? c : a));
  const nm = s => `${esc(s)} (${esc(inputInfo(s).desc)})`;

  body.innerHTML = `
    <h3>The best ${task} model: ${esc(b.model.label)}</h3>
    <p class="note">${b.method === "gain" ? "XGBoost gain: how much the model's loss drops, in total, at splits on each input." :
      "Share of the model's importance per sensor."} Shares add up to 100% over the 52 sensors.</p>
    <div class="plot"><div id="chart-sens-top"></div></div>
    ${b.hist ? `<h3>Current reading or recent behaviour?</h3>
    <div class="plot"><div id="legend-sens-kind"></div><div id="chart-sens-kind"></div></div>` : ""}
    <details class="learn"><summary><span class="learn-tag">Table</span>Do the other models agree? Rank of these sensors in every model</summary>
    <div class="learn-body">
    <p class="note" style="margin-top:0">The same 15 sensors in the best model of each family, each column scaled to its own largest sensor. Logistic
      regression and PCA + logistic regression use coefficient size on standardized inputs, the tree models their importances; gradient
      boosting (scikit-learn), PCA monitoring and the RBF SVM have no per-sensor weights.</p>
    <div class="plot"><div id="chart-sens-agree"></div></div></div></details>
    <p class="callout" id="sens-callout"></p>
    <p class="note">Importance shows what a model uses, not what causes a fault. Correlated sensors share credit (a model may lean on one
      of two nearly identical readings and ignore the other), and different importance measures are not directly comparable, so look
      at the ranking rather than the exact numbers.</p>`;

  barChart($("chart-sens-top"), {
    categories: top.map(s => `${s} · ${inputInfo(s).desc}`),
    series: [{ name: "share", color: COLORS.data, values: top.map(s => b.bySensor[s]) }],
    x: { domain: [0, b.bySensor[top[0]] * 1.15], fmt: x => pct(x, 0), label: "Share of the model's importance" },
    labelWidth: 260, valueFmt: x => pct(x, 1),
    tip: i => { const s = top[i], p = b.perSensorKind[s];
      return `<b>${esc(s)}</b> · ${esc(inputInfo(s).desc)}<br>Share: <b>${pct(b.bySensor[s], 1)}</b>${b.hist ? `<br>${KINDS.map(([kk, l]) =>
        `${l}: ${pct(p[kk], 1)}`).join("<br>")}` : ""}`; },
  });

  if (b.hist) {
    const both = ["detection", "diagnosis"].map(t => sens.cache[t]?.best).filter(m => m?.hist);
    legend($("legend-sens-kind"), both.map((m, i) => ({ label: m === b ? `Best ${task} model` : `Best ${task === "detection" ? "diagnosis" : "detection"} model`, color: i ? COLORS.b : COLORS.data })));
    barChart($("chart-sens-kind"), {
      categories: KINDS.map(c => c[1]),
      series: both.map((m, i) => ({ name: m.id, color: i ? COLORS.b : COLORS.data, values: KINDS.map(c => m.byKind[c[0]]) })),
      x: { domain: [0, 1], fmt: x => pct(x, 0), label: "Share of the best model's importance" },
      labelWidth: 200, valueFmt: x => pct(x, 0),
    });
    if (both.length < 2) loadSensorTask(manifest, task === "detection" ? "diagnosis" : "detection").then(() => {
      if (sens.task === task) drawSensorSummary(manifest); }).catch(() => {});
  }

  const cols = [b, ...others];
  const mx = cols.map(m => Math.max(...f.map(s => m.bySensor[s])) || 1);
  const rankIn = (m, s) => [...f].sort((x, y) => m.bySensor[y] - m.bySensor[x]).indexOf(s) + 1;
  $("chart-sens-agree").innerHTML = `<div class="table-wrap"><table class="sens-agree"><thead><tr><th>Sensor</th>${cols.map(m =>
      `<th class="n" title="${esc(m.model?.label || famName(m.fam))}">${esc(short(m.fam))}</th>`).join("")}</tr></thead><tbody>${
    top.map(s => `<tr><td title="${esc(inputInfo(s).desc)}"><code>${esc(s)}</code> <span class="note">${esc(inputInfo(s).desc)}</span></td>${
      cols.map((m, c) => { const v = m.bySensor[s] / mx[c], a = Math.pow(Math.min(1, v), 0.6);
        return `<td class="n" title="${esc(m.model?.label || famName(m.fam))}: ${pct(m.bySensor[s], 1)} of its weight, rank ${rankIn(m, s)} of 52"
          style="background:color-mix(in srgb, var(--c-heat) ${(a * 100).toFixed(0)}%, var(--surface));color:${a > 0.55 ? "var(--on-heat)" : "var(--text)"}">${rankIn(m, s)}</td>`; }).join("")}</tr>`).join("")}
    </tbody></table></div><p class="note">Numbers are each sensor's rank (1 = most important) in that model; a stronger colour = a larger share of the model's weight.</p>`;

  const avgOv = ov.length ? ov.reduce((a, x) => a + x[1], 0) / ov.length : null;
  $("sens-callout").innerHTML = `<b>A handful of sensors carry most of the weight.</b> For the best ${task} model, the top 5 sensors
    hold ${pct(topShare(5), 0)} of the importance and the top 10 hold ${pct(topShare(10), 0)}; the leaders are ${ranked.slice(0, 3).map(nm).join(", ")}.
    ${b.hist ? `${kindTop[0] === "now" ? "Most of it comes from the <b>current readings</b>" : kindTop[0] === "std"
      ? "The largest share comes from <b>how much a reading varies</b>" : "The largest share comes from <b>the rolling means</b>"}
      (${KINDS.map(([kk, l]) => `${l.toLowerCase()} ${pct(k[kk], 0)}`).join(", ")}).` : ""}
    ${avgOv != null ? `The other families share on average ${num(avgOv, 1)} of its top 10 sensors (most: ${esc(famName(ov.reduce((a, x) => (x[1] > a[1] ? x : a))[0].fam))},
      ${ov.reduce((a, x) => (x[1] > a[1] ? x : a))[1]}; fewest: ${esc(famName(ov.reduce((a, x) => (x[1] < a[1] ? x : a))[0].fam))}, ${ov.reduce((a, x) => (x[1] < a[1] ? x : a))[1]}).` : ""}`;
}

// ---------- The whole system (end to end) ----------
const E2E_ROWS = [["chain", "Two models in a chain"], ["single", "One model, 21 answers"], ["single_argmax", "One model, most likely answer"]];
const e2e = { data: null, split: "val", which: "chain" };

async function renderE2E(manifest) {
  const body = $("e2e-body");
  const data = await getJSON("data/end_to_end.json").catch(() => null);
  if (!data) {
    body.innerHTML = `<div class="empty" style="min-height:100px">These results appear after the next pipeline run.</div>`;
    return;
  }
  e2e.data = data;
  const m = data.models;
  const ci = (v, c) => ciCell(c, x => num(x, 3));
  const row = (key, name, d) => `<tr class="${key === "chain" ? "highlight" : ""}"><td>${name}</td>
      <td class="n">${num(d.macro_F1, 3)}${ci(d.macro_F1, d.ci?.macro_F1)}</td><td class="n">${pct(d.accuracy)}</td>
      <td class="n">${pct(d.false_alarm_rate)}</td><td class="n">${pct(d.faults_caught)}</td><td class="n">${pct(d.right_fault)}</td></tr>`;
  const table = (split, title) => `<h3>${title}</h3><div class="table-wrap"><table>
      <thead><tr><th>Approach</th><th class="n">Macro-F1, 21 answers</th><th class="n">Accuracy</th><th class="n">False alarms</th>
        <th class="n">Faulty rows flagged</th><th class="n">Right fault named</th></tr></thead>
      <tbody>${E2E_ROWS.filter(([k]) => data[split][k]).map(([k, n]) => row(k, n, data[split][k])).join("")}
      ${split === "test" ? ["chain_as_validated", "single_as_validated"].filter(k => data.test[k]).map(k => row(k,
        `${k.startsWith("chain") ? "Chain" : "One model"} <span class="note">as validated</span>`, data.test[k])).join("") : ""}</tbody></table></div>`;
  const v = data.val, gap = v.single.macro_F1 - v.chain.macro_F1;
  body.innerHTML = `
    <p class="note">Chain: <b>${esc(m.detection_label)}</b> then <b>${esc(m.diagnosis_label)}</b>. One model: ${esc(m.single_settings)}, retrained
      with 21 answers. "Right fault named" counts a faulty row as correct only when the fault number is right; a row flagged with the
      wrong fault counts against it.</p>
    ${table("val", "Validation")}
    ${data.test ? table("test", "Final test") : ""}
    <p class="callout">${Math.abs(gap) < 0.005 ? "<b>The two approaches are practically tied</b>" :
      `<b>${gap > 0 ? "One model with 21 answers" : "Two specialised models in a chain"} does better</b>`} on validation: macro-F1
      ${num(v.chain.macro_F1, 3)} for the chain against ${num(v.single.macro_F1, 3)} for one model, at
      ${pct(v.chain.false_alarm_rate)} and ${pct(v.single.false_alarm_rate)} false alarms. ${v.single_argmax ? `Without false alarm control,
      the single model's most likely answer raises false alarms on ${pct(v.single_argmax.false_alarm_rate)} of normal rows: "no fault" is
      only 1 of 21 answers in training, so the model rarely picks it.` : ""}${data.test?.chain && data.test?.single ? ` On the final test
      ${(data.test.single.macro_F1 - data.test.chain.macro_F1) * gap > 0 ? "the order holds" : "the order changes"}: ${num(data.test.chain.macro_F1, 3)} for the chain against ${num(data.test.single.macro_F1, 3)} for one model.` : ""}
      Compared with diagnosis alone (which only ever sees truly faulty rows), the chain shows the real cost of missed detections and
      false alarms.</p>
    <details class="learn"><summary><span class="learn-tag">Chart</span>Where each answer ends up (21 × 21 confusion matrix)</summary>
    <div class="learn-body">
    <div class="modelbar" style="position:static;margin-top:4px">
      <div class="control"><div class="control-head">Approach</div><div class="seg" id="e2e-which">${E2E_ROWS.map(([k, n]) =>
        `<button type="button" data-v="${k}" class="${k === e2e.which ? "on" : ""}">${n}</button>`).join("")}</div></div>
      ${data.test ? `<div class="control"><div class="control-head">Data</div><div class="seg" id="e2e-split">
        <button type="button" data-v="val" class="on">Validation</button><button type="button" data-v="test">Final test</button></div></div>` : ""}
    </div>
    <div class="plot"><div class="plot-title" id="e2e-title"></div><div id="chart-e2e"></div></div>
    <p class="note" id="e2e-note"></p></div></details>`;
  const seg = (id, key) => $(id)?.addEventListener("click", ev => {
    const b = ev.target.closest("button"); if (!b) return;
    e2e[key] = b.dataset.v;
    $(id).querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b));
    renderE2EMatrix();
  });
  seg("e2e-which", "which"); seg("e2e-split", "split");
  renderE2EMatrix();
}

function renderE2EMatrix() {
  const d = e2e.data[e2e.split][e2e.which];
  const labels = e2e.data.labels, cm = d.confusion_matrix;
  const shares = cm.map(r => { const t = r.reduce((a, b) => a + b, 0) || 1; return r.map(c => c / t); });
  const name = l => (l === 0 ? "no fault" : `fault ${l}`);
  $("e2e-title").textContent = `Where each true answer ends up: ${E2E_ROWS.find(r => r[0] === e2e.which)[1].toLowerCase()}, ${e2e.split === "val" ? "validation" : "final test"}`;
  heatmap($("chart-e2e"), {
    matrix: shares, rowLabels: labels.map(l => (l === 0 ? "none" : String(l))), colLabels: labels.map(l => (l === 0 ? "none" : String(l))),
    rowTitle: "True answer", colTitle: "Predicted answer",
    tip: (r, c) => `True <b>${name(labels[r])}</b>, predicted <b>${name(labels[c])}</b><br>${int(cm[r][c])} rows (${pct(shares[r][c])} of ${name(labels[r])})`,
  });
  const missed = labels.slice(1).map((l, k) => [l, shares[k + 1][0]]).sort((a, b) => b[1] - a[1]).slice(0, 3);
  $("e2e-note").innerHTML = `The first column is "no fault": a faulty row there was missed by detection. Most missed:
    ${missed.map(([l, s_]) => `fault ${l} (${pct(s_, 0)} of its rows)`).join(", ")}. The first row is normal operation: anything off its
    first cell is a false alarm.`;
}

// ---------- How much data (learning curve) ----------
function renderDataCurve(manifest) {
  const body = $("data-body"), lc = manifest.learning_curve;
  if (!lc || !lc.detection) {
    body.innerHTML = `<div class="empty" style="min-height:100px">These results appear after the next pipeline run.</div>`;
    return;
  }
  const tasks = [["detection", "Detection: faults caught at 5% false alarms", COLORS.fault, v => pct(v)],
                 ["diagnosis", "Diagnosis: macro-F1", COLORS.data, v => num(v, 3)]].filter(t => lc[t[0]]);
  body.innerHTML = `
    <p class="note">Detection model: ${esc(lc.detection?.label || "")}. Diagnosis model: ${esc(lc.diagnosis?.label || "")}.</p>
    <div class="split even">
      <div class="plot"><div class="plot-title">Fewer simulation runs</div><div id="legend-lc1"></div><div id="chart-lc1"></div></div>
      <div class="plot"><div class="plot-title">Sparser sampling inside each run</div><div id="legend-lc2"></div><div id="chart-lc2"></div></div>
    </div>
    <p class="callout" id="callout-lc"></p>`;
  const items = tasks.map(([, n, c]) => ({ label: n, color: c }));
  const ciTip = (p, fmt) => (p.ci && p.ci[1] > p.ci[0] ? `<br>95% interval ${fmt(p.ci[0])} – ${fmt(p.ci[1])}` : "");
  const shares = Object.keys(lc.detection.runs).map(Number).sort((a, b) => a - b);
  // Zoom the score axis to the range the points use, so differences of a few points are visible (not a 0–100% axis).
  const allVals = tasks.flatMap(([t]) => [...Object.values(lc[t].runs), ...Object.values(lc[t].stride)].map(p => p.metric));
  const yDom = [Math.max(0, Math.floor((Math.min(...allVals) - 0.02) * 20) / 20), Math.min(1, Math.ceil((Math.max(...allVals) + 0.01) * 20) / 20)];
  legend($("legend-lc1"), items);
  lineChart($("chart-lc1"), {
    height: 280,
    x: { type: "log", domain: [shares[0] * 0.8, 1.15], ticks: shares, fmt: x => pct(x, 0), label: "Share of training simulation runs used (log scale)" },
    y: { domain: yDom, fmt: y => pct(y, 0), label: "Validation score (axis zoomed)" },
    series: tasks.map(([t, n, c, fmt]) => ({ name: n, color: c, width: 2.2, marker: 4,
      points: shares.map(sh => [sh, lc[t].runs[String(sh)].metric, lc[t].runs[String(sh)], fmt]) })),
    tip: (s_, p) => `<b>${s_.name}</b><br>${p[2].sims} of ${lc.detection.train_sims} simulation numbers (${int(p[2].rows)} training rows)<br>score <b>${p[3](p[1])}</b>${ciTip(p[2], p[3])}`,
    hoverRadius: 18,
  });
  const strides = Object.keys(lc.detection.stride).map(Number).sort((a, b) => b - a);
  legend($("legend-lc2"), items);
  lineChart($("chart-lc2"), {
    height: 280,
    x: { type: "log", domain: [Math.max(...strides) * 1.25, Math.min(...strides) * 0.8], ticks: strides,
         fmt: k => (k === 1 ? "every" : `1 in ${k}`), label: "Samples used for training (every 3 min = every sample)" },
    y: { domain: yDom, fmt: y => pct(y, 0), label: "Validation score (axis zoomed)" },
    series: tasks.map(([t, n, c, fmt]) => ({ name: n, color: c, width: 2.2, marker: 4,
      points: strides.map(k => [k, lc[t].stride[String(k)].metric, lc[t].stride[String(k)], fmt]) })),
    tip: (s_, p) => `<b>${s_.name}</b><br>every ${p[0] === 1 ? "" : p[0] + "th "}sample: one every ${fmtMinutes(p[0] * 3)} (${int(p[2].rows)} training rows)<br>score <b>${p[3](p[1])}</b>${ciTip(p[2], p[3])}`,
    hoverRadius: 18,
  });
  // Callout: how much each axis moves the score.
  const moves = tasks.map(([t]) => {
    const r = lc[t].runs, st = lc[t].stride, full = r["1"].metric;
    return { t, runs: full - r[String(shares[0])].metric, sparse: full - st[String(Math.max(...strides))].metric };
  });
  const say = (t, d) => (t === "detection" ? `${d >= 0 ? "+" : "−"}${Math.abs(d * 100).toFixed(1)} percentage points` : `${d >= 0 ? "+" : "−"}${num(Math.abs(d), 3)}`);
  const parts = moves.map(m => `<b>${m.t === "detection" ? "Detection" : "Diagnosis"}:</b> all runs instead of ${pct(shares[0], 0)} of them:
    ${say(m.t, m.runs)}; every sample instead of 1 in ${Math.max(...strides)}: ${say(m.t, m.sparse)}.`);
  const runsWin = moves.every(m => m.runs > 2 * Math.abs(m.sparse));
  $("callout-lc").innerHTML = parts.join(" ") + (runsWin
    ? " More runs help far more than denser sampling: the useful information is in the variety of disturbances, not in the number of rows."
    : " Sampling density matters too here, so the extra rows are not simply redundant.");
}
