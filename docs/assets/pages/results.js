import {
  renderChrome, showError, loadManifest, loadModel, loadTestModel, bestModel, dataNote, detectionView, findModel as findAny,
  familyOf, familyLabel, modelLink, COLORS, pct, num, int, tile, esc,
  metricsAtFar, loadAllDetection, farControl, OFFICIAL_FAR, isHist, windowText, windowOf, getJSON, fmtMinutes,
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
  const bestOf = (models, f) => models.reduce((a, b) => (f(b) > f(a) ? b : a));
  const regDet = bestOf(det.filter(m => m.penalty === "l1" || m.penalty === "l2"), m => m.val.recall);
  const regDia = bestOf(dia.filter(m => m.penalty === "l1" || m.penalty === "l2"), m => m.val.macro_F1);
  const noneDia = dia.find(m => m.penalty === "none");
  const lassoSmall = det.filter(m => m.penalty === "l1" && m.coef_nonzero > 0).sort((a, b) => a.coef_nonzero - b.coef_nonzero)[0];

  const items = [
    (() => {
      // Like-for-like: best non-linear vs best logistic regression, with the same inputs.
      const all = manifest.detection.models, allD = manifest.diagnosis.models;
      const pick = (ms, fs, lin, key) => {
        const c = ms.filter(m => (fs === "hist" ? isHist(m.features) : (m.features || "inst") === fs) && familyOf(m) !== "baseline" && (familyOf(m) === "logreg") === lin);
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
      const all = manifest.detection.models, h = all.filter(m => isHist(m.features));
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
    `<b>Detection speed varies enormously by fault:</b> from minutes for sudden faults to many hours, or never, for subtle ones.
     See <a href="detection.html#speed">Detection</a>.`,
  ];
  $("findings-list").innerHTML = items.filter(Boolean).map(t => `<li>${t}</li>`).join("");
}

// ---------- Best model of each family ----------
const fam = { far: OFFICIAL_FAR, all: null };
const ciCell = (ci, fmt) => (ci && ci[1] > ci[0] ? `<div class="note">${fmt(ci[0])} – ${fmt(ci[1])}</div>` : "");

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
    ? "Detection ranked at the official 5% false alarm rate." + (rows.some(r => r.det?.ci) ? " Small grey ranges are 95% bootstrap intervals over validation runs." : "")
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
    <p class="intro">The chosen models, retrained on all training runs and evaluated once on ${int(test.rows)} test rows
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
    <h3>Validation against test, per fault (best detection model)</h3>
    <div class="plot"><div id="legend-test"></div><div id="chart-test"></div></div>`;

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
  const ci = (v, c) => (c && c[1] > c[0] ? `<div class="note">${num(c[0], 3)} – ${num(c[1], 3)}</div>` : "");
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
      the single model's most likely answer raises false alarms on ${pct(v.single_argmax.false_alarm_rate)} of normal rows.` : ""}
      Compared with diagnosis alone (which only ever sees truly faulty rows), the chain shows the real cost of missed detections and
      false alarms.</p>
    <div class="modelbar" style="position:static;margin-top:16px">
      <div class="control"><div class="control-head">Approach</div><div class="seg" id="e2e-which">${E2E_ROWS.map(([k, n]) =>
        `<button type="button" data-v="${k}" class="${k === e2e.which ? "on" : ""}">${n}</button>`).join("")}</div></div>
      ${data.test ? `<div class="control"><div class="control-head">Data</div><div class="seg" id="e2e-split">
        <button type="button" data-v="val" class="on">Validation</button><button type="button" data-v="test">Final test</button></div></div>` : ""}
    </div>
    <div class="plot"><div class="plot-title" id="e2e-title"></div><div id="chart-e2e"></div></div>
    <p class="note" id="e2e-note"></p>`;
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
    <div class="split">
      <div class="plot"><div class="plot-title">Fewer simulation runs</div><div id="legend-lc1"></div><div id="chart-lc1"></div></div>
      <div class="plot"><div class="plot-title">Sparser sampling inside each run</div><div id="legend-lc2"></div><div id="chart-lc2"></div></div>
    </div>
    <p class="callout" id="callout-lc"></p>`;
  const items = tasks.map(([, n, c]) => ({ label: n, color: c }));
  const ciTip = (p, fmt) => (p.ci && p.ci[1] > p.ci[0] ? `<br>95% interval ${fmt(p.ci[0])} – ${fmt(p.ci[1])}` : "");
  const shares = Object.keys(lc.detection.runs).map(Number).sort((a, b) => a - b);
  legend($("legend-lc1"), items);
  lineChart($("chart-lc1"), {
    height: 280,
    x: { type: "log", domain: [shares[0] * 0.8, 1.15], ticks: shares, fmt: x => pct(x, 0), label: "Share of training simulation runs used (log scale)" },
    y: { domain: [0, 1], fmt: y => pct(y, 0), label: "Validation score" },
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
    y: { domain: [0, 1], fmt: y => pct(y, 0), label: "Validation score" },
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
