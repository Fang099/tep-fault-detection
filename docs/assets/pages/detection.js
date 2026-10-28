import {
  renderChrome, showError, loadManifest, loadModel, getJSON, dataNote, modelBar, readSelection, writeSelection, findModel, familyOf, selectionOf, isHist, windowOf, windowText,
  detectionView, logit, PROB_TICKS, COLORS, bestModel, familyLabel, modelLink, lowerLabel, FAULTS, faultDesc, pct, num, int, fmtMinutes, fmtC, tile, esc,
} from "../site.js";
import { explainer, renderSensors, historyFormula } from "../explain.js";
import { lineChart, barChart, legend } from "../charts.js";
import * as M from "../math.js";
import { renderWindowSweep } from "../window.js";

renderChrome();

const state = { manifest: null, model: null, file: null, traces: null, view: null, t: 0, far: 0.05, farTarget: 0.05, fault: null,
                speedFar: "0.05", speedK: "3", runIndex: 1, token: 0 };

// Delays for the chosen false alarm rate and "k alarms in a row" (older exports only have the default k).
function speedEntry(far = state.speedFar, k = state.speedK) {
  const d = state.file.val.delays[far];
  if (!d) return null;
  if (d.by_k) return d.by_k[k] || null;
  return +k === (state.manifest.settings.consecutive || 3) ? d : null;
}
const inARow = k => `${k} alarm${+k === 1 ? "" : "s"} in a row`;
const COST_RATIOS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000];
const PREVALENCES = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5];
const $ = id => document.getElementById(id);

main().catch(showError);

async function main() {
  const manifest = state.manifest = await loadManifest();
  $("data-note").innerHTML = dataNote(manifest);
  renderStaticFormulas();
  $("f-history").innerHTML = historyFormula();
  const q = new URLSearchParams(location.search);
  if (q.get("fault")) state.fault = +q.get("fault");

  const slider = $("th");
  slider.addEventListener("input", () => {
    state.t = +slider.value;
    state.far = state.view.at(state.t).far;
    state.farTarget = null;
    renderThresholdDependent();
  });
  $("far-buttons").addEventListener("click", e => {
    const b = e.target.closest("button");
    if (!b) return;
    state.far = state.farTarget = +b.dataset.far;
    state.t = state.view.indexForFar(state.far);
    renderThresholdDependent();
  });
  const levels = manifest.settings.far_levels || [0.01, 0.05, 0.1];
  $("speed-far").innerHTML = levels.map(f => `<button type="button" data-far="${String(f)}"${f === 0.05 ? ' class="on"' : ""}>${pct(f, 0)}</button>`).join("");
  ["cost-ratio", "cost-prev"].forEach(id => $(id).addEventListener("input", renderCost));
  $("cost-apply").addEventListener("click", () => {
    if (state.costBest == null) return;
    state.t = state.costBest; state.far = state.view.at(state.t).far; state.farTarget = null;
    renderThresholdDependent();
    document.getElementById("threshold").scrollIntoView({ behavior: "smooth" });
  });
  $("run-fault").addEventListener("change", e => { state.runIndex = +e.target.value; renderRun(); });
  const ks = manifest.settings.consecutive_levels || [];
  state.speedK = String(manifest.settings.consecutive || 3);
  $("speed-k-wrap").hidden = ks.length < 2;
  $("speed-k").innerHTML = ks.map(k => `<button type="button" data-k="${k}"${String(k) === state.speedK ? ' class="on"' : ""}>${k}</button>`).join("");
  $("speed-k").addEventListener("click", e => {
    const b = e.target.closest("button");
    if (!b) return;
    state.speedK = b.dataset.k;
    $("speed-k").querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b));
    renderSpeed();
    renderChance();
    renderRun();
  });
  $("speed-far").addEventListener("click", e => {
    const b = e.target.closest("button");
    if (!b) return;
    state.speedFar = b.dataset.far;
    $("speed-far").querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b));
    renderSpeed();
    renderChance();
  });

  state.bar = modelBar($("modelbar"), {
    task: "detection", manifest, selection: readSelection(manifest, "detection"),
    onChange: (sel, model) => selectModel(sel, model).catch(showError),
  });
}

async function selectModel(sel, model) {
  const token = ++state.token;
  writeSelection(sel, { fault: state.fault });
  const [file, traces] = await Promise.all([
    loadModel("detection", model.id),
    getJSON(`data/detection/traces/${model.id}.json`).catch(() => null),   // older results have no traces
  ]);
  if (token !== state.token) return;
  state.model = model;
  state.file = file;
  state.traces = traces;
  state.view = detectionView(file.val);
  $("th").max = state.view.B;
  // Keep the same false alarm rate when switching models; the baseline can only alarm on everything or nothing.
  state.t = model.penalty === "baseline" ? 0 : state.view.indexForFar(state.farTarget ?? state.far);
  renderAll();
}

function renderAll() {
  renderExplainer();
  renderCalibration();
  renderRunOptions();
  renderThresholdDependent();
  renderSensors({ chartEl: $("chart-sensors"), titleEl: $("sensors-title"), introEl: $("sensors-intro"), file: state.file, model: state.model });
  renderHistoryLive();
  renderWindow();
  renderSpeed();
  renderWeighting();
  renderAucLive();
  renderChance();
}

function renderThresholdDependent() {
  $("th").value = state.t;
  renderThreshold();
  renderThresholdLive();
  renderWorked();
  renderScores();
  renderRoc();
  renderCost();
  renderFaults();
  renderFaultPanel();
  renderRun();
}

const isBaseline = () => familyOf(state.model) === "baseline";

// ---------- Alarm threshold and confusion matrix ----------
function renderThreshold() {
  const v = state.view, r = v.at(state.t);
  $("th-val").textContent = r.threshold == null ? "above every score" : `score ≥ ${num(r.threshold, 4)}`;
  $("far-buttons").querySelectorAll("button").forEach(b => b.classList.toggle("on", state.farTarget != null && +b.dataset.far === state.farTarget));

  const cell = (count, total, kind, title) => {
    const share = total ? count / total : 0;
    return `<div class="cm-cell ${kind}" style="--a:${(0.1 + 0.55 * share).toFixed(3)}">
      <div class="cm-title">${title}</div><div class="cm-count">${int(count)}</div><div class="cm-share">${pct(share)} of row</div></div>`;
  };
  $("cm").innerHTML = `<div class="cm">
    <div></div><div class="cm-head">No alarm</div><div class="cm-head">Alarm</div>
    <div class="cm-side">Actually normal</div>${cell(r.TN, v.totalN, "ok", "Correctly quiet")}${cell(r.FP, v.totalN, "bad", "False alarm")}
    <div class="cm-side">Actually faulty</div>${cell(r.FN, v.totalF, "bad", "Missed fault")}${cell(r.TP, v.totalF, "ok", "Fault caught")}
  </div>`;

  // 95% intervals exist for the model's own 5% threshold only.
  const ci = state.model.ci, atSel = ci && r.threshold != null && Math.abs(r.threshold - state.model.val.threshold) < 1e-9;
  const range = (k, f) => (atSel && ci[k] ? `95% interval ${f(ci[k][0])} to ${f(ci[k][1])}` : "");
  $("tiles").innerHTML = [
    tile("Faults caught (recall)", pct(r.recall), range("recall", x => pct(x))),
    tile("False alarm rate", pct(r.far), range("far", x => pct(x))),
    tile("Precision", pct(r.precision), range("precision", x => pct(x))),
    tile("F1 (fault class)", num(r.f1, 3), range("F1", x => num(x, 3))),
    tile("Normal recall", pct(r.normalRecall)),
    tile("Validation rows", int(v.totalN + v.totalF), `${pct(v.totalF / (v.totalN + v.totalF), 0)} faulty`),
  ].join("");

  renderCiLive();
  $("callout-threshold").innerHTML = isBaseline()
    ? `<b>The majority baseline</b> gives every row the same score, so it can only alarm on everything or on nothing.
       Alarming on everything gives an F1 of ${num(v.at(0).f1, 3)} and catches every fault, which looks excellent,
       but every single normal row is a false alarm. This is why F1 alone is misleading here.`
    : `<b>At this threshold</b> the model catches ${pct(r.recall)} of faulty rows while raising a false alarm on
       ${pct(r.far)} of normal rows. ${r.precision > 0.9 ? `Precision stays high (${pct(r.precision)}) because ${pct(v.totalF / (v.totalN + v.totalF), 0)} of rows are faulty anyway,
       so even a random alarm is usually right.` : ""}${coarseNote(v, r)}`;
}

// Tree models give only a limited set of scores. Deep trees put many rows at exactly their top score, so a
// false-alarm limit can be impossible to meet without switching alarms off entirely.
function coarseNote(v, r) {
  if (isBaseline()) return "";
  const topShare = v.normal[v.B - 1] / v.totalN;
  if (state.farTarget && r.far === 0 && r.recall === 0 && topShare > state.farTarget) {
    return ` <b>No threshold can meet this limit.</b> ${pct(topShare)} of normal rows share this model's highest score
      (a deep tree gives many rows a score of almost exactly 1), so any alarm at all would exceed ${pct(state.farTarget, 0)} false alarms.
      This is why the deepest trees score 0% here even though their ROC-AUC is reasonable; a forest averages many trees and avoids it.`;
  }
  if (v.B < 40) return ` <b>This model gives only ${v.B} different scores</b> (a tree has one score per leaf), so the threshold can only move in big steps.`;
  return "";
}

// ---------- Learn: static formulas ----------
function renderStaticFormulas() {
  const FAR = M.row(M.t("FAR"), M.paren(M.i("t"))), rec = M.row(M.t("recall"), M.paren(M.i("t")));
  $("f-cost").innerHTML = M.block(M.t("cost"), M.paren(M.i("t")), M.o("="), M.paren(M.n(1), M.o("−"), M.i("π")), M.sp(0.15), FAR,
    M.o("+"), M.i("π"), M.sp(0.1), M.i("r"), M.sp(0.15), M.paren(M.n(1), M.o("−"), rec));
  $("f-calib").innerHTML = M.block(M.t("Brier"), M.o("="), M.frac(M.n(1), M.i("N")), M.sum(M.row(M.i("i"), M.o("="), M.n(1)), M.i("N")),
    M.sup(M.paren(M.sub(M.i("p"), M.i("i")), M.o("−"), M.sub(M.i("y"), M.i("i"))), M.n(2)), M.sp(2), M.t("ECE"), M.o("="),
    M.sum(M.i("b"), null), M.frac(M.sub(M.i("n"), M.i("b")), M.i("N")), M.abs(M.sub(M.t("mean score"), M.i("b")), M.o("−"), M.sub(M.t("share faulty"), M.i("b"))));
  $("f-weights").innerHTML =
    M.block(M.i("L"), M.o("="), M.sum(M.row(M.i("i"), M.o("="), M.n(1)), M.i("N")),
            M.sub(M.i("w"), M.sub(M.i("y"), M.i("i"))), M.sp(0.15), M.sub(M.i("ℓ"), M.i("i")),
            M.sp(2), M.sub(M.i("w"), M.i("c")), M.o("="), M.frac(M.i("N"), M.row(M.n(2), M.sp(0.1), M.sub(M.i("N"), M.i("c")))));
  $("f-auc").innerHTML =
    M.block(M.t("AUC"), M.o("="), M.t("P"), M.paren(M.sub(M.i("s"), M.t("fault")), M.o(">"), M.sub(M.i("s"), M.t("normal"))),
            M.o("+"), M.frac(M.n(1), M.n(2)), M.t("P"), M.paren(M.sub(M.i("s"), M.t("fault")), M.o("="), M.sub(M.i("s"), M.t("normal"))));
  $("f-chance").innerHTML =
    M.block(M.t("P("), M.i("k"), M.sp(0.3), M.t("in a row within"), M.sp(0.3), M.i("n"), M.sp(0.3), M.t("samples)"), M.o("≈"), M.n(1), M.o("−"),
            M.sup(M.paren(M.n(1), M.o("−"), M.paren(M.n(1), M.o("−"), M.i("α")), M.sup(M.i("α"), M.i("k"))), M.i("n")));
}

// ---------- Learn: how the selected model decides ----------
function renderExplainer() {
  const ex = explainer("detection", state.model, state.manifest);
  $("explainer-title").textContent = ex.title;
  $("model-explainer").innerHTML = ex.html + (ex.sigmoid
    ? `<p data-live="threshold"></p><div class="plot"><div id="chart-sigmoid"></div></div>` : "");
  state.sigmoid = !!ex.sigmoid;
}

// The logistic curve with the current threshold, for models whose score is a logistic function of a sum
// (logistic regression: z; gradient boosting and XGBoost: the boosted total F).
function renderThresholdLive() {
  const live = $("model-explainer").querySelector('[data-live="threshold"]');
  if (!state.sigmoid || !live) return;
  const r = state.view.at(state.t);
  const sym = familyOf(state.model) === "logreg" ? "z" : "F";
  if (r.threshold == null) {
    live.innerHTML = `The threshold is above every score, so no row raises an alarm.`;
  } else {
    const t = Math.min(1 - 1e-9, Math.max(1e-9, r.threshold)), zt = Math.log(t / (1 - t));
    live.innerHTML = `The curve only ever rises, so a threshold on <i>p</i> is the same as a cutoff on <i>${sym}</i>.
      At the current threshold <i>t</i> = ${num(t, 4)}, an alarm goes off when <i>${sym}</i> ≥ ${num(zt, 2)}.`;
  }
  const pts = [];
  for (let z = -8; z <= 8.001; z += 0.1) pts.push([z, 1 / (1 + Math.exp(-z))]);
  const spec = {
    height: 220, margin: { left: 50 },
    x: { domain: [-8, 8], ticks: [-8, -4, 0, 4, 8], label: sym === "z" ? "z (log-odds)" : "F (boosted total, log-odds)" },
    y: { domain: [0, 1], ticks: [0, 0.25, 0.5, 0.75, 1], label: "p" },
    series: [{ name: "logistic", color: COLORS.data, width: 2.4, points: pts }],
  };
  if (r.threshold != null) {
    const t = Math.min(1 - 1e-9, Math.max(1e-9, r.threshold)), zt = Math.log(t / (1 - t));
    spec.hlines = [{ y: t, color: COLORS.amber, label: `t = ${num(t, 3)}`, labelX: "left" }];
    spec.vlines = [{ x: Math.max(-8, Math.min(8, zt)), color: COLORS.amber, dash: "4 3", label: `alarm when ${sym} ≥ ${num(zt, 2)}` }];
  }
  lineChart($("chart-sigmoid"), spec);
}

// ---------- Learn: metrics worked out ----------
function renderWorked() {
  const r = state.view.at(state.t);
  const T = M.t, f = M.frac, row = M.row, o = M.o, num0 = x => M.num(x);
  const line = (name, top, bottom, a, b, c, value) =>
    M.block(T(name), o("="), f(T(top), row(T(bottom[0]), o("+"), T(bottom[1]))), o("="),
            f(num0(a), row(num0(b), o("+"), num0(c))), o("="), M.n(value));
  const P = r.precision, R = r.recall;
  $("worked-metrics").innerHTML = [
    line("Recall", "TP", ["TP", "FN"], r.TP, r.TP, r.FN, pct(R)),
    line("False alarm rate", "FP", ["FP", "TN"], r.FP, r.FP, r.TN, pct(r.far)),
    r.TP + r.FP ? line("Precision", "TP", ["TP", "FP"], r.TP, r.TP, r.FP, pct(P)) : M.block(T("Precision is undefined: no row raises an alarm.")),
    M.block(T("F1"), o("="), f(row(M.n(2), o("·"), M.i("P"), o("·"), M.i("R")), row(M.i("P"), o("+"), M.i("R"))), o("="),
            f(row(M.n(2), o("×"), M.n(num(P, 3)), o("×"), M.n(num(R, 3))), row(M.n(num(P, 3)), o("+"), M.n(num(R, 3)))), o("="), M.n(num(r.f1, 3))),
  ].join("");
  $("worked-note").innerHTML = `Normal recall is TN / (TN + FP) = ${pct(r.normalRecall)}, which is simply 1 minus the false alarm rate.
    F1 combines precision and recall for the fault class only, so it never looks at how well normal operation is recognized.`;
}

// ---------- Learn: history inputs, for the selected family ----------
function renderHistoryLive() {
  const { manifest, model } = state, fam = familyOf(model);
  const models = manifest.detection.models.filter(m => familyOf(m) === fam);
  const best = fs => models.filter(m => (m.features || "inst") === fs).sort((a, b) => b.val.recall - a.val.recall)[0];
  const inst = best("inst"), hist = best("hist");
  const el = $("history-live");
  if (!hist || !inst) {
    const trained = [...new Set(manifest.detection.models.filter(m => isHist(m.features)).map(m => familyLabel(manifest, familyOf(m))))];
    el.innerHTML = trained.length
      ? `<p class="note">${esc(familyLabel(manifest, fam))} was trained on instantaneous readings only. History versions exist for: ${trained.map(esc).join(", ")}.</p>`
      : `<p class="note">These results don't include history models yet.</p>`;
    return;
  }
  const row = (name, m) => `<tr class="${m.id === model.id ? "highlight" : ""}"><td>${name}</td><td><a href="${modelLink("detection.html", m)}">${esc(m.label)}</a></td>
    <td class="n">${pct(m.val.recall)}</td><td class="n">${num(m.val.ROC_AUC, 3)}</td></tr>`;
  const gain = hist.val.recall - inst.val.recall;
  el.innerHTML = `<p>Best ${esc(lowerLabel(familyLabel(manifest, fam)))} with and without history, at 5% false alarms:</p>
    <div class="table-wrap"><table><thead><tr><th>Inputs</th><th>Best settings</th><th class="n">Faults caught</th><th class="n">ROC-AUC</th></tr></thead>
    <tbody>${row("Instantaneous (52)", inst)}${row("With 30-min history (156)", hist)}${isHist(model.features) && model.features !== "hist" && familyOf(model) === fam
      ? row(`With ${windowText(windowOf(model.features))} history (156)`, model) : ""}</tbody></table></div>
    <p style="margin-top:10px">History ${gain >= 0 ? `adds <b>${(100 * gain).toFixed(1)} percentage points</b>` : `costs <b>${(-100 * gain).toFixed(1)} percentage points</b>`}
      of faults caught for this model. Compare the fault-by-fault and detection-speed charts with the inputs switch to see which faults gain,
      and see <a href="#window">How much history helps</a> for other window lengths.</p>`;
}

// ---------- Learn: class weighting ----------
function renderWeighting() {
  const { manifest, model } = state;
  const fam = familyOf(model);
  const f = manifest.dataset.val_fault_share;
  const wN = 1 / (2 * (1 - f)), wF = 1 / (2 * f);
  const base = fam === "logreg" ? model : { penalty: "none", C: null };
  const pair = ["none", "balanced"].map(w => findModel(manifest, "detection", { family: "logreg", penalty: base.penalty, weighting: w, C: base.C }));
  const name = base.penalty === "none" ? "no penalty" : `${base.penalty === "l1" ? "lasso" : "ridge"}, C = ${fmtC(base.C)}`;
  const note = fam === "logreg" ? "" : fam === "pcamon" ? `<p><b>PCA monitoring doesn't use labels at all</b>, so class weighting
    doesn't apply: it learns normal operation from fault-free runs and flags anything far from it.
    For reference, here is what weighting does to logistic regression:</p>` : `<p><b>Class weighting was only explored for logistic regression.</b> The ${esc(model.label.split(",")[0].toLowerCase())}
    models here are trained without it; choosing each model's threshold at 5% false alarms deals with the imbalance instead.
    For reference, here is what weighting does to logistic regression:</p>`;
  $("weighting-live").innerHTML = note + `
    <p>Here about ${pct(1 - f, 1)} of training rows are normal and ${pct(f, 1)} faulty. Balanced weighting therefore gives each normal row
      a weight of ${num(wN, 2)} and each faulty row ${num(wF, 3)}: missing the point on a normal row costs ${num(wN / wF, 1)} times as much.</p>
    <p>The model mostly responds by moving its intercept <i>b</i>, which slides every score up or down together. The scores change a lot,
      but their order barely does, and at a fixed false alarm rate only the order matters. Logistic regression with ${esc(name)}:</p>
    <div class="table-wrap"><table>
      <thead><tr><th>Weighting</th><th class="n">Threshold for 5% false alarms</th><th class="n">Faults caught there</th><th class="n">ROC-AUC</th></tr></thead>
      <tbody>${pair.map((m, k) => m ? `<tr class="${m.id === model.id ? "highlight" : ""}"><td>${k ? "Balanced" : "Unweighted"}</td>
        <td class="n">${num(m.val.threshold, 3)}</td><td class="n">${pct(m.val.recall)}</td><td class="n">${num(m.val.ROC_AUC, 3)}</td></tr>` : "").join("")}</tbody>
    </table></div>
    <p class="note" style="margin-top:10px">Switch the class weighting in the bar at the top and watch the two distributions slide while their overlap stays about the same.</p>`;
}

// ---------- Learn: AUC as a probability ----------
function renderAucLive() {
  const v = state.view, m = state.model;
  if (v.constant) {
    $("auc-live").innerHTML = `For the majority baseline every pair is a tie, so the AUC is exactly 0.5: no better than a coin flip.`;
    return;
  }
  const faultCounts = new Array(v.B).fill(0);
  for (const fl of v.faults) v.split.fault_counts[fl].forEach((c, k) => (faultCounts[k] += c));
  let below = 0, wins = 0;
  for (let k = 0; k < v.B; k++) {
    wins += faultCounts[k] * (below + 0.5 * v.normal[k]);
    below += v.normal[k];
  }
  const aucBins = wins / (v.totalF * v.totalN);
  const pairs = v.totalF * v.totalN;
  $("auc-live").innerHTML = `For this model the validation data has ${int(v.totalF)} faulty and ${int(v.totalN)} normal rows,
    so ${(pairs / 1e9).toFixed(1)} billion possible pairs. In ${pct(aucBins)} of them the faulty row scores higher.
    The pipeline computed the exact value, ${num(m.val.ROC_AUC, 3)}; this website groups scores into about 200 bins, which explains any small difference.`;
}

// ---------- Learn: chance alarms ----------
function renderChance() {
  const n = 500 - state.manifest.dataset.fault_onset.train;
  const k = +state.speedK;
  const est = a => 1 - Math.pow(1 - (1 - a) * a ** k, n);
  const delays = state.file.val.delays;
  const levels = Object.keys(delays).sort((a, b) => +a - +b);
  const entry = a => speedEntry(a) || { by_fault: {}, reference: null };
  const hasRef = levels.some(a => entry(a).reference);
  $("chance-live").innerHTML = `
    <p>Validation runs have <i>n</i> = ${n} samples after the fault starts; here <i>k</i> = ${k} (${inARow(k)}). Faults 3, 9 and 15
      barely change the measurements, so their "detections" are a good place to look for chance:</p>
    <div class="table-wrap"><table>
      <thead><tr><th>False alarm rate</th><th class="n">Estimate if alarms were independent</th>${hasRef ? '<th class="n">Fault-free runs, measured</th>' : ""}<th class="n">Fault 3: runs "detected"</th><th class="n">Fault 9</th><th class="n">Fault 15</th></tr></thead>
      <tbody>${levels.map(a => `<tr class="${a === state.speedFar ? "highlight" : ""}"><td>${pct(+a, 0)}</td><td class="n">${pct(est(+a), 1)}</td>${hasRef ? `<td class="n">${entry(a).reference ? pct(entry(a).reference[1], 0) : "–"}</td>` : ""}
        ${[3, 9, 15].map(fl => `<td class="n">${entry(a).by_fault[fl] ? pct(entry(a).by_fault[fl][1], 0) : "–"}</td>`).join("")}</tr>`).join("")}</tbody>
    </table></div>
    <p style="margin-top:10px">${hasRef ? "The measured share on fault-free runs is usually well above the independent estimate." : "The observed shares are well above the estimate."}
      That is expected when alarms cluster in time: once the process wanders into an unusual-looking stretch, several consecutive samples
      score high together, so ${k} in a row is far more likely than independent alarms would suggest. Requiring more alarms in a row
      makes chance detections rarer but adds at least 3 minutes per extra alarm to every real detection.
      ${hasRef ? "The measured column is the honest reference: a fault whose share of runs detected isn't clearly above it isn't really being detected." : "A reference measured directly on fault-free runs will replace this rough estimate after the next pipeline run."}</p>`;
}

// ---------- Score distributions ----------
// Tick values for a log10 axis spanning [lo, hi] (in log10 units): powers of ten, plus 2 and 5 when the range is short.
function logTicks(lo, hi) {
  const out = [];
  for (let e = Math.floor(lo) - 1; e <= Math.ceil(hi) + 1; e++)
    for (const m of hi - lo > 3 ? [1] : [1, 2, 5]) { const v = m * 10 ** e; if (Math.log10(v) >= lo && Math.log10(v) <= hi) out.push(v); }
  return out;
}

// Probability scores are spread out with the log-odds; T² and SPE (positive statistics) with log10.
let scoreX = logit;
function densitySteps(edges, counts, total) {
  const pts = [];
  for (let k = 0; k < counts.length; k++) {
    const lo = scoreX(edges[k]), hi = scoreX(edges[k + 1]);
    if (!(hi > lo)) continue;
    const d = total ? counts[k] / total / (hi - lo) : 0;
    pts.push([lo, d], [hi, d]);
  }
  return pts;
}

function renderScores() {
  const v = state.view, el = $("chart-scores");
  if (v.constant) {
    el.innerHTML = `<div class="empty">The majority baseline gives every row the same score, so there is no distribution to plot.</div>`;
    $("legend-scores").innerHTML = "";
    $("callout-scores").innerHTML = "";
    return;
  }
  const stat = state.model.score_kind === "stat";
  scoreX = stat ? (x => Math.log10(Math.max(x, 1e-6))) : logit;
  const faultCounts = new Array(v.B).fill(0);
  for (const f of v.faults) v.split.fault_counts[f].forEach((c, k) => (faultCounts[k] += c));
  const series = [
    { name: "Normal rows", color: COLORS.normal, points: densitySteps(v.edges, v.normal, v.totalN), area: true, width: 1.8 },
    { name: "Faulty rows", color: COLORS.fault, points: densitySteps(v.edges, faultCounts, v.totalF), area: true, width: 1.8 },
  ];
  const items = [{ label: "Normal rows", color: COLORS.normal }, { label: "Faulty rows (all faults)", color: COLORS.fault }];
  if (state.fault) {
    const fc = v.split.fault_counts[state.fault];
    series.push({ name: `Fault ${state.fault}`, color: COLORS.b, dash: "5 3", width: 2.2,
                  points: densitySteps(v.edges, fc, fc.reduce((a, b) => a + b, 0)) });
    items.push({ label: `Fault ${state.fault} only`, color: COLORS.b, dash: true });
  }
  legend($("legend-scores"), items);

  const lo = stat ? scoreX(v.edges[0]) : Math.max(-12, logit(v.edges[0]));
  const hi = stat ? scoreX(v.edges[v.B]) : Math.min(12, logit(v.edges[v.B]));
  const ticks = stat ? logTicks(lo, hi) : PROB_TICKS.filter(p => logit(p) >= lo && logit(p) <= hi);
  const ys = series.flatMap(s => s.points.map(p => p[1])).sort((a, b) => a - b);
  const yMax = ys[Math.floor(ys.length * 0.995)] * 1.1 || 1;
  const r = v.at(state.t);
  lineChart(el, {
    height: 290,
    x: { domain: [lo, hi], ticks: ticks.map(scoreX), fmt: x => { const t = ticks[ticks.map(scoreX).indexOf(x)]; return stat ? (t >= 10000 ? t.toExponential(0) : String(+t.toPrecision(3))) : String(t); },
         label: stat ? `${state.model.params.stat === "T2" ? "Hotelling T²" : "SPE"} (log scale)` : "Model's fault score (probability, spaced by log-odds)" },
    y: { domain: [0, yMax], fmt: y => y.toFixed(2), label: "Density" },
    series,
    vlines: r.threshold == null ? [] : [{ x: scoreX(r.threshold), color: COLORS.amber, label: "threshold", width: 2 }],
  });

  $("callout-scores").innerHTML = `<b>The overlap is the problem.</b> Normal rows to the right of the line are false alarms
    (${pct(r.far)} of normal rows); faulty rows to the left are missed (${pct(1 - r.recall)} of faulty rows).
    Many faulty rows score just like normal ones, because a single snapshot of a slowly developing fault can look normal.`;
}

// ---------- ROC ----------
function renderRoc() {
  const v = state.view, r = v.at(state.t), m = state.model;
  const pts = [];
  for (let t = v.B; t >= 0; t--) { const a = v.at(t); pts.push([a.far, a.recall, t]); }
  lineChart($("chart-roc"), {
    height: 330,
    x: { domain: [0, 1], fmt: x => pct(x, 0), label: "False alarm rate (normal rows flagged)" },
    y: { domain: [0, 1], fmt: y => pct(y, 0), label: "Faults caught (recall)" },
    series: [
      { name: "chance", color: COLORS.faint, dash: "4 4", width: 1.2, points: [[0, 0], [1, 1]], hover: false },
      { name: "ROC", color: COLORS.data, width: 2.4, points: pts },
    ],
    markers: [{ x: r.far, y: r.recall, color: COLORS.amber, label: "current" }],
    tip: (s, p) => `False alarms <b>${pct(p[0])}</b><br>Faults caught <b>${pct(p[1])}</b><br><span class="note">Click to use this threshold</span>`,
    onClick: (s, p) => { state.t = p[2]; state.far = v.at(p[2]).far; state.farTarget = null; renderThresholdDependent(); },
  });
  $("tiles-roc").innerHTML = [
    tile("ROC-AUC", num(m.val.ROC_AUC, 3), "0.5 is chance"),
    tile("PR-AUC", num(m.val.PR_AUC, 3), "fault as positive"),
    tile("Current point", `${pct(r.far, 1)} → ${pct(r.recall, 1)}`, "false alarms → caught"),
  ].join("");
}

// ---------- Fault by fault ----------
function renderFaults() {
  const v = state.view, rec = v.perFault(state.t), totals = v.faultTotals();
  barChart($("chart-faults"), {
    categories: v.faults.map(f => `Fault ${f}`),
    series: [{ name: "caught", color: COLORS.fault, values: rec }],
    x: { domain: [0, 1], fmt: x => pct(x, 0), label: "Share of the fault's rows caught" },
    labelWidth: 70, valueFmt: x => pct(x, 0),
    highlight: state.fault ? v.faults.indexOf(state.fault) : null,
    tip: k => `<b>Fault ${v.faults[k]}</b>: ${faultDesc(v.faults[k])}<br>Caught ${pct(rec[k])} of ${int(totals[k])} rows`,
    onClick: k => { state.fault = state.fault === v.faults[k] ? null : v.faults[k]; writeFault(); renderThresholdDependent(); },
  });
}

function writeFault() {
  const q = new URLSearchParams(location.search);
  state.fault ? q.set("fault", state.fault) : q.delete("fault");
  history.replaceState(null, "", "?" + q.toString() + location.hash);
}

function renderFaultPanel() {
  const v = state.view, rec = v.perFault(state.t);
  const panel = $("fault-panel");
  if (!state.fault) {
    const order = v.faults.map((f, k) => [f, rec[k]]).sort((a, b) => a[1] - b[1]);
    panel.innerHTML = `<p style="margin-top:0"><b>Click a fault</b> to see what it is, how often it is caught, and how its scores compare
      with normal rows in the score chart above.</p>
      <p class="note" style="margin:0">Hardest to catch at this threshold: ${order.slice(0, 3).map(([f, r]) => `fault ${f} (${pct(r, 0)})`).join(", ")}.
      Faults 3, 9 and 15 are reported in the TEP literature as nearly undetectable, because they barely change the
      statistics of the measured variables.</p>`;
    return;
  }
  const k = v.faults.indexOf(state.fault);
  const d = state.file.val.delays["0.05"].by_fault[state.fault];
  panel.innerHTML = `<h3 style="margin:0 0 2px">Fault ${state.fault}: ${esc(FAULTS[state.fault][0])}</h3>
    <p class="note" style="margin:0 0 12px">Type: ${FAULTS[state.fault][1]}</p>
    <div class="tiles">
      ${tile("Rows caught", pct(rec[k]))}
      ${tile("Fault rows", int(v.faultTotals()[k]))}
      ${tile("Median delay at 5%", d ? fmtMinutes(d[0]) : "–")}
      ${tile("Runs detected at 5%", d ? pct(d[1], 0) : "–")}
    </div>
    <button class="btn" type="button" style="margin-top:12px" id="clear-fault">Clear selection</button>`;
  $("clear-fault").onclick = () => { state.fault = null; writeFault(); renderThresholdDependent(); };
}

// ---------- Detection speed ----------
function renderSpeed() {
  const entry = speedEntry();
  if (!entry) {
    $("chart-speed").innerHTML = `<div class="empty">Not computed for ${inARow(state.speedK)} in these results.</div>`;
    $("callout-speed").innerHTML = ""; $("tiles-speed").innerHTML = "";
    return;
  }
  const delays = entry.by_fault || {};
  const faults = Object.keys(delays).map(Number).sort((a, b) => a - b);
  const med = faults.map(f => delays[f][0]), share = faults.map(f => delays[f][1]);
  const finite = med.filter(x => x != null).sort((a, b) => a - b);
  const ref = entry.reference;     // fault-free runs, as if a fault had started
  barChart($("chart-speed"), {
    categories: faults.map(f => `Fault ${f}`),
    series: [{ name: "median delay", color: COLORS.data, values: med, opacity: share.map(s => 0.25 + 0.7 * s), missingLabel: "not detected" }],
    x: { type: "log", domain: [Math.min(3, ...finite.filter(v => v > 0)), Math.max(1000, ...finite) * 1.3], ticks: [3, 10, 30, 60, 180, 600], fmt: x => fmtMinutes(x), label: "Median time to detection (log scale)" },
    labelWidth: 70, rowHeight: 21,
    valueFmt: (x, k) => `${pct(share[k], 0)} of runs`,
    highlight: state.fault ? faults.indexOf(state.fault) : null,
    vlines: ref && ref[0] != null ? [{ x: ref[0], color: COLORS.muted, label: "chance", dash: "4 3" }] : [],
    tip: k => `<b>Fault ${faults[k]}</b>: ${faultDesc(faults[k])}<br>Median ${fmtMinutes(med[k])}, detected in ${pct(share[k], 0)} of runs`,
  });
  const chance = ref ? faults.filter((f, k) => share[k] <= Math.min(1, ref[1] + 0.15) && (med[k] == null || ref[0] == null || med[k] >= 0.5 * ref[0])) : [];
  $("callout-speed").innerHTML = ref
    ? `<b>Chance reference:</b> on fault-free validation runs, measured as if a fault had started after the first hour, ${inARow(state.speedK)}
       ${state.speedK === "1" ? "happens" : "happen"} in ${pct(ref[1], 0)} of runs${ref[0] != null ? `, after a median of ${fmtMinutes(ref[0])} (the dashed "chance" line)` : ""}.
       ${chance.length ? `Fault${chance.length > 1 ? "s" : ""} ${chance.join(", ")} ${chance.length > 1 ? "are" : "is"} not detected clearly more often or sooner than that, so ${chance.length > 1 ? "their" : "its"} "detections" may be chance alarms.` : "Every fault is detected clearly more often or sooner than that."}`
    : `<b>Read long delays with care.</b> At a 5% false alarm rate a normal process also raises occasional alarms, and over many hours three
       in a row eventually happen by chance. A reference measured on fault-free runs appears here after the next pipeline run.`;
  const allRuns = faults.filter((f, k) => share[k] >= 0.999);
  const fastest = faults.map((f, k) => [f, med[k]]).filter(x => x[1] != null).sort((a, b) => a[1] - b[1])[0];
  $("tiles-speed").innerHTML = [
    tile("Detected in every run", `${allRuns.length} / ${faults.length}`, "fault types"),
    ...(ref ? [tile("Fault-free runs “detected”", pct(ref[1], 0), "by chance")] : []),
    tile("Fastest", fastest ? fmtMinutes(fastest[1]) : "–", fastest ? `fault ${fastest[0]}` : ""),
    tile("Median over faults", finite.length ? fmtMinutes(finite[Math.floor(finite.length / 2)]) : "–"),
  ].join("");
}

// ---------- Uncertainty ----------
function renderCiLive() {
  const m = state.model, ci = m.ci, el = $("ci-live");
  if (!ci) {
    el.innerHTML = `<p class="note">${isBaseline() ? "The majority baseline has no interval: it gives the same answer for every sample." :
      "These results were exported before intervals were added; they appear after the next pipeline run."}</p>`;
    return;
  }
  el.innerHTML = `<p>For this model at its 5% threshold: faults caught <b>${pct(m.val.recall)}</b>, 95% interval
    <b>${pct(ci.recall[0])} to ${pct(ci.recall[1])}</b>. Two models whose intervals overlap heavily are not reliably different on this
    data; the <a href="compare.html">Compare</a> page shows both side by side.</p>`;
}

// ---------- Threshold by cost ----------
function renderCost() {
  const v = state.view;
  const r = COST_RATIOS[+$("cost-ratio").value], pi = PREVALENCES[+$("cost-prev").value];
  $("cost-ratio-val").textContent = `${r} false alarm${r === 1 ? "" : "s"}`;
  $("cost-prev-val").textContent = pct(pi, pi < 0.01 ? 1 : 0);
  if (isBaseline()) {
    $("chart-cost").innerHTML = `<div class="empty">The baseline can only alarm on everything or nothing, so there is no threshold to choose.</div>`;
    $("legend-cost").innerHTML = ""; $("tiles-cost").innerHTML = ""; $("callout-cost").innerHTML = ""; state.costBest = null;
    return;
  }
  const none = pi * r;                                  // cost of never raising an alarm
  const pts = [];
  let best = null;
  for (let t = 0; t <= v.B; t++) {
    const a = v.at(t), c = ((1 - pi) * a.far + pi * r * (1 - a.recall)) / none;
    pts.push([a.far, c, t]);
    if (!best || c < best[1] - 1e-12) best = [a.far, c, t];
  }
  pts.sort((a, b) => a[0] - b[0]);
  state.costBest = best[2];
  const cur = v.at(state.t), curCost = ((1 - pi) * cur.far + pi * r * (1 - cur.recall)) / none;
  const yMax = Math.min(3, Math.max(1.1, ...pts.filter(p => p[0] <= 0.5).map(p => p[1])));
  legend($("legend-cost"), [{ label: "Expected cost at each threshold", color: COLORS.data }, { label: "Cheapest", color: COLORS.l2 }, { label: "Current threshold", color: COLORS.amber }]);
  lineChart($("chart-cost"), {
    height: 280,
    x: { domain: [0, 0.5], fmt: x => pct(x, 0), label: "False alarm rate of the threshold" },
    y: { domain: [0, yMax], fmt: y => pct(y, 0), label: "Cost, % of no alarms at all" },
    series: [{ name: "cost", color: COLORS.data, width: 2.2, points: pts.filter(p => p[0] <= 0.5) }],
    hlines: [{ y: 1, color: COLORS.muted, label: "no alarms at all", dash: "4 4" }],
    markers: [{ x: best[0], y: best[1], color: COLORS.l2, label: "cheapest" }, { x: cur.far, y: Math.min(curCost, yMax), color: COLORS.amber }],
    tip: (s, p) => `False alarms <b>${pct(p[0])}</b><br>Cost <b>${pct(p[1], 0)}</b> of no alarms`,
    onClick: (s, p) => { state.t = p[2]; state.far = v.at(p[2]).far; state.farTarget = null; renderThresholdDependent(); },
  });
  const b = v.at(best[2]);
  $("tiles-cost").innerHTML = [
    tile("Cheapest false alarm rate", pct(b.far)),
    tile("Faults caught there", pct(b.recall)),
    tile("Cost vs no alarms", pct(best[1], 0)),
  ].join("");
  $("callout-cost").innerHTML = best[1] >= 0.999
    ? `<b>With these costs, raising no alarms at all is cheapest.</b> Faults are so rare, or misses so cheap, that every alarm this model raises costs more than it saves.`
    : `<b>With a miss costing ${r} false alarms and a fault present ${pct(pi, pi < 0.01 ? 1 : 0)} of the time,</b> the cheapest threshold accepts
       ${pct(b.far)} false alarms to catch ${pct(b.recall)} of faulty samples, cutting the expected cost to ${pct(best[1], 0)} of having no monitoring.
       ${Math.abs(b.far - 0.05) < 0.01 ? "That is close to the 5% used everywhere else on this site." : b.far > 0.05 ? "That is looser than the 5% used elsewhere on this site." : "That is stricter than the 5% used elsewhere on this site."}`;
}

// ---------- Run viewer ----------
function renderRunOptions() {
  const sel = $("run-fault"), tr = state.traces;
  if (!tr) { sel.innerHTML = ""; sel.disabled = true; return; }
  sel.disabled = false;
  const prev = state.runIndex;
  sel.innerHTML = tr.runs.map((r, k) => `<option value="${k}">${r.fault === 0 ? "Fault-free run (no fault)" : `Fault ${r.fault}: ${esc(FAULTS[r.fault][0])}`}</option>`).join("");
  state.runIndex = Math.min(prev, tr.runs.length - 1);
  if (state.fault) {
    const k = tr.runs.findIndex(r => r.fault === state.fault);
    if (k >= 0) state.runIndex = k;
  }
  sel.value = state.runIndex;
}

function renderRun() {
  const tr = state.traces, el = $("chart-run");
  if (!tr) {
    el.innerHTML = `<div class="empty">Run traces aren't in these results yet. They are exported by the latest version of the pipeline.</div>`;
    $("legend-run").innerHTML = ""; $("tiles-run").innerHTML = ""; $("callout-run").innerHTML = "";
    return;
  }
  const run = tr.runs[state.runIndex], r = state.view.at(state.t);
  const hours = k => ((k + 1) * tr.sample_minutes) / 60;          // sample k+1 at 3-minute steps
  const level = 100 * (1 - r.far);                                // percentile above which a sample alarms
  const onsetH = (tr.onset * tr.sample_minutes) / 60;
  // Rarity: a sample at percentile q scores higher than all but 1 in N normal samples, N = 1 / (1 - q).
  // Plotted as log10(N), so the 5% false-alarm threshold sits at "1 in 20" and extreme scores spread out.
  const rarity = q => Math.log10(1 / Math.max(1 - q / 100, 0.001));
  const pts = run.rank.map((v, k) => [hours(k), rarity(v / 10), v / 10]);
  const alarms = r.threshold == null ? [] : pts.filter(p => p[2] >= level);
  const N = [1, 2, 5, 10, 20, 50, 100, 1000];
  legend($("legend-run"), [{ label: "Score", color: COLORS.data }, { label: "Alarm", color: COLORS.fault },
                           { label: "Threshold", color: COLORS.amber, dash: true }]);
  lineChart(el, {
    height: 320, margin: { left: 74 },
    x: { domain: [0, hours(run.rank.length - 1)], ticks: [0, 5, 10, 15, 20, 25].filter(h => h <= hours(run.rank.length - 1)), fmt: h => `${h} h`, label: "Time since the run started" },
    y: { domain: [0, 3.05], ticks: N.map(n => Math.log10(n)), fmt: y => `1 in ${Math.round(10 ** y)}`, label: "How unusual (log scale)" },
    series: [
      { name: "score", color: COLORS.data, width: 1.4, points: pts },
      { name: "alarm", color: COLORS.fault, width: 0.001, marker: 2.6, points: alarms, hover: false },
    ],
    hlines: r.threshold == null || r.far <= 0 ? [] : [{ y: rarity(level), color: COLORS.amber, label: `threshold: 1 in ${Math.round(1 / r.far)}` }],
    vlines: [{ x: onsetH, color: COLORS.text, dash: "4 3", label: run.fault ? "fault starts" : "(no fault in this run)" }],
    tip: (s, p) => `${(p[0] * 60).toFixed(0)} min: higher than <b>${p[2].toFixed(1)}%</b> of normal samples${p[2] >= level && r.threshold != null ? "<br><b>alarm</b>" : ""}`,
    hoverRadius: 14,
  });
  // Summary: alarms before and after the fault starts, first alarm, first 3 in a row.
  const on = tr.onset, isAlarm = v => r.threshold != null && v / 10 >= level;
  const before = run.rank.slice(0, on).filter(isAlarm).length;
  const after = run.rank.slice(on);
  const firstIdx = after.findIndex(isAlarm);
  let streakIdx = -1;
  const need = +state.speedK;
  for (let k = 0, run_ = 0; k < after.length; k++) { run_ = isAlarm(after[k]) ? run_ + 1 : 0; if (run_ >= need) { streakIdx = k; break; } }
  const mins = k => (k < 0 ? "never" : fmtMinutes((k + 1) * tr.sample_minutes));
  const shareAfter = after.filter(isAlarm).length / after.length;
  $("tiles-run").innerHTML = run.fault
    ? [tile("False alarms before the fault", `${before} of ${on}`), tile("First alarm after the fault", mins(firstIdx)),
       tile(`${inARow(state.speedK)} after`, mins(streakIdx)), tile("Faulty samples with an alarm", pct(shareAfter))].join("")
    : [tile("False alarms in the whole run", `${run.rank.filter(isAlarm).length} of ${run.rank.length}`),
       tile("Share of samples alarmed", pct(run.rank.filter(isAlarm).length / run.rank.length)),
       tile(`${state.speedK} in a row after hour 1`, mins(streakIdx))].join("");
  $("callout-run").innerHTML = run.fault
    ? (streakIdx < 0 ? `<b>This run's fault is never reliably detected</b> at this threshold: the score rarely stays above the line.`
       : `<b>The fault is caught ${mins(streakIdx)} after it starts</b> (${inARow(state.speedK)}). This is one run; the speed chart below summarizes every validation run.`)
    : `<b>A fault-free run shows the false alarms an operator would see.</b> Every dot here is an alarm with nothing wrong.`;
}

// ---------- Calibration ----------
function renderCalibration() {
  const cal = state.file.calibration, el = $("chart-calib");
  const clear = msg => { el.innerHTML = `<div class="empty">${msg}</div>`; $("legend-calib").innerHTML = ""; $("tiles-calib").innerHTML = ""; $("calib-note").textContent = ""; $("callout-calib").innerHTML = ""; };
  if (isBaseline()) return clear("The baseline gives every sample the same score, so there is nothing to calibrate.");
  if (state.model.score_kind === "stat") return clear("T² and SPE are distances, not probabilities, so calibration doesn't apply.");
  if (!cal) return clear("Calibration data isn't in these results yet. It is exported by the latest version of the pipeline.");
  const pts = cal.bins.map(([p, f, n]) => [p, f, n]);
  legend($("legend-calib"), [{ label: "This model", color: COLORS.data }, { label: "Perfect calibration", color: COLORS.faint, dash: true }]);
  lineChart(el, {
    height: 320,
    x: { domain: [0, 1], fmt: x => num(x, 1), label: "Model's score (mean within each bin)" },
    y: { domain: [0, 1], fmt: y => pct(y, 0), label: "Share actually faulty" },
    series: [
      { name: "perfect", color: COLORS.faint, dash: "5 4", width: 1.4, points: [[0, 0], [1, 1]], hover: false },
      { name: "model", color: COLORS.data, width: 2.2, marker: 5, points: pts },
    ],
    tip: (s, p) => `Mean score <b>${num(p[0], 3)}</b><br>Actually faulty <b>${pct(p[1])}</b><br>${int(p[2])} samples`,
  });
  $("tiles-calib").innerHTML = [tile("Brier score", num(cal.brier, 3), "0 is perfect"), tile("ECE", num(cal.ece, 3), "0 is perfect")].join("");
  $("calib-note").textContent = "Hover over a dot to see how many samples it holds; dots built from few samples are less reliable.";
  const total = pts.reduce((a, x) => a + x[2], 0);
  const gap = pts.reduce((a, [p, f, n]) => a + (f - p) * n, 0) / total;
  // The bin furthest from the diagonal among bins holding at least 0.5% of samples.
  const big = pts.filter(x => x[2] >= 0.005 * total);
  const worst = big.length ? big.reduce((a, x) => (Math.abs(x[1] - x[0]) > Math.abs(a[1] - a[0]) ? x : a)) : null;
  const worstText = worst && Math.abs(worst[1] - worst[0]) >= 0.1
    ? ` The biggest mismatch: samples scored around ${num(worst[0], 2)} (${pct(worst[2] / total)} of all samples) are faulty
       ${pct(worst[1], 0)} of the time, so a score there ${worst[1] < worst[0] ? "overstates" : "understates"} the risk.`
    : "";
  $("callout-calib").innerHTML = (cal.ece < 0.02
    ? `<b>Well calibrated overall</b> (ECE ${num(cal.ece, 3)}): averaged over all samples, scores match the true share of faults within ${pct(cal.ece)}.
       Most samples get scores near 0 or 1, where the model is right, so this average is dominated by them.`
    : Math.abs(gap) < 0.01
      ? `<b>Off in both directions</b> (ECE ${num(cal.ece, 3)}): some score ranges overstate the risk and others understate it, so the
         errors cancel in the overall average but not bin by bin.`
    : gap > 0
      ? `<b>The scores are too low:</b> samples are faulty ${pct(gap, 0)} more often, on average, than the scores say.
         ${state.model.class_weight === "balanced" ? "That is the expected effect of balanced class weighting, which deliberately pulls scores down when most samples are faulty." : ""}`
      : `<b>The scores are too high:</b> samples are faulty ${pct(-gap, 0)} less often, on average, than the scores say.`) + worstText;
}

function renderWindow() {
  renderWindowSweep({
    task: "detection", manifest: state.manifest, model: state.model,
    els: { section: $("window"), legend: $("legend-window"), chart: $("chart-window"), legend2: $("legend-window2"),
           chart2: $("chart-window2"), speedNote: $("window-speed-note"), callout: $("callout-window") },
    onPick: m => state.bar.set(selectionOf(m)),
  }).catch(showError);
}
