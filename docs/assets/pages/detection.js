import {
  renderChrome, showError, loadManifest, loadModel, dataNote, modelBar, readSelection, writeSelection, findModel, familyOf,
  detectionView, logit, PROB_TICKS, COLORS, FAULTS, faultDesc, pct, num, int, fmtMinutes, fmtC, tile, esc,
} from "../site.js";
import { explainer, renderSensors } from "../explain.js";
import { lineChart, barChart, legend } from "../charts.js";
import * as M from "../math.js";

renderChrome();

const state = { manifest: null, model: null, file: null, view: null, t: 0, far: 0.05, farTarget: 0.05, fault: null, speedFar: "0.05", token: 0 };
const $ = id => document.getElementById(id);

main().catch(showError);

async function main() {
  const manifest = state.manifest = await loadManifest();
  $("data-note").innerHTML = dataNote(manifest);
  renderStaticFormulas();
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
  $("speed-far").addEventListener("click", e => {
    const b = e.target.closest("button");
    if (!b) return;
    state.speedFar = b.dataset.far;
    $("speed-far").querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b));
    renderSpeed();
    renderChance();
  });

  modelBar($("modelbar"), {
    task: "detection", manifest, selection: readSelection(manifest, "detection"),
    onChange: (sel, model) => selectModel(sel, model).catch(showError),
  });
}

async function selectModel(sel, model) {
  const token = ++state.token;
  writeSelection(sel, { fault: state.fault });
  const file = await loadModel("detection", model.id);
  if (token !== state.token) return;
  state.model = model;
  state.file = file;
  state.view = detectionView(file.val);
  $("th").max = state.view.B;
  // Keep the same false alarm rate when switching models; the baseline can only alarm on everything or nothing.
  state.t = model.penalty === "baseline" ? 0 : state.view.indexForFar(state.farTarget ?? state.far);
  renderAll();
}

function renderAll() {
  renderExplainer();
  renderThresholdDependent();
  renderSensors({ chartEl: $("chart-sensors"), titleEl: $("sensors-title"), introEl: $("sensors-intro"), file: state.file, model: state.model });
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
  renderFaults();
  renderFaultPanel();
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

  $("tiles").innerHTML = [
    tile("Faults caught (recall)", pct(r.recall)),
    tile("False alarm rate", pct(r.far)),
    tile("Precision", pct(r.precision)),
    tile("F1 (fault class)", num(r.f1, 3)),
    tile("Normal recall", pct(r.normalRecall)),
    tile("Validation rows", int(v.totalN + v.totalF), `${pct(v.totalF / (v.totalN + v.totalF), 0)} faulty`),
  ].join("");

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
  $("f-weights").innerHTML =
    M.block(M.i("L"), M.o("="), M.sum(M.row(M.i("i"), M.o("="), M.n(1)), M.i("N")),
            M.sub(M.i("w"), M.sub(M.i("y"), M.i("i"))), M.sp(0.15), M.sub(M.i("ℓ"), M.i("i")),
            M.sp(2), M.sub(M.i("w"), M.i("c")), M.o("="), M.frac(M.i("N"), M.row(M.n(2), M.sp(0.1), M.sub(M.i("N"), M.i("c")))));
  $("f-auc").innerHTML =
    M.block(M.t("AUC"), M.o("="), M.t("P"), M.paren(M.sub(M.i("s"), M.t("fault")), M.o(">"), M.sub(M.i("s"), M.t("normal"))),
            M.o("+"), M.frac(M.n(1), M.n(2)), M.t("P"), M.paren(M.sub(M.i("s"), M.t("fault")), M.o("="), M.sub(M.i("s"), M.t("normal"))));
  $("f-chance").innerHTML =
    M.block(M.t("P(3 in a row within"), M.sp(0.3), M.i("n"), M.sp(0.3), M.t("samples)"), M.o("≈"), M.n(1), M.o("−"),
            M.sup(M.paren(M.n(1), M.o("−"), M.paren(M.n(1), M.o("−"), M.i("α")), M.sup(M.i("α"), M.n(3))), M.i("n")));
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

// ---------- Learn: class weighting ----------
function renderWeighting() {
  const { manifest, model } = state;
  const fam = familyOf(model);
  const f = manifest.dataset.val_fault_share;
  const wN = 1 / (2 * (1 - f)), wF = 1 / (2 * f);
  const base = fam === "logreg" ? model : { penalty: "none", C: null };
  const pair = ["none", "balanced"].map(w => findModel(manifest, "detection", { family: "logreg", penalty: base.penalty, weighting: w, C: base.C }));
  const name = base.penalty === "none" ? "no penalty" : `${base.penalty === "l1" ? "lasso" : "ridge"}, C = ${fmtC(base.C)}`;
  const note = fam === "logreg" ? "" : `<p><b>Class weighting was only explored for logistic regression.</b> The ${esc(model.label.split(",")[0].toLowerCase())}
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
  const est = a => 1 - Math.pow(1 - (1 - a) * a ** 3, n);
  const delays = state.file.val.delays;
  const levels = Object.keys(delays).sort((a, b) => +a - +b);
  $("chance-live").innerHTML = `
    <p>Validation runs have <i>n</i> = ${n} samples after the fault starts. Faults 3, 9 and 15 barely change the measurements,
      so their "detections" are a good place to look for chance:</p>
    <div class="table-wrap"><table>
      <thead><tr><th>False alarm rate</th><th class="n">Estimate if alarms were independent</th><th class="n">Fault 3: runs "detected"</th><th class="n">Fault 9</th><th class="n">Fault 15</th></tr></thead>
      <tbody>${levels.map(a => `<tr class="${a === state.speedFar ? "highlight" : ""}"><td>${pct(+a, 0)}</td><td class="n">${pct(est(+a), 1)}</td>
        ${[3, 9, 15].map(fl => `<td class="n">${delays[a].by_fault[fl] ? pct(delays[a].by_fault[fl][1], 0) : "–"}</td>`).join("")}</tr>`).join("")}</tbody>
    </table></div>
    <p style="margin-top:10px">The observed shares are well above the estimate. That is expected when alarms cluster in time: once the process
      wanders into an unusual-looking stretch, several consecutive samples score high together, so three in a row is far more likely than
      independent alarms would suggest. A reference measured directly on fault-free runs will replace this rough estimate.</p>`;
}

// ---------- Score distributions ----------
function densitySteps(edges, counts, total) {
  const pts = [];
  for (let k = 0; k < counts.length; k++) {
    const lo = logit(edges[k]), hi = logit(edges[k + 1]);
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

  const lo = Math.max(-12, logit(v.edges[0])), hi = Math.min(12, logit(v.edges[v.B]));
  const ticks = PROB_TICKS.filter(p => logit(p) >= lo && logit(p) <= hi);
  const ys = series.flatMap(s => s.points.map(p => p[1])).sort((a, b) => a - b);
  const yMax = ys[Math.floor(ys.length * 0.995)] * 1.1 || 1;
  const r = v.at(state.t);
  lineChart(el, {
    height: 290,
    x: { domain: [lo, hi], ticks: ticks.map(logit), fmt: x => String(ticks[ticks.map(logit).indexOf(x)]), label: "Model's fault score (probability, spaced by log-odds)" },
    y: { domain: [0, yMax], fmt: y => y.toFixed(2), label: "Density" },
    series,
    vlines: r.threshold == null ? [] : [{ x: logit(r.threshold), color: COLORS.amber, label: "threshold", width: 2 }],
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
  const delays = state.file.val.delays[state.speedFar]?.by_fault || {};
  const faults = Object.keys(delays).map(Number).sort((a, b) => a - b);
  const med = faults.map(f => delays[f][0]), share = faults.map(f => delays[f][1]);
  const finite = med.filter(x => x != null).sort((a, b) => a - b);
  barChart($("chart-speed"), {
    categories: faults.map(f => `Fault ${f}`),
    series: [{ name: "median delay", color: COLORS.data, values: med, opacity: share.map(s => 0.25 + 0.7 * s), missingLabel: "not detected" }],
    x: { type: "log", domain: [3, Math.max(1000, ...finite) * 1.3], ticks: [3, 10, 30, 60, 180, 600], fmt: x => fmtMinutes(x), label: "Median time to detection (log scale)" },
    labelWidth: 70, rowHeight: 21,
    valueFmt: (x, k) => `${pct(share[k], 0)} of runs`,
    highlight: state.fault ? faults.indexOf(state.fault) : null,
    tip: k => `<b>Fault ${faults[k]}</b>: ${faultDesc(faults[k])}<br>Median ${fmtMinutes(med[k])}, detected in ${pct(share[k], 0)} of runs`,
  });
  const allRuns = faults.filter((f, k) => share[k] >= 0.999);
  const fastest = faults.map((f, k) => [f, med[k]]).filter(x => x[1] != null).sort((a, b) => a[1] - b[1])[0];
  $("tiles-speed").innerHTML = [
    tile("Detected in every run", `${allRuns.length} / ${faults.length}`, "fault types"),
    tile("Fastest", fastest ? fmtMinutes(fastest[1]) : "–", fastest ? `fault ${fastest[0]}` : ""),
    tile("Median over faults", finite.length ? fmtMinutes(finite[Math.floor(finite.length / 2)]) : "–"),
  ].join("");
}
