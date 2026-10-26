import {
  renderChrome, showError, loadManifest, loadModel, dataNote, modelBar, readSelection, writeSelection, selectionOf,
  COLORS, FAULTS, faultDesc, pct, num, int, fmtMinutes, tile, esc,
} from "../site.js";
import { barChart, heatmap } from "../charts.js";
import { explainer, renderSensors } from "../explain.js";
import * as M from "../math.js";
import { renderWindowSweep } from "../window.js";

renderChrome();

const state = { manifest: null, model: null, file: null, row: null, token: 0 };
const $ = id => document.getElementById(id);

main().catch(showError);

async function main() {
  const manifest = state.manifest = await loadManifest();
  $("data-note").innerHTML = dataNote(manifest);
  renderStaticFormulas();
  const q = new URLSearchParams(location.search);
  if (q.get("fault")) state.row = +q.get("fault") - 1;
  state.bar = modelBar($("modelbar"), {
    task: "diagnosis", manifest, selection: readSelection(manifest, "diagnosis"),
    onChange: (sel, model) => selectModel(sel, model).catch(showError),
  });
}

async function selectModel(sel, model) {
  const token = ++state.token;
  writeSelection(sel, { fault: state.row != null ? state.row + 1 : null });
  const file = await loadModel("diagnosis", model.id);
  if (token !== state.token) return;
  state.model = model;
  state.file = file;
  render();
}

function render() {
  renderOverall();
  renderExplainer();
  renderSensorsChart();
  renderMacro();
  renderMatrix();
  renderRecall();
  renderSpeed();
  renderWindow();
}

function selectRow(r) {
  state.row = state.row === r ? null : r;
  writeSelection(selectionOf(state.model), { fault: state.row != null ? state.row + 1 : null });
  renderMatrix();
  renderRecall();
  renderSensorsChart();
}

function renderExplainer() {
  const ex = explainer("diagnosis", state.model, state.manifest);
  $("explainer-title").textContent = ex.title;
  $("model-explainer").innerHTML = ex.html;
}

function renderSensorsChart() {
  renderSensors({ chartEl: $("chart-sensors"), titleEl: $("sensors-title"), introEl: $("sensors-intro"),
                  file: state.file, model: state.model, faultIndex: state.row });
}

function rowShares() {
  return state.file.val.confusion_matrix.map(row => {
    const total = row.reduce((a, b) => a + b, 0) || 1;
    return row.map(c => c / total);
  });
}

// Per-fault precision, recall and F1 from the confusion matrix.
function perFaultScores() {
  const cm = state.file.val.confusion_matrix, K = cm.length;
  return cm.map((row, k) => {
    const tp = row[k], rowSum = row.reduce((a, b) => a + b, 0);
    let colSum = 0;
    for (let r = 0; r < K; r++) colSum += cm[r][k];
    const P = colSum ? tp / colSum : 0, R = rowSum ? tp / rowSum : 0;
    return { k, fault: state.file.val.labels[k], P, R, F1: P + R ? (2 * P * R) / (P + R) : 0, named: colSum };
  });
}

function renderOverall() {
  const m = state.model, base = state.manifest.diagnosis.models.find(x => x.penalty === "baseline");
  const cm = state.file.val.confusion_matrix;
  const rows = cm.reduce((a, r) => a + r.reduce((x, y) => x + y, 0), 0);
  const recall = state.file.val.recall;
  const good = recall.filter(r => r >= 0.8).length, poor = recall.filter(r => r < 0.2).length;
  $("tiles").innerHTML = [
    tile("Macro-F1", num(m.val.macro_F1, 3), m.ci ? `95% interval ${num(m.ci.macro_F1[0], 3)} to ${num(m.ci.macro_F1[1], 3)}` : `baseline ${num(base.val.macro_F1, 3)}`, "big"),
    tile("Accuracy", pct(m.val.accuracy), m.ci ? `95% interval ${pct(m.ci.accuracy[0])} to ${pct(m.ci.accuracy[1])}` : `baseline ${pct(base.val.accuracy)}`, "big"),
    tile("Faults with recall of 80% or more", `${good} / 20`),
    tile("Faults with recall under 20%", `${poor} / 20`),
    tile("Rows evaluated", int(rows)),
  ].join("");
  $("callout").innerHTML = m.family === "baseline"
    ? `<b>The majority baseline</b> names the same fault for every row, so it gets one fault right and the other 19 wrong.`
    : `<b>This model reaches a macro-F1 of ${num(m.val.macro_F1, 3)}</b>, against ${num(base.val.macro_F1, 3)} for the baseline.
       ${good} faults are recognized at least 80% of the time; ${poor} are recognized less than 20% of the time.`;
}

// ---------- Learn ----------
function renderStaticFormulas() {
  $("f-macro").innerHTML =
    M.block(M.sub(M.t("F1"), M.i("k")), M.o("="), M.frac(M.row(M.n(2), M.sub(M.i("P"), M.i("k")), M.sub(M.i("R"), M.i("k"))),
            M.row(M.sub(M.i("P"), M.i("k")), M.o("+"), M.sub(M.i("R"), M.i("k")))),
            M.sp(2), M.t("macro-F1"), M.o("="), M.frac(M.n(1), M.n(20)), M.sum(M.row(M.i("k"), M.o("="), M.n(1)), M.n(20)), M.sub(M.t("F1"), M.i("k")));
}

function renderMacro() {
  const s = perFaultScores();
  const sorted = [...s].sort((a, b) => b.F1 - a.F1);
  const mean = s.reduce((a, x) => a + x.F1, 0) / s.length;
  const meanR = s.reduce((a, x) => a + x.R, 0) / s.length;
  const show = x => `<tr><td>Fault ${x.fault}</td><td class="n">${pct(x.P)}</td><td class="n">${pct(x.R)}</td><td class="n">${num(x.F1, 3)}</td></tr>`;
  $("macro-live").innerHTML = `
    <div class="table-wrap"><table>
      <thead><tr><th>Fault (best and worst three)</th><th class="n">Precision</th><th class="n">Recall</th><th class="n">F1</th></tr></thead>
      <tbody>${sorted.slice(0, 3).map(show).join("")}
        <tr><td colspan="4" class="note" style="text-align:center">… ${s.length - 6} more …</td></tr>
        ${sorted.slice(-3).map(show).join("")}</tbody>
    </table></div>
    <p style="margin-top:10px">Average of all 20 F1 values: <b>${num(mean, 3)}</b>. Because every fault has the same number of rows,
      accuracy works out to exactly the average of the 20 recalls: ${pct(meanR)}.</p>`;
}

// ---------- Confusion matrix ----------
function renderMatrix() {
  const shares = rowShares(), labels = state.file.val.labels, cm = state.file.val.confusion_matrix;
  heatmap($("chart-matrix"), {
    matrix: shares, rowLabels: labels.map(String), colLabels: labels.map(String),
    rowTitle: "True fault", colTitle: "Predicted fault", highlightRow: state.row,
    tip: (r, c) => `True <b>fault ${labels[r]}</b>, predicted <b>fault ${labels[c]}</b><br>${int(cm[r][c])} rows (${pct(shares[r][c])} of fault ${labels[r]})`,
    onClick: r => selectRow(r),
  });
  renderRowPanel(shares, labels);
}

function renderRowPanel(shares, labels) {
  const panel = $("row-panel");
  if (state.row == null) {
    const recall = state.file.val.recall;
    const worst = labels.map((f, k) => [f, recall[k]]).sort((a, b) => a[1] - b[1]).slice(0, 4);
    panel.innerHTML = `<p style="margin-top:0"><b>Click a row</b> to see where that fault's rows end up.</p>
      <p class="note" style="margin:0">Least recognized: ${worst.map(([f, r]) => `fault ${f} (${pct(r, 0)})`).join(", ")}.
      Faults that barely change the measurements tend to be spread across many columns.</p>`;
    return;
  }
  const r = state.row, f = labels[r];
  const others = shares[r].map((s, c) => [labels[c], s]).filter(([g]) => g !== f).sort((a, b) => b[1] - a[1]).slice(0, 4);
  panel.innerHTML = `<h3 style="margin:0 0 2px">Fault ${f}: ${esc(FAULTS[f][0])}</h3>
    <p class="note" style="margin:0 0 12px">Type: ${FAULTS[f][1]}</p>
    <div class="tiles">${tile("Named correctly", pct(shares[r][r]))}${tile("Rows", int(state.file.val.confusion_matrix[r].reduce((a, b) => a + b, 0)))}</div>
    <h3>Most often mistaken for</h3>
    <table><tbody>${others.map(([g, s]) => `<tr><td>Fault ${g}<div class="note">${esc(FAULTS[g][0])}</div></td><td class="n">${pct(s)}</td></tr>`).join("")}</tbody></table>`;
}

// ---------- Per-fault recall ----------
function renderRecall() {
  const labels = state.file.val.labels, recall = state.file.val.recall;
  barChart($("chart-recall"), {
    categories: labels.map(f => `Fault ${f}`),
    series: [{ name: "recall", color: COLORS.data, values: recall }],
    x: { domain: [0, 1], fmt: x => pct(x, 0), label: "Share of the fault's rows named correctly" },
    labelWidth: 70, valueFmt: x => pct(x, 0),
    highlight: state.row,
    vlines: [{ x: 0.05, color: COLORS.muted, label: "chance" }],
    tip: k => `<b>Fault ${labels[k]}</b>: ${faultDesc(labels[k])}<br>Recall ${pct(recall[k])}`,
    onClick: k => selectRow(k),
  });
}

// ---------- Speed ----------
function renderSpeed() {
  const delays = state.file.val.delays;
  const faults = Object.keys(delays).map(Number).sort((a, b) => a - b);
  const med = faults.map(f => delays[f][0]), share = faults.map(f => delays[f][1]);
  const finite = med.filter(x => x != null).sort((a, b) => a - b);
  barChart($("chart-speed"), {
    categories: faults.map(f => `Fault ${f}`),
    series: [{ name: "median", color: COLORS.data, values: med, opacity: share.map(s => 0.25 + 0.7 * s), missingLabel: "never named correctly" }],
    x: { type: "log", domain: [3, Math.max(1000, ...finite) * 1.3], ticks: [3, 10, 30, 60, 180, 600], fmt: x => fmtMinutes(x), label: "Median time to a correct diagnosis (log scale)" },
    labelWidth: 70, rowHeight: 21,
    valueFmt: (x, k) => `${pct(share[k], 0)} of runs`,
    tip: k => `<b>Fault ${faults[k]}</b>: ${faultDesc(faults[k])}<br>Median ${fmtMinutes(med[k])}, in ${pct(share[k], 0)} of runs`,
  });
  const all = faults.filter((f, k) => share[k] >= 0.999).length;
  $("tiles-speed").innerHTML = [
    tile("Named in every run", `${all} / ${faults.length}`, "fault types"),
    tile("Median over faults", finite.length ? fmtMinutes(finite[Math.floor(finite.length / 2)]) : "–"),
  ].join("");
}

function renderWindow() {
  renderWindowSweep({
    task: "diagnosis", manifest: state.manifest, model: state.model,
    els: { section: $("window"), legend: $("legend-window"), chart: $("chart-window"), legend2: $("legend-window2"),
           chart2: $("chart-window2"), speedNote: $("window-speed-note"), callout: $("callout-window") },
    onPick: m => state.bar.set(selectionOf(m)),
  }).catch(showError);
}
