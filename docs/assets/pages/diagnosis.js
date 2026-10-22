import {
  renderChrome, showError, loadManifest, loadModel, dataNote, modelBar, readSelection, writeSelection,
  COLORS, FAULTS, faultDesc, pct, num, int, fmtMinutes, tile,
} from "../site.js";
import { barChart, heatmap } from "../charts.js";

renderChrome();

const state = { manifest: null, model: null, file: null, row: null, token: 0 };
const $ = id => document.getElementById(id);

main().catch(showError);

async function main() {
  const manifest = state.manifest = await loadManifest();
  $("data-note").innerHTML = dataNote(manifest);
  const q = new URLSearchParams(location.search);
  if (q.get("fault")) state.row = +q.get("fault") - 1;
  modelBar($("modelbar"), {
    task: "diagnosis", manifest, selection: readSelection(manifest, "diagnosis"),
    onChange: (sel, model) => selectModel(sel, model).catch(showError),
  });
}

async function selectModel(sel, model) {
  const token = ++state.token;
  writeSelection(sel, "", { fault: state.row != null ? state.row + 1 : null });
  const file = await loadModel("diagnosis", model.id);
  if (token !== state.token) return;
  state.model = model;
  state.file = file;
  render();
}

function render() {
  renderOverall();
  renderMatrix();
  renderRecall();
  renderSpeed();
}

function rowShares() {
  return state.file.val.confusion_matrix.map(row => {
    const total = row.reduce((a, b) => a + b, 0) || 1;
    return row.map(c => c / total);
  });
}

function renderOverall() {
  const m = state.model, base = state.manifest.diagnosis.models.find(x => x.penalty === "baseline");
  const cm = state.file.val.confusion_matrix;
  const rows = cm.reduce((a, r) => a + r.reduce((x, y) => x + y, 0), 0);
  const recall = state.file.val.recall;
  const good = recall.filter(r => r >= 0.8).length, poor = recall.filter(r => r < 0.2).length;
  $("tiles").innerHTML = [
    tile("Macro-F1", num(m.val.macro_F1, 3), `baseline ${num(base.val.macro_F1, 3)}`, "big"),
    tile("Accuracy", pct(m.val.accuracy), `baseline ${pct(base.val.accuracy)}`, "big"),
    tile("Faults with recall ≥ 80%", `${good} / 20`),
    tile("Faults with recall < 20%", `${poor} / 20`),
    tile("Rows evaluated", int(rows)),
  ].join("");
  $("callout").innerHTML = m.penalty === "baseline"
    ? `<b>The majority baseline</b> names the same fault for every row, so it gets exactly one fault right and the other 19 wrong.`
    : `<b>Macro-F1 averages the F1 score of all 20 faults equally,</b> so a model can't hide poor performance on rare-looking faults
       behind good performance on easy ones. This model reaches ${num(m.val.macro_F1, 3)}, against ${num(base.val.macro_F1, 3)} for the baseline.`;
}

function renderMatrix() {
  const shares = rowShares(), labels = state.file.val.labels, cm = state.file.val.confusion_matrix;
  heatmap($("chart-matrix"), {
    matrix: shares, rowLabels: labels.map(String), colLabels: labels.map(String),
    rowTitle: "True fault", colTitle: "Predicted fault", highlightRow: state.row,
    tip: (r, c) => `True <b>fault ${labels[r]}</b>, predicted <b>fault ${labels[c]}</b><br>${int(cm[r][c])} rows (${pct(shares[r][c])} of fault ${labels[r]})`,
    onClick: r => { state.row = state.row === r ? null : r; writeSelection(selFromModel(), "", { fault: state.row != null ? state.row + 1 : null }); renderMatrix(); renderRecall(); },
  });
  renderRowPanel(shares, labels);
}

function selFromModel() {
  const m = state.model;
  return { penalty: m.penalty, weighting: "none", C: m.C };
}

function renderRowPanel(shares, labels) {
  const panel = $("row-panel");
  if (state.row == null) {
    const recall = state.file.val.recall;
    const worst = labels.map((f, i) => [f, recall[i]]).sort((a, b) => a[1] - b[1]).slice(0, 4);
    panel.innerHTML = `<p style="margin-top:0"><b>Click a row</b> to see where that fault's rows end up.</p>
      <p class="note">Least recognized: ${worst.map(([f, r]) => `fault ${f} (${pct(r, 0)})`).join(", ")}.
      Faults that barely change the measurements tend to be spread across many columns.</p>`;
    return;
  }
  const r = state.row, f = labels[r];
  const others = shares[r].map((s, c) => [labels[c], s]).filter(([g]) => g !== f).sort((a, b) => b[1] - a[1]).slice(0, 4);
  panel.innerHTML = `<div class="kicker">Fault ${f}</div>
    <h3 style="margin:2px 0 4px">${FAULTS[f][0]}</h3>
    <p class="note" style="margin:0 0 10px">Type: ${FAULTS[f][1]}</p>
    <div class="tiles">${tile("Named correctly", pct(shares[r][r]))}${tile("Recall", pct(state.file.val.recall[r]))}</div>
    <h3>Most often mistaken for</h3>
    <table><tbody>${others.map(([g, s]) => `<tr><td>Fault ${g}<div class="note">${FAULTS[g][0]}</div></td><td class="n">${pct(s)}</td></tr>`).join("")}</tbody></table>`;
}

function renderRecall() {
  const labels = state.file.val.labels, recall = state.file.val.recall;
  barChart($("chart-recall"), {
    categories: labels.map(f => `Fault ${f}`),
    series: [{ name: "recall", color: COLORS.teal, values: recall }],
    x: { domain: [0, 1], fmt: x => pct(x, 0), label: "Share of the fault's rows named correctly" },
    labelWidth: 70, valueFmt: x => pct(x, 0),
    highlight: state.row,
    vlines: [{ x: 0.05, color: "#64748b", label: "chance" }],
    tip: i => `<b>Fault ${labels[i]}</b>: ${faultDesc(labels[i])}<br>Recall ${pct(recall[i])}`,
    onClick: i => { state.row = state.row === i ? null : i; writeSelection(selFromModel(), "", { fault: state.row != null ? state.row + 1 : null }); renderMatrix(); renderRecall(); },
  });
}

function renderSpeed() {
  const delays = state.file.val.delays;
  const faults = Object.keys(delays).map(Number).sort((a, b) => a - b);
  const med = faults.map(f => delays[f][0]), share = faults.map(f => delays[f][1]);
  const finite = med.filter(x => x != null).sort((a, b) => a - b);
  barChart($("chart-speed"), {
    categories: faults.map(f => `Fault ${f}`),
    series: [{ name: "median", color: COLORS.purple, values: med, opacity: share.map(s => 0.25 + 0.7 * s), missingLabel: "never named correctly" }],
    x: { type: "log", domain: [3, Math.max(1000, ...finite) * 1.3], ticks: [3, 10, 30, 60, 180, 600], fmt: x => fmtMinutes(x), label: "Median time to a correct diagnosis (log scale)" },
    labelWidth: 70, rowHeight: 21,
    valueFmt: (x, i) => `${pct(share[i], 0)} of runs`,
    tip: i => `<b>Fault ${faults[i]}</b>: ${faultDesc(faults[i])}<br>Median ${fmtMinutes(med[i])}, in ${pct(share[i], 0)} of runs`,
  });
  const all = faults.filter((f, i) => share[i] >= 0.999).length;
  $("tiles-speed").innerHTML = [
    tile("Named in every run", `${all} / ${faults.length}`, "fault types"),
    tile("Median over faults", finite.length ? fmtMinutes(finite[Math.floor(finite.length / 2)]) : "–"),
  ].join("");
}
