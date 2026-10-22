import {
  renderChrome, showError, loadManifest, loadModel, dataNote, modelBar, readSelection, writeSelection,
  detectionView, logit, PROB_TICKS, COLORS, FAULTS, faultDesc, pct, num, int, fmtMinutes, tile,
} from "../site.js";
import { lineChart, barChart, legend } from "../charts.js";

renderChrome();

const state = { model: null, file: null, view: null, t: 0, far: 0.05, farTarget: 0.05, fault: null, speedFar: "0.05", token: 0 };
const $ = id => document.getElementById(id);

main().catch(showError);

async function main() {
  const manifest = await loadManifest();
  $("data-note").innerHTML = dataNote(manifest);
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
  });

  modelBar($("modelbar"), {
    task: "detection", manifest, selection: readSelection(manifest, "detection"),
    onChange: (sel, model) => selectModel(sel, model).catch(showError),
  });
}

async function selectModel(sel, model) {
  const token = ++state.token;
  writeSelection(sel, "", { fault: state.fault });
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
  renderThresholdDependent();
  renderSpeed();
}

function renderThresholdDependent() {
  $("th").value = state.t;
  renderThreshold();
  renderScores();
  renderRoc();
  renderFaults();
  renderFaultPanel();
}

// ---------- 1. Threshold and confusion matrix ----------
function renderThreshold() {
  const v = state.view, r = v.at(state.t), m = state.model;
  $("th-val").textContent = r.threshold == null ? "above every score" : `score ≥ ${num(r.threshold, 4)}`;
  $("far-buttons").querySelectorAll("button").forEach(b => b.classList.toggle("on", state.farTarget != null && +b.dataset.far === state.farTarget));

  const cell = (count, total, kind, title) => {
    const share = total ? count / total : 0;
    return `<div class="cm-cell ${kind}" style="--a:${0.12 + 0.6 * share}">
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

  const isBase = m.penalty === "baseline";
  $("callout-threshold").innerHTML = isBase
    ? `<b>The majority baseline.</b> It gives every row the same score, so it can only alarm on everything or on nothing.
       Alarming on everything gives an F1 of ${num(v.at(0).f1, 3)} and catches every fault, which looks excellent,
       but every single normal row is a false alarm. This is why F1 alone is misleading here.`
    : `<b>At this threshold,</b> the model catches ${pct(r.recall)} of faulty rows while raising a false alarm on
       ${pct(r.far)} of normal rows. ${r.precision > 0.9 ? `Precision stays high (${pct(r.precision)}) because 91% of rows are faulty anyway:
       even a random alarm is usually right.` : ""}`;
}

// ---------- 2. Score distributions ----------
function densitySteps(edges, counts, total) {
  const pts = [];
  for (let i = 0; i < counts.length; i++) {
    const lo = logit(edges[i]), hi = logit(edges[i + 1]);
    if (!(hi > lo)) continue;
    const d = total ? counts[i] / total / (hi - lo) : 0;
    pts.push([lo, d], [hi, d]);
  }
  return pts;
}

function renderScores() {
  const v = state.view, el = $("chart-scores");
  if (v.constant) {
    el.innerHTML = `<div class="empty">The majority baseline gives every row the same score, so there is nothing to plot.</div>`;
    $("legend-scores").innerHTML = "";
    $("callout-scores").innerHTML = "";
    return;
  }
  const faultCounts = new Array(v.B).fill(0);
  for (const f of v.faults) v.split.fault_counts[f].forEach((c, i) => (faultCounts[i] += c));
  const series = [
    { name: "Normal rows", color: COLORS.normal, points: densitySteps(v.edges, v.normal, v.totalN), area: true, width: 1.8 },
    { name: "Faulty rows", color: COLORS.fault, points: densitySteps(v.edges, faultCounts, v.totalF), area: true, width: 1.8 },
  ];
  const items = [{ label: "Normal rows", color: COLORS.normal }, { label: "Faulty rows (all faults)", color: COLORS.fault }];
  if (state.fault) {
    const fc = v.split.fault_counts[state.fault];
    series.push({ name: `Fault ${state.fault}`, color: COLORS.purple, dash: "5 3", width: 2.2,
                  points: densitySteps(v.edges, fc, fc.reduce((a, b) => a + b, 0)) });
    items.push({ label: `Fault ${state.fault} only`, color: COLORS.purple, dash: true });
  }
  legend($("legend-scores"), items);

  const lo = Math.max(-12, logit(v.edges[0])), hi = Math.min(12, logit(v.edges[v.B]));
  const ticks = PROB_TICKS.filter(p => logit(p) >= lo && logit(p) <= hi);
  const ys = series.flatMap(s => s.points.map(p => p[1])).sort((a, b) => a - b);
  const yMax = ys[Math.floor(ys.length * 0.995)] * 1.1 || 1;
  const r = v.at(state.t);
  lineChart($("chart-scores"), {
    height: 290,
    x: { domain: [lo, hi], ticks: ticks.map(logit), fmt: x => String(ticks[ticks.map(logit).indexOf(x)]), label: "Model's fault score (probability, log-odds spacing)" },
    y: { domain: [0, yMax], fmt: y => y.toFixed(2), label: "Density" },
    series,
    vlines: r.threshold == null ? [] : [{ x: logit(r.threshold), color: "#111", label: "threshold" }],
  });

  // Overlap: share of normal rows above / faulty rows below the threshold, already in the tiles; describe the shape here.
  $("callout-scores").innerHTML = `<b>The overlap is the problem.</b> Normal rows to the right of the line are false alarms
    (${pct(r.far)} of normal rows); faulty rows to the left are missed (${pct(1 - r.recall)} of faulty rows).
    Many faulty rows score just like normal ones, because a single snapshot of a slowly developing fault can look normal.`;
}

// ---------- 3. ROC ----------
function renderRoc() {
  const v = state.view, r = v.at(state.t), m = state.model;
  const pts = [];
  for (let t = v.B; t >= 0; t--) { const a = v.at(t); pts.push([a.far, a.recall, t]); }
  lineChart($("chart-roc"), {
    height: 330,
    x: { domain: [0, 1], fmt: x => pct(x, 0), label: "False alarm rate (normal rows flagged)" },
    y: { domain: [0, 1], fmt: y => pct(y, 0), label: "Faults caught (recall)" },
    series: [
      { name: "chance", color: "#94a3b8", dash: "4 4", width: 1.2, points: [[0, 0], [1, 1]], hover: false },
      { name: "ROC", color: COLORS.teal, width: 2.4, points: pts },
    ],
    markers: [{ x: r.far, y: r.recall, color: COLORS.amber, label: "you" }],
    tip: (s, p) => `False alarms <b>${pct(p[0])}</b><br>Faults caught <b>${pct(p[1])}</b><br><span class="note">click to use this threshold</span>`,
    onClick: (s, p) => { state.t = p[2]; state.far = v.at(p[2]).far; state.farTarget = null; renderThresholdDependent(); },
  });
  $("tiles-roc").innerHTML = [
    tile("ROC-AUC", num(m.val.ROC_AUC, 3), "0.5 = chance"),
    tile("PR-AUC", num(m.val.PR_AUC, 3), "fault = positive"),
    tile("Current point", `${pct(r.far, 1)} → ${pct(r.recall, 1)}`, "false alarms → caught"),
  ].join("");
}

// ---------- 4. Fault by fault ----------
function renderFaults() {
  const v = state.view, rec = v.perFault(state.t), totals = v.faultTotals();
  barChart($("chart-faults"), {
    categories: v.faults.map(f => `Fault ${f}`),
    series: [{ name: "caught", color: COLORS.fault, values: rec }],
    x: { domain: [0, 1], fmt: x => pct(x, 0), label: "Share of the fault's rows caught" },
    labelWidth: 70, valueFmt: x => pct(x, 0),
    highlight: state.fault ? v.faults.indexOf(state.fault) : null,
    tip: i => `<b>Fault ${v.faults[i]}</b>: ${faultDesc(v.faults[i])}<br>Caught ${pct(rec[i])} of ${int(totals[i])} rows`,
    onClick: i => { state.fault = state.fault === v.faults[i] ? null : v.faults[i]; writeFault(); renderThresholdDependent(); },
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
    const order = v.faults.map((f, i) => [f, rec[i]]).sort((a, b) => a[1] - b[1]);
    panel.innerHTML = `<p style="margin-top:0"><b>Click a fault</b> to see what it is, how often it is caught, and how its scores look in section 2.</p>
      <p class="note">Hardest to catch at this threshold: ${order.slice(0, 3).map(([f, r]) => `fault ${f} (${pct(r, 0)})`).join(", ")}.
      Faults 3, 9 and 15 are reported in the TEP literature as nearly undetectable, because they barely change the
      statistics of the measured variables.</p>`;
    return;
  }
  const i = v.faults.indexOf(state.fault);
  const d = state.file.val.delays["0.05"].by_fault[state.fault];
  panel.innerHTML = `<div class="kicker">Fault ${state.fault}</div>
    <h3 style="margin:2px 0 4px">${FAULTS[state.fault][0]}</h3>
    <p class="note" style="margin:0 0 10px">Type: ${FAULTS[state.fault][1]}</p>
    <div class="tiles">
      ${tile("Rows caught", pct(rec[i]))}
      ${tile("Fault rows", int(v.faultTotals()[i]))}
      ${tile("Median delay at 5%", d ? fmtMinutes(d[0]) : "–")}
      ${tile("Runs detected at 5%", d ? pct(d[1], 0) : "–")}
    </div>
    <button class="btn" style="margin-top:12px" id="clear-fault">Clear selection</button>`;
  $("clear-fault").onclick = () => { state.fault = null; writeFault(); renderThresholdDependent(); };
}

// ---------- 5. Detection speed ----------
function renderSpeed() {
  const delays = state.file.val.delays[state.speedFar]?.by_fault || {};
  const faults = Object.keys(delays).map(Number).sort((a, b) => a - b);
  const med = faults.map(f => delays[f][0]), share = faults.map(f => delays[f][1]);
  const finite = med.filter(x => x != null);
  barChart($("chart-speed"), {
    categories: faults.map(f => `Fault ${f}`),
    series: [{ name: "median delay", color: COLORS.teal, values: med, opacity: share.map(s => 0.25 + 0.7 * s), missingLabel: "not detected" }],
    x: { type: "log", domain: [3, Math.max(1000, ...finite) * 1.3], ticks: [3, 10, 30, 60, 180, 600], fmt: x => fmtMinutes(x), label: "Median time to detection (log scale)" },
    labelWidth: 70, rowHeight: 21,
    valueFmt: (x, i) => `${pct(share[i], 0)} of runs`,
    highlight: state.fault ? faults.indexOf(state.fault) : null,
    tip: i => `<b>Fault ${faults[i]}</b>: ${faultDesc(faults[i])}<br>Median ${fmtMinutes(med[i])}, detected in ${pct(share[i], 0)} of runs`,
  });
  const allRuns = faults.filter((f, i) => share[i] >= 0.999);
  const fastest = faults.map((f, i) => [f, med[i]]).filter(x => x[1] != null).sort((a, b) => a[1] - b[1])[0];
  $("tiles-speed").innerHTML = [
    tile("Detected in every run", `${allRuns.length} / ${faults.length}`, "fault types"),
    tile("Fastest", fastest ? fmtMinutes(fastest[1]) : "–", fastest ? `fault ${fastest[0]}` : ""),
    tile("Median over faults", fmtMinutes(finite.sort((a, b) => a - b)[Math.floor(finite.length / 2)])),
  ].join("");
}
