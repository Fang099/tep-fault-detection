import { renderChrome, showError, loadManifest, getJSON, bestModel, VARIABLES, FAULTS, COLORS, pct, num, int, tile, esc } from "../site.js";
import { lineChart, heatmap, legend } from "../charts.js";

renderChrome();
const $ = id => document.getElementById(id);

// ---------- Plant diagram: which variables measure each unit ----------
const range = (prefix, a, b) => Array.from({ length: b - a + 1 }, (_, k) => `${prefix}_${a + k}`);
const UNITS = {
  feeds: { name: "Feeds", text: "Fresh reactants A, D and E flow into the reactor. A mix of A and C (stream 4) enters at the stripper.",
           vars: [...range("xmeas", 1, 4), ...range("xmv", 1, 4)] },
  reactor: { name: "Reactor", text: "Gas-phase reactions turn A, C, D and E into the products G and H. A cooling water loop removes the heat.",
             vars: ["xmeas_6", "xmeas_7", "xmeas_8", "xmeas_9", "xmeas_21", ...range("xmeas", 23, 28), "xmv_10"] },
  condenser: { name: "Condenser", text: "Cools the reactor outlet so the products condense into liquid.",
               vars: ["xmeas_22", "xmv_11"] },
  separator: { name: "Separator", text: "Splits the cooled stream: vapour goes back to the reactor, liquid goes on to the stripper.",
               vars: range("xmeas", 11, 14).concat(["xmv_7"]) },
  compressor: { name: "Compressor", text: "Pushes unreacted gas from the separator back to the reactor (the recycle, stream 8).",
                vars: ["xmeas_5", "xmeas_20", "xmv_5"] },
  purge: { name: "Purge", text: "Bleeds off some recycle gas so inert gases and by-products don't build up. Its composition is analyzed.",
           vars: ["xmeas_10", ...range("xmeas", 29, 36), "xmv_6"] },
  stripper: { name: "Stripper", text: "Strips leftover reactants out of the liquid, leaving the product stream (G and H). The product is analyzed.",
              vars: [...range("xmeas", 15, 19), ...range("xmeas", 37, 41), "xmv_8", "xmv_9"] },
};

const info = $("plant-info");
function showUnit(key) {
  document.querySelectorAll(".unit").forEach(u => u.classList.toggle("on", u.dataset.unit === key));
  if (!key) {
    info.innerHTML = `<h3>52 sensors, 7 places</h3>
      <p class="note" style="margin:6px 0 0">Hover over or select a unit to see which of the 52 variables describe it.
      These readings, taken every 3 minutes, are everything the models get to see.</p>`;
    return;
  }
  const u = UNITS[key];
  info.innerHTML = `<h3>${u.name}</h3><p class="note" style="margin:4px 0 0">${u.text}</p>
    <ul class="vars">${u.vars.map(v => `<li><code>${v}</code><span>${esc(VARIABLES[v])}</span></li>`).join("")}</ul>`;
}
let pinned = null;
document.querySelectorAll(".unit").forEach(g => {
  const key = g.dataset.unit;
  g.addEventListener("mouseenter", () => showUnit(key));
  g.addEventListener("mouseleave", () => showUnit(pinned));
  g.addEventListener("focus", () => showUnit(key));
  g.addEventListener("blur", () => showUnit(pinned));
  const pin = () => { pinned = pinned === key ? null : key; showUnit(pinned ?? key); };
  g.addEventListener("click", pin);
  g.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pin(); } });
});
showUnit(null);

// ---------- Reference tables ----------
$("var-table").innerHTML = `<thead><tr><th>Variable</th><th>Description</th></tr></thead><tbody>${
  Object.entries(VARIABLES).map(([k, v]) => `<tr><td><code>${k}</code></td><td>${esc(v)}</td></tr>`).join("")}</tbody>`;
$("fault-table").innerHTML = `<thead><tr><th class="n">Fault</th><th>Process variable</th><th>Type</th></tr></thead><tbody>${
  Object.entries(FAULTS).map(([k, [d, t]]) => `<tr><td class="n">${k}</td><td>${esc(d)}</td><td>${t}</td></tr>`).join("")}</tbody>`;

// ---------- Label timelines ----------
function timeline(title, total, onset, sub) {
  const normalW = onset == null ? 100 : (onset / total) * 100;
  return `<div style="margin:0 0 18px">
    <div style="display:flex;justify-content:space-between;gap:12px;font-size:15px"><b>${title}</b><span class="note">${sub}</span></div>
    <div style="display:flex;height:24px;border-radius:4px;overflow:hidden;margin:5px 0 3px;border:1px solid var(--border)">
      <div style="width:${normalW}%;background:${COLORS.normal}"></div>
      ${onset == null ? "" : `<div style="flex:1;background:${COLORS.fault}"></div>`}
    </div>
    <div class="ticks" style="position:relative"><span>sample 1</span>${onset == null ? "" :
      `<span style="position:absolute;left:${normalW}%;padding-left:5px;border-left:2px solid ${COLORS.fault}">fault starts after sample ${onset}</span>`}<span>sample ${total}</span></div>
  </div>`;
}
$("timelines").innerHTML = `
  <div class="legend" style="margin:0 0 12px"><span style="--c:${COLORS.normal}">labelled normal (0)</span><span style="--c:${COLORS.fault}">labelled faulty (1)</span></div>
  ${timeline("Fault-free run (training or test)", 500, null, "every row normal")}
  ${timeline("Faulty training run", 500, 20, "25 hours, fault after 1 hour")}
  ${timeline("Faulty test run", 960, 160, "48 hours, fault after 8 hours")}`;

// ---------- Numbers from the results ----------
loadManifest().then(manifest => {
  const d = manifest.dataset, s = manifest.settings;
  $("data-tiles").innerHTML = [
    tile("Runs used per fault type", `${d.runs_per_fault} / 500`),
    tile("Training runs", int(d.train_runs)),
    tile("Validation runs", int(d.val_runs)),
    tile("Validation rows", int(d.val_rows), `${pct(d.val_fault_share, 0)} faulty`),
    tile("Inputs", "52", "process variables"),
  ].join("");
  $("subset-note").innerHTML = `Current results use ${d.runs_per_fault < 500 ? `the first ${d.runs_per_fault} of 500 runs per fault type` : "all 500 runs per fault type"}.
    To keep training time manageable, models train on every ${ordinal(s.det_stride)} time step for detection and every
    ${ordinal(s.diag_stride)} for diagnosis, since neighbouring 3-minute samples are nearly identical. Validation and test always use every row.`;

  const det = bestModel(manifest, "detection"), dia = bestModel(manifest, "diagnosis");
  const detBase = manifest.detection.models.find(m => m.penalty === "baseline");
  const diaBase = manifest.diagnosis.models.find(m => m.penalty === "baseline");
  $("results-intro").textContent = `Validation results for ${manifest.detection.models.length + manifest.diagnosis.models.length} models.
    ${manifest.test ? "Test results are on the Results page." : "The final test has not been run yet."}`;
  $("result-tiles").innerHTML = [
    tile("Best detection model", esc(det.label), "", "text"),
    tile("Faults caught at 5% false alarms", pct(det.val.recall), "baseline: 0%, it can only alarm on everything", "big"),
    tile("Detection ROC-AUC", num(det.val.ROC_AUC, 3), `baseline ${num(detBase.val.ROC_AUC, 3)}`, "big"),
    tile("Best diagnosis model", esc(dia.label), "", "text"),
    tile("Diagnosis macro-F1", num(dia.val.macro_F1, 3), `baseline ${num(diaBase.val.macro_F1, 3)}`, "big"),
  ].join("");
}).catch(showError);

function ordinal(k) { return k === 1 ? "" : k === 2 ? "2nd" : k === 3 ? "3rd" : `${k}th`; }

// ---------- Explore the data ----------
const ex = { summary: null, traces: null, fault: 1, sensor: null, pinnedPair: null };
Promise.all([getJSON("data/data_summary.json"), getJSON("data/data_traces.json")])
  .then(([summary, traces]) => { ex.summary = summary; ex.traces = traces; initExplore(); })
  .catch(() => {
    $("explore-body").innerHTML = `<div class="empty">The data plots appear once the results include the data summary
      (exported by the latest version of the pipeline).</div>`;
  });

const short = v => v.replace("xmeas_", "m").replace("xmv_", "v");

function initExplore() {
  const { summary, traces } = ex;
  const q = summary.quality;
  const copies = q.val_rows_copied_from_train;
  $("quality-tiles").innerHTML = [
    tile("Rows checked", int(q.rows), `${int(q.runs)} runs`),
    tile("Missing values", int(q.missing_values)),
    tile("Repeated rows", int(q.duplicate_rows), q.rows ? `${pct(q.duplicate_rows / q.rows)} of rows` : ""),
    tile("Constant variables", String(q.constant_columns.length)),
    tile("Normal / faulty rows", `${pct(q.class_counts.normal / q.rows, 0)} / ${pct(q.class_counts.faulty / q.rows, 0)}`),
    ...(copies != null ? [tile("Validation rows also in training", int(copies), copies === 0 ? "leakage check passed" : "should be 0", copies === 0 ? "" : "warn")] : []),
  ].join("");
  $("quality-note").innerHTML = `${q.missing_values === 0 && q.constant_columns.length === 0
    ? "Nothing is missing and no variable is constant, so nothing needed imputing or dropping."
    : "Some quality issues were found; see the counts above."}
    ${q.duplicate_rows > 0 ? ` The repeated rows are not errors: every faulty run starts as an exact copy of the fault-free run with the
      same simulation number, and stays identical until the fault starts to show. They are kept (they are genuine normal readings),
      but they decide how the data is split: <a href="#protocol">all copies of a simulation stay on the same side</a>.` : ""}
    The other real issues are the strong correlation between neighbouring samples (handled by splitting by run) and the class
    imbalance (handled by the choice of metrics).`;

  const faults = traces.runs.filter(r => r.fault > 0).map(r => r.fault);
  $("trace-fault").innerHTML = faults.map(f => `<option value="${f}">Fault ${f}: ${esc(FAULTS[f][0])} (${FAULTS[f][1].toLowerCase()})</option>`).join("");
  $("trace-fault").value = ex.fault;
  $("trace-fault").addEventListener("change", e => { ex.fault = +e.target.value; ex.sensor = null; renderSensorList(); renderTrace(); });
  $("trace-sensor").addEventListener("change", e => { ex.sensor = e.target.value; renderTrace(); });
  renderSensorList();
  renderTrace();
  renderCorr();
  renderPca();
}

// How much a fault changes each sensor in its run: mean shift after the fault starts (in normal standard deviations)
// and the ratio of its spread to normal spread.
function sensorChanges(fault) {
  const { summary, traces } = ex, run = traces.runs.find(r => r.fault === fault);
  return summary.features.map(f => {
    const v = run.values[f], st = summary.variables[f];
    const after = v.filter((_, k) => run.sample[k] > traces.onset);
    const mean = after.reduce((a, b) => a + b, 0) / after.length;
    const sd = Math.sqrt(after.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, after.length - 1));
    const shift = st.std > 0 ? (mean - st.mean) / st.std : 0, spread = st.std > 0 ? sd / st.std : 1;
    return { f, shift, spread, score: Math.abs(shift) + Math.abs(Math.log(Math.max(spread, 1e-9))) * 2 };
  }).sort((a, b) => b.score - a.score);
}

function renderSensorList() {
  const ch = sensorChanges(ex.fault);
  ex.changes = ch;
  if (!ex.sensor) ex.sensor = ch[0].f;
  $("trace-sensor").innerHTML = ch.map(c => `<option value="${c.f}">${c.f} · ${esc(VARIABLES[c.f])} (shift ${c.shift >= 0 ? "+" : ""}${num(c.shift, 1)}σ, spread ×${num(c.spread, 1)})</option>`).join("");
  $("trace-sensor").value = ex.sensor;
}

function renderTrace() {
  const { summary, traces } = ex, f = ex.sensor;
  const run = traces.runs.find(r => r.fault === ex.fault), normal = traces.runs.find(r => r.fault === 0);
  const hours = s => (s * traces.sample_minutes) / 60;
  const st = summary.variables[f];
  const series = [
    ...(normal ? [{ name: "Fault-free run", color: COLORS.normal, width: 1.5, points: normal.sample.map((s_, k) => [hours(s_), normal.values[f][k]]) }] : []),
    { name: `Fault ${ex.fault} run`, color: COLORS.fault, width: 1.8, points: run.sample.map((s_, k) => [hours(s_), run.values[f][k]]) },
  ];
  legend($("legend-trace"), series.map(x => ({ label: x.name, color: x.color })).concat([{ label: "Normal range (mean ± 2σ)", color: COLORS.muted, dash: true }]));
  const all = series.flatMap(x => x.points.map(p => p[1])).concat([st.mean - 2 * st.std, st.mean + 2 * st.std]);
  const lo = Math.min(...all), hi = Math.max(...all), pad = (hi - lo) * 0.06 || 1;
  lineChart($("chart-trace"), {
    height: 300, margin: { left: 70 },
    x: { domain: [0, 25], ticks: [0, 5, 10, 15, 20, 25], fmt: h => `${h} h`, label: "Time since the run started" },
    y: { domain: [lo - pad, hi + pad], fmt: y => (Math.abs(y) >= 1000 ? y.toFixed(0) : +y.toPrecision(4)), label: f },
    series,
    hlines: [{ y: st.mean + 2 * st.std, color: COLORS.muted, dash: "4 4" }, { y: st.mean - 2 * st.std, color: COLORS.muted, dash: "4 4" }],
    vlines: [{ x: hours(traces.onset), color: COLORS.text, dash: "4 3", label: "fault starts" }],
    tip: (s_, p) => `<b>${s_.name}</b><br>${(p[0] * 60).toFixed(0)} min: ${+p[1].toPrecision(5)}`,
    hoverRadius: 16,
  });
  const c = ex.changes.find(x => x.f === f);
  $("trace-tiles").innerHTML = [
    tile("Mean shift after the fault", `${c.shift >= 0 ? "+" : ""}${num(c.shift, 1)}σ`),
    tile("Spread vs normal", `×${num(c.spread, 2)}`),
    tile("Normal mean", String(+st.mean.toPrecision(5))),
    tile("Normal σ", String(+st.std.toPrecision(3))),
  ].join("");
  const top = ex.changes[0];
  $("trace-callout").innerHTML = top.score < 1
    ? `<b>Fault ${ex.fault} barely changes any sensor:</b> the largest shift is ${num(Math.abs(top.shift), 1)}σ. This is why some faults are nearly impossible to detect from a single snapshot.`
    : Math.abs(c.shift) < 0.5 && c.spread > 1.5
      ? `<b>This sensor's average hardly moves, but it fluctuates ${num(c.spread, 1)} times as much as normal.</b> A single snapshot can't show that; a rolling standard deviation (the history inputs) can.`
      : `<b>${f} ${Math.abs(c.shift) >= 2 ? "leaves" : "stays mostly inside"} its normal range after fault ${ex.fault} starts.</b>
         Note how long the change can take to build up: the readings respond with a delay that depends on the plant's dynamics.`;
}

function renderCorr() {
  const { summary } = ex, feats = summary.features, C = summary.correlation;
  const pairs = [];
  for (let i = 0; i < feats.length; i++) for (let j = i + 1; j < feats.length; j++) pairs.push([i, j, C[i][j]]);
  pairs.sort((a, b) => Math.abs(b[2]) - Math.abs(a[2]));
  heatmap($("chart-corr"), {
    matrix: C, rowLabels: feats.map(short), colLabels: feats.map(short), diverging: true, diagonal: false, labelEvery: window.innerWidth < 600 ? 10 : 5,
    ariaLabel: "correlation matrix of the 52 variables", rowTitle: "", colTitle: "",
    tip: (r, c) => `<b>${feats[r]}</b> ${esc(VARIABLES[feats[r]])}<br><b>${feats[c]}</b> ${esc(VARIABLES[feats[c]])}<br>correlation <b>${num(C[r][c], 2)}</b>`,
    onClick: (r, c) => { ex.pinnedPair = [r, c]; ex.sensor = feats[r]; $("trace-sensor").value = ex.sensor; renderTrace(); renderCorrPanel(pairs);
                         document.getElementById("chart-trace").scrollIntoView({ behavior: "smooth", block: "center" }); },
  });
  renderCorrPanel(pairs);
}

function renderCorrPanel(pairs) {
  const { summary } = ex, feats = summary.features;
  const strong = pairs.filter(p => Math.abs(p[2]) >= 0.8).length;
  $("corr-panel").innerHTML = `<h3 style="margin-top:0">Most strongly related pairs</h3>
    <table><tbody>${pairs.slice(0, 6).map(([i, j, r]) => `<tr><td>${feats[i]} and ${feats[j]}<div class="note">${esc(VARIABLES[feats[i]])} · ${esc(VARIABLES[feats[j]])}</div></td>
      <td class="n">${num(r, 2)}</td></tr>`).join("")}</tbody></table>
    <p class="note" style="margin:10px 0 0">${strong === 0 ? "No pair reaches a correlation of 0.8 in this data." : `${strong} pairs have`} ${strong === 0 ? "" : "a correlation of 0.8 or more in either direction."} Many sensors measure
      linked parts of the plant, so the 52 readings carry much less independent information than their number suggests. That is what PCA
      exploits, and why a model's individual coefficients can be unstable without regularization.</p>`;
}

function renderPca() {
  const v = ex.summary.pca_variance;
  let run = 0;
  const cum = v.map((x, k) => [k + 1, (run += x)]), ind = v.map((x, k) => [k + 1, x]);
  const k90 = cum.find(p => p[1] >= 0.9)?.[0], k99 = cum.find(p => p[1] >= 0.99)?.[0];
  legend($("legend-pca"), [{ label: "Running total", color: COLORS.data }, { label: "Each component", color: COLORS.fault }]);
  lineChart($("chart-pca"), {
    height: 280,
    x: { domain: [0.5, v.length + 0.5], ticks: [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 52].filter(t => t <= v.length), label: "Principal component" },
    y: { domain: [0, 1], fmt: y => pct(y, 0), label: "Share of normal variation" },
    series: [
      { name: "Each component", color: COLORS.fault, width: 1.5, marker: 2.5, points: ind },
      { name: "Running total", color: COLORS.data, width: 2.4, marker: 3, points: cum },
    ],
    hlines: [{ y: 0.9, color: COLORS.muted, label: "90%", dash: "4 4" }],
    vlines: k90 ? [{ x: k90, color: COLORS.amber, dash: "4 3", label: `${k90} components` }] : [],
    tip: (s_, p) => `Component ${p[0]}: <b>${pct(ind[p[0] - 1][1])}</b><br>First ${p[0]} together: <b>${pct(cum[p[0] - 1][1])}</b>`,
    hoverRadius: 14,
  });
  $("pca-callout").innerHTML = `<b>${k90} of the 52 components hold 90% of normal variation</b>${k99 ? `, and ${k99} hold 99%` : ""}.
    The first component alone captures ${pct(v[0])}. ${k90 <= v.length * 0.6
      ? "Normal operation varies along far fewer directions than there are sensors, which is the idea behind"
      : "Here the variation is spread over many directions, so a few components can't summarize normal operation well. That limits"} <a href="detection.html?family=pcamon">PCA monitoring</a>${k90 <= v.length * 0.6 ? ": anything far outside those directions is suspicious" : ""}.`;
}
