import {
  renderChrome, showError, loadManifest, loadModel, dataNote, detectionView, modelLink, auc,
  metricsAtFar, loadAllDetection, nearestDelayLevel, farControl, OFFICIAL_FAR,
  COLORS, bestModel, familyOf, familyLabel, settingsText, faultDesc, pct, num, fmtC, fmtMinutes, esc,
} from "../site.js";
import { barChart, lineChart, legend } from "../charts.js";

renderChrome();

const state = { manifest: null, task: "detection", a: null, b: null, sort: null, token: 0, far: OFFICIAL_FAR, rocZoom: false, files: null, all: null };
const $ = id => document.getElementById(id);
const A = COLORS.a, B = COLORS.b;

main().catch(showError);

async function main() {
  const manifest = state.manifest = await loadManifest();
  $("data-note").innerHTML = dataNote(manifest);
  const q = new URLSearchParams(location.search);
  if (q.get("task") === "diagnosis") state.task = "diagnosis";
  const levels = manifest.settings?.far_levels || [0.01, 0.02, 0.05, 0.1, 0.2];
  if (q.get("far") && levels.some(l => Math.abs(l - +q.get("far")) < 1e-9)) state.far = +q.get("far");
  farControl($("far"), levels, state.far, far => { state.far = far; rerender(); });
  $("roc-zoom").addEventListener("click", e => {
    const b = e.target.closest("button"); if (!b) return;
    state.rocZoom = b.dataset.v === "zoom";
    $("roc-zoom").querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b));
    renderRoc();
  });
  setTask(state.task, q.get("a"), q.get("b"));

  $("task").addEventListener("click", e => {
    const b = e.target.closest("button");
    if (b) setTask(b.dataset.v);
  });
  $("pick-a").addEventListener("change", e => { state.a = e.target.value; update(); });
  $("pick-b").addEventListener("change", e => { state.b = e.target.value; update(); });
  $("swap").addEventListener("click", () => { [state.a, state.b] = [state.b, state.a]; update(); });
}

// Default comparison: the best model overall against the best model of a different family
// (logistic regression if the winner is a tree model, otherwise the best of the other families).
function defaults(task) {
  const m = state.manifest, a = bestModel(m, task);
  const famA = familyOf(a);
  let b = famA !== "logreg" ? bestModel(m, task, "logreg") : null;
  if (!b) {
    const others = m[task].models.filter(x => !["baseline", famA].includes(familyOf(x)));
    b = others.sort((x, y) => (task === "detection" ? y.val.recall - x.val.recall : y.val.macro_F1 - x.val.macro_F1))[0]
        || m[task].models.find(x => familyOf(x) === "baseline");
  }
  return [a.id, b.id];
}

function setTask(task, a, b) {
  state.task = task;
  state.sort = null;
  $("task").querySelectorAll("button").forEach(x => x.classList.toggle("on", x.dataset.v === task));
  $("far-wrap").hidden = task !== "detection";
  $("roc").hidden = task !== "detection";
  document.querySelector('#sidetoc a[href="#roc"]')?.parentElement?.toggleAttribute("hidden", task !== "detection");
  const models = state.manifest[task].models;
  const ids = new Set(models.map(m => m.id));
  const [da, db] = defaults(task);
  state.a = ids.has(a) ? a : da;
  state.b = ids.has(b) ? b : db;
  // Group the menu by model family.
  const opts = state.manifest.families.map(f => {
    const items = models.filter(m => familyOf(m) === f.key);
    return items.length ? `<optgroup label="${f.label}">${items.map(m =>
      `<option value="${m.id}">${esc(m.label)}${m.id === state.manifest[task].best ? " (best)" : ""}</option>`).join("")}</optgroup>` : "";
  }).join("");
  $("pick-a").innerHTML = opts;
  $("pick-b").innerHTML = opts;
  update();
}

async function update() {
  const token = ++state.token;
  $("pick-a").value = state.a;
  $("pick-b").value = state.b;
  const q = new URLSearchParams({ task: state.task, a: state.a, b: state.b });
  if (state.task === "detection" && Math.abs(state.far - OFFICIAL_FAR) > 1e-9) q.set("far", state.far);
  history.replaceState(null, "", "?" + q + location.hash);
  const models = state.manifest[state.task].models;
  const ma = models.find(m => m.id === state.a), mb = models.find(m => m.id === state.b);
  const [fa, fb] = await Promise.all([loadModel(state.task, ma.id), loadModel(state.task, mb.id)]).catch(e => { showError(e); return []; });
  if (token !== state.token || !fa) return;
  state.files = { ma, fa, mb, fb };
  rerender(true);
}

function rerender(fromUpdate) {
  if (!state.files) return;
  if (!fromUpdate) {
    const q = new URLSearchParams(location.search);
    if (Math.abs(state.far - OFFICIAL_FAR) > 1e-9) q.set("far", state.far); else q.delete("far");
    history.replaceState(null, "", "?" + q + location.hash);
  }
  const { ma, fa, mb, fb } = state.files;
  renderHead(ma, fa, mb, fb);
  renderFaults(ma, fa, mb, fb);
  if (state.task === "detection") renderRoc();
  renderSpeed(fa, fb);
  renderTable();
}

const farText = () => `${+(100 * state.far).toFixed(1)}%`;
const official = () => Math.abs(state.far - OFFICIAL_FAR) < 1e-9;

// Detection metrics at 5% false alarms (from the summary, so it works for every model without loading its file).
// The baseline gives every row the same score, so to stay within any false alarm budget it raises no alarm at all.
function at5(m) {
  if (m.penalty !== "baseline") return m.val;
  return { ...m.val, recall: 0, normal_recall: 1, precision: null, F1: 0 };
}
const ciText = (ci, fmt) => (ci && ci[1] > ci[0] ? `<div class="note">${fmt(ci[0])} – ${fmt(ci[1])}</div>` : "");

// ---------- 1. Head to head ----------
function renderHead(ma, fa, mb, fb) {
  const det = state.task === "detection";
  const mt = det ? new Map([[ma.id, metricsAtFar(ma, fa, state.far)], [mb.id, metricsAtFar(mb, fb, state.far)]]) : null;
  const used = v => (v == null ? "–" : v), inputsOf = m => m.n_inputs || 52;
  const usedRow = ["Inputs used", m => m.coef_nonzero, (v, m) => (v == null ? `all ${inputsOf(m)}` : `${v} / ${inputsOf(m)}`), null];
  const rows = det ? [
    ["Faults caught (recall)", m => mt.get(m.id).recall, pct, true, m => mt.get(m.id).ci?.recall],
    ["False alarm rate", m => mt.get(m.id).far, pct, false, m => mt.get(m.id).ci?.far],
    ["Precision", m => mt.get(m.id).precision, pct, true, m => mt.get(m.id).ci?.precision],
    ["F1 (fault class)", m => mt.get(m.id).F1, v => num(v, 3), true, m => mt.get(m.id).ci?.F1],
    ["ROC-AUC (all rates)", m => m.val.ROC_AUC, v => num(v, 3), true],
    ["PR-AUC (all rates)", m => m.val.PR_AUC, v => num(v, 3), true],
    usedRow,
    ["Training time", m => m.fit_seconds, v => (v == null ? "–" : `${num(v, 0)} s`), false],
  ] : [
    ["Macro-F1", m => m.val.macro_F1, v => num(v, 3), true, m => m.ci?.macro_F1],
    ["Accuracy", m => m.val.accuracy, pct, true, m => m.ci?.accuracy],
    usedRow,
    ["Training time", m => m.fit_seconds, v => (v == null ? "–" : `${num(v, 0)} s`), false],
  ];
  const hasCi = det ? official() && (ma.ci || mb.ci) : (ma.ci || mb.ci);
  $("head-intro").innerHTML = (det
    ? `Validation metrics with each model at its own ${farText()} false-alarm threshold${official() ? " (the official comparison)" : ""}. The last column shows A minus B; green means A is better.`
    : "Validation metrics on rows where a fault is active. The last column shows A minus B; green means A is better.")
    + (hasCi ? " Small grey ranges are 95% bootstrap intervals over validation runs." : "")
    + (det && !official() ? " Intervals are computed only at the official 5% rate." : "");
  $("head-table").innerHTML = `<thead><tr><th></th><th class="n"><span class="badge a">A</span> ${esc(ma.label)}</th>
      <th class="n"><span class="badge b">B</span> ${esc(mb.label)}</th><th class="n">A − B</th></tr></thead><tbody>${
    rows.map(([name, get, fmt, higherBetter, ciOf]) => {
      const va = get(ma), vb = get(mb);
      let delta = "";
      if (va != null && vb != null && higherBetter !== null) {
        const d = va - vb, tol = name === "Training time" ? 0.5 : 5e-4;
        const better = higherBetter ? d > tol : d < -tol, worse = higherBetter ? d < -tol : d > tol;
        const shown = fmt === pct ? `${d >= 0 ? "+" : ""}${(100 * d).toFixed(1)} pts` : name === "Training time" ? `${d >= 0 ? "+" : ""}${num(d, 0)} s` : `${d >= 0 ? "+" : ""}${num(d, 3)}`;
        delta = `<span class="${better ? "delta-up" : worse ? "delta-down" : ""}">${shown}</span>`;
      }
      const cf = fmt === pct ? x => pct(x) : x => num(x, 3);
      return `<tr><td>${name}</td><td class="n">${fmt(va, ma)}${ciOf ? ciText(ciOf(ma), cf) : ""}</td><td class="n">${fmt(vb, mb)}${ciOf ? ciText(ciOf(mb), cf) : ""}</td><td class="n">${delta}</td></tr>`;
    }).join("")}</tbody>`;
  const main = det ? [`faults caught at ${farText()} false alarms`, mt.get(ma.id).recall - mt.get(mb.id).recall, v => `${(100 * v).toFixed(1)} percentage points`]
                   : ["macro-F1", ma.val.macro_F1 - mb.val.macro_F1, v => num(v, 3)];
  // Do the 95% intervals overlap? A rough guide to whether a difference is more than run-to-run noise.
  const ciA = det ? (official() ? ma.ci?.recall : null) : ma.ci?.macro_F1, ciB = det ? (official() ? mb.ci?.recall : null) : mb.ci?.macro_F1;
  const overlap = ciA && ciB && ciA[1] > ciA[0] ? ciA[0] <= ciB[1] && ciB[0] <= ciA[1] : null;
  const noise = overlap == null ? "" : overlap
    ? " Their 95% intervals overlap, so with this many validation runs the gap may not hold on new runs."
    : " Their 95% intervals don't overlap, so the gap is larger than the run-to-run variation.";
  $("head-callout").innerHTML = (Math.abs(main[1]) < 0.005
    ? `<b>Practically a tie</b> on ${main[0]}: the difference is ${main[2](Math.abs(main[1]))}.`
    : `<b>Model ${main[1] > 0 ? "A" : "B"} is ahead</b> on ${main[0]} by ${main[2](Math.abs(main[1]))}.`) + noise
    + (det && !official() ? ` <span class="note">The official ranking stays at 5%; other rates show how robust the comparison is.</span>` : "");
}

// ---------- 2. Fault by fault ----------
function perFault(model, file) {
  if (state.task === "detection") {
    const v = detectionView(file.val);
    const r = metricsAtFar(model, file, state.far);
    return { faults: v.faults, values: r.perFault || v.faults.map(() => 0) };
  }
  return { faults: file.val.labels, values: file.val.recall };
}

function renderFaults(ma, fa, mb, fb) {
  const pa = perFault(ma, fa), pb = perFault(mb, fb);
  $("faults-intro").textContent = state.task === "detection"
    ? `Share of each fault's rows caught, with each model at its own ${farText()} false-alarm threshold.`
    : "Share of each fault's rows named correctly.";
  legend($("legend-faults"), [{ label: `A · ${ma.label}`, color: A }, { label: `B · ${mb.label}`, color: B }]);
  barChart($("chart-faults"), {
    categories: pa.faults.map(f => `Fault ${f}`),
    series: [{ name: "A", color: A, values: pa.values }, { name: "B", color: B, values: pb.values }],
    x: { domain: [0, 1], fmt: x => pct(x, 0), label: state.task === "detection" ? "Share of rows caught" : "Recall" },
    labelWidth: 70, valueFmt: x => pct(x, 0),
    tip: i => `<b>Fault ${pa.faults[i]}</b>: ${faultDesc(pa.faults[i])}<br>A: <b>${pct(pa.values[i])}</b> · B: <b>${pct(pb.values[i])}</b>`,
  });
}

// ---------- ROC overlay ----------
function renderRoc() {
  const { ma, fa, mb, fb } = state.files;
  const curve = (m, f) => (familyOf(m) === "baseline" ? [[0, 0], [1, 1]] : detectionView(f.val).roc());
  const ca = curve(ma, fa), cb = curve(mb, fb);
  const ra = metricsAtFar(ma, fa, state.far), rb = metricsAtFar(mb, fb, state.far);
  const xmax = state.rocZoom ? 0.2 : 1;
  const clipTo = c => c.filter(p => p[0] <= xmax + 1e-9);
  legend($("legend-roc"), [{ label: `A · ${ma.label}`, color: A }, { label: `B · ${mb.label}`, color: B },
                           { label: "Random guessing", color: COLORS.muted, dash: true }]);
  lineChart($("chart-roc"), {
    height: 340,
    x: { domain: [0, xmax], fmt: x => pct(x, 0), label: "False alarm rate (share of normal rows flagged)" },
    y: { domain: [0, 1], fmt: y => pct(y, 0), label: "Faults caught (recall)" },
    series: [
      { name: "Random", color: COLORS.muted, width: 1, dash: "4 4", points: [[0, 0], [xmax, xmax]], hover: false },
      { name: "A", color: A, width: 2.4, points: clipTo(ca) },
      { name: "B", color: B, width: 2.4, points: clipTo(cb) },
    ],
    vlines: [{ x: state.far, color: COLORS.muted, dash: "3 3", label: `${farText()} false alarms` }],
    markers: [{ x: ra.far, y: ra.recall, color: A, label: "A" }, { x: rb.far, y: rb.recall, color: B, label: "B" }],
    tip: (s_, p) => `<b>Model ${s_.name}</b><br>False alarms ${pct(p[0])}<br>Faults caught <b>${pct(p[1])}</b>`,
    onClick: (s_, p) => {
      const levels = state.manifest.settings?.far_levels || [0.01, 0.02, 0.05, 0.1, 0.2];
      const near = levels.reduce((a, b) => (Math.abs(b - p[0]) < Math.abs(a - p[0]) ? b : a));
      $("far").querySelector(`button[data-v="${near}"]`)?.click();
    },
    hoverRadius: 18,
  });
  const aa = auc(ca), ab = auc(cb);
  // Do the curves cross? Compare recall at each listed rate.
  const levels = state.manifest.settings?.far_levels || [0.01, 0.02, 0.05, 0.1, 0.2];
  const wins = levels.map(l => metricsAtFar(ma, fa, l).recall - metricsAtFar(mb, fb, l).recall);
  const aAll = wins.every(d => d >= -0.005), bAll = wins.every(d => d <= 0.005);
  $("roc-callout").innerHTML = aAll && bAll ? "<b>The two curves practically coincide</b> over the listed false alarm rates."
    : aAll || bAll ? `<b>Model ${aAll ? "A" : "B"} catches at least as many faults at every listed false alarm rate</b> (${pct(levels[0], 0)} to ${pct(levels[levels.length - 1], 0)}), so the choice of rate doesn't change the winner. Area under the curve: A ${num(aa, 3)}, B ${num(ab, 3)}.`
    : `<b>The curves cross:</b> ${levels.filter((_, k) => wins[k] > 0.005).map(l => pct(l, 0)).join(", ") || "none"} favour A and
       ${levels.filter((_, k) => wins[k] < -0.005).map(l => pct(l, 0)).join(", ") || "none"} favour B. Which model is better depends on how many false alarms the plant can tolerate.`;
}

// ---------- 3. Speed ----------
function delaysOf(file) {
  if (state.task !== "detection") return file.val.delays;
  return file.val.delays[nearestDelayLevel(file.val.delays, state.far)].by_fault;
}
function renderSpeed(fa, fb) {
  const da = delaysOf(fa), db = delaysOf(fb);
  const faults = Object.keys(da).map(Number).sort((a, b) => a - b);
  const all = faults.flatMap(f => [da[f][0], db[f]?.[0]]).filter(x => x != null);
  $("speed-intro").textContent = state.task === "detection"
    ? `Median minutes from fault start to 3 alarms in a row, at ${pct(+nearestDelayLevel(fa.val.delays, state.far), 0)} false alarms. Faded bars: detected in only some runs.`
    : "Median minutes from fault start to 3 correct diagnoses in a row. Faded bars: achieved in only some runs.";
  legend($("legend-speed"), [{ label: "A", color: A }, { label: "B", color: B }]);
  barChart($("chart-speed"), {
    categories: faults.map(f => `Fault ${f}`),
    series: [
      { name: "A", color: A, values: faults.map(f => da[f][0]), opacity: faults.map(f => 0.25 + 0.7 * da[f][1]), missingLabel: "never" },
      { name: "B", color: B, values: faults.map(f => db[f]?.[0] ?? null), opacity: faults.map(f => 0.25 + 0.7 * (db[f]?.[1] ?? 0)), missingLabel: "never" },
    ],
    x: { type: "log", domain: [3, Math.max(1000, ...all) * 1.3], ticks: [3, 10, 30, 60, 180, 600], fmt: fmtMinutes, label: "Median time (log scale)" },
    labelWidth: 70,
    tip: i => { const f = faults[i]; return `<b>Fault ${f}</b><br>A: ${fmtMinutes(da[f][0])} (${pct(da[f][1], 0)} of runs)<br>B: ${fmtMinutes(db[f]?.[0])} (${pct(db[f]?.[1] ?? 0, 0)} of runs)`; },
  });
}

// ---------- 4. All models ----------
function atFar(m) {
  if (official() || familyOf(m) === "baseline") { const v = at5(m); return { recall: v.recall, precision: v.precision }; }
  const f = state.all?.[m.id];
  return f ? metricsAtFar(m, f, state.far) : { recall: null, precision: null };
}

function renderTable() {
  const det = state.task === "detection";
  if (det && !official() && !state.all) {
    $("table-intro").textContent = "Loading every model's scores to rank them at this false alarm rate…";
    loadAllDetection(state.manifest, (k, n) => { $("table-intro").textContent = `Loading every model's scores… ${k} of ${n}`; })
      .then(all => { state.all = all; renderTable(); })
      .catch(showError);
  }
  $("table-intro").textContent = det
    ? `All trained detection models, ranked on validation at ${farText()} false alarms${official() ? " (the official rate)" : ""}. Click a column to sort, or use the buttons to load a model as A or B.`
    : "All trained diagnosis models, ranked on validation. Click a column to sort, or use the buttons to load a model as A or B.";
  const cols = [
    ["Model", m => m.label, "s"],
    ["Family", m => familyLabel(state.manifest, familyOf(m)), "s"],
    ["Settings", m => settingsText(m), "s"],
    ["Sensors used", m => m.coef_nonzero, "n", v => (v == null ? "–" : v)],
    ...(det ? [
      // The baseline (one score for every row) cannot raise any alarm within the budget, so it catches 0%.
      [`Caught @${farText()} FA`, m => atFar(m).recall, "n", pct],
      ["Precision", m => atFar(m).precision, "n", pct],
      ["ROC-AUC", m => m.val.ROC_AUC, "n", v => num(v, 3)],
    ] : [
      ["Macro-F1", m => m.val.macro_F1, "n", v => num(v, 3)],
      ["Accuracy", m => m.val.accuracy, "n", pct],
    ]),
  ];
  const key = det ? 4 : 4;
  if (!state.sort) state.sort = { col: key, desc: true };
  const { col, desc } = state.sort;
  const models = [...state.manifest[state.task].models].sort((x, y) => {
    let a = cols[col][1](x), b = cols[col][1](y);
    if (a == null) return 1; if (b == null) return -1;
    const r = typeof a === "string" ? a.localeCompare(b) : a - b;
    return desc ? -r : r;
  });
  const best = state.manifest[state.task].best;
  const page = det ? "detection.html" : "diagnosis.html";
  $("all-table").innerHTML = `<thead><tr>${cols.map(([name, , type], i) =>
      `<th class="sortable ${type === "n" ? "n" : ""}" data-col="${i}">${name}${i === col ? (desc ? " ▾" : " ▴") : ""}</th>`).join("")}<th></th></tr></thead>
    <tbody>${models.map(m => `<tr class="${m.id === best ? "highlight" : ""}">${cols.map(([, get, type, fmt]) => {
      const v = get(m);
      return `<td class="${type === "n" ? "n" : ""}">${fmt ? fmt(v) : esc(v)}${type === "s" && get === cols[0][1] && m.id === best ? ' <span class="badge best">best</span>' : ""}</td>`;
    }).join("")}<td style="white-space:nowrap"><button class="btn small" type="button" data-a="${m.id}" aria-label="Set as model A">A</button>
      <button class="btn small" type="button" data-b="${m.id}" aria-label="Set as model B">B</button>
      <a href="${modelLink(page, m)}" style="font-size:13px;margin-left:4px">open</a></td></tr>`).join("")}</tbody>`;
  $("all-table").onclick = e => {
    const th = e.target.closest("th[data-col]");
    if (th) {
      const c = +th.dataset.col;
      state.sort = { col: c, desc: state.sort.col === c ? !state.sort.desc : cols[c][2] === "n" };
      renderTable();
      return;
    }
    const b = e.target.closest("button");
    if (b?.dataset.a) { state.a = b.dataset.a; update(); window.scrollTo({ top: 0, behavior: "smooth" }); }
    if (b?.dataset.b) { state.b = b.dataset.b; update(); window.scrollTo({ top: 0, behavior: "smooth" }); }
  };
}
