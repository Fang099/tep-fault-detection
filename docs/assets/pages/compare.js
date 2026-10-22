import {
  renderChrome, showError, loadManifest, loadModel, dataNote, detectionView, modelLink,
  COLORS, faultDesc, pct, num, fmtC, fmtMinutes, esc,
} from "../site.js";
import { barChart, legend } from "../charts.js";

renderChrome();

const state = { manifest: null, task: "detection", a: null, b: null, sort: null, token: 0 };
const $ = id => document.getElementById(id);
const A = COLORS.teal, B = COLORS.purple;

main().catch(showError);

async function main() {
  const manifest = state.manifest = await loadManifest();
  $("data-note").innerHTML = dataNote(manifest);
  const q = new URLSearchParams(location.search);
  if (q.get("task") === "diagnosis") state.task = "diagnosis";
  setTask(state.task, q.get("a"), q.get("b"));

  $("task").addEventListener("click", e => {
    const b = e.target.closest("button");
    if (b) setTask(b.dataset.v);
  });
  $("pick-a").addEventListener("change", e => { state.a = e.target.value; update(); });
  $("pick-b").addEventListener("change", e => { state.b = e.target.value; update(); });
  $("swap").addEventListener("click", () => { [state.a, state.b] = [state.b, state.a]; update(); });
}

function defaults(task) {
  const models = state.manifest[task].models;
  const a = state.manifest[task].best;
  const b = task === "detection"
    ? models.find(m => m.penalty === "none" && m.class_weight === "none")?.id
    : models.find(m => m.penalty === "baseline")?.id;
  return [a, b || models[0].id];
}

function setTask(task, a, b) {
  state.task = task;
  state.sort = null;
  $("task").querySelectorAll("button").forEach(x => x.classList.toggle("on", x.dataset.v === task));
  const models = state.manifest[task].models;
  const ids = new Set(models.map(m => m.id));
  const [da, db] = defaults(task);
  state.a = ids.has(a) ? a : da;
  state.b = ids.has(b) ? b : db;
  const opts = models.map(m => `<option value="${m.id}">${esc(m.label)}${m.id === state.manifest[task].best ? " ★ best" : ""}</option>`).join("");
  $("pick-a").innerHTML = opts;
  $("pick-b").innerHTML = opts;
  update();
}

async function update() {
  const token = ++state.token;
  $("pick-a").value = state.a;
  $("pick-b").value = state.b;
  const q = new URLSearchParams({ task: state.task, a: state.a, b: state.b });
  history.replaceState(null, "", "?" + q + location.hash);
  const models = state.manifest[state.task].models;
  const ma = models.find(m => m.id === state.a), mb = models.find(m => m.id === state.b);
  const [fa, fb] = await Promise.all([loadModel(state.task, ma.id), loadModel(state.task, mb.id)]).catch(e => { showError(e); return []; });
  if (token !== state.token || !fa) return;
  renderHead(ma, mb);
  renderFaults(ma, fa, mb, fb);
  renderSpeed(fa, fb);
  renderTable();
}

// ---------- 1. Head to head ----------
function renderHead(ma, mb) {
  const det = state.task === "detection";
  const rows = det ? [
    ["Faults caught (recall)", m => m.val.recall, pct, true],
    ["False alarm rate", m => 1 - m.val.normal_recall, pct, false],
    ["Precision", m => m.val.precision, pct, true],
    ["F1 (fault class)", m => m.val.F1, v => num(v, 3), true],
    ["ROC-AUC", m => m.val.ROC_AUC, v => num(v, 3), true],
    ["PR-AUC", m => m.val.PR_AUC, v => num(v, 3), true],
    ["Sensors used", m => m.coef_nonzero, v => (v == null ? "–" : `${v} / 52`), null],
    ["Training time", m => m.fit_seconds, v => (v == null ? "–" : `${num(v, 0)} s`), false],
  ] : [
    ["Macro-F1", m => m.val.macro_F1, v => num(v, 3), true],
    ["Accuracy", m => m.val.accuracy, pct, true],
    ["Sensors used", m => m.coef_nonzero, v => (v == null ? "–" : `${v} / 52`), null],
    ["Training time", m => m.fit_seconds, v => (v == null ? "–" : `${num(v, 0)} s`), false],
  ];
  $("head-intro").textContent = det
    ? "Validation metrics at each model's own 5% false-alarm threshold. The last column shows A minus B; green means A is better."
    : "Validation metrics on rows where a fault is active. The last column shows A minus B; green means A is better.";
  $("head-table").innerHTML = `<thead><tr><th></th><th class="n"><span class="badge">A</span> ${esc(ma.label)}</th>
      <th class="n"><span class="badge purple">B</span> ${esc(mb.label)}</th><th class="n">A − B</th></tr></thead><tbody>${
    rows.map(([name, get, fmt, higherBetter]) => {
      const va = get(ma), vb = get(mb);
      let delta = "";
      if (va != null && vb != null && higherBetter !== null) {
        const d = va - vb, tol = name === "Training time" ? 0.5 : 5e-4;
        const better = higherBetter ? d > tol : d < -tol, worse = higherBetter ? d < -tol : d > tol;
        const shown = fmt === pct ? `${d >= 0 ? "+" : ""}${(100 * d).toFixed(1)} pts` : name === "Training time" ? `${d >= 0 ? "+" : ""}${num(d, 0)} s` : `${d >= 0 ? "+" : ""}${num(d, 3)}`;
        delta = `<span class="${better ? "delta-up" : worse ? "delta-down" : ""}">${shown}</span>`;
      }
      return `<tr><td>${name}</td><td class="n">${fmt(va)}</td><td class="n">${fmt(vb)}</td><td class="n">${delta}</td></tr>`;
    }).join("")}</tbody>`;
  const main = det ? ["faults caught at 5% false alarms", ma.val.recall - mb.val.recall, pct]
                   : ["macro-F1", ma.val.macro_F1 - mb.val.macro_F1, v => num(v, 3)];
  $("head-callout").innerHTML = Math.abs(main[1]) < 0.005
    ? `<b>Practically a tie</b> on ${main[0]}: the difference is ${main[2](Math.abs(main[1]))}.`
    : `<b>Model ${main[1] > 0 ? "A" : "B"} is ahead</b> on ${main[0]} by ${main[2](Math.abs(main[1]))}.`;
}

// ---------- 2. Fault by fault ----------
function perFault(model, file) {
  if (state.task === "detection") {
    const v = detectionView(file.val);
    const t = model.penalty === "baseline" ? 0 : v.indexForFar(0.05);
    return { faults: v.faults, values: v.perFault(t) };
  }
  return { faults: file.val.labels, values: file.val.recall };
}

function renderFaults(ma, fa, mb, fb) {
  const pa = perFault(ma, fa), pb = perFault(mb, fb);
  $("faults-intro").textContent = state.task === "detection"
    ? "Share of each fault's rows caught, with each model at its own 5% false-alarm threshold (read from binned scores, so within about half a point of the exact value)."
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

// ---------- 3. Speed ----------
function delaysOf(file) {
  return state.task === "detection" ? file.val.delays["0.05"].by_fault : file.val.delays;
}
function renderSpeed(fa, fb) {
  const da = delaysOf(fa), db = delaysOf(fb);
  const faults = Object.keys(da).map(Number).sort((a, b) => a - b);
  const all = faults.flatMap(f => [da[f][0], db[f]?.[0]]).filter(x => x != null);
  $("speed-intro").textContent = state.task === "detection"
    ? "Median minutes from fault start to 3 alarms in a row, at 5% false alarms. Faded bars: detected in only some runs."
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
function renderTable() {
  const det = state.task === "detection";
  const cols = [
    ["Model", m => m.label, "s"],
    ...(det ? [["Weighting", m => (m.penalty === "baseline" ? "–" : m.class_weight), "s"]] : []),
    ["Penalty", m => ({ baseline: "baseline", none: "none", l1: "lasso", l2: "ridge" }[m.penalty]), "s"],
    ["C", m => m.C, "n", v => (v == null ? "–" : fmtC(v))],
    ["Sensors", m => m.coef_nonzero, "n", v => (v == null ? "–" : v)],
    ...(det ? [
      // At 5% false alarms the baseline (one score for every row) cannot raise any alarm, so it catches 0%.
      ["Caught @5% FA", m => (m.penalty === "baseline" ? 0 : m.val.recall), "n", pct],
      ["Precision", m => (m.penalty === "baseline" ? null : m.val.precision), "n", pct],
      ["ROC-AUC", m => m.val.ROC_AUC, "n", v => num(v, 3)],
    ] : [
      ["Macro-F1", m => m.val.macro_F1, "n", v => num(v, 3)],
      ["Accuracy", m => m.val.accuracy, "n", pct],
    ]),
  ];
  const key = det ? 5 : 4;
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
    }).join("")}<td style="white-space:nowrap"><button class="btn" data-a="${m.id}" style="padding:2px 8px">A</button>
      <button class="btn" data-b="${m.id}" style="padding:2px 8px">B</button>
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
