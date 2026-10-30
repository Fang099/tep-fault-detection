import {
  renderChrome, showError, loadManifest, loadModel, dataNote, findModel as findAny, modelLink,
  COLORS, PALETTE, VARIABLES, pct, num, int, fmtC, tile, esc,
} from "../site.js";
import { lineChart, barChart, legend } from "../charts.js";
import * as M from "../math.js";

// Every model on this page is a logistic regression.
const findModel = (manifest, task, sel) => findAny(manifest, task, { family: "logreg", ...sel });

renderChrome();

const state = { manifest: null, task: "detection", penalty: "l1", weighting: "balanced", fault: "all", ci: 3, token: 0, files: null };
const $ = id => document.getElementById(id);

main().catch(showError);

async function main() {
  const manifest = state.manifest = await loadManifest();
  $("data-note").innerHTML = dataNote(manifest);
  const grid = manifest.settings.C_grid;
  const slider = $("path-c");
  slider.max = grid.length - 1;
  slider.value = state.ci;
  $("path-ticks").innerHTML = grid.map(c => `<span>${fmtC(c)}</span>`).join("");
  $("path-fault").innerHTML = `<option value="all">All faults (size of each sensor's coefficients)</option>` +
    Array.from({ length: 20 }, (_, i) => `<option value="${i + 1}">Fault ${i + 1}</option>`).join("");

  segment("task", v => { state.task = v; renderPerformance(); renderObjective(); loadPaths(); });
  segment("path-penalty", v => { state.penalty = v; loadPaths(); });
  segment("path-weight", v => { state.weighting = v; loadPaths(); });
  $("path-fault").addEventListener("change", e => { state.fault = e.target.value; renderPaths(); });
  slider.addEventListener("input", () => { state.ci = +slider.value; renderPaths(); renderObjective(); });

  renderStaticFormulas();
  $("lambda").addEventListener("input", renderShrink);
  renderShrink();
  renderPerformance();
  renderObjective();
  await loadPaths();
}

function segment(id, onPick) {
  $(id).addEventListener("click", e => {
    const b = e.target.closest("button");
    if (!b) return;
    $(id).querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b));
    onPick(b.dataset.v);
  });
}

const metricOf = (task, m) => (task === "detection" ? m.val.recall : m.val.macro_F1);
const metricName = task => (task === "detection" ? "Faults caught at 5% false alarms" : "Macro-F1");
const fmtMetric = (task, v) => (task === "detection" ? pct(v) : num(v, 3));

// ---------- 1. Performance against C ----------
function renderPerformance() {
  const { manifest, task } = state;
  const grid = manifest.settings.C_grid;
  // Regularization is studied on logistic regression with instantaneous inputs.
  const models = manifest[task].models.filter(m => (m.family || "logreg") === "logreg" && (m.features || "inst") === "inst");
  const weightings = task === "detection" ? ["balanced", "none"] : [null];
  const series = [], hlines = [], items = [];
  for (const pen of ["l1", "l2"]) {
    for (const w of weightings) {
      const pts = grid.map(C => {
        const m = findModel(manifest, task, { penalty: pen, weighting: w || "none", C });
        return m ? [C, metricOf(task, m), m] : null;
      }).filter(Boolean);
      const name = `${pen === "l1" ? "Lasso" : "Ridge"}${w ? (w === "balanced" ? ", balanced" : ", unweighted") : ""}`;
      const color = pen === "l1" ? COLORS.l1 : COLORS.l2;
      series.push({ name, color, points: pts, marker: 4, width: 2.2, dash: w === "none" ? "6 4" : null, opacity: w === "none" ? 0.6 : 1 });
      items.push({ label: name, color, dash: w === "none" });
    }
  }
  for (const w of weightings) {
    const m = findModel(manifest, task, { penalty: "none", weighting: w || "none" });
    if (m) hlines.push({ y: metricOf(task, m), color: COLORS.none, label: `no penalty${w ? (w === "balanced" ? ", balanced" : ", unweighted") : ""}`,
                         dash: w === "none" ? "2 4" : "7 4" });
  }
  items.push({ label: "No penalty", color: COLORS.none, dash: true });
  legend($("legend-perf"), items);
  const ys = series.flatMap(s => s.points.map(p => p[1])).concat(hlines.map(h => h.y));
  const page = task === "detection" ? "detection.html" : "diagnosis.html";
  lineChart($("chart-perf"), {
    height: 320,
    x: { type: "log", domain: [grid[0] / 2, grid[grid.length - 1] * 2], ticks: grid, fmt: fmtC, label: "C (left = stronger regularization)" },
    y: { domain: [0, ys.length ? Math.max(...ys) * 1.08 : 1], fmt: y => fmtMetric(task, y), label: metricName(task) },
    series, hlines,
    tip: (s, p) => `<b>${esc(p[2].label)}</b><br>${metricName(task)}: <b>${fmtMetric(task, p[1])}</b><br>Sensors used: ${p[2].coef_nonzero} of 52<br><span class="note">click to open</span>`,
    onClick: (s, p) => { location.href = modelLink(page, p[2]); },
  });

  // Sensors used by lasso.
  const nzSeries = weightings.map(w => ({
    name: w || "lasso", color: COLORS.l1, marker: 3, dash: w === "none" ? "6 4" : null, opacity: w === "none" ? 0.6 : 1,
    points: grid.map(C => findModel(manifest, task, { penalty: "l1", weighting: w || "none", C })).filter(Boolean).map(m => [m.C, m.coef_nonzero, m]),
  }));
  lineChart($("chart-nonzero"), {
    height: 180, margin: { left: 44 },
    x: { type: "log", domain: [grid[0] / 2, grid[grid.length - 1] * 2], ticks: grid.filter((_, i) => i % 2 === 0), fmt: fmtC },
    y: { domain: [0, 54], ticks: [0, 13, 26, 39, 52] },
    series: nzSeries,
    tip: (s, p) => `${esc(p[2].label)}<br><b>${p[1]}</b> of 52 sensors`,
  });

  // Tiles and callout.
  const regs = models.filter(m => m.penalty === "l1" || m.penalty === "l2");
  const nones = models.filter(m => m.penalty === "none");
  const missing = grid.length * 2 * weightings.length + weightings.length - models.length;
  $("partial-note").textContent = missing > 0
    ? `${missing} of the logistic regression models for this task are not in these results yet, so some points are missing.` : "";
  if (!regs.length || !nones.length) {
    $("tiles-perf").innerHTML = "";
    $("callout-perf").innerHTML = "The comparison appears once both regularized and unregularized models are in the results.";
    return;
  }
  const bestReg = regs.reduce((a, b) => (metricOf(task, b) > metricOf(task, a) ? b : a));
  const bestNone = nones.reduce((a, b) => (metricOf(task, b) > metricOf(task, a) ? b : a));
  const best = metricOf(task, bestReg) > metricOf(task, bestNone) ? bestReg : bestNone;
  const tol = task === "detection" ? 0.02 : 0.02;
  const frugal = models.filter(m => m.penalty === "l1" && metricOf(task, m) >= metricOf(task, best) - tol)
                       .sort((a, b) => a.coef_nonzero - b.coef_nonzero)[0];
  $("tiles-perf").innerHTML = [
    tile("Best regularized", fmtMetric(task, metricOf(task, bestReg)), esc(bestReg.label)),
    tile("Best unregularized", fmtMetric(task, metricOf(task, bestNone)), esc(bestNone.label)),
    frugal ? tile("Fewest sensors within 0.02 of best", `${frugal.coef_nonzero} / 52`, esc(frugal.label)) : "",
  ].join("");
  const gain = metricOf(task, bestReg) - metricOf(task, bestNone);
  $("callout-perf").innerHTML = gain <= 0.005
    ? `<b>Regularization doesn't improve accuracy here.</b> The best regularized model is within ${fmtMetric(task, Math.abs(gain))} of the
       unregularized one, and stronger regularization only makes things worse. That is expected: with 52 inputs and hundreds of
       thousands of training rows, a linear model has little room to overfit. What lasso does offer is a smaller model:
       ${frugal ? `${frugal.coef_nonzero} sensors get within 0.02 of the best result.` : ""}`
    : `<b>Mild regularization helps a little:</b> the best regularized model beats the unregularized one by ${fmtMetric(task, gain)}.
       Stronger regularization (smaller C) eventually removes too much information.`;
}

// ---------- 2. Coefficient paths ----------
async function loadPaths() {
  const { manifest, task } = state;
  $("path-weight-ctl").hidden = task !== "detection";
  $("path-fault-ctl").hidden = task !== "diagnosis";
  const token = ++state.token;
  const grid = manifest.settings.C_grid;
  const models = grid.map(C => findModel(manifest, task, { penalty: state.penalty, weighting: task === "detection" ? state.weighting : "none", C }));
  const files = await Promise.all(models.map(m => (m ? loadModel(task, m.id) : null)));
  if (token !== state.token) return;
  state.files = files.map((f, i) => ({ model: models[i], file: f }));
  renderPaths();
}

function coefVector(file) {
  const vals = file.coefficients.values;
  if (!Array.isArray(vals[0])) return vals;                       // detection: one vector
  if (state.fault === "all") return vals[0].map((_, j) => Math.sqrt(vals.reduce((s, row) => s + row[j] ** 2, 0)));
  return vals[+state.fault - 1];
}

function renderPaths() {
  const { manifest, task } = state;
  const grid = manifest.settings.C_grid, features = manifest.dataset.features;
  const entries = state.files.filter(e => e.file);
  const vecs = entries.map(e => coefVector(e.file));
  const cur = state.files[state.ci];
  const curVec = cur?.file ? coefVector(cur.file) : null;
  $("path-cval").textContent = `C = ${fmtC(grid[state.ci])}`;

  const ranked = curVec ? features.map((f, j) => [j, Math.abs(curVec[j])]).filter(x => x[1] > 0).sort((a, b) => b[1] - a[1]) : [];
  const top = ranked.slice(0, PALETTE.length).map(x => x[0]);
  const series = features.map((f, j) => {
    const k = top.indexOf(j);
    return { name: f, color: k >= 0 ? PALETTE[k] : COLORS.faint, width: k >= 0 ? 2.2 : 1, j,
             points: entries.map((e, i) => [e.model.C, vecs[i][j]]) };
  });
  series.sort((a, b) => (top.indexOf(a.j) >= 0) - (top.indexOf(b.j) >= 0));   // draw highlighted lines on top
  legend($("legend-paths"), top.map((j, k) => ({ label: `${features[j]} · ${VARIABLES[features[j]]}`, color: PALETTE[k] })));
  if (!entries.length) {
    $("chart-paths").innerHTML = `<div class="empty">None of these models are in the results yet. Try the other penalty or weighting.</div>`;
    $("legend-paths").innerHTML = ""; $("tiles-path").innerHTML = ""; $("top-title").textContent = ""; $("chart-top").innerHTML = "";
    return;
  }
  const all = vecs.flat();
  const lim = Math.max(...all.map(Math.abs)) || 1;
  lineChart($("chart-paths"), {
    height: 340,
    x: { type: "log", domain: [grid[0] / 2, grid[grid.length - 1] * 2], ticks: grid, fmt: fmtC, label: "C (left = stronger regularization)" },
    y: { domain: task === "diagnosis" && state.fault === "all" ? [0, lim * 1.05] : [-lim * 1.05, lim * 1.05], fmt: y => num(y, Math.abs(lim) < 1 ? 2 : 1), label: "Coefficient" },
    series, vlines: [{ x: grid[state.ci], color: COLORS.amber, dash: "4 3" }], hoverRadius: 18,
    tip: (s, p) => `<b>${s.name}</b> · ${VARIABLES[s.name]}<br>C = ${fmtC(p[0])}: coefficient ${num(p[1], 4)}`,
  });

  const model = cur?.model;
  $("tiles-path").innerHTML = model ? [
    tile("Sensors used", `${ranked.length} / 52`),
    tile(task === "detection" ? "Caught at 5% FA" : "Macro-F1", fmtMetric(task, metricOf(task, model))),
  ].join("") : "";

  const shown = ranked.slice(0, 15);
  $("top-title").textContent = shown.length
    ? `Largest coefficients at C = ${fmtC(grid[state.ci])} (${ranked.length} of 52 sensors non-zero)`
    : `At C = ${fmtC(grid[state.ci])} every coefficient is zero: the model ignores all sensors.`;
  if (!shown.length) { $("chart-top").innerHTML = ""; return; }
  const values = shown.map(([j]) => curVec[j]);
  const m = Math.max(...values.map(Math.abs));
  barChart($("chart-top"), {
    categories: shown.map(([j]) => `${features[j]} · ${VARIABLES[features[j]]}`),
    series: [{ name: "coef", color: COLORS.data, values }],
    x: { domain: values.some(v => v < 0) ? [-m * 1.1, m * 1.1] : [0, m * 1.1], fmt: x => num(x, m < 1 ? 2 : 1),
         label: task === "detection" ? "Coefficient (positive pushes toward 'fault')" : "Coefficient" },
    labelWidth: 330, valueFmt: x => num(x, 3), rowHeight: 22,
    tip: i => `<b>${features[shown[i][0]]}</b>: ${VARIABLES[features[shown[i][0]]]}<br>coefficient ${num(values[i], 4)}`,
  });
}

// ---------- Learn: the objective and what C changes ----------
const hat = x => `<mover accent="true">${M.row(x)}${M.o("^")}</mover>`;
const beta = M.i("β");

function renderStaticFormulas() {
  const sumN = M.sum(M.row(M.i("i"), M.o("="), M.n(1)), M.i("N"));
  const l = M.sub(M.i("ℓ"), M.i("i"));
  const minw = M.under(M.t("min"), M.row(M.i("w"), M.o(","), M.i("b")));
  $("f-objective").innerHTML =
    M.block(M.t("Ridge:"), M.sp(0.6), minw, M.sp(0.3), M.frac(M.n(1), M.n(2)), M.sup(M.norm(M.i("w"), 2), M.n(2)), M.o("+"), M.i("C"), sumN, l) +
    M.block(M.t("Lasso:"), M.sp(0.6), minw, M.sp(0.3), M.norm(M.i("w"), 1), M.o("+"), M.i("C"), sumN, l) +
    M.block(l, M.o("="), M.o("−"), M.row(M.o("["), M.sub(M.i("y"), M.i("i")), M.t("ln"), M.sp(0.15), M.sub(M.i("p"), M.i("i")), M.o("+"),
            M.paren(M.n(1), M.o("−"), M.sub(M.i("y"), M.i("i"))), M.t("ln"), M.sp(0.15), M.paren(M.n(1), M.o("−"), M.sub(M.i("p"), M.i("i"))), M.o("]")),
            M.sp(1), M.t("(log-loss of row"), M.sp(0.3), M.i("i"), M.t(")"));
  $("f-scaled").innerHTML =
    M.block(M.frac(M.n(1), M.i("N")), M.sum(M.row(M.i("i"), M.o("="), M.n(1)), M.i("N")), M.sub(M.i("ℓ"), M.i("i")), M.sp(0.3), M.o("+"), M.sp(0.3),
            M.frac(M.n(1), M.row(M.i("C"), M.o("·"), M.i("N"))), M.sp(0.15), M.o("×"), M.sp(0.15), M.t("penalty"), M.paren(M.i("w")));
  $("f-shrink").innerHTML =
    M.block(M.sub(beta, M.t("ridge")), M.o("="), M.frac(hat(beta), M.row(M.n(1), M.o("+"), M.i("λ")))) +
    M.block(M.sub(beta, M.t("lasso")), M.o("="), M.t("sign"), M.paren(hat(beta)), M.sp(0.2), M.o("·"), M.sp(0.2),
            M.t("max"), M.paren(M.abs(hat(beta)), M.o("−"), M.i("λ"), M.o(","), M.sp(0.2), M.n(0)));
}

// Number of training rows each model was fitted on (training runs subsample every k-th time step).
function trainingRows(task) {
  const d = state.manifest.dataset, s = state.manifest.settings;
  if (task === "detection") return d.train_runs * Math.floor(500 / s.det_stride);
  const faultyRuns = Math.round(d.train_runs * 20 / 21);
  return faultyRuns * (Math.floor(500 / s.diag_stride) - Math.floor(d.fault_onset.train / s.diag_stride));
}
const sci = x => {
  const e = Math.floor(Math.log10(x)), m = x / 10 ** e;
  return e >= -2 && e <= 2 ? num(x, Math.max(0, 2 - e)) : `${m.toFixed(1)} × 10<sup>${e < 0 ? "−" : ""}${Math.abs(e)}</sup>`;
};

function renderObjective() {
  const { manifest, task } = state;
  const N = trainingRows(task), grid = manifest.settings.C_grid;
  const w = task === "detection" ? "balanced" : "none";
  const metric = m => (m ? (task === "detection" ? pct(m.val.recall) : num(m.val.macro_F1, 3)) : "–");
  const none = findModel(manifest, task, { penalty: "none", weighting: w });
  $("objective-live").innerHTML = `
    <p>${task === "detection" ? "Detection" : "Diagnosis"} models here were trained on <i>N</i> = ${int(N)} rows. Smaller C makes the
      penalty count for more:</p>
    <div class="table-wrap"><table>
      <thead><tr><th class="n">C</th><th class="n">Penalty weight 1 / (C·N)</th><th class="n">Ridge${task === "detection" ? ", balanced" : ""}</th>
        <th class="n">Lasso${task === "detection" ? ", balanced" : ""}</th><th class="n">Lasso sensors</th></tr></thead>
      <tbody>${grid.map((C, k) => {
        const r = findModel(manifest, task, { penalty: "l2", weighting: w, C }), l = findModel(manifest, task, { penalty: "l1", weighting: w, C });
        return `<tr class="${k === state.ci ? "highlight" : ""}"><td class="n">${fmtC(C)}</td><td class="n">${sci(1 / (C * N))}</td>
          <td class="n">${metric(r)}</td><td class="n">${metric(l)}</td><td class="n">${l ? l.coef_nonzero : "–"}</td></tr>`;
      }).join("")}
        <tr><td class="n">∞</td><td class="n">0</td><td class="n" colspan="2" style="text-align:center">no penalty: ${metric(none)}</td><td class="n">52</td></tr></tbody>
    </table></div>
    <p style="margin-top:10px">With this many rows, the average loss dominates until the penalty weight grows large, which is why the curves
      above stay flat for large C and only fall away at the left. It also means the same C regularizes less on more data:
      ${manifest.dataset.runs_per_fault < 500 ? `with all 500 runs per fault, <i>N</i> would be about ${Math.round(500 / manifest.dataset.runs_per_fault)} times larger, so each C would behave roughly like a C that many times larger does here.`
        : "doubling the rows has the same effect as doubling C."}
      The highlighted row follows the C slider in the coefficient paths below.</p>`;
}

// ---------- Learn: one-coefficient shrinkage demo ----------
function renderShrink() {
  const lam = +$("lambda").value;
  $("lambda-val").textContent = `λ = ${lam.toFixed(2)}`;
  const xs = [];
  for (let x = -3; x <= 3.0001; x += 0.05) xs.push(+x.toFixed(2));
  const ridge = x => x / (1 + lam), lasso = x => Math.sign(x) * Math.max(Math.abs(x) - lam, 0);
  legend($("legend-shrink"), [{ label: "No penalty", color: COLORS.faint, dash: true }, { label: "Ridge", color: COLORS.l2 }, { label: "Lasso", color: COLORS.l1 }]);
  lineChart($("chart-shrink"), {
    height: 250, margin: { left: 48 },
    x: { domain: [-3, 3], ticks: [-3, -2, -1, 0, 1, 2, 3], label: "Coefficient without a penalty" },
    y: { domain: [-3, 3], ticks: [-3, -2, -1, 0, 1, 2, 3], label: "With the penalty" },
    series: [
      { name: "none", color: COLORS.faint, dash: "4 4", width: 1.5, points: xs.map(x => [x, x]), hover: false },
      { name: "Ridge", color: COLORS.l2, width: 2.4, points: xs.map(x => [x, ridge(x)]) },
      { name: "Lasso", color: COLORS.l1, width: 2.4, points: xs.map(x => [x, lasso(x)]) },
    ],
    tip: (s, p) => `<b>${s.name}</b>: ${num(p[0], 2)} becomes ${num(p[1], 2)}`,
    hoverRadius: 16,
  });
  $("shrink-live").innerHTML = `At λ = ${lam.toFixed(2)}, a coefficient of 0.5 becomes ${num(ridge(0.5), 2)} under ridge and
    ${lasso(0.5) === 0 ? "exactly 0" : num(lasso(0.5), 2)} under lasso. A coefficient of 2 becomes ${num(ridge(2), 2)} under ridge and
    ${num(lasso(2), 2)} under lasso. Every coefficient between −${lam.toFixed(2)} and ${lam.toFixed(2)} is switched off by lasso.`;
}
