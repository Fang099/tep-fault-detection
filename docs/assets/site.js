// Shared site code: navigation, data loading, model selection, TEP metadata and detection maths.

export const PAGES = [
  { key: "overview", href: "index.html", label: "Overview" },
  { key: "detection", href: "detection.html", label: "Detection" },
  { key: "diagnosis", href: "diagnosis.html", label: "Diagnosis" },
  { key: "regularization", href: "regularization.html", label: "Regularization" },
  { key: "compare", href: "compare.html", label: "Compare" },
  { key: "results", href: "results.html", label: "Results" },
  { key: "reproduce", href: "reproduce.html", label: "Reproduce" },
];

// Colours are CSS variables defined in style.css, so they follow the light/dark theme.
export const COLORS = {
  normal: "var(--c-normal)", fault: "var(--c-fault)", data: "var(--c-data)", amber: "var(--c-amber)",
  a: "var(--c-a)", b: "var(--c-b)", l1: "var(--c-l1)", l2: "var(--c-l2)", none: "var(--c-none)",
  muted: "var(--muted)", faint: "var(--faint)", text: "var(--text)",
};
export const PALETTE = ["var(--p1)", "var(--p2)", "var(--p3)", "var(--p4)", "var(--p5)", "var(--p6)", "var(--p7)", "var(--p8)"];

// ---------- TEP metadata (Downs & Vogel, 1993) ----------
export const VARIABLES = {
  xmeas_1: "A feed (stream 1)", xmeas_2: "D feed (stream 2)", xmeas_3: "E feed (stream 3)",
  xmeas_4: "A and C feed (stream 4)", xmeas_5: "Recycle flow (stream 8)", xmeas_6: "Reactor feed rate (stream 6)",
  xmeas_7: "Reactor pressure", xmeas_8: "Reactor level", xmeas_9: "Reactor temperature",
  xmeas_10: "Purge rate (stream 9)", xmeas_11: "Product separator temperature", xmeas_12: "Product separator level",
  xmeas_13: "Product separator pressure", xmeas_14: "Product separator underflow (stream 10)",
  xmeas_15: "Stripper level", xmeas_16: "Stripper pressure", xmeas_17: "Stripper underflow (stream 11)",
  xmeas_18: "Stripper temperature", xmeas_19: "Stripper steam flow", xmeas_20: "Compressor work",
  xmeas_21: "Reactor cooling water outlet temperature", xmeas_22: "Separator cooling water outlet temperature",
  xmeas_23: "Reactor feed: component A", xmeas_24: "Reactor feed: component B", xmeas_25: "Reactor feed: component C",
  xmeas_26: "Reactor feed: component D", xmeas_27: "Reactor feed: component E", xmeas_28: "Reactor feed: component F",
  xmeas_29: "Purge gas: component A", xmeas_30: "Purge gas: component B", xmeas_31: "Purge gas: component C",
  xmeas_32: "Purge gas: component D", xmeas_33: "Purge gas: component E", xmeas_34: "Purge gas: component F",
  xmeas_35: "Purge gas: component G", xmeas_36: "Purge gas: component H",
  xmeas_37: "Product: component D", xmeas_38: "Product: component E", xmeas_39: "Product: component F",
  xmeas_40: "Product: component G", xmeas_41: "Product: component H",
  xmv_1: "D feed flow valve (stream 2)", xmv_2: "E feed flow valve (stream 3)", xmv_3: "A feed flow valve (stream 1)",
  xmv_4: "A and C feed flow valve (stream 4)", xmv_5: "Compressor recycle valve", xmv_6: "Purge valve (stream 9)",
  xmv_7: "Separator pot liquid flow valve (stream 10)", xmv_8: "Stripper liquid product flow valve (stream 11)",
  xmv_9: "Stripper steam valve", xmv_10: "Reactor cooling water flow valve", xmv_11: "Condenser cooling water flow valve",
};

export const FAULTS = {
  1: ["A/C feed ratio, B composition constant (stream 4)", "Step"],
  2: ["B composition, A/C ratio constant (stream 4)", "Step"],
  3: ["D feed temperature (stream 2)", "Step"],
  4: ["Reactor cooling water inlet temperature", "Step"],
  5: ["Condenser cooling water inlet temperature", "Step"],
  6: ["A feed loss (stream 1)", "Step"],
  7: ["C header pressure loss (stream 4)", "Step"],
  8: ["A, B, C feed composition (stream 4)", "Random variation"],
  9: ["D feed temperature (stream 2)", "Random variation"],
  10: ["C feed temperature (stream 4)", "Random variation"],
  11: ["Reactor cooling water inlet temperature", "Random variation"],
  12: ["Condenser cooling water inlet temperature", "Random variation"],
  13: ["Reaction kinetics", "Slow drift"],
  14: ["Reactor cooling water valve", "Sticking"],
  15: ["Condenser cooling water valve", "Sticking"],
  16: ["Unknown", "Unknown"], 17: ["Unknown", "Unknown"], 18: ["Unknown", "Unknown"],
  19: ["Unknown", "Unknown"], 20: ["Unknown", "Unknown"],
};
export const faultName = f => `Fault ${f}`;
export const faultDesc = f => `${FAULTS[f][0]} · ${FAULTS[f][1].toLowerCase()}`;
export const varLabel = v => `${v.replace("_", " ")}`;

// ---------- Formatting ----------
export const pct = (x, d = 1) => (x == null || !isFinite(x) ? "–" : (100 * x).toFixed(d) + "%");
export const num = (x, d = 3) => (x == null || !isFinite(x) ? "–" : Number(x).toFixed(d));
export const int = x => (x == null ? "–" : Math.round(x).toLocaleString("en-US"));
export function fmtC(C) {
  if (C == null) return "∞";
  const e = Math.round(Math.log10(C));
  if (Math.abs(10 ** e - C) / C < 1e-9) return e >= -3 ? String(+(10 ** e).toFixed(3)) : `1e${e}`;
  return String(C);
}
export function fmtMinutes(m) {
  if (m == null || !isFinite(m)) return "not detected";
  if (m < 120) return `${Math.round(m)} min`;
  return `${(m / 60).toFixed(1)} h`;
}
export function esc(s) { return String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

// ---------- Data loading ----------
const cache = new Map();
export function getJSON(path) {
  if (!cache.has(path)) {
    cache.set(path, fetch(path).then(r => {
      if (!r.ok) throw new Error(`Could not load ${path} (${r.status})`);
      return r.json();
    }));
  }
  return cache.get(path);
}
export const loadManifest = () => getJSON("data/manifest.json");
export const loadModel = (task, id) => getJSON(`data/${task}/${id}.json`);
export const loadTestModel = (task, id) => getJSON(`data/${task}/test_${id}.json`);

// ---------- Page chrome ----------
const MARK = `<svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true">
  <rect x="6" y="2.5" width="10" height="17" rx="5" fill="none" stroke="currentColor" stroke-width="1.6"/>
  <path d="M1 11h5M16 7h5M16 15h5" stroke="currentColor" stroke-width="1.6"/>
  <circle cx="11" cy="13.5" r="2.2" fill="var(--c-fault)"/></svg>`;

function effectiveTheme() {
  const set = document.documentElement.getAttribute("data-theme");
  if (set) return set;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}
function themeButton(btn) {
  const dark = effectiveTheme() === "dark";
  btn.innerHTML = dark
    ? `<svg width="15" height="15" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4.5" fill="currentColor"/><g stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 2v2.5M12 19.5V22M2 12h2.5M19.5 12H22M4.9 4.9l1.8 1.8M17.3 17.3l1.8 1.8M4.9 19.1l1.8-1.8M17.3 6.7l1.8-1.8"/></g></svg><span class="tt-label">Light</span>`
    : `<svg width="15" height="15" viewBox="0 0 24 24" aria-hidden="true"><path d="M20 14.5A8.5 8.5 0 0 1 9.5 4a8.5 8.5 0 1 0 10.5 10.5z" fill="currentColor"/></svg><span class="tt-label">Dark</span>`;
  btn.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
}

export function renderChrome() {
  const page = document.body.dataset.page;
  const nav = document.getElementById("topnav");
  if (nav) {
    nav.className = "topnav";
    nav.innerHTML = `<div class="inner">
      <a class="brand" href="index.html">${MARK}<span>TEP fault monitor</span></a>
      <div class="navlinks">${PAGES.map(p => `<a href="${p.href}"${p.key === page ? ' class="active" aria-current="page"' : ""}>${p.label}</a>`).join("")}</div>
      <button class="theme-toggle" type="button"></button></div>`;
    const btn = nav.querySelector(".theme-toggle");
    themeButton(btn);
    btn.addEventListener("click", () => {
      const next = effectiveTheme() === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      try { localStorage.setItem("tep-theme", next); } catch (e) { /* storage unavailable: theme lasts for this page only */ }
      themeButton(btn);
    });
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", () => themeButton(btn));
  }
  buildSideToc();
  const foot = document.querySelector("footer");
  if (foot) foot.innerHTML = `ES 335 machine learning project. Tennessee Eastman Process data from Kaggle
    (<a href="https://www.kaggle.com/datasets/afrniomelo/tep-csv">afrniomelo/tep-csv</a>).
    <a href="https://github.com/Fang099/tep-fault-detection">Source code on GitHub</a>.`;
}

// "On this page" list built from the page's sections, with the current section highlighted while scrolling.
function buildSideToc() {
  const toc = document.getElementById("sidetoc");
  if (!toc) return;
  const sections = [...document.querySelectorAll("main section.section[id]")];
  toc.innerHTML = `<p class="toc-title">On this page</p><ol>${
    sections.map(s => `<li><a href="#${s.id}">${esc(s.dataset.toc || s.querySelector("h2")?.textContent || s.id)}</a></li>`).join("")}</ol>`;
  const links = new Map(sections.map(s => [s.id, toc.querySelector(`a[href="#${s.id}"]`)]));
  const visible = new Set();
  let current = sections[0]?.id;
  const update = () => {
    const first = sections.find(s => visible.has(s.id));
    if (first) current = first.id;
    links.forEach((a, id) => a.classList.toggle("current", current === id));
  };
  update();
  const io = new IntersectionObserver(entries => {
    for (const e of entries) e.isIntersecting ? visible.add(e.target.id) : visible.delete(e.target.id);
    update();
  }, { rootMargin: "-90px 0px -55% 0px" });
  sections.forEach(s => io.observe(s));
}

export function showError(err) {
  console.error(err);
  const main = document.querySelector("main");
  const box = document.createElement("div");
  box.className = "panel";
  box.style.marginBottom = "20px";
  box.innerHTML = `<b>Something went wrong loading the results.</b><p class="note">${esc(err.message || err)}.
    If you opened this file directly from disk, serve the folder instead (for example <code>python -m http.server</code> inside <code>docs/</code>).</p>`;
  main.prepend(box);
}

// Banner explaining which data the numbers come from.
export function dataNote(manifest) {
  const runs = manifest.dataset.runs_per_fault;
  const subset = runs < 500 ? `the first ${runs} of 500 simulation runs per fault type` : "all 500 simulation runs per fault type";
  return `Numbers on this page are <b>validation</b> results: models trained on ${int(manifest.dataset.train_runs)} runs and
    evaluated on ${int(manifest.dataset.val_runs)} held-out runs (${subset}).`;
}

// ---------- Models ----------
// A selection is { family: "baseline" | "logreg", penalty: "none" | "l1" | "l2", weighting: "none" | "balanced", C }.
// In the exported data the majority baseline is stored with penalty "baseline".
export const FAMILIES = [
  { key: "baseline", label: "Majority baseline" },
  { key: "logreg", label: "Logistic regression" },
];
const sameC = (a, b) => (a == null && b == null) || (a != null && b != null && Math.abs(Math.log10(a) - Math.log10(b)) < 1e-6);
export const familyOf = m => (m.penalty === "baseline" ? "baseline" : "logreg");

export function findModel(manifest, task, sel) {
  return manifest[task].models.find(m => {
    if (sel.family === "baseline") return m.penalty === "baseline";
    if (m.penalty !== sel.penalty) return false;
    if (task === "detection" && m.class_weight !== sel.weighting) return false;
    return sel.penalty === "none" ? true : sameC(m.C, sel.C);
  });
}
export function selectionOf(model) {
  if (familyOf(model) === "baseline") return { family: "baseline", penalty: "none", weighting: "none", C: null };
  return { family: "logreg", penalty: model.penalty, weighting: model.class_weight || "none", C: model.C };
}
export function bestModel(manifest, task) {
  return manifest[task].models.find(m => m.id === manifest[task].best);
}

// Read / write the selected model in the page URL so links and reloads keep it.
export function readSelection(manifest, task) {
  const q = new URLSearchParams(location.search);
  if (q.get("family") === "baseline" || q.get("penalty") === "baseline") return selectionOf(manifest[task].models.find(m => m.penalty === "baseline"));
  const penalty = q.get("penalty");
  if (!penalty) return selectionOf(bestModel(manifest, task));
  const sel = { family: "logreg", penalty, weighting: q.get("w") || "none", C: q.get("C") ? +q.get("C") : null };
  if (sel.penalty !== "none" && sel.C == null) sel.C = 1;
  return findModel(manifest, task, sel) ? sel : selectionOf(bestModel(manifest, task));
}
function selectionParams(sel) {
  const q = new URLSearchParams();
  if (sel.family === "baseline") { q.set("family", "baseline"); return q; }
  q.set("penalty", sel.penalty);
  if (sel.weighting && sel.weighting !== "none") q.set("w", sel.weighting);
  if (sel.C != null && sel.penalty !== "none") q.set("C", sel.C);
  return q;
}
export function writeSelection(sel, extra = {}) {
  const q = selectionParams(sel);
  const old = new URLSearchParams(location.search);
  for (const [k, v] of Object.entries(extra)) v == null ? old.delete(k) : q.set(k, v);
  history.replaceState(null, "", "?" + q.toString() + location.hash);
}
export function modelLink(page, model) {
  return `${page}?${selectionParams(selectionOf(model))}`;
}

// ---------- Model selector (sticky bar) ----------
// opts: { task, manifest, selection, onChange(selection, model) }
export function modelBar(container, opts) {
  const { task, manifest } = opts;
  const grid = manifest.settings.C_grid;
  let sel = { ...opts.selection };
  // Logistic-regression settings to return to after looking at the baseline.
  let lastLR = sel.family === "logreg" ? { ...sel } : selectionOf(bestModel(manifest, task));

  container.classList.add("modelbar");
  container.setAttribute("aria-label", "Model selection");
  container.innerHTML = `
    <div class="control"><div class="control-head">Model family</div>
      <div class="seg" data-k="family">${FAMILIES.map(f => `<button type="button" data-v="${f.key}">${f.label}</button>`).join("")}</div></div>
    <p class="baseline-note" data-k="bnote">Predicts the most common class for every row. It has no settings and ignores the sensors entirely.</p>
    <div class="lr-settings" data-k="lr">
      ${task === "detection" ? `<div class="control"><div class="control-head">Class weighting</div>
        <div class="seg" data-k="weighting"><button type="button" data-v="none">Unweighted</button><button type="button" data-v="balanced">Balanced</button></div></div>` : ""}
      <div class="control"><div class="control-head">Penalty</div>
        <div class="seg" data-k="penalty"><button type="button" data-v="none">None</button><button type="button" data-v="l1">Lasso (L1)</button><button type="button" data-v="l2">Ridge (L2)</button></div></div>
      <div class="control grow"><div class="control-head"><span>Strength C <span class="note">(smaller is stronger)</span></span><span class="val" data-k="cval"></span></div>
        <input type="range" min="0" max="${grid.length - 1}" step="1" data-k="C" aria-label="Regularization strength C">
        <div class="ticks">${grid.map(c => `<span>${fmtC(c)}</span>`).join("")}</div></div>
    </div>
    <div class="current"><span class="note">Showing</span> <b data-k="label"></b> <span data-k="badges"></span>
      <button class="btn" type="button" data-k="best" style="margin-left:auto">Use best model</button></div>`;

  const q = k => container.querySelector(`[data-k="${k}"]`);
  const slider = q("C");

  function sync() {
    const model = findModel(manifest, task, sel);
    const isBase = sel.family === "baseline";
    q("lr").hidden = isBase;
    q("lr").style.display = isBase ? "none" : "";
    q("bnote").hidden = !isBase;
    for (const k of ["family", "weighting", "penalty"]) {
      q(k)?.querySelectorAll("button").forEach(b => {
        b.classList.toggle("on", b.dataset.v === sel[k]);
        b.setAttribute("aria-pressed", b.dataset.v === sel[k]);
      });
    }
    const usesC = sel.penalty === "l1" || sel.penalty === "l2";
    slider.disabled = !usesC;
    slider.value = usesC ? Math.max(0, grid.findIndex(c => sameC(c, sel.C))) : grid.length - 1;
    q("cval").textContent = usesC ? `C = ${fmtC(sel.C)}` : "no penalty";
    q("label").textContent = model ? model.label : "not trained";
    const badges = [];
    if (model && model.id === manifest[task].best) badges.push(`<span class="badge best">best on validation</span>`);
    if (model && model.coef_nonzero != null) badges.push(`<span class="badge">${model.coef_nonzero} of 52 sensors used</span>`);
    q("badges").innerHTML = badges.join(" ");
    return model;
  }
  function changed() {
    if (sel.family === "logreg") lastLR = { ...sel };
    const model = sync();
    if (model) opts.onChange(sel, model);
  }
  container.addEventListener("click", e => {
    const b = e.target.closest("button");
    if (!b) return;
    if (b.dataset.k === "best") { sel = selectionOf(bestModel(manifest, task)); changed(); return; }
    const seg = b.closest(".seg");
    if (!seg || !b.dataset.v) return;
    const k = seg.dataset.k;
    if (k === "family") {
      if (b.dataset.v === sel.family) return;
      sel = b.dataset.v === "baseline" ? { family: "baseline", penalty: "none", weighting: "none", C: null } : { ...lastLR };
    } else {
      sel[k] = b.dataset.v;
      if ((sel.penalty === "l1" || sel.penalty === "l2") && sel.C == null) sel.C = grid[grid.length - 1];
      if (sel.penalty === "none") sel.C = null;
    }
    changed();
  });
  slider.addEventListener("input", () => { sel.C = grid[+slider.value]; changed(); });
  const model = sync();
  opts.onChange(sel, model);
  return { set(newSel) { sel = { ...newSel }; changed(); } };
}

// ---------- Detection maths from score histograms ----------
// Scores in bin i lie in [edges[i], edges[i+1]). With threshold index t, rows in bins t.. are flagged as faulty.
export function detectionView(split) {
  const edges = split.bin_edges, normal = split.normal_counts, B = normal.length;
  const faults = Object.keys(split.fault_counts).map(Number).sort((a, b) => a - b);
  const suffix = arr => { const s = new Array(B + 1).fill(0); for (let i = B - 1; i >= 0; i--) s[i] = s[i + 1] + arr[i]; return s; };
  const sufN = suffix(normal);
  const sufF = {};
  for (const f of faults) sufF[f] = suffix(split.fault_counts[f]);
  const sufAll = new Array(B + 1).fill(0);
  for (const f of faults) for (let i = 0; i <= B; i++) sufAll[i] += sufF[f][i];
  const totalN = sufN[0], totalF = sufAll[0];

  const at = t => {
    const TP = sufAll[t], FP = sufN[t], FN = totalF - TP, TN = totalN - FP;
    const precision = TP + FP ? TP / (TP + FP) : 0, recall = totalF ? TP / totalF : 0;
    return { t, threshold: t < B ? edges[t] : null, TP, FP, FN, TN, precision, recall,
             f1: precision + recall ? (2 * precision * recall) / (precision + recall) : 0,
             normalRecall: totalN ? TN / totalN : 0, far: totalN ? FP / totalN : 0 };
  };
  return {
    B, edges, faults, totalN, totalF, normal, split, at,
    indexForFar(far) { for (let t = 0; t <= B; t++) if (sufN[t] / totalN <= far + 1e-12) return t; return B; },
    indexForThreshold(th) { for (let t = 0; t < B; t++) if (edges[t] >= th - 1e-12) return t; return B; },
    perFault(t) { return faults.map(f => sufF[f][t] / (sufF[f][0] || 1)); },
    faultTotals() { return faults.map(f => sufF[f][0]); },
    roc() { const pts = []; for (let t = 0; t <= B; t++) pts.push([sufN[t] / totalN, sufAll[t] / totalF]); return pts.reverse(); },
    constant: B <= 1,
  };
}

// Area under a curve given as [[x,y]] sorted by x.
export function auc(points) {
  let a = 0;
  for (let i = 1; i < points.length; i++) a += (points[i][0] - points[i - 1][0]) * (points[i][1] + points[i - 1][1]) / 2;
  return a;
}

// Logit transform used to spread probability scores for plotting.
export const logit = p => { const e = 1e-7, q = Math.min(1 - e, Math.max(e, p)); return Math.log(q / (1 - q)); };
export const PROB_TICKS = [0.0001, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 0.9999];

// Tile HTML helper.
export function tile(label, value, sub = "", cls = "") {
  return `<div class="tile ${cls}"><div class="label">${label}</div><div class="value">${value}</div>${sub ? `<div class="sub">${sub}</div>` : ""}</div>`;
}
