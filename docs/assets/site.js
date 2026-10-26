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
// Manifests written before tree models existed have no family information; fill it in so every page can rely on it.
const FAMILY_LABELS = { baseline: "Majority baseline", logreg: "Logistic regression", pcalr: "PCA + logistic regression",
  pcamon: "PCA monitoring (T² / SPE)", tree: "Decision tree", rf: "Random forest", gb: "Gradient boosting", ada: "AdaBoost", xgb: "XGBoost" };
function normalizeManifest(m) {
  for (const task of ["detection", "diagnosis"]) {
    for (const s of m[task].models) {
      if (!s.family) s.family = s.penalty === "baseline" ? "baseline" : "logreg";
      if (!s.params) s.params = {};
      if (!s.features) s.features = "inst";
      if (!s.score_kind) s.score_kind = "prob";
    }
    m[task].best_by_family ||= {};
  }
  if (!m.families) {
    const present = new Set([...m.detection.models, ...m.diagnosis.models].map(s => s.family));
    m.families = Object.keys(FAMILY_LABELS).filter(k => present.has(k)).map(k => ({ key: k, label: FAMILY_LABELS[k], params: [] }));
  }
  return m;
}
export const loadManifest = () => getJSON("data/manifest.json").then(m => (m._normalized ? m : Object.assign(normalizeManifest(m), { _normalized: true })));
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

// ---------- Model inputs ----------
// Inputs are the 52 readings, plus (for history models) a trailing mean and standard deviation of each, named
// like "xmeas_9_mean10". inputInfo turns a name into a readable description.
export function inputInfo(name) {
  const m = /^(xm(?:eas|v)_\d+)(?:_(mean|std)(\d+))?$/.exec(name);
  if (!m) return { code: name, desc: name };
  const base = VARIABLES[m[1]] || m[1];
  if (!m[2]) return { code: name, desc: base };
  return { code: name, desc: `${base} (${windowText(+m[3])} ${m[2] === "mean" ? "mean" : "std. dev."})`, history: true };
}
// Input sets: "inst" (52 readings), "hist" (plus 30-minute history) or "hist{w}" (history over w samples).
export const isHist = fs => (fs || "inst").startsWith("hist");
export const windowOf = fs => (!isHist(fs) ? 0 : fs === "hist" ? 10 : +fs.slice(4));
export const fsForWindow = w => (w === 0 ? "inst" : w === 10 ? "hist" : `hist${w}`);
export function windowText(w) {
  const m = w * 3;
  return m < 60 ? `${m}-min` : `${+(m / 60).toFixed(1)}-hour`;
}
export const windowShort = w => { const m = w * 3; return m < 60 ? `${m} min` : `${+(m / 60).toFixed(1)} h`; };
// Lower-case a model name for use mid-sentence, but keep names like XGBoost, AdaBoost and PCA as they are.
export const lowerLabel = l => (/[A-Z]/.test(l.slice(1)) ? l : l.charAt(0).toLowerCase() + l.slice(1));
export const INPUT_SETS = { inst: "Instantaneous", hist: "With 30-min history" };

// ---------- Models ----------
// A selection is { family, features, ...settings }. Logistic regression uses { penalty, weighting, C }; other
// families use the parameters listed for them in the manifest (e.g. { depth, trees } or { stat, k }).
// features is "inst" (52 readings), "hist" (readings plus 30-minute history) or "hist{w}" (another window).
const sameC = (a, b) => (a == null && b == null) || (a != null && b != null && Math.abs(Math.log10(a) - Math.log10(b)) < 1e-6);
export const familyOf = m => m.family || (m.penalty === "baseline" ? "baseline" : "logreg");
export const familyInfo = (manifest, key) => manifest.families.find(f => f.key === key) || { key, label: FAMILY_LABELS[key] || key, params: [] };
export const familyLabel = (manifest, key) => familyInfo(manifest, key).label;

export function findModel(manifest, task, sel) {
  return manifest[task].models.find(m => {
    const fam = familyOf(m);
    if (fam !== sel.family) return false;
    if (fam === "baseline") return true;
    if ((m.features || "inst") !== (sel.features || "inst")) return false;
    if (fam === "logreg") {
      if (m.penalty !== sel.penalty) return false;
      if (task === "detection" && m.class_weight !== sel.weighting) return false;
      return sel.penalty === "none" ? true : sameC(m.C, sel.C);
    }
    return familyInfo(manifest, fam).params.every(p => m.params[p.key] === sel[p.key]);
  });
}
export function selectionOf(model) {
  const fam = familyOf(model);
  if (fam === "baseline") return { family: "baseline" };
  const features = model.features || "inst";
  if (fam === "logreg") return { family: "logreg", features, penalty: model.penalty, weighting: model.class_weight || "none", C: model.C };
  return { family: fam, features, ...model.params };
}
export function bestModel(manifest, task, family) {
  const id = family ? manifest[task].best_by_family[family] : manifest[task].best;
  return manifest[task].models.find(m => m.id === id) ||
         (family ? manifest[task].models.filter(m => familyOf(m) === family).sort(byScore(task))[0] : null);
}
const byScore = task => (a, b) => (task === "detection" ? b.val.recall - a.val.recall : b.val.macro_F1 - a.val.macro_F1);

// Read / write the selected model in the page URL so links and reloads keep it.
export function readSelection(manifest, task) {
  const q = new URLSearchParams(location.search);
  let sel;
  const fam = q.get("family") || (q.get("penalty") === "baseline" ? "baseline" : q.get("penalty") ? "logreg" : null);
  const h = q.get("hist");
  const features = !h ? "inst" : h === "1" ? "hist" : fsForWindow(+h);
  if (fam === "baseline") sel = { family: "baseline" };
  else if (fam === "logreg") {
    const penalty = q.get("penalty") || "none";
    sel = { family: "logreg", features, penalty, weighting: q.get("w") || "none", C: penalty === "none" ? null : +(q.get("C") || 1) };
  } else if (fam) {
    const params = familyInfo(manifest, fam).params;
    const best = bestModel(manifest, task, fam);
    // No settings in the link: open the family's best model; partial settings: fill the rest from it.
    sel = best ? selectionOf(best) : { family: fam };
    const given = params.some(p => q.has(p.key)) || q.has("hist");
    if (given) sel.features = features;
    for (const p of params) if (q.has(p.key)) sel[p.key] = p.type === "choice" ? q.get(p.key) : +q.get(p.key);
  }
  return sel && findModel(manifest, task, sel) ? sel : selectionOf(bestModel(manifest, task));
}
function selectionParams(sel) {
  const q = new URLSearchParams();
  if (sel.family === "logreg") {
    q.set("penalty", sel.penalty);
    if (sel.weighting && sel.weighting !== "none") q.set("w", sel.weighting);
    if (sel.C != null && sel.penalty !== "none") q.set("C", sel.C);
  } else {
    q.set("family", sel.family);
    for (const [k, v] of Object.entries(sel)) if (!["family", "features"].includes(k) && v != null) q.set(k, v);
  }
  if (isHist(sel.features)) q.set("hist", sel.features === "hist" ? "1" : windowOf(sel.features));
  return q;
}
export function writeSelection(sel, extra = {}) {
  const q = selectionParams(sel);
  for (const [k, v] of Object.entries(extra)) if (v != null) q.set(k, v);
  history.replaceState(null, "", "?" + q.toString() + location.hash);
}
export function modelLink(page, model) {
  return `${page}?${selectionParams(selectionOf(model))}`;
}
// Short description of a model's settings, e.g. "depth 6, 100 rounds" or "lasso, balanced, C = 0.01".
export function settingsText(model) {
  const fam = familyOf(model), p = model.params || {};
  if (fam === "baseline") return "–";
  const hist = isHist(model.features) ? `${windowText(windowOf(model.features))} history` : null;
  if (fam === "logreg") {
    const pen = { none: "no penalty", l1: "lasso", l2: "ridge" }[model.penalty];
    return [pen, model.class_weight === "balanced" ? "balanced" : null, model.C != null ? `C = ${fmtC(model.C)}` : null, hist].filter(Boolean).join(", ");
  }
  return [p.stat ? (p.stat === "T2" ? "T²" : "SPE") : null, p.k != null ? `${p.k} components` : null,
          p.depth != null ? `depth ${p.depth}` : null, p.trees != null ? `${p.trees} trees` : null,
          p.rounds != null ? `${p.rounds} rounds` : null, hist].filter(Boolean).join(", ");
}

// ---------- Model selector (sticky bar) ----------
// A dropdown picks the model family; the controls next to it change with the family.
// opts: { task, manifest, selection, onChange(selection, model) }
export function modelBar(container, opts) {
  const { task, manifest } = opts;
  const grid = manifest.settings.C_grid;
  const families = manifest.families.filter(f => (!f.tasks || f.tasks.includes(task)) && manifest[task].models.some(m => familyOf(m) === f.key));
  const hasHistory = fam => manifest[task].models.some(m => familyOf(m) === fam && isHist(m.features));
  // History windows trained for a family (the full grid uses 30 minutes; other windows only the best settings).
  const windowsFor = fam => [...new Set(manifest[task].models.filter(m => familyOf(m) === fam && isHist(m.features))
    .map(m => windowOf(m.features)))].sort((a, b) => a - b);
  let sel = { ...opts.selection };
  // Settings to return to when switching back to a family: start each family at its best model.
  const last = new Map(families.map(f => [f.key, selectionOf(bestModel(manifest, task, f.key) || manifest[task].models.find(m => familyOf(m) === f.key))]));
  last.set(sel.family, { ...sel });

  container.classList.add("modelbar");
  container.setAttribute("aria-label", "Model selection");
  container.innerHTML = `
    <div class="control"><div class="control-head"><label for="mb-family-${task}">Model</label></div>
      <select id="mb-family-${task}" data-k="family">${families.map(f => `<option value="${f.key}">${esc(f.label)}</option>`).join("")}</select></div>
    <div class="family-controls" data-k="controls"></div>
    <div class="current"><span class="note">Showing</span> <b data-k="label"></b> <span data-k="badges"></span>
      <span style="margin-left:auto;display:flex;gap:8px;flex-wrap:wrap">
        <button class="btn" type="button" data-k="best-family">Best of this model</button>
        <button class="btn" type="button" data-k="best">Best overall</button></span></div>`;
  const q = k => container.querySelector(`[data-k="${k}"]`);
  const controls = q("controls");

  function renderControls() {
    const fam = sel.family;
    const wins = fam !== "baseline" ? windowsFor(fam) : [];
    const inputs = fam !== "baseline" && hasHistory(fam) ? `<div class="control"><div class="control-head">Inputs</div>
      <div class="seg" data-k="features"><button type="button" data-v="inst">Instantaneous</button><button type="button" data-v="hist">+ history</button></div></div>
      ${wins.length > 1 ? `<div class="control grow" data-k="window-wrap"><div class="control-head"><span>History window
        <span class="note" data-k="window-note"></span></span><span class="val" data-k="wval"></span></div>
        <input type="range" min="0" max="${wins.length - 1}" step="1" data-k="window" aria-label="History window">
        <div class="ticks">${wins.map(w => `<span>${windowShort(w)}</span>`).join("")}</div></div>` : ""}` : "";
    renderFamilyControls(fam);
    controls.insertAdjacentHTML("afterbegin", inputs);
  }
  function renderFamilyControls(fam) {
    if (fam === "baseline") {
      controls.innerHTML = `<p class="baseline-note">Predicts the most common class for every row. It has no settings and ignores the sensors entirely.</p>`;
    } else if (fam === "logreg") {
      controls.innerHTML = `
        ${task === "detection" ? `<div class="control"><div class="control-head">Class weighting</div>
          <div class="seg" data-k="weighting"><button type="button" data-v="none">Unweighted</button><button type="button" data-v="balanced">Balanced</button></div></div>` : ""}
        <div class="control"><div class="control-head">Penalty</div>
          <div class="seg" data-k="penalty"><button type="button" data-v="none">None</button><button type="button" data-v="l1">Lasso (L1)</button><button type="button" data-v="l2">Ridge (L2)</button></div></div>
        <div class="control grow"><div class="control-head"><span>Strength C <span class="note">(smaller is stronger)</span></span><span class="val" data-k="cval"></span></div>
          <input type="range" min="0" max="${grid.length - 1}" step="1" data-k="C" aria-label="Regularization strength C">
          <div class="ticks">${grid.map(c => `<span>${fmtC(c)}</span>`).join("")}</div></div>`;
    } else {
      controls.innerHTML = familyInfo(manifest, fam).params.map(p => p.type === "choice"
        ? `<div class="control"><div class="control-head">${esc(p.label)}</div>
            <div class="seg" data-k="${p.key}">${p.values.map(v => `<button type="button" data-v="${v}">${v === "T2" ? "Hotelling T²" : esc(v)}</button>`).join("")}</div></div>`
        : `<div class="control grow"><div class="control-head"><span>${esc(p.label)}</span><span class="val" data-val="${p.key}"></span></div>
          <input type="range" min="0" max="${p.values.length - 1}" step="1" data-param="${p.key}" aria-label="${esc(p.label)}">
          <div class="ticks">${p.values.map(v => `<span>${v}</span>`).join("")}</div></div>`).join("");
    }
  }

  function sync() {
    const model = findModel(manifest, task, sel);
    q("family").value = sel.family;
    // Inputs switch: disable an option when no model with these settings was trained on it.
    const cur = isHist(sel.features) ? "hist" : "inst";
    q("features")?.querySelectorAll("button").forEach(b => {
      b.classList.toggle("on", b.dataset.v === cur);
      b.setAttribute("aria-pressed", b.dataset.v === cur);
      b.disabled = b.dataset.v !== cur && !findModel(manifest, task, { ...sel, features: b.dataset.v });
      b.title = b.disabled ? "Not trained with these settings" : "";
    });
    const ww = q("window-wrap");
    if (ww) {
      const wins = windowsFor(sel.family), w = windowOf(sel.features);
      ww.hidden = !isHist(sel.features);
      q("window").value = Math.max(0, wins.indexOf(w));
      q("wval").textContent = isHist(sel.features) ? windowShort(w) : "";
      q("window-note").textContent = w === 10 ? "" : "(other windows: best 30-min settings only)";
    }
    if (sel.family === "logreg") {
      for (const k of ["weighting", "penalty"]) {
        q(k)?.querySelectorAll("button").forEach(b => {
          b.classList.toggle("on", b.dataset.v === sel[k]);
          b.setAttribute("aria-pressed", b.dataset.v === sel[k]);
        });
      }
      const usesC = sel.penalty === "l1" || sel.penalty === "l2";
      const slider = q("C");
      slider.disabled = !usesC;
      slider.value = usesC ? Math.max(0, grid.findIndex(c => sameC(c, sel.C))) : grid.length - 1;
      q("cval").textContent = usesC ? `C = ${fmtC(sel.C)}` : "no penalty";
    } else if (sel.family !== "baseline") {
      for (const p of familyInfo(manifest, sel.family).params) {
        if (p.type === "choice") {
          controls.querySelectorAll(`[data-k="${p.key}"] button`).forEach(b => b.classList.toggle("on", b.dataset.v === sel[p.key]));
          continue;
        }
        controls.querySelector(`[data-param="${p.key}"]`).value = Math.max(0, p.values.indexOf(sel[p.key]));
        controls.querySelector(`[data-val="${p.key}"]`).textContent = sel[p.key];
      }
    }
    q("label").textContent = model ? model.label : "not trained";
    const badges = [];
    if (model && model.id === manifest[task].best) badges.push(`<span class="badge best">best overall</span>`);
    else if (model && model.id === manifest[task].best_by_family[sel.family]) badges.push(`<span class="badge best">best ${esc(lowerLabel(familyLabel(manifest, sel.family)))}</span>`);
    if (model && model.coef_nonzero != null) {
      const n = model.n_inputs || 52;
      badges.push(`<span class="badge">${model.coef_nonzero} of ${n} ${n > 52 ? "inputs" : "sensors"} used</span>`);
    }
    q("badges").innerHTML = badges.join(" ");
    q("best-family").hidden = sel.family === "baseline";
    return model;
  }
  function changed(rebuild = false) {
    if (rebuild) renderControls();
    // Windows other than 30 minutes exist only for one setting: changing another setting returns to 30 minutes.
    if (isHist(sel.features) && sel.features !== "hist" && !findModel(manifest, task, sel)) sel.features = "hist";
    // A setting that only exists without history (e.g. lasso) switches the inputs back to instantaneous.
    if (isHist(sel.features) && !findModel(manifest, task, sel) && findModel(manifest, task, { ...sel, features: "inst" })) sel.features = "inst";
    last.set(sel.family, { ...sel });
    const model = sync();
    if (model) opts.onChange(sel, model);
  }
  q("family").addEventListener("change", e => { sel = { ...last.get(e.target.value) }; changed(true); });
  container.addEventListener("click", e => {
    const b = e.target.closest("button");
    if (!b) return;
    if (b.dataset.k === "best" || b.dataset.k === "best-family") {
      const m = b.dataset.k === "best" ? bestModel(manifest, task) : bestModel(manifest, task, sel.family);
      const famChanged = familyOf(m) !== sel.family;
      sel = selectionOf(m);
      changed(famChanged);
      return;
    }
    const seg = b.closest(".seg");
    if (!seg || !b.dataset.v) return;
    sel[seg.dataset.k] = b.dataset.v;
    if ((sel.penalty === "l1" || sel.penalty === "l2") && sel.C == null) sel.C = grid[grid.length - 1];
    if (sel.penalty === "none") sel.C = null;
    changed();
  });
  container.addEventListener("input", e => {
    const t = e.target;
    if (t.dataset.k === "C") { sel.C = grid[+t.value]; changed(); }
    else if (t.dataset.k === "window") {
      const w = windowsFor(sel.family)[+t.value], fs = fsForWindow(w);
      if (findModel(manifest, task, { ...sel, features: fs })) sel = { ...sel, features: fs };
      else {
        // Jump to the settings that were trained with this window.
        const m = manifest[task].models.find(x => familyOf(x) === sel.family && (x.features || "inst") === fs);
        if (m) sel = selectionOf(m);
      }
      changed();
    }
    else if (t.dataset.param) {
      const p = familyInfo(manifest, sel.family).params.find(x => x.key === t.dataset.param);
      sel[p.key] = p.values[+t.value];
      changed();
    }
  });
  renderControls();
  const model = sync();
  opts.onChange(sel, model);
  return { set(newSel) { const reb = newSel.family !== sel.family; sel = { ...newSel }; changed(reb); } };
}

// ---------- Which sensors a model relies on ----------
// Logistic regression: coefficients on standardized inputs (signed). Tree models: importances (share of the
// error reduction from splits on each sensor). Returns null if the model has neither.
export function sensorWeights(file, faultIndex = null) {
  if (file.coefficients) {
    const v = file.coefficients.values;
    if (!Array.isArray(v[0])) return { kind: "coef", values: v };
    if (faultIndex != null) return { kind: "coef", values: v[faultIndex] };
    return { kind: "coefsize", values: v[0].map((_, j) => Math.sqrt(v.reduce((a, row) => a + row[j] ** 2, 0))) };
  }
  if (file.importances) return { kind: "importance", values: file.importances.values, method: file.importances.kind };
  return null;
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

// ---------- Metrics at any false-alarm rate ----------
// The official comparison uses 5% false alarms with the exact numbers from the summary. Other rates are read from the
// binned validation scores; the bin edges include the exact thresholds for the listed rates, so these are exact too.
export const OFFICIAL_FAR = 0.05;
export function metricsAtFar(model, file, far) {
  if (model.penalty === "baseline" || familyOf(model) === "baseline")
    return { recall: 0, far: 0, precision: null, F1: 0, perFault: null, threshold: null };
  const official = Math.abs(far - OFFICIAL_FAR) < 1e-9 && model.val.recall != null;
  const summary = official ? { recall: model.val.recall, far: 1 - model.val.normal_recall, precision: model.val.precision, F1: model.val.F1,
                               threshold: model.val.threshold, ci: model.ci || null } : null;
  if (!file) return summary;
  const v = detectionView(file.val);
  const t = official ? v.indexForThreshold(model.val.threshold) : v.indexForFar(far), r = v.at(t);
  const fromBins = { recall: r.recall, far: r.far, precision: r.TP + r.FP ? r.precision : null, F1: r.f1, threshold: r.threshold };
  return { ...fromBins, ...(summary || {}), perFault: v.perFault(t), view: v, t };
}

// Every detection model file, loaded once and cached (used when ranking at a rate other than 5%).
let allDetCache = null;
export function loadAllDetection(manifest, onProgress) {
  if (!allDetCache) {
    const models = manifest.detection.models.filter(m => familyOf(m) !== "baseline");
    let done = 0;
    allDetCache = Promise.all(models.map(m => loadModel("detection", m.id).then(f => { onProgress?.(++done, models.length); return [m.id, f]; })))
      .then(pairs => Object.fromEntries(pairs));
  }
  return allDetCache;
}

// Nearest delay level available in a file for a requested rate.
export function nearestDelayLevel(delays, far) {
  const keys = Object.keys(delays);
  return keys.reduce((a, b) => (Math.abs(+b - far) < Math.abs(+a - far) ? b : a));
}

// False-alarm rate selector (buttons). The official rate is marked.
export function farControl(el, levels, value, onChange) {
  el.innerHTML = `<div class="seg" role="group" aria-label="False alarm rate">${levels.map(l =>
    `<button type="button" data-v="${l}" class="${Math.abs(l - value) < 1e-9 ? "on" : ""}">${+(100 * l).toFixed(1)}%${Math.abs(l - OFFICIAL_FAR) < 1e-9 ? " ★" : ""}</button>`).join("")}</div>`;
  el.onclick = e => {
    const b = e.target.closest("button[data-v]"); if (!b) return;
    el.querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b));
    onChange(+b.dataset.v);
  };
}
