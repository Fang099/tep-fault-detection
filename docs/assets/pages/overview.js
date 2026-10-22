import { renderChrome, showError, loadManifest, bestModel, VARIABLES, FAULTS, COLORS, pct, num, int, tile, esc } from "../site.js";

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
