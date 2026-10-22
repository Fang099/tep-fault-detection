import { renderChrome, showError, loadManifest, bestModel, VARIABLES, FAULTS, COLORS, pct, num, int, tile, esc } from "../site.js";

renderChrome();
const $ = id => document.getElementById(id);

$("var-table").innerHTML = `<thead><tr><th>Variable</th><th>Description</th></tr></thead><tbody>${
  Object.entries(VARIABLES).map(([k, v]) => `<tr><td><code>${k}</code></td><td>${esc(v)}</td></tr>`).join("")}</tbody>`;
$("fault-table").innerHTML = `<thead><tr><th>Fault</th><th>Process variable</th><th>Type</th></tr></thead><tbody>${
  Object.entries(FAULTS).map(([k, [d, t]]) => `<tr><td class="n">${k}</td><td>${esc(d)}</td><td>${t}</td></tr>`).join("")}</tbody>`;

// Timelines: one bar per run type, coloured by label.
function timeline(title, total, onset, sub) {
  const normalW = onset == null ? 100 : (onset / total) * 100;
  return `<div style="margin:0 0 16px">
    <div style="display:flex;justify-content:space-between;font-size:14.5px"><b>${title}</b><span class="note">${sub}</span></div>
    <div style="display:flex;height:26px;border-radius:6px;overflow:hidden;margin:4px 0 2px;border:1px solid var(--border)">
      <div style="width:${normalW}%;background:${COLORS.normal};opacity:.85"></div>
      ${onset == null ? "" : `<div style="flex:1;background:${COLORS.fault};opacity:.85"></div>`}
    </div>
    <div class="ticks" style="position:relative"><span>sample 1</span>${onset == null ? "" :
      `<span style="position:absolute;left:${normalW}%;padding-left:4px;border-left:1.5px solid ${COLORS.fault}">↑ fault starts after sample ${onset}</span>`}<span>sample ${total}</span></div>
  </div>`;
}
$("timelines").innerHTML = `
  <div class="legend" style="margin:0 0 10px"><span style="--c:${COLORS.normal}">labelled normal (0)</span><span style="--c:${COLORS.fault}">labelled faulty (1)</span></div>
  ${timeline("Fault-free run (training or test)", 500, null, "every row normal")}
  ${timeline("Faulty training run", 500, 20, "25 hours, fault after 1 hour")}
  ${timeline("Faulty test run", 960, 160, "48 hours, fault after 8 hours")}`;

loadManifest().then(manifest => {
  const d = manifest.dataset, s = manifest.settings;
  $("data-tiles").innerHTML = [
    tile("Runs used per fault", `${d.runs_per_fault} / 500`),
    tile("Training runs", int(d.train_runs)),
    tile("Validation runs", int(d.val_runs)),
    tile("Validation rows", int(d.val_rows), `${pct(d.val_fault_share, 0)} faulty`),
    tile("Inputs", "52", "process variables"),
  ].join("");
  $("subset-note").innerHTML = `Current results use ${d.runs_per_fault < 500 ? `the first ${d.runs_per_fault} of 500 runs per fault type` : "all 500 runs per fault type"}.
    To keep training time manageable, models train on every ${s.det_stride === 1 ? "" : `${ordinal(s.det_stride)} `}time step for detection and every
    ${ordinal(s.diag_stride)} for diagnosis (neighbouring 3-minute samples are nearly identical); validation and test always use every row.`;

  const det = bestModel(manifest, "detection"), dia = bestModel(manifest, "diagnosis");
  const detBase = manifest.detection.models.find(m => m.penalty === "baseline");
  const diaBase = manifest.diagnosis.models.find(m => m.penalty === "baseline");
  $("results-intro").textContent = `Validation results for ${manifest.detection.models.length + manifest.diagnosis.models.length} logistic-regression
    models and baselines. ${manifest.test ? "Test results are on the Results page." : "The final test has not been run yet."}`;
  $("result-tiles").innerHTML = [
    tile("Best detection model", esc(det.label), "", "text"),
    tile("Faults caught at 5% false alarms", pct(det.val.recall), "baseline: 0% (it can only alarm on everything)", "big"),
    tile("Detection ROC-AUC", num(det.val.ROC_AUC, 3), `baseline ${num(detBase.val.ROC_AUC, 3)}`, "big"),
    tile("Best diagnosis model", esc(dia.label), "", "text"),
    tile("Diagnosis macro-F1", num(dia.val.macro_F1, 3), `baseline ${num(diaBase.val.macro_F1, 3)}`, "big"),
  ].join("");
}).catch(showError);

function ordinal(n) { return n === 2 ? "2nd" : n === 3 ? "3rd" : `${n}th`; }
