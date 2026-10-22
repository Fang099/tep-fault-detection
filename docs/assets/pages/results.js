import {
  renderChrome, showError, loadManifest, loadModel, loadTestModel, bestModel, dataNote, detectionView, findModel as findAny,
  familyOf, familyLabel, modelLink, COLORS, pct, num, int, tile, esc,
} from "../site.js";
import { barChart, legend } from "../charts.js";

renderChrome();
const $ = id => document.getElementById(id);
const findModel = (manifest, task, sel) => findAny(manifest, task, { family: "logreg", ...sel });

main().catch(showError);

async function main() {
  const manifest = await loadManifest();
  $("data-note").innerHTML = dataNote(manifest);
  renderFamilies(manifest);
  renderFindings(manifest);
  await renderTest(manifest);
}

// ---------- Findings (computed from the manifest so they stay true when results change) ----------
function renderFindings(manifest) {
  const det = manifest.detection.models, dia = manifest.diagnosis.models;
  const best = bestModel(manifest, "detection"), bestD = bestModel(manifest, "diagnosis");
  const lrDet = bestModel(manifest, "detection", "logreg"), lrDia = bestModel(manifest, "diagnosis", "logreg");
  const base = det.find(m => m.penalty === "baseline");
  const noneU = findModel(manifest, "detection", { penalty: "none", weighting: "none" });
  const noneB = findModel(manifest, "detection", { penalty: "none", weighting: "balanced" });
  const bestOf = (models, f) => models.reduce((a, b) => (f(b) > f(a) ? b : a));
  const regDet = bestOf(det.filter(m => m.penalty === "l1" || m.penalty === "l2"), m => m.val.recall);
  const regDia = bestOf(dia.filter(m => m.penalty === "l1" || m.penalty === "l2"), m => m.val.macro_F1);
  const noneDia = dia.find(m => m.penalty === "none");
  const lassoSmall = det.filter(m => m.penalty === "l1" && m.coef_nonzero > 0).sort((a, b) => a.coef_nonzero - b.coef_nonzero)[0];

  const nonlinear = familyOf(best) !== "logreg" && lrDet;
  const items = [
    nonlinear ? `<b>Non-linear models do better than logistic regression.</b> The best detection model, ${esc(best.label)}, catches
     ${pct(best.val.recall)} of faulty rows at 5% false alarms against ${pct(lrDet.val.recall)} for the best logistic regression.
     ${lrDia && familyOf(bestD) !== "logreg" ? `For diagnosis, ${esc(bestD.label)} reaches macro-F1 ${num(bestD.val.macro_F1, 3)} against
     ${num(lrDia.val.macro_F1, 3)}.` : ""} Some faults change the readings in ways no single weighted sum can separate.` : "",
    `<b>Accuracy and F1 are misleading here.</b> The majority baseline scores F1 ${num(base.val.F1, 3)} by alarming on every row,
     including every normal one. Compared at 5% false alarms, it catches nothing.`,
    `<b>The best detection model (${esc(best.label)}) catches ${pct(best.val.recall)} of faulty rows at 5% false alarms</b>
     (ROC-AUC ${num(best.val.ROC_AUC, 3)}). Roughly half of all faulty rows look normal in a single snapshot.`,
    noneU && noneB ? `<b>Class weighting matters only a little.</b> Without a penalty, balanced weights catch ${pct(noneB.val.recall)} against
     ${pct(noneU.val.recall)} unweighted. At a fixed false-alarm rate, weighting mostly moves the threshold rather than changing the ranking of rows.` : "",
    (() => {
      const noneBest = [noneU, noneB].filter(Boolean).reduce((a, b) => (b.val.recall > a.val.recall ? b : a));
      const gain = Math.max(regDet.val.recall - noneBest.val.recall, regDia.val.macro_F1 - noneDia.val.macro_F1);
      return gain <= 0.005
        ? `<b>Regularization does not improve accuracy.</b> The best regularized detection model reaches ${pct(regDet.val.recall)}
           (${esc(regDet.label)}) against ${pct(noneBest.val.recall)} without a penalty; for diagnosis, macro-F1 ${num(regDia.val.macro_F1, 3)}
           against ${num(noneDia.val.macro_F1, 3)}. With 52 inputs and hundreds of thousands of rows, there is little overfitting to prevent.`
        : `<b>Mild regularization helps slightly.</b> Detection: ${pct(regDet.val.recall)} (${esc(regDet.label)}) against
           ${pct(noneBest.val.recall)} without a penalty. Diagnosis: macro-F1 ${num(regDia.val.macro_F1, 3)} against ${num(noneDia.val.macro_F1, 3)}.`;
    })(),
    lassoSmall ? `<b>Lasso shows how few sensors carry the signal.</b> ${esc(lassoSmall.label)} uses only ${lassoSmall.coef_nonzero} of 52 sensors
     and still catches ${pct(lassoSmall.val.recall)} of faulty rows at 5% false alarms. See <a href="regularization.html">Regularization</a>.` : "",
    `<b>Diagnosis is far above chance but uneven.</b> The best diagnosis model (${esc(bestD.label)}) reaches macro-F1
     ${num(bestD.val.macro_F1, 3)} against ${num(dia.find(m => m.penalty === "baseline").val.macro_F1, 3)} for the baseline;
     some faults are named almost perfectly while others are barely above chance. See <a href="diagnosis.html">Diagnosis</a>.`,
    `<b>Detection speed varies enormously by fault:</b> from minutes for sudden faults to many hours, or never, for subtle ones.
     See <a href="detection.html#speed">Detection</a>.`,
  ];
  $("findings-list").innerHTML = items.filter(Boolean).map(t => `<li>${t}</li>`).join("");
}

// ---------- Best model of each family ----------
function renderFamilies(manifest) {
  const fams = manifest.families.filter(f => f.key !== "baseline");
  const rows = fams.map(f => ({ f, det: bestModel(manifest, "detection", f.key), dia: bestModel(manifest, "diagnosis", f.key) }));
  $("families-table").innerHTML = `<thead><tr><th>Model</th><th>Best detection settings</th><th class="n">Caught at 5% false alarms</th>
      <th class="n">ROC-AUC</th><th>Best diagnosis settings</th><th class="n">Macro-F1</th></tr></thead><tbody>${
    rows.map(({ f, det, dia }) => `<tr class="${det?.id === manifest.detection.best || dia?.id === manifest.diagnosis.best ? "highlight" : ""}">
      <td><b>${esc(f.label)}</b></td>
      <td>${det ? `<a href="${modelLink("detection.html", det)}">${esc(det.label)}</a>` : "–"}</td>
      <td class="n">${det ? pct(det.val.recall) : "–"}</td><td class="n">${det ? num(det.val.ROC_AUC, 3) : "–"}</td>
      <td>${dia ? `<a href="${modelLink("diagnosis.html", dia)}">${esc(dia.label)}</a>` : "–"}</td>
      <td class="n">${dia ? num(dia.val.macro_F1, 3) : "–"}</td></tr>`).join("")}</tbody>`;
  legend($("legend-families"), [{ label: "Detection: faults caught at 5% false alarms", color: COLORS.fault },
                                { label: "Diagnosis: macro-F1", color: COLORS.data }]);
  barChart($("chart-families"), {
    categories: rows.map(r => r.f.label),
    series: [{ name: "det", color: COLORS.fault, values: rows.map(r => r.det?.val.recall ?? null) },
             { name: "dia", color: COLORS.data, values: rows.map(r => r.dia?.val.macro_F1 ?? null) }],
    x: { domain: [0, 1], fmt: x => num(x, 1), label: "Validation score (higher is better)" },
    labelWidth: 150, valueFmt: (x, k, s) => (s.name === "det" ? pct(x) : num(x, 3)),
    tip: k => `<b>${esc(rows[k].f.label)}</b><br>Detection: ${rows[k].det ? pct(rows[k].det.val.recall) : "–"}<br>Diagnosis: ${rows[k].dia ? num(rows[k].dia.val.macro_F1, 3) : "–"}`,
  });
}

// ---------- Final test ----------
async function renderTest(manifest) {
  const body = $("test-body");
  const test = manifest.test;
  if (!test) {
    body.innerHTML = `<div class="empty" style="min-height:120px">The final test has not been run yet.<br>
      It runs once, after all models have been chosen on validation data.</div>`;
    return;
  }
  const detIds = Object.keys(test.detection), diaIds = Object.keys(test.diagnosis);
  const label = (task, id) => manifest[task].models.find(m => m.id === id)?.label || id;
  body.innerHTML = `
    <p class="intro">The chosen models, retrained on all training runs and evaluated once on ${int(test.rows)} test rows
      (${test.runs_per_fault} faulty runs per fault type plus the fault-free test runs; ${pct(test.fault_share, 0)} faulty).
      Detection uses the threshold chosen on validation.</p>
    <h3>Detection</h3>
    <div class="table-wrap"><table><thead><tr><th>Model</th><th class="n">Caught</th><th class="n">False alarms</th><th class="n">Precision</th>
      <th class="n">F1</th><th class="n">ROC-AUC</th><th class="n">PR-AUC</th></tr></thead><tbody>${
      detIds.map(id => { const r = test.detection[id]; return `<tr><td>${esc(label("detection", id))}</td><td class="n">${pct(r.recall)}</td>
        <td class="n">${pct(1 - r.normal_recall)}</td><td class="n">${pct(r.precision)}</td><td class="n">${num(r.F1, 3)}</td>
        <td class="n">${num(r.ROC_AUC, 3)}</td><td class="n">${num(r.PR_AUC, 3)}</td></tr>`; }).join("")}</tbody></table></div>
    <h3>Diagnosis</h3>
    <div class="table-wrap"><table><thead><tr><th>Model</th><th class="n">Macro-F1</th><th class="n">Accuracy</th></tr></thead><tbody>${
      diaIds.map(id => { const r = test.diagnosis[id]; return `<tr><td>${esc(label("diagnosis", id))}</td><td class="n">${num(r.macro_F1, 3)}</td>
        <td class="n">${pct(r.accuracy)}</td></tr>`; }).join("")}</tbody></table></div>
    <h3>Validation against test, per fault (best detection model)</h3>
    <div class="plot"><div id="legend-test"></div><div id="chart-test"></div></div>`;

  const best = bestModel(manifest, "detection");
  const [valFile, testFile] = await Promise.all([loadModel("detection", best.id), loadTestModel("detection", best.id)]);
  const vv = detectionView(valFile.val), tv = detectionView(testFile.test);
  const valRec = vv.perFault(vv.indexForThreshold(best.val.threshold));
  const testRec = tv.perFault(tv.indexForThreshold(best.val.threshold));
  legend($("legend-test"), [{ label: "Validation", color: COLORS.data }, { label: "Test", color: COLORS.b }]);
  barChart($("chart-test"), {
    categories: vv.faults.map(f => `Fault ${f}`),
    series: [{ name: "val", color: COLORS.data, values: valRec }, { name: "test", color: COLORS.b, values: testRec }],
    x: { domain: [0, 1], fmt: x => pct(x, 0), label: "Share of the fault's rows caught at the validation threshold" },
    labelWidth: 70, valueFmt: x => pct(x, 0),
  });
}
