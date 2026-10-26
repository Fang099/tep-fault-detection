// "How much history?" — the history window sweep, shared by the Detection and Diagnosis pages.
// For each family trained with history, the best 30-minute settings were retrained with other windows. This plots
// the score and the typical time to detection against the window, with "no history" (same settings) as the start.
import { loadModel, familyOf, familyLabel, findModel, selectionOf, isHist, windowOf, windowShort, windowText,
         PALETTE, pct, num, fmtMinutes, esc, lowerLabel } from "./site.js";
import { lineChart, legend } from "./charts.js";

// Chains of models: [{family, base, points:[{w, model}]}], one per family that has a sweep.
export function sweepChains(manifest, task) {
  const models = manifest[task].models;
  const chains = [];
  for (const base of models.filter(m => m.features === "hist")) {
    const sweep = models.filter(m => m.sweep_of === base.id);
    if (!sweep.length) continue;
    const inst = findModel(manifest, task, { ...selectionOf(base), features: "inst" });
    const points = [...(inst ? [{ w: 0, model: inst }] : []), { w: 10, model: base },
                    ...sweep.map(m => ({ w: windowOf(m.features), model: m }))].sort((a, b) => a.w - b.w);
    chains.push({ family: familyOf(base), base, points });
  }
  return chains;
}

const metricOf = (task, m) => (task === "detection" ? m.val.recall : m.val.macro_F1);

// Typical time to detection for one model file: the median, over faults that every window of the chain detects in at
// least half of the runs, of each fault's median delay (so all windows are compared on the same faults).
function delaysOf(task, file) {
  return task === "detection" ? (file.val.delays["0.05"] || Object.values(file.val.delays)[0]).by_fault : file.val.delays;
}
const median = a => { const s = [...a].sort((x, y) => x - y), n = s.length; return n ? (n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2) : null; };

export async function renderWindowSweep({ task, manifest, model, els, onPick }) {
  const chains = sweepChains(manifest, task);
  const section = els.section;
  if (!chains.length) {
    section.hidden = true;
    document.querySelector(`#sidetoc a[href="#${section.id}"]`)?.parentElement?.setAttribute("hidden", "");
    return;
  }
  section.hidden = false;
  const fam = familyOf(model);
  const color = k => PALETTE[(k + 1) % PALETTE.length];
  const wins = [...new Set(chains.flatMap(c => c.points.map(p => p.w)))].sort((a, b) => a - b);
  const xOf = w => wins.indexOf(w);
  const label = w => (w === 0 ? "none" : windowShort(w));
  const isCurrent = m => m.id === model.id;
  const items = chains.map((c, k) => ({ label: familyLabel(manifest, c.family), color: color(k) }));

  // Chart 1: score against the window.
  legend(els.legend, items);
  const metricName = task === "detection" ? "Faults caught at 5% false alarms" : "Macro-F1";
  const fmt = task === "detection" ? v => pct(v) : v => num(v, 3);
  const series = chains.map((c, k) => ({
    name: familyLabel(manifest, c.family), color: color(k), width: c.family === fam ? 3 : 1.8, marker: c.family === fam ? 4.5 : 3.5,
    opacity: c.family === fam ? 1 : 0.75, points: c.points.map(p => [xOf(p.w), metricOf(task, p.model), p.model]),
  }));
  const vals = series.flatMap(s => s.points.map(p => p[1]));
  const lo = Math.max(0, Math.min(...vals) - 0.05), hi = Math.min(1, Math.max(...vals) + 0.03);
  const cur = chains.flatMap(c => c.points).find(p => isCurrent(p.model));
  lineChart(els.chart, {
    height: 300,
    x: { domain: [-0.3, wins.length - 0.7], ticks: wins.map((_, i) => i), fmt: i => label(wins[i]), label: "History window (none = single snapshot)" },
    y: { domain: [lo, hi], fmt: task === "detection" ? y => pct(y, 0) : y => num(y, 2), label: metricName },
    series,
    markers: cur ? [{ x: xOf(cur.w), y: metricOf(task, cur.model), color: "var(--text)", r: 6 }] : [],
    tip: (s, p) => `<b>${esc(p[2].label)}</b><br>${metricName}: <b>${fmt(p[1])}</b><br><span class="note">click to open</span>`,
    onClick: (s, p) => onPick(p[2]),
    hoverRadius: 22,
  });

  // Chart 2: typical time to detection (or to a correct diagnosis) against the window.
  els.speedNote.textContent = "Loading detection times…";
  const files = await Promise.all(chains.map(c => Promise.all(c.points.map(p => loadModel(task, p.model.id).catch(() => null)))));
  const speed = chains.map((c, k) => {
    const del = files[k].map(f => (f ? delaysOf(task, f) : null));
    if (del.some(d => !d)) return null;
    const faults = Object.keys(del[0]).filter(f => del.every(d => d[f] && d[f][0] != null && d[f][1] >= 0.5));
    if (!faults.length) return null;
    return { faults, points: c.points.map((p, i) => [xOf(p.w), median(faults.map(f => del[i][f][0])), p.model]) };
  });
  legend(els.legend2, items.filter((_, k) => speed[k]?.points.some(p => p[1] != null)));
  const sv = speed.filter(Boolean).flatMap(s => s.points.map(p => p[1])).filter(v => v != null);
  if (!sv.length) { els.chart2.innerHTML = `<div class="empty">No detection times available.</div>`; els.speedNote.textContent = ""; return; }
  lineChart(els.chart2, {
    height: 260,
    x: { domain: [-0.3, wins.length - 0.7], ticks: wins.map((_, i) => i), fmt: i => label(wins[i]), label: "History window" },
    y: { domain: [0, Math.max(...sv) * 1.15], fmt: y => fmtMinutes(y), label: task === "detection" ? "Typical time to detect" : "Typical time to diagnose" },
    series: chains.map((c, k) => speed[k] && {
      name: familyLabel(manifest, c.family), color: color(k), width: c.family === fam ? 3 : 1.8, marker: c.family === fam ? 4.5 : 3.5,
      opacity: c.family === fam ? 1 : 0.75, points: speed[k].points.filter(p => p[1] != null),
    }).filter(Boolean),
    tip: (s, p) => `<b>${esc(p[2].label)}</b><br>typical time: <b>${fmtMinutes(p[1])}</b><br><span class="note">click to open</span>`,
    onClick: (s, p) => onPick(p[2]),
    hoverRadius: 22,
  });
  const nF = speed.find((s, k) => s && chains[k].family === fam)?.faults.length ?? speed.find(Boolean)?.faults.length;
  els.speedNote.textContent = `Median over the ${nF ?? ""} faults that every window of a model catches in at least half of the runs
    (${task === "detection" ? "3 alarms in a row at 5% false alarms" : "3 correct diagnoses in a row"}), so all windows are compared on the same faults.`;

  // Callout for the selected model's family (or the first family).
  const k = Math.max(0, chains.findIndex(c => c.family === fam));
  const c = chains[k], pts = c.points;
  const best = pts.filter(p => p.model.selectable !== false).reduce((a, b) => (metricOf(task, b.model) > metricOf(task, a.model) ? b : a));
  const none = pts.find(p => p.w === 0), longest = pts[pts.length - 1];
  const sp = speed[k]?.points;
  const tOf = w => sp?.[pts.findIndex(p => p.w === w)]?.[1];
  const name = lowerLabel(familyLabel(manifest, c.family));
  let text = `<b>For ${esc(name)}, ${best.w === 0 ? "no history" : `a ${windowText(best.w)} window`} works best</b>
    (${fmt(metricOf(task, best.model))}${none && best.w !== 0 ? `, against ${fmt(metricOf(task, none.model))} from single snapshots` : ""}).`;
  if (longest.w !== best.w && metricOf(task, longest.model) < metricOf(task, best.model))
    text += ` Going longer, to ${windowShort(longest.w)}, gives ${fmt(metricOf(task, longest.model))}: the window dilutes a fresh fault with readings from before it.`;
  if (sp && tOf(longest.w) != null && tOf(pts[0].w) != null && tOf(longest.w) > tOf(pts[0].w))
    text += ` Longer windows also react later: typically ${fmtMinutes(tOf(pts[0].w))} with ${pts[0].w === 0 ? "no history" : windowShort(pts[0].w)}
      against ${fmtMinutes(tOf(longest.w))} with ${windowShort(longest.w)}.`;
  const unselectable = chains.flatMap(c => c.points).filter(p => p.model.selectable === false).map(p => p.w);
  const capNote = unselectable.length ? ` Windows longer than 1 hour (${[...new Set(unselectable)].map(windowShort).join(", ")}) are shown but never
    picked as the best model: training and validation faults start after 1 hour, so such a window is still partly empty when the fault
    starts, while in the test files (fault after 8 hours) it is full. Validation would flatter them.` : "";
  els.callout.innerHTML = text + ` <span class="note">Windows other than 30 minutes use the best 30-minute settings, so they are compared on equal terms.${capNote}</span>`;
}
