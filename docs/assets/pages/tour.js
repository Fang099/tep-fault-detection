// Guided tour: the whole project in nine short steps, each with one small interactive piece built from the real results.
import {
  renderChrome, showError, loadManifest, loadModel, loadTestModel, getJSON, pct, num, int, esc, fmtMinutes, COLORS,
  familyLabel, familyOf, isHist, detectionView, logit, VARIABLES, windowShort, windowOf,
} from "../site.js";
import { lineChart, barChart, legend } from "../charts.js";

renderChrome();
const $ = id => document.getElementById(id);
const D = {};                                  // everything loaded once
let cur = 0;

// ---------- helpers ----------
const median = a => { const s = [...a].sort((x, y) => x - y), n = s.length; return n ? (n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2) : null; };
const seg = (id, opts, on) => `<div class="seg" id="${id}">${opts.map(([v, l]) => `<button type="button" data-v="${v}" class="${v === on ? "on" : ""}">${l}</button>`).join("")}</div>`;
function onSeg(id, fn) {
  const el = $(id);
  el.addEventListener("click", e => {
    const b = e.target.closest("button"); if (!b) return;
    el.querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b));
    fn(b.dataset.v);
  });
}
const tiles = items => `<div class="tour-tiles">${items.map(([v, l, cls]) => `<div class="tour-tile ${cls || ""}"><b>${v}</b><span>${l}</span></div>`).join("")}</div>`;
const sensorName = s => `${s.replace("_", " ")}`;

// Detection-score density on a log-odds axis (as on the Detection page).
const X_LIM = 16.2;
const xOf = p => Math.max(-X_LIM, Math.min(X_LIM, logit(p)));
function density(edges, counts, total) {
  const pts = [];
  for (let k = 0; k < counts.length; k++) {
    const lo = xOf(edges[k]), hi = xOf(edges[k + 1]);
    if (!(hi > lo)) continue;
    const d = total ? counts[k] / total / (hi - lo) : 0;
    pts.push([lo, d], [hi, d]);
  }
  return pts;
}

// ---------- the steps ----------
const STEPS = [
  {
    kicker: "The problem",
    title: "Is something wrong, and what is it?",
    text: () => `<p>A simulated industrial plant is watched by <b>52 sensors</b>, read every 3 minutes. Now and then something
      goes wrong: one of <b>20 kinds of fault</b>. The project asks two questions of every set of readings, and trains machine
      learning models to answer them.</p>`,
    draw(el) {
      const q = D.manifest.dataset;
      el.innerHTML = `
        <div class="tour-qs">
          <button type="button" class="tour-q" data-k="det"><span class="tour-q-n">1</span><b>Is something wrong?</b><span>Detection: fault or normal</span></button>
          <span class="tour-arrow" aria-hidden="true">→</span>
          <button type="button" class="tour-q" data-k="dia"><span class="tour-q-n">2</span><b>Which fault is it?</b><span>Diagnosis: one of 20 faults</span></button>
        </div>
        <p class="tour-reveal" id="t1-reveal">Click a question to see how well the best model answers it.</p>
        ${tiles([[q.features.length, "sensors"], ["20", "fault types"], [int(D.quality.runs), "simulated runs"], [`${(D.quality.rows / 1e6).toFixed(2)} M`, "rows of readings"]])}`;
      el.querySelectorAll(".tour-q").forEach(b => b.addEventListener("click", () => {
        el.querySelectorAll(".tour-q").forEach(x => x.classList.toggle("on", x === b));
        $("t1-reveal").innerHTML = b.dataset.k === "det"
          ? `The best model catches <b>${pct(D.det.val.recall)}</b> of faulty readings while raising a false alarm on only 5% of normal ones.`
          : `Once a fault is flagged, the best model names the right one with a macro-F1 of <b>${num(D.dia.val.macro_F1, 3)}</b> (1 = always right).`;
      }));
    },
  },
  {
    kicker: "Why it is hard",
    title: "Can you spot the fault?",
    text: () => `<p>Below is one sensor from a real simulation run. At some point a fault is switched on. <b>Click the chart where you
      think it starts.</b> Pick a different fault to try again: some are obvious, some are almost invisible.</p>`,
    draw(el) {
      const levels = D.spotFaults;
      el.innerHTML = `<div class="tour-row">${seg("t2-fault", levels.map(l => [String(l.f), `${l.tag} (fault ${l.f})`]), String(levels[0].f))}</div>
        <div class="plot"><div id="t2-legend"></div><div id="t2-chart"></div></div><p class="tour-reveal" id="t2-out"></p>`;
      let f = levels[0].f, guess = null;
      const draw = () => {
        const run = D.traces.runs.find(r => r.fault === f), base = D.traces.runs.find(r => r.fault === 0);
        const s = D.spotSensor[f], minutes = D.traces.sample_minutes, onset = D.traces.onset;
        const keep = run.sample.map((t, i) => [t, i]).filter(([t]) => t <= 160);
        const pts = keep.map(([t, i]) => [t * minutes / 60, run.values[s][i]]);
        const normPts = keep.map(([t, i]) => [t * minutes / 60, base.values[s][i]]);
        const delay = D.det.delays.by_fault[f]?.[0];
        const vl = guess == null ? [] : [
          { x: onset * minutes / 60, color: COLORS.fault, dash: "5 4", width: 2 },
          { x: guess, color: COLORS.text, width: 2 },
          ...(delay != null ? [{ x: (onset * minutes + delay) / 60, color: COLORS.normal, dash: "2 3", width: 2 }] : []),
        ];
        legend($("t2-legend"), guess == null ? [{ label: "this run", color: COLORS.data }] : [
          { label: "this run", color: COLORS.data }, { label: "same run, no fault", color: COLORS.muted },
          { label: "fault starts", color: COLORS.fault, dash: true }, { label: "your guess", color: COLORS.text },
          ...(delay != null ? [{ label: "model's typical alarm", color: COLORS.normal, dash: true }] : [])]);
        const all = [...pts, ...(guess != null ? normPts : [])].map(p => p[1]);
        const lo = Math.min(...all), hi = Math.max(...all), pad = (hi - lo) * 0.12 || 1;
        lineChart($("t2-chart"), {
          height: 260,
          x: { domain: [0, 8], label: "Hours into the run", fmt: v => `${v} h` },
          y: { domain: [lo - pad, hi + pad], label: sensorName(s), fmt: v => +v.toPrecision(4) },
          series: [...(guess != null ? [{ name: "fault-free run", color: COLORS.muted, width: 1.4, opacity: 0.7, points: normPts, hover: false }] : []),
                   { name: "this run", color: COLORS.data, width: 1.8, points: pts }],
          vlines: vl, hoverRadius: 2000,
          onClick: (sr, p) => { guess = p[0]; draw(); },
        });
        if (guess == null) { $("t2-out").textContent = "Click anywhere on the chart to place your guess."; return; }
        const start = onset * minutes / 60, diff = Math.round((guess - start) * 60);
        $("t2-out").innerHTML = `${diff < 0 ? `Your guess was <b>${fmtMinutes(-diff)} before</b> the fault started: that would have been a false alarm.`
          : `You spotted it <b>${fmtMinutes(diff)}</b> after it started.`}
          ${delay != null ? ` The best model typically raises the alarm <b>${fmtMinutes(delay)}</b> after the start (3 alarms in a row, across all validation runs of this fault).` : ""}
          The grey line is the same run without the fault: the two are identical until the fault begins.
          <span class="note">This shows one sensor; the models see all 52 at once.</span>`;
      };
      onSeg("t2-fault", v => { f = +v; guess = null; draw(); });
      draw();
    },
  },
  {
    kicker: "Fair testing",
    title: "Train on some runs, test on others",
    text: () => `<p>The models learn from 80% of the simulation runs and are checked on the other 20%, which they never see while
      learning. Each run stays <b>whole on one side</b>. Splitting at random by row instead would let a model peek: rows 3 minutes
      apart are nearly identical, so it would be tested on near-copies of what it learned.</p>`,
    draw(el) {
      const runs = 5, n = 40, checkRun = 3, focus = { run: 1, i: 22 };
      // A fixed pseudo-random 20% of readings for the random split (the same every time), always including the highlighted one.
      let seed = 11;
      const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
      const randomVal = Array.from({ length: runs }, (_, r) => Array.from({ length: n }, (_, i) =>
        (r === focus.run && i === focus.i) ? true : (r === focus.run && Math.abs(i - focus.i) === 1) ? false : rnd() < 0.2));
      el.innerHTML = `<div class="tour-row">${seg("t3-mode", [["run", "Split by run (this project)"], ["random", "Split at random"]], "run")}</div>
        <div class="t3-runs" id="t3-runs"></div>
        <div class="legend" style="margin:8px 0 0"><span style="--c:${COLORS.normal}">used to learn</span><span style="--c:${COLORS.fault}">used to check</span></div>
        <p class="note" style="margin:2px 0 0">Each bar is one simulation run, each block one reading, in time order (a real run has 500 readings, 3 minutes apart).</p>
        <p class="tour-reveal" id="t3-out"></p>`;
      const draw = mode => {
        $("t3-runs").innerHTML = Array.from({ length: runs }, (_, r) => {
          const whole = mode === "run" ? (r === checkRun ? "check" : "learn") : "both";
          const blocks = Array.from({ length: n }, (_, i) => {
            const val = mode === "run" ? r === checkRun : randomVal[r][i];
            const mark = mode === "random" && r === focus.run ? (i === focus.i ? " focus" : Math.abs(i - focus.i) === 1 ? " nb" : "") : "";
            return `<i class="${val ? "val" : ""}${mark}"></i>`;
          }).join("");
          return `<div class="t3-run"><span class="t3-name">Run ${r + 1}</span><div class="t3-bar" style="--n:${n}">${blocks}</div>
            <span class="t3-tag ${whole}">${whole === "check" ? "checked" : whole === "learn" ? "learned" : "both"}</span></div>`;
        }).join("");
        const s = D.split;
        $("t3-out").innerHTML = mode === "run"
          ? `Each run is used <b>entirely</b> for learning or entirely for checking. The checked run is completely new to the model, just like a
             new day at a real plant.`
          : `Now every run is cut up. Look at run ${focus.run + 1}: the outlined reading is used to check the model, but the readings
             3 minutes before and after it (dashed) were used to learn, and they are almost identical. The model is partly tested on what it
             has already seen.${s ? ` Measured on this data, that makes a deep decision tree look better than it is: ROC-AUC
             <b>${num(s.detection.tree.random.ROC_AUC, 3)}</b> instead of the honest <b>${num(s.detection.tree.run.ROC_AUC, 3)}</b>.` : ""}`;
      };
      onSeg("t3-mode", draw);
      draw("run");
    },
  },
  {
    kicker: "Detection",
    title: "From a score to an alarm",
    text: () => `<p>A detection model gives every set of readings a <b>fault score</b>. Above a threshold, it raises an alarm.
      Drag the threshold: catching more faults always means more false alarms. This project compares models at the threshold that
      gives <b>5% false alarms</b>.</p>`,
    draw(el) {
      const v = D.detView, file = D.detFile.val;
      el.innerHTML = `<div class="plot"><div id="t4-legend"></div><div id="t4-chart"></div></div>
        <div class="tour-row" style="align-items:center"><input type="range" id="t4-slider" min="0" max="${v.B}" step="1" aria-label="Threshold" style="flex:1;min-width:180px">
          <button class="btn" type="button" id="t4-five">Set 5% false alarms</button></div>
        <div id="t4-tiles"></div>`;
      legend($("t4-legend"), [{ label: "normal readings", color: COLORS.normal }, { label: "faulty readings", color: COLORS.fault }]);
      const totF = v.totalF, faultCounts = file.normal_counts.map((_, k) => Object.values(file.fault_counts).reduce((a, c) => a + c[k], 0));
      const normalPts = density(v.edges, v.normal, v.totalN), faultPts = density(v.edges, faultCounts, totF);
      const ys = [...normalPts, ...faultPts].map(p => p[1]).sort((a, b) => a - b);
      const yMax = (ys[Math.floor(ys.length * 0.985)] || 1) * 1.25;
      const ticks = [1e-4, 0.01, 0.5, 0.99, 0.9999, 0.999999];
      let t = v.indexForFar(0.05);
      const draw = () => {
        const r = v.at(t), x = t < v.B ? xOf(v.edges[t]) : X_LIM;
        lineChart($("t4-chart"), {
          height: 240,
          x: { domain: [Math.max(-X_LIM, xOf(v.edges[0])), X_LIM], ticks: ticks.map(xOf), fmt: x => String(ticks.find(p => Math.abs(xOf(p) - x) < 1e-6)),
               label: "Fault score (spaced by log-odds)" },
          y: { domain: [0, yMax], ticks: [], label: "How many readings" },
          series: [{ color: COLORS.normal, points: normalPts, area: true, width: 1.6 }, { color: COLORS.fault, points: faultPts, area: true, width: 1.6 }],
          vlines: [{ x, label: "threshold", color: COLORS.text }],
        });
        $("t4-tiles").innerHTML = tiles([[pct(r.recall), "of faulty readings caught"], [pct(r.far), "of normal readings raise a false alarm",
          Math.abs(r.far - 0.05) < 0.003 ? "hl" : ""]]);
        $("t4-slider").value = t;
      };
      $("t4-slider").addEventListener("input", e => { t = +e.target.value; draw(); });
      $("t4-five").addEventListener("click", () => { t = v.indexForFar(0.05); draw(); });
      draw();
    },
  },
  {
    kicker: "Detection",
    title: "Which kind of model works best?",
    text: () => `<p>Nine kinds of model were trained, from logistic regression (one straight boundary) to tree ensembles like XGBoost.
      Each can also be given <b>history</b>: besides the current readings, the average and spread of each sensor over the last
      half hour or hour. Switch history on and off.</p>`,
    draw(el) {
      el.innerHTML = `<div class="tour-row">${seg("t5-hist", [["inst", "Current readings only"], ["hist", "With history"]], "hist")}</div>
        <div class="plot"><div id="t5-chart"></div></div><p class="tour-reveal" id="t5-out"></p>`;
      const draw = mode => {
        const rows = D.families.map(r => ({ ...r, v: mode === "hist" && r.hist ? r.hist : r.inst })).filter(r => r.v).sort((a, b) => b.v.val.recall - a.v.val.recall);
        const top = rows[0];
        barChart($("t5-chart"), {
          categories: rows.map(r => r.label + (r.subset ? " *" : "")),
          series: [{ name: "caught", color: COLORS.data, values: rows.map(r => r.v.val.recall), opacity: rows.map(r => (mode === "hist" && !r.hist ? 0.4 : 1)) }],
          x: { domain: [0, 1], fmt: x => pct(x, 0), label: "Faulty readings caught at 5% false alarms" },
          labelWidth: 200, valueFmt: x => pct(x, 1), highlight: 0,
          tip: i => `<b>${esc(rows[i].v.label)}</b><br>${pct(rows[i].v.val.recall)} caught${mode === "hist" && !rows[i].hist ? "<br>(no history version)" : ""}`,
        });
        const gain = D.bestHist.val.recall - D.bestInst.val.recall;
        $("t5-out").innerHTML = mode === "hist"
          ? `With history, <b>${esc(top.label)}</b> leads at <b>${pct(top.v.val.recall)}</b>, ${num(gain * 100, 1)} points more than the best model without history.
             Paler bars have no history version. <span class="note">* trained on a small subset of the data.</span>`
          : `On the current readings alone the best is ${esc(top.label)} at ${pct(top.v.val.recall)}. Flexible models beat logistic regression
             because faults often show up in combinations of sensors that a straight boundary can't follow.`;
      };
      onSeg("t5-hist", draw);
      draw("hist");
    },
  },
  {
    kicker: "Diagnosis",
    title: "Which fault is it?",
    text: () => `<p>Once something is wrong, a second model names the fault. Most faults are recognized almost every time; a few
      barely change the readings and get confused with each other. <b>Click a fault</b> to see what it is mistaken for.</p>`,
    draw(el) {
      const g = D.diaFile.val, labels = g.labels, rec = g.recall;
      const order = labels.map((f, i) => i).sort((a, b) => rec[a] - rec[b]);
      el.innerHTML = `${tiles([[num(D.dia.val.macro_F1, 3), "macro-F1 (1 = always right)"], [pct(D.dia.val.accuracy), "of faulty readings named correctly"]])}
        <div class="plot"><div id="t6-chart"></div></div><p class="tour-reveal" id="t6-out"></p>`;
      const show = i => {
        const row = g.confusion_matrix[i], tot = row.reduce((a, b) => a + b, 0) || 1;
        const others = row.map((c, j) => [j, c / tot]).filter(([j]) => j !== i).sort((a, b) => b[1] - a[1]).slice(0, 2).filter(x => x[1] >= 0.01);
        $("t6-out").innerHTML = `<b>Fault ${labels[i]}</b> is named correctly in <b>${pct(rec[i])}</b> of its readings.${others.length
          ? ` It is most often mistaken for ${others.map(([j, s]) => `fault ${labels[j]} (${pct(s)})`).join(" and ")}.` : " It is almost never mistaken for another fault."}`;
      };
      barChart($("t6-chart"), {
        categories: order.map(i => `Fault ${labels[i]}`),
        series: [{ name: "recall", color: COLORS.data, values: order.map(i => rec[i]), opacity: order.map((i, k) => (k < 3 ? 1 : 0.55)) }],
        x: { domain: [0, 1], fmt: x => pct(x, 0), label: "Share of the fault's readings named correctly" },
        labelWidth: 90, valueFmt: x => pct(x, 0), rowHeight: 16,
        tip: k => `Fault ${labels[order[k]]}: ${pct(rec[order[k]])} correct<br><span class="note">click for details</span>`,
        onClick: k => show(order[k]),
      });
      show(order[0]);
    },
  },
  {
    kicker: "Speed",
    title: "How quickly is a fault caught?",
    text: () => `<p>To avoid jumping at single noisy readings, an alarm is only raised after several alarming readings <b>in a row</b>.
      Requiring more in a row makes alarms more trustworthy but later. Try it. The dashed line is when 5%-false-alarm noise alone
      would set off an alarm in a normal run: faults detected close to it are barely distinguishable from chance.</p>`,
    draw(el) {
      const byK = D.det.delays.by_k, ks = Object.keys(byK).map(Number).sort((a, b) => a - b);
      el.innerHTML = `<div class="tour-row"><span class="note">Alarms in a row:</span>${seg("t7-k", ks.map(k => [String(k), String(k)]), "3")}</div>
        <div class="plot"><div id="t7-chart"></div></div><p class="tour-reveal" id="t7-out"></p>`;
      const draw = k => {
        const d = byK[k], faults = Object.keys(d.by_fault).filter(f => d.by_fault[f][0] != null).sort((a, b) => d.by_fault[a][0] - d.by_fault[b][0]);
        const vals = faults.map(f => d.by_fault[f][0]), ref = d.reference?.[0];
        barChart($("t7-chart"), {
          categories: faults.map(f => `Fault ${f}`),
          series: [{ name: "delay", color: COLORS.data, values: vals, opacity: vals.map(v => (ref && v > ref * 0.6 ? 0.45 : 1)) }],
          x: { domain: [0, Math.max(...vals, ref || 0) * 1.08], fmt: v => fmtMinutes(v), label: "Typical time from fault start to alarm" },
          labelWidth: 90, valueFmt: v => fmtMinutes(v), rowHeight: 16,
          vlines: ref ? [{ x: ref, label: "chance", color: COLORS.muted, dash: "5 4" }] : [],
          tip: i => `Fault ${faults[i]}: typically ${fmtMinutes(vals[i])}`,
        });
        $("t7-out").innerHTML = `With <b>${k}</b> in a row, half of the faults are caught within <b>${fmtMinutes(median(vals))}</b>; the fastest within
          ${fmtMinutes(vals[0])}. ${ref ? `By chance alone, a normal run would set off an alarm after about ${fmtMinutes(ref)}.` : ""}`;
      };
      onSeg("t7-k", v => draw(v));
      draw("3");
    },
  },
  {
    kicker: "The final test",
    title: "Does it hold up on new data?",
    text: () => `<p>All choices were made on the validation runs. Only then were the chosen models run, once, on a separate test set:
      ${int(D.manifest.test.rows)} readings from longer runs they had never seen. Switch between the two.</p>`,
    draw(el) {
      const t = D.manifest.test, bd = D.manifest.detection.best, bg = D.manifest.diagnosis.best;
      const dt = t.detection[bd], gt = t.diagnosis[bg];
      const e = D.e2e, ev = e?.val?.single, et = e?.test?.single;
      const d = D.manifest.dataset, on = d.fault_onset?.train ?? 20, v = D.det.val;
      const valFirst = (d.val_runs * on) / (v.TN + v.FP), testFirst = (t.runs_per_fault * 21 * on) / (dt.TN + dt.FP);
      const rows = [
        ["Faulty readings caught", D.det.val.recall, dt.recall, pct],
        ["False alarms (same threshold)", 1 - D.det.val.normal_recall, 1 - dt.normal_recall, pct],
        ["Diagnosis macro-F1", D.dia.val.macro_F1, gt.macro_F1, v => num(v, 3)],
        ...(ev && et ? [["Whole system (21 answers) macro-F1", ev.macro_F1, et.macro_F1, v => num(v, 3)]] : []),
      ];
      el.innerHTML = `<div class="tour-row">${seg("t8-which", [["val", "Validation"], ["test", "Final test"]], "val")}</div><div id="t8-tiles"></div>
        <p class="tour-reveal" id="t8-out"></p>`;
      const draw = which => {
        $("t8-tiles").innerHTML = tiles(rows.map(([l, a, b, f]) => [f(which === "val" ? a : b), l, which === "test" && Math.abs(b - a) > 0.015 ? "hl" : ""]));
        $("t8-out").innerHTML = which === "val" ? "These are the numbers the models were chosen on."
          : `The threshold was <b>not</b> re-tuned on the test set: it is the one chosen on validation, kept exactly as it was. With it the
             model flags a little more of everything: more faulty readings, but also ${pct(1 - dt.normal_recall)} of normal readings instead of 5%.
             ${D.like ? `Compared fairly, at the same 5% false alarms, it catches <b>${pct(D.like.recall)}</b> on test: the same as on validation, not better.` : ""}
             <br><br><b>Why more false alarms?</b> In the training data every fault starts after exactly one hour, so the model has learned
             that the first hour of a run is always fault-free and almost never raises an alarm there. ${pct(valFirst, 0)} of the normal
             validation readings come from that easy first hour, but only ${pct(testFirst, 0)} of the normal test readings, where runs are
             longer and faults start after 8 hours. <a href="results.html#far-note">The full explanation</a>.`;
      };
      onSeg("t8-which", draw);
      draw("val");
    },
  },
  {
    kicker: "Summary",
    title: "What we learned",
    text: () => `<p>Five takeaways. Each links to the page with the full story.</p>`,
    draw(el) {
      const lr = D.families.find(f => f.key === "logreg"), top = D.bestHist, hard = D.hardFaults;
      const td = D.manifest.test.detection[D.manifest.detection.best];
      const cards = [
        ["History helps most", `Adding the last hour of each sensor lifts detection from ${pct(D.bestInst.val.recall)} to ${pct(top.val.recall)}.`, "detection.html#window"],
        ["Flexible models win", `${esc(top.label.split(",")[0])} beats logistic regression (${pct(lr?.hist?.val.recall ?? lr?.inst?.val.recall)}): faults hide in combinations of sensors.`, "compare.html"],
        ["Some faults are nearly invisible", `Faults ${hard.join(", ")} barely change the readings, so they are the hardest to name or catch.`, "diagnosis.html"],
        ["Test fairly", "Splitting by run, not by row, keeps the score honest; a random split overstates it.", "index.html#protocol"],
        ["It holds up, with one catch", `On the unseen test runs it catches ${pct(D.like?.recall ?? td.recall)} of faulty readings at the same 5% false alarms. With the validation threshold kept, false alarms rise to ${pct(1 - td.normal_recall)}: the model treats the first hour of a run as always fault-free.`, "results.html#far-note"],
      ];
      el.innerHTML = `<div class="tour-cards">${cards.map(([h, p, href], i) => `<a class="tour-card" href="${href}"><span class="tour-q-n">${i + 1}</span><b>${h}</b><span>${p}</span></a>`).join("")}</div>
        <div class="tour-row" style="margin-top:18px"><a class="btn primary" href="index.html">Go to the overview</a><a class="btn" href="detection.html">Explore the models</a>
        <button class="btn" type="button" id="t9-restart">Start the tour again</button></div>`;
      $("t9-restart").addEventListener("click", () => go(0));
    },
  },
];

// ---------- navigation ----------
function go(i, focus = true) {
  cur = Math.max(0, Math.min(STEPS.length - 1, i));
  const s = STEPS[cur];
  $("tour-count").textContent = `Step ${cur + 1} of ${STEPS.length}`;
  $("tour-bar").style.width = `${((cur + 1) / STEPS.length) * 100}%`;
  $("tour-kicker").textContent = s.kicker;
  $("tour-title").textContent = s.title;
  $("tour-text").innerHTML = s.text();
  const stage = $("tour-stage");
  stage.innerHTML = "";
  try { s.draw(stage); } catch (e) { showError(e); }
  $("tour-prev").disabled = cur === 0;
  $("tour-next").textContent = cur === STEPS.length - 1 ? "Finish" : "Next →";
  $("tour-dots").querySelectorAll("button").forEach((b, k) => { b.classList.toggle("on", k === cur); b.setAttribute("aria-selected", String(k === cur)); });
  history.replaceState(null, "", `#step-${cur + 1}`);
  if (focus) { $("tour-title").focus({ preventScroll: true }); window.scrollTo({ top: 0, behavior: "smooth" }); }
}

function setupNav() {
  $("tour-dots").innerHTML = STEPS.map((s, k) => `<button type="button" role="tab" aria-label="Step ${k + 1}: ${esc(s.title)}" title="${esc(s.title)}"></button>`).join("");
  $("tour-dots").addEventListener("click", e => { const b = e.target.closest("button"); if (b) go([...b.parentNode.children].indexOf(b)); });
  $("tour-prev").addEventListener("click", () => go(cur - 1));
  $("tour-next").addEventListener("click", () => (cur === STEPS.length - 1 ? (location.href = "index.html") : go(cur + 1)));
  document.addEventListener("keydown", e => {
    if (e.target.closest("input, select, textarea") || e.altKey || e.ctrlKey || e.metaKey) return;
    if (e.key === "ArrowRight") go(cur + 1);
    if (e.key === "ArrowLeft") go(cur - 1);
  });
}

// ---------- data ----------
async function load() {
  const m = await loadManifest();
  D.manifest = m;
  const det = m.detection.models.find(x => x.id === m.detection.best), dia = m.diagnosis.models.find(x => x.id === m.diagnosis.best);
  const [detFile, diaFile, traces, summary, split, e2e, testFile] = await Promise.all([
    loadModel("detection", det.id), loadModel("diagnosis", dia.id), getJSON("data/data_traces.json"), getJSON("data/data_summary.json"),
    getJSON("data/split_demo.json").catch(() => null), getJSON("data/end_to_end.json").catch(() => null),
    m.test ? loadTestModel("detection", det.id).catch(() => null) : null]);
  Object.assign(D, { detFile, diaFile, traces, split, e2e, quality: summary.quality });
  // Like for like: what the retrained model catches on test at the same 5% false alarms as on validation.
  if (testFile?.test) { const tv = detectionView(testFile.test); D.like = tv.at(tv.indexForFar(0.05)); }
  D.det = { ...det, delays: detFile.val.delays["0.05"] || Object.values(detFile.val.delays)[0] };
  D.dia = dia;
  D.detView = detectionView(detFile.val);

  // Best model of each family, with and without history (selectable models only).
  const ok = x => x.selectable !== false && familyOf(x) !== "baseline";
  const best = list => list.reduce((a, b) => (!a || b.val.recall > a.val.recall ? b : a), null);
  const fams = [...new Set(m.detection.models.filter(ok).map(familyOf))];
  const subset = m.settings?.subset_families || [];
  D.families = fams.map(f => ({
    key: f, label: familyLabel(m, f), subset: subset.includes(f),
    inst: best(m.detection.models.filter(x => ok(x) && familyOf(x) === f && !isHist(x.features))),
    hist: best(m.detection.models.filter(x => ok(x) && familyOf(x) === f && isHist(x.features))),
  }));
  const main = m.detection.models.filter(x => ok(x) && !subset.includes(familyOf(x)));
  D.bestInst = best(main.filter(x => !isHist(x.features)));
  D.bestHist = det;

  // Spot-the-fault: an easy, a middling and the hardest fault by detection recall; for each, the sensor it moves most.
  const perFault = D.detView.perFault(D.detView.indexForFar(0.05)), faults = D.detView.faults;
  const ranked = faults.map((f, i) => [f, perFault[i]]).sort((a, b) => b[1] - a[1]);
  const pick = [ranked[0], ranked[Math.floor(ranked.length / 2)], ranked[ranked.length - 1]];
  D.spotFaults = pick.map(([f], i) => ({ f, tag: ["Obvious", "Subtle", "Nearly invisible"][i] }));
  const base = traces.runs.find(r => r.fault === 0);
  D.spotSensor = {};
  for (const { f } of D.spotFaults) {
    const run = traces.runs.find(r => r.fault === f);
    let bestS = m.dataset.features[0], bestZ = -1;
    for (const s of m.dataset.features) {
      const b = base.values[s], v = run.values[s];
      const mu = b.reduce((a, c) => a + c, 0) / b.length, sd = Math.sqrt(b.reduce((a, c) => a + (c - mu) ** 2, 0) / b.length) || 1;
      const post = v.filter((_, i) => run.sample[i] > traces.onset && run.sample[i] <= 160);
      const z = Math.abs(post.reduce((a, c) => a + c, 0) / post.length - mu) / sd;
      if (z > bestZ) { bestZ = z; bestS = s; }
    }
    D.spotSensor[f] = bestS;
  }
  // Hardest faults to name (lowest diagnosis recall).
  D.hardFaults = diaFile.val.labels.map((f, i) => [f, diaFile.val.recall[i]]).sort((a, b) => a[1] - b[1]).slice(0, 3).map(x => x[0]).sort((a, b) => a - b);
}

load().then(() => {
  setupNav();
  const m = /step-(\d+)/.exec(location.hash);
  go(m ? +m[1] - 1 : 0, false);
  // A link or typed address with a different #step-N moves the tour there.
  window.addEventListener("hashchange", () => {
    const h = /step-(\d+)/.exec(location.hash);
    if (h && +h[1] - 1 !== cur) go(+h[1] - 1);
  });
}).catch(showError);
