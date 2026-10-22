// Explanations that depend on the selected model family: how the model turns readings into a score (Learn box)
// and which sensors it relies on (bar chart). Shared by the Detection and Diagnosis pages.
import { COLORS, VARIABLES, familyOf, num, pct, esc, sensorWeights } from "./site.js";
import { barChart } from "./charts.js";
import * as M from "./math.js";

const sub = (a, b) => M.sub(M.i(a), typeof b === "string" ? M.i(b) : b);
const sumTo = (i, n) => M.sum(M.row(M.i(i), M.o("="), M.n(1)), M.i(n));

// ---------- How the model makes its decision ----------
// Returns { title, html, sigmoid } where sigmoid says whether to draw the logistic curve below the text.
export function explainer(task, model, manifest) {
  const fam = familyOf(model), p = model.params || {};
  const det = task === "detection";
  const lr = manifest.settings.learning_rate ?? 0.1;
  const formula = (...blocks) => `<div class="formula">${blocks.join("")}</div>`;

  if (fam === "baseline") {
    return { title: "How the majority baseline decides", html: det
      ? `<p>It doesn't look at the readings at all. It gives every row the same probability, the share of faulty rows in training,
         so a threshold can only switch alarms on for every row or for none.</p>`
      : `<p>It doesn't look at the readings at all. Every fault has the same number of training rows, so it simply names the first one,
         fault 1, for every row.</p>` };
  }

  if (fam === "logreg") {
    if (det) {
      const wx = k => [sub("w", M.n(k)), sub("x", M.n(k))];
      return { title: "How 52 readings become one alarm", sigmoid: true, html: `
        <p>Each reading is first standardized: shifted and scaled so that, on the training runs, every variable has mean 0 and
          standard deviation 1. Logistic regression then adds the readings up with one weight each, giving a single number <i>z</i>
          (the log-odds of a fault), and squashes it into a probability between 0 and 1:</p>
        ${formula(
          M.block(M.i("z"), M.o("="), ...wx(1), M.o("+"), ...wx(2), M.o("+"), M.o("⋯"), M.o("+"), ...wx(52), M.o("+"), M.i("b")),
          M.block(M.i("p"), M.o("="), M.frac(M.n(1), M.row(M.n(1), M.o("+"), M.sup(M.i("e"), M.row(M.o("−"), M.i("z")))))),
          M.block(M.t("alarm if"), M.sp(0.4), M.i("p"), M.o("≥"), M.i("t"), M.sp(1), M.o("⟺"), M.sp(1), M.i("z"), M.o("≥"),
                  M.t("ln"), M.sp(0.15), M.frac(M.i("t"), M.row(M.n(1), M.o("−"), M.i("t")))))}
        <p data-live="threshold"></p>` };
    }
    const k = M.i("k");
    return { title: "One model, twenty scores", html: `
      <p>For diagnosis, logistic regression keeps a separate set of weights for every fault <i>k</i>. Each set turns the 52 readings into
        a score <i>z<sub>k</sub></i>, and the softmax function turns the 20 scores into probabilities that add up to 1. The model names the
        fault with the highest probability:</p>
      ${formula(
        M.block(sub("z", k), M.o("="), M.sub(M.i("w"), M.row(k, M.o(","), M.n(1))), sub("x", M.n(1)), M.o("+"), M.o("⋯"), M.o("+"),
                M.sub(M.i("w"), M.row(k, M.o(","), M.n(52))), sub("x", M.n(52)), M.o("+"), sub("b", k)),
        M.block(sub("p", k), M.o("="), M.frac(M.sup(M.i("e"), sub("z", k)), M.row(M.sum(M.row(M.i("j"), M.o("="), M.n(1)), M.n(20)), M.sup(M.i("e"), sub("z", "j")))),
                M.sp(1.5), M.t("predicted fault"), M.o("="), M.under(M.t("arg max"), k), M.sp(0.2), sub("p", k)))}
      <p>That is 20 × 52 = 1,040 weights plus 20 intercepts, 1,060 numbers in all, compared with 53 for detection.
        ${model.coef_nonzero < 52 ? `This lasso model uses ${model.coef_nonzero} of the 52 sensors for at least one fault.` : "This model uses all 52 sensors."}</p>` };
  }

  if (fam === "tree") {
    const leaves = 2 ** p.depth;
    return { title: "How a decision tree decides", html: `
      <p>A decision tree asks a chain of yes-or-no questions about the readings, such as "is <code>xmeas_9</code> above 0.8?"
        (on standardized values). Each answer leads to the next question until it reaches a leaf. Training picks each question so the
        rows on either side are as pure as possible.</p>
      ${formula(det
        ? M.block(M.t("score"), M.paren(M.i("x")), M.o("="), M.frac(M.t("faulty training rows in the leaf"), M.t("all training rows in the leaf")))
        : M.block(M.t("predicted fault"), M.o("="), M.t("the most common fault among the training rows in the leaf")))}
      <p>With depth ${p.depth} the tree asks at most ${p.depth} questions in a row, so it has at most ${leaves.toLocaleString("en-US")}
        leaves${det ? ` and so at most ${leaves.toLocaleString("en-US")} different scores. That is why shallow trees give only a few spikes in the
        score chart and the threshold slider jumps in big steps` : ""}. Deeper trees fit the training runs more closely, but past some depth
        they start memorizing those runs instead of learning what a fault looks like.</p>` };
  }

  if (fam === "rf") {
    return { title: "How a random forest decides", html: `
      <p>A random forest grows many decision trees, each on a random ${pct(manifest.settings.rf_sample ?? 0.25, 0)} of the training rows,
        and each choosing from a random subset of the sensors at every split. That makes the trees different from each other.
        ${det ? "The forest's score is the average of the trees' scores:" : "Each tree gives a share for every fault; the forest averages them and names the highest:"}</p>
      ${formula(det
        ? M.block(M.i("p"), M.paren(M.i("x")), M.o("="), M.frac(M.n(1), M.i("T")), sumTo("t", "T"), sub("p", "t"), M.paren(M.i("x")))
        : M.block(M.t("predicted fault"), M.o("="), M.under(M.t("arg max"), M.i("k")), M.sp(0.2), M.frac(M.n(1), M.i("T")), sumTo("t", "T"),
                  M.sub(M.i("p"), M.row(M.i("t"), M.o(","), M.i("k"))), M.paren(M.i("x"))))}
      <p>This forest averages <i>T</i> = ${p.trees} tree${p.trees === 1 ? "" : "s"} of depth ${p.depth}. Each tree overfits in its own way;
        averaging cancels much of that out. This is bagging: <b>b</b>ootstrap <b>agg</b>regat<b>ing</b>.
        Adding trees never makes a forest overfit more; it just stops helping after a while.</p>` };
  }

  if (fam === "gb" || fam === "xgb") {
    const xgb = fam === "xgb";
    const F = M.sub(M.i("F"), M.i("M"));
    return { title: xgb ? "How XGBoost decides" : "How gradient boosting decides", sigmoid: det, html: `
      <p>Boosting builds trees one after another. Each new tree is fitted to the errors the trees so far still make (more precisely, to the
        gradient of the log-loss), and its output is added to the total, scaled down by the learning rate <i>η</i> = ${lr}:</p>
      ${formula(
        M.block(F, M.paren(M.i("x")), M.o("="), sumTo("m", "M"), M.i("η"), M.sp(0.1), sub("f", "m"), M.paren(M.i("x"))),
        det ? M.block(M.i("p"), M.o("="), M.frac(M.n(1), M.row(M.n(1), M.o("+"), M.sup(M.i("e"), M.row(M.o("−"), F))))) :
              M.block(M.t("one total"), M.sp(0.3), M.sub(M.i("F"), M.row(M.i("M"), M.o(","), M.i("k"))), M.sp(0.3), M.t("per fault, then softmax as in logistic regression")),
        xgb ? M.block(M.t("XGBoost minimizes"), M.sp(0.5), sumTo("i", "N"), sub("ℓ", "i"), M.o("+"), sumTo("m", "M"),
                      M.paren(M.i("γ"), sub("T", "m"), M.o("+"), M.frac(M.n(1), M.n(2)), M.i("λ"), M.sup(M.norm(sub("w", "m"), 2), M.n(2)))) : "")}
      <p>Here <i>M</i> = ${p.rounds} rounds of trees with depth ${p.depth}${det ? "" : `, and for diagnosis every round builds one tree per fault:
        ${(20 * p.rounds).toLocaleString("en-US")} trees in all`}. More rounds fit the training runs more closely; after some point extra rounds
        stop helping on new runs.
        ${xgb ? `The last formula is XGBoost's training objective: the log-loss plus a penalty on each tree's number of leaves <i>T<sub>m</sub></i>
        and the size of its leaf values <i>w<sub>m</sub></i>, a built-in ridge-style regularization (defaults λ = 1, γ = 0 are used).` :
        "This is scikit-learn's histogram version: it bins each sensor's values first, which makes finding splits much faster on large data."}</p>
      ${det ? `<p class="note">The score on the website is <i>p</i>, the boosted total pushed through the same logistic curve as logistic regression.</p>` : ""}` };
  }

  if (fam === "ada") {
    return { title: "How AdaBoost decides", html: `
      <p>AdaBoost also builds small trees one after another, but it works by re-weighting the training rows: rows the last tree got
        wrong count more for the next tree. Each tree <i>h<sub>m</sub></i> gets a vote weight <i>α<sub>m</sub></i> based on its weighted
        error rate, and the final answer combines the weighted votes:</p>
      ${formula(
        M.block(M.i("F"), M.paren(M.i("x")), M.o("="), sumTo("m", "M"), sub("α", "m"), M.sp(0.1), sub("h", "m"), M.paren(M.i("x")),
                M.sp(2), sub("α", "m"), M.o("="), M.t("ln"), M.sp(0.15), M.frac(M.row(M.n(1), M.o("−"), sub("err", "m")), sub("err", "m")),
                det ? "" : M.row(M.o("+"), M.t("ln"), M.paren(M.n(20), M.o("−"), M.n(1)))))}
      <p>Here <i>M</i> = ${p.rounds} rounds of depth-${p.depth} trees${p.depth === 1 ? ' (one question each, called "stumps")' : ""}.
        A tree that is barely better than chance gets a vote near zero. Because the rounds can't run in parallel, this model was trained
        on a random ${pct(manifest.settings.ada_sample ?? 0.25, 0)} of the training rows to keep training time manageable.</p>` };
  }
  return { title: "How this model decides", html: "" };
}

// ---------- Which sensors the model relies on ----------
const KIND_TEXT = {
  coef: "Coefficients on standardized readings. Positive values push toward \"fault\", negative toward \"normal\"; the bigger the bar, the more the sensor matters.",
  coefsize: "For diagnosis, logistic regression has one coefficient per sensor for each of the 20 faults. Bars show the overall size of each sensor's 20 coefficients; pick a fault in the confusion matrix to see its own coefficients.",
  impurity: "Feature importance: the share of the total improvement in purity that comes from splits on each sensor. The bars add up to 100%. They say how much a sensor is used, not in which direction it pushes.",
  "weighted impurity": "Feature importance averaged over the boosting rounds, weighting each small tree by its vote. The bars add up to 100%.",
  gain: "XGBoost's gain importance: the average improvement in the loss each time a sensor is used for a split, scaled to add up to 100%.",
};

export function renderSensors({ chartEl, titleEl, introEl, file, model, faultIndex = null }) {
  const w = sensorWeights(file, faultIndex);
  const fam = familyOf(model);
  if (fam === "baseline") {
    introEl.textContent = "The majority baseline ignores every sensor.";
    titleEl.textContent = "";
    chartEl.innerHTML = `<div class="empty">Nothing to show: the baseline uses none of the 52 sensors.</div>`;
    return;
  }
  if (!w) {
    introEl.textContent = "scikit-learn's histogram gradient boosting doesn't report feature importances. Measuring them would need permutation importance (shuffling one sensor at a time and re-scoring every row), which isn't computed for this site.";
    titleEl.textContent = "";
    chartEl.innerHTML = `<div class="empty">Not available for gradient boosting. XGBoost, a closely related model, does report them.</div>`;
    return;
  }
  const kind = w.kind === "importance" ? (w.method || "impurity") : w.kind;
  introEl.textContent = KIND_TEXT[kind] || KIND_TEXT.impurity;
  const ranked = w.values.map((v, j) => [j, v]).filter(x => x[1] !== 0).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const top = ranked.slice(0, 15);
  titleEl.textContent = `Top ${top.length} of the ${ranked.length} sensors this model uses${faultIndex != null && w.kind === "coef" && Array.isArray(file.coefficients?.values?.[0]) ? ` (fault ${faultIndex + 1})` : ""}`;
  if (!top.length) { chartEl.innerHTML = `<div class="empty">This model gives every sensor a weight of zero.</div>`; return; }
  const features = (file.coefficients || file.importances).features;
  const values = top.map(([j]) => w.values[j]);
  const isImp = w.kind === "importance";
  const mx = Math.max(...values.map(Math.abs));
  barChart(chartEl, {
    categories: top.map(([j]) => `${features[j]} · ${VARIABLES[features[j]]}`),
    series: [{ name: "weight", color: isImp ? COLORS.data : COLORS.fault, values }],
    x: { domain: values.some(v => v < 0) ? [-mx * 1.1, mx * 1.1] : [0, mx * 1.1],
         fmt: x => (isImp ? pct(x, 0) : num(x, mx < 1 ? 2 : 1)),
         label: isImp ? "Share of importance" : w.kind === "coefsize" ? "Size of the sensor's coefficients" : "Coefficient (positive pushes toward fault)" },
    labelWidth: 330, rowHeight: 22, valueFmt: x => (isImp ? pct(x, 1) : num(x, 3)),
    tip: k => `<b>${features[top[k][0]]}</b>: ${esc(VARIABLES[features[top[k][0]]])}<br>${isImp ? `${pct(values[k], 1)} of importance` : `coefficient ${num(values[k], 4)}`}`,
  });
}
