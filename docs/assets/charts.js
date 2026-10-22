// Small SVG chart helpers: line charts, horizontal bar charts and heatmaps.
// Colours are CSS custom properties (e.g. "var(--c-fault)"), so charts follow the light/dark theme automatically.
// Each chart re-renders itself when its container is resized.

const NS = "http://www.w3.org/2000/svg";
const COLOR_PROPS = new Set(["fill", "stroke"]);

function svgEl(tag, attrs = {}, parent) {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null) continue;
    // CSS variables only work through the style property, not presentation attributes.
    if (COLOR_PROPS.has(k) && String(v).includes("var(")) node.style.setProperty(k, v);
    else node.setAttribute(k, v);
  }
  if (parent) parent.appendChild(node);
  return node;
}

// ---------- Tooltip ----------
let tipNode;
export function showTip(evt, html) {
  if (!tipNode) {
    tipNode = document.createElement("div");
    tipNode.className = "tooltip";
    tipNode.setAttribute("role", "status");
    document.body.appendChild(tipNode);
  }
  tipNode.innerHTML = html;
  tipNode.style.opacity = 1;
  const pad = 14, w = tipNode.offsetWidth, h = tipNode.offsetHeight;
  let x = evt.clientX + pad, y = evt.clientY + pad;
  if (x + w > window.innerWidth - 8) x = evt.clientX - w - pad;
  if (y + h > window.innerHeight - 8) y = evt.clientY - h - pad;
  tipNode.style.left = x + "px";
  tipNode.style.top = y + "px";
}
export function hideTip() { if (tipNode) tipNode.style.opacity = 0; }

// ---------- Scales ----------
function linearScale(domain, range) {
  const [d0, d1] = domain, [r0, r1] = range;
  const f = v => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);
  f.invert = p => d0 + ((p - r0) / (r1 - r0 || 1)) * (d1 - d0);
  return f;
}
function logScale(domain, range) {
  const lin = linearScale([Math.log10(domain[0]), Math.log10(domain[1])], range);
  const f = v => lin(Math.log10(v));
  f.invert = p => 10 ** lin.invert(p);
  return f;
}
export function niceTicks(lo, hi, count = 5) {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map(s => s * mag).find(s => s >= raw) || raw;
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) ticks.push(+v.toFixed(12));
  return ticks;
}
function niceDomain(lo, hi, padFrac = 0.05) {
  if (lo === hi) return [lo - 1, hi + 1];
  const pad = (hi - lo) * padFrac;
  return [lo - pad, hi + pad];
}
function makeScale(axis, range) {
  return axis.type === "log" ? logScale(axis.domain, range) : linearScale(axis.domain, range);
}
function axisTicks(axis) {
  if (axis.ticks) return axis.ticks;
  if (axis.type === "log") {
    const out = [];
    for (let e = Math.ceil(Math.log10(axis.domain[0])); e <= Math.floor(Math.log10(axis.domain[1])); e++) out.push(10 ** e);
    return out;
  }
  return niceTicks(axis.domain[0], axis.domain[1], axis.count || 5);
}

// ---------- Frame: svg, margins, axes, grid ----------
function frame(el, spec) {
  const W = Math.max(240, Math.floor(el.clientWidth || 600));
  const H = spec.height || 300;
  const m = Object.assign({ top: 12, right: 18, bottom: spec.x?.label ? 44 : 28, left: spec.y?.label ? 62 : 46 }, spec.margin);
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img", "aria-label": spec.ariaLabel || "chart" });
  el.replaceChildren(svg);
  const w = W - m.left - m.right, h = H - m.top - m.bottom;
  const g = svgEl("g", { transform: `translate(${m.left},${m.top})` }, svg);
  return { svg, g, w, h, m, W, H };
}

function drawAxes(ctx, spec, xs, ys) {
  const { g, w, h } = ctx;
  const grid = svgEl("g", { class: "grid" }, g);
  const xFmt = spec.x.fmt || (v => String(v));
  const yFmt = spec.y.fmt || (v => String(v));
  // Skip x tick labels that would overlap their neighbour.
  let lastRight = -Infinity;
  for (const t of axisTicks(spec.x)) {
    const x = xs(t);
    if (x < -0.5 || x > w + 0.5) continue;
    if (spec.x.grid !== false) svgEl("line", { x1: x, x2: x, y1: 0, y2: h }, grid);
    const text = xFmt(t), half = text.length * 3.3;
    if (x - half < lastRight + 4) continue;
    lastRight = x + half;
    const lbl = svgEl("text", { x, y: h + 16, "text-anchor": "middle" }, g);
    lbl.textContent = text;
  }
  for (const t of axisTicks(spec.y)) {
    const y = ys(t);
    if (y < -0.5 || y > h + 0.5) continue;
    if (spec.y.grid !== false) svgEl("line", { x1: 0, x2: w, y1: y, y2: y }, grid);
    const lbl = svgEl("text", { x: -8, y: y + 4, "text-anchor": "end" }, g);
    lbl.textContent = yFmt(t);
  }
  svgEl("line", { class: "domain", x1: 0, x2: w, y1: h, y2: h }, g);
  svgEl("line", { class: "domain", x1: 0, x2: 0, y1: 0, y2: h }, g);
  if (spec.x.label) {
    const t = svgEl("text", { class: "axis-label", x: w / 2, y: h + 36, "text-anchor": "middle" }, g);
    t.textContent = spec.x.label;
  }
  if (spec.y.label) {
    const t = svgEl("text", { class: "axis-label", transform: `translate(${-ctx.m.left + 14},${h / 2}) rotate(-90)`, "text-anchor": "middle" }, g);
    t.textContent = spec.y.label;
  }
}

function keepResponsive(el, render) {
  el._render = render;
  if (!el._observer) {
    let lastW = el.clientWidth;
    el._observer = new ResizeObserver(() => {
      if (Math.abs(el.clientWidth - lastW) > 4) { lastW = el.clientWidth; el._render(); }
    });
    el._observer.observe(el);
  }
  render();
}

// ---------- Line chart ----------
// spec: { height, x:{type,domain,label,fmt,ticks}, y:{...}, series:[{name,color,points:[[x,y,...]],width,dash,area,opacity,marker,hover}],
//         vlines:[{x,color,label,dash}], hlines:[{y,color,label,dash}], markers:[{x,y,color,r,label}],
//         tip:(series,point)=>html, onClick:(series,point)=>void, hoverRadius }
export function lineChart(el, spec) {
  keepResponsive(el, () => {
    const ctx = frame(el, spec);
    const { g, w, h } = ctx;
    const allPts = spec.series.flatMap(s => s.points);
    if (!spec.x.domain) spec.x.domain = niceDomain(Math.min(...allPts.map(p => p[0])), Math.max(...allPts.map(p => p[0])), 0);
    if (!spec.y.domain) spec.y.domain = niceDomain(Math.min(0, ...allPts.map(p => p[1])), Math.max(...allPts.map(p => p[1])));
    const xs = makeScale(spec.x, [0, w]), ys = makeScale(spec.y, [h, 0]);
    drawAxes(ctx, spec, xs, ys);

    const clip = "clip" + Math.random().toString(36).slice(2);
    svgEl("rect", { x: 0, y: -2, width: w, height: h + 4 }, svgEl("clipPath", { id: clip }, g));
    const plotG = svgEl("g", { "clip-path": `url(#${clip})` }, g);

    for (const s of spec.series) {
      if (!s.points.length) continue;
      const d = s.points.map((p, i) => `${i ? "L" : "M"}${xs(p[0]).toFixed(2)},${ys(p[1]).toFixed(2)}`).join("");
      if (s.area) {
        const base = ys(Math.max(spec.y.domain[0], 0));
        const first = s.points[0], last = s.points[s.points.length - 1];
        svgEl("path", { d: `${d}L${xs(last[0])},${base}L${xs(first[0])},${base}Z`, fill: s.color, "fill-opacity": s.areaOpacity ?? 0.16, stroke: "none" }, plotG);
      }
      svgEl("path", { d, fill: "none", stroke: s.color, "stroke-width": s.width || 2, "stroke-dasharray": s.dash || null,
                      "stroke-opacity": s.opacity ?? 1, "stroke-linejoin": "round" }, plotG);
      if (s.marker) for (const p of s.points) svgEl("circle", { cx: xs(p[0]), cy: ys(p[1]), r: s.marker, fill: s.color, "fill-opacity": s.opacity ?? 1 }, plotG);
    }
    for (const v of spec.vlines || []) {
      const x = xs(v.x);
      if (x < -1 || x > w + 1) continue;
      svgEl("line", { x1: x, x2: x, y1: 0, y2: h, stroke: v.color || "var(--text)", "stroke-width": v.width || 1.5, "stroke-dasharray": v.dash || null }, g);
      if (v.label) {
        const right = x > w * 0.7;
        const t = svgEl("text", { x: right ? x - 5 : x + 5, y: 12, "text-anchor": right ? "end" : "start", style: `fill:${v.color || "var(--text)"};font-weight:600` }, g);
        t.textContent = v.label;
      }
    }
    const labelYs = [];
    for (const hl of spec.hlines || []) {
      const y = ys(hl.y);
      svgEl("line", { x1: 0, x2: w, y1: y, y2: y, stroke: hl.color || "var(--text)", "stroke-width": hl.width || 1.5, "stroke-dasharray": hl.dash || "5 4" }, g);
      if (hl.label) {
        let ly = y - 5;
        if (labelYs.some(o => Math.abs(o - ly) < 12)) ly = y + 13;   // avoid overlapping an earlier label
        labelYs.push(ly);
        const t = svgEl("text", { x: hl.labelX === "left" ? 6 : w - 4, y: ly, "text-anchor": hl.labelX === "left" ? "start" : "end", style: `fill:${hl.color || "var(--text)"};font-weight:600` }, g);
        t.textContent = hl.label;
      }
    }
    for (const mk of spec.markers || []) {
      svgEl("circle", { cx: xs(mk.x), cy: ys(mk.y), r: mk.r || 6, fill: mk.color || "var(--text)", stroke: "var(--surface)", "stroke-width": 2 }, g);
      if (mk.label) {
        const t = svgEl("text", { x: xs(mk.x) + 9, y: ys(mk.y) - 8, style: `fill:${mk.color || "var(--text)"};font-weight:700` }, g);
        t.textContent = mk.label;
      }
    }

    if (spec.tip || spec.onClick) {
      const hoverDot = svgEl("circle", { r: 5, fill: "none", stroke: "var(--text)", "stroke-width": 2, visibility: "hidden" }, g);
      const overlay = svgEl("rect", { x: 0, y: 0, width: w, height: h, fill: "transparent", style: spec.onClick ? "cursor:pointer" : null }, g);
      const nearest = evt => {
        const r = overlay.getBoundingClientRect();
        const px = (evt.clientX - r.left) * (w / r.width), py = (evt.clientY - r.top) * (h / r.height);
        let best = null, bd = Infinity;
        for (const s of spec.series) {
          if (s.hover === false) continue;
          for (const p of s.points) {
            const dx = xs(p[0]) - px, dy = ys(p[1]) - py, dd = dx * dx + dy * dy;
            if (dd < bd) { bd = dd; best = { s, p }; }
          }
        }
        return bd < (spec.hoverRadius || 40) ** 2 ? best : null;
      };
      overlay.addEventListener("mousemove", evt => {
        const hit = nearest(evt);
        if (!hit) { hoverDot.setAttribute("visibility", "hidden"); hideTip(); return; }
        hoverDot.setAttribute("cx", xs(hit.p[0])); hoverDot.setAttribute("cy", ys(hit.p[1]));
        hoverDot.setAttribute("visibility", "visible");
        if (spec.tip) showTip(evt, spec.tip(hit.s, hit.p));
      });
      overlay.addEventListener("mouseleave", () => { hoverDot.setAttribute("visibility", "hidden"); hideTip(); });
      if (spec.onClick) overlay.addEventListener("click", evt => { const hit = nearest(evt); if (hit) spec.onClick(hit.s, hit.p); });
    }
  });
}

// ---------- Horizontal bar chart ----------
// spec: { categories:[label], series:[{name,color,values:[v],opacity:[0..1],missingLabel}], x:{type,domain,label,fmt,ticks},
//         labelWidth, rowHeight, highlight: index, valueFmt, tip:(i)=>html, onClick:(i)=>void, vlines }
export function barChart(el, spec) {
  keepResponsive(el, () => {
    const n = spec.categories.length, k = spec.series.length;
    const rowH = spec.rowHeight || (k > 1 ? 26 : 20);
    const avail = el.clientWidth || 600;
    const left = Math.round(Math.min(spec.labelWidth || 130, Math.max(64, avail * 0.42)));
    const maxChars = Math.floor(left / 6.3);
    const fit = s => (s.length > maxChars ? s.slice(0, maxChars - 1) + "…" : s);
    const fullSpec = { ...spec, height: 10 + n * rowH + (spec.x.label ? 46 : 30),
                       margin: { left, top: 10, right: spec.valueFmt ? 74 : 18, ...spec.margin },
                       y: { domain: [0, 1], ticks: [], grid: false } };
    const ctx = frame(el, fullSpec);
    const { g, w, h } = ctx;
    const vals = spec.series.flatMap(s => s.values.filter(v => v != null && isFinite(v)));
    const domain = spec.x.domain || [spec.x.type === "log" ? Math.min(...vals) * 0.8 : 0, Math.max(...vals, 1e-9) * 1.02];
    const xAxis = { ...spec.x, domain };
    const xs = makeScale(xAxis, [0, w]);
    drawAxes(ctx, { ...fullSpec, x: xAxis }, xs, () => 0);
    const band = h / n, inner = band * 0.78, barH = inner / k;
    const x0 = spec.x.type === "log" ? 0 : xs(Math.max(0, domain[0]));

    spec.categories.forEach((cat, i) => {
      const y0 = i * band;
      if (i === spec.highlight) svgEl("rect", { x: -ctx.m.left + 4, y: y0, width: w + ctx.m.left + ctx.m.right - 8, height: band, fill: "var(--hl-row)", rx: 3 }, g);
      const lbl = svgEl("text", { x: -8, y: y0 + band / 2 + 4, "text-anchor": "end", style: i === spec.highlight ? "fill:var(--text);font-weight:700" : null }, g);
      lbl.textContent = fit(cat);
      spec.series.forEach((s, j) => {
        const v = s.values[i];
        const y = y0 + (band - inner) / 2 + j * barH;
        if (v == null || !isFinite(v)) {
          const t = svgEl("text", { x: 4, y: y + barH / 2 + 4, style: "font-style:italic" }, g);
          t.textContent = s.missingLabel || "n/a";
          return;
        }
        const xv = xs(v);
        svgEl("rect", { x: Math.min(x0, xv), y, width: Math.max(1, Math.abs(xv - x0)), height: Math.max(2, barH - 2), rx: 1.5,
                        fill: s.color, "fill-opacity": s.opacity ? s.opacity[i] : 0.9 }, g);
        if (spec.valueFmt) {
          const t = svgEl("text", { x: Math.max(xv, x0) + 5, y: y + barH / 2 + 4, style: "font-size:11px" }, g);
          t.textContent = spec.valueFmt(v, i, s);
        }
      });
      const hit = svgEl("rect", { x: -ctx.m.left, y: y0, width: w + ctx.m.left + ctx.m.right, height: band, fill: "transparent",
                                  style: spec.onClick ? "cursor:pointer" : null }, g);
      if (spec.tip) {
        hit.addEventListener("mousemove", evt => showTip(evt, spec.tip(i)));
        hit.addEventListener("mouseleave", hideTip);
      }
      if (spec.onClick) hit.addEventListener("click", () => spec.onClick(i));
    });
    for (const v of spec.vlines || []) {
      const x = xs(v.x);
      svgEl("line", { x1: x, x2: x, y1: 0, y2: h, stroke: v.color || "var(--text)", "stroke-width": 1.5, "stroke-dasharray": v.dash || "4 3" }, g);
      if (v.label) { const t = svgEl("text", { x: x + 4, y: -2, style: `fill:${v.color || "var(--text)"};font-weight:600` }, g); t.textContent = v.label; }
    }
  });
}

// ---------- Heatmap ----------
// spec: { matrix:[[v]] (0..1), rowLabels, colLabels, rowTitle, colTitle, highlightRow, tip:(r,c)=>html, onClick:(r,c)=>void }
export function heatmap(el, spec) {
  keepResponsive(el, () => {
    const n = spec.matrix.length, mcols = spec.matrix[0].length;
    const W = Math.max(260, Math.floor(el.clientWidth || 600));
    const m = { left: 56, top: 14, right: 12, bottom: 50 };
    const cell = Math.floor(Math.min((W - m.left - m.right) / mcols, 30));
    const w = cell * mcols, h = cell * n;
    const svg = svgEl("svg", { viewBox: `0 0 ${w + m.left + m.right} ${h + m.top + m.bottom}`, class: "chart",
                               style: `max-width:${w + m.left + m.right}px`, role: "img", "aria-label": "confusion matrix" });
    el.replaceChildren(svg);
    const g = svgEl("g", { transform: `translate(${m.left},${m.top})` }, svg);
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < mcols; c++) {
        const v = spec.matrix[r][c];
        const a = Math.pow(Math.min(1, Math.max(0, v)), 0.6);
        const rect = svgEl("rect", { x: c * cell, y: r * cell, width: cell - 1, height: cell - 1,
                                     style: `fill:color-mix(in srgb, var(--c-heat) ${(a * 100).toFixed(1)}%, var(--surface));${spec.onClick ? "cursor:pointer" : ""}` }, g);
        if (r === c) svgEl("rect", { x: c * cell + 0.5, y: r * cell + 0.5, width: cell - 2, height: cell - 2, fill: "none", stroke: "var(--muted)", "stroke-width": 1, style: "pointer-events:none" }, g);
        if (spec.tip) {
          rect.addEventListener("mousemove", evt => showTip(evt, spec.tip(r, c)));
          rect.addEventListener("mouseleave", hideTip);
        }
        if (spec.onClick) rect.addEventListener("click", () => spec.onClick(r, c));
        if (cell >= 24 && v >= 0.1) {
          const t = svgEl("text", { x: c * cell + cell / 2 - 0.5, y: r * cell + cell / 2 + 3.5, "text-anchor": "middle",
                                    style: `font-size:9.5px;fill:${a > 0.55 ? "var(--on-heat)" : "var(--text)"};pointer-events:none` }, g);
          t.textContent = Math.round(v * 100);
        }
      }
    }
    if (spec.highlightRow != null) {
      svgEl("rect", { x: -1, y: spec.highlightRow * cell - 1, width: w + 1, height: cell + 1, fill: "none", stroke: "var(--c-amber)", "stroke-width": 2.5, rx: 2 }, g);
    }
    spec.rowLabels.forEach((lbl, r) => {
      const t = svgEl("text", { x: -6, y: r * cell + cell / 2 + 4, "text-anchor": "end",
                                style: r === spec.highlightRow ? "fill:var(--text);font-weight:700" : null }, g);
      t.textContent = lbl;
    });
    spec.colLabels.forEach((lbl, c) => {
      const t = svgEl("text", { x: c * cell + cell / 2, y: h + 14, "text-anchor": "middle" }, g);
      t.textContent = lbl;
    });
    const xt = svgEl("text", { class: "axis-label", x: w / 2, y: h + 38, "text-anchor": "middle" }, g);
    xt.textContent = spec.colTitle || "";
    const yt = svgEl("text", { class: "axis-label", transform: `translate(${-42},${h / 2}) rotate(-90)`, "text-anchor": "middle" }, g);
    yt.textContent = spec.rowTitle || "";
  });
}

// Legend helper: items [{label,color,dash}]
export function legend(el, items) {
  el.className = "legend";
  el.innerHTML = items.map(i => `<span class="${i.dash ? "dash" : ""}" style="--c:${i.color}">${i.label}</span>`).join("");
}
