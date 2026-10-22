// Tiny MathML builder for the "Learn" boxes. Browsers render MathML natively, so no maths library is needed.
//   block(row(i("p"), o("="), frac(n(1), row(n(1), o("+"), sup(i("e"), row(o("−"), i("z")))))))

const wrap = (tag, content) => `<${tag}>${content}</${tag}>`;
const join = parts => parts.flat().join("");

export const i = s => wrap("mi", s);                       // identifier (italic): x, w, z
export const n = s => wrap("mn", s);                       // number
export const o = s => wrap("mo", s);                       // operator: = + − ×
export const t = s => wrap("mtext", s);                    // upright text
export const row = (...p) => wrap("mrow", join(p));
export const frac = (a, b) => `<mfrac>${row(a)}${row(b)}</mfrac>`;
export const sub = (a, b) => `<msub>${row(a)}${row(b)}</msub>`;
export const sup = (a, b) => `<msup>${row(a)}${row(b)}</msup>`;
export const subsup = (a, b, c) => `<msubsup>${row(a)}${row(b)}${row(c)}</msubsup>`;
export const under = (a, b) => `<munder>${row(a)}${row(b)}</munder>`;
export const underover = (a, b, c) => `<munderover>${row(a)}${row(b)}${row(c)}</munderover>`;
export const paren = (...p) => row(o("("), ...p, o(")"));
export const abs = (...p) => row(o("|"), ...p, o("|"));
export const norm = (x, k) => sub(row(o("‖"), x, o("‖")), n(k));
export const sp = (em = 0.4) => `<mspace width="${em}em"></mspace>`;
export const sum = (lo, hi) => (hi ? underover(o("∑"), lo, hi) : under(o("∑"), lo));
export const block = (...p) => `<math display="block">${join(p)}</math>`;
export const inline = (...p) => `<math>${join(p)}</math>`;

// A number formatted for MathML (thousands separators, fixed decimals).
export const num = (x, d = 0) => n(Number(x).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }));
