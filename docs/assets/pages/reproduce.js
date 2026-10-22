import { renderChrome, loadManifest } from "../site.js";

renderChrome();
loadManifest().then(m => {
  const n = m.dataset.runs_per_fault;
  document.getElementById("current-data").textContent =
    `The results currently on this site come from a run with ${n < 500 ? `--runs-per-fault ${n}` : "all 500 runs per fault"}` +
    `${m.test ? ", including the final test" : "; the final test has not been run yet"}.`;
}).catch(() => {});
