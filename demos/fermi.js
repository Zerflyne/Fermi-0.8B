// Tiny client for the local FERMI server (server.py). Works from the server itself or from file:// (then it
// talks to http://127.0.0.1:8765).
const FERMI_URL = location.protocol === "file:" ? "http://127.0.0.1:8765" : "";

async function fermi(state, questions) {
  const r = await fetch(FERMI_URL + "/api/classify", {method: "POST", headers: {"Content-Type": "application/json"},
                                                        body: JSON.stringify({state, questions})});
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}

async function fermiInfo() {
  try { return await (await fetch(FERMI_URL + "/api/info")).json(); } catch (e) { return null; }
}

const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const pct = p => (100 * p).toFixed(p >= 0.995 || p < 0.005 ? 0 : 1) + "%";

// probability bars: items = [[label, p], ...]
function bars(items) {
  const best = Math.max(...items.map(x => x[1]));
  return `<div class="bars">` + items.map(([l, p]) =>
    `<div class="lab" title="${esc(l)}">${esc(l)}</div><div class="b ${p === best ? "win" : ""}"><div style="width:${(100 * p).toFixed(1)}%"></div></div><div class="v">${pct(p)}</div>`
  ).join("") + `</div>`;
}

// any answer -> [[label, p]]
function answerItems(q, a) {
  if (a.type === "noul") return [["yes", a.p_yes], ["no", 1 - a.p_yes]];
  if (a.type === "choice") return Object.entries(a.probabilities);
  return a.probabilities.map((p, i) => [q.criteria[i], p]);
}

async function deviceBadge(el) {
  const i = await fermiInfo();
  el.innerHTML = i ? `<span class="pill">FERMI-0.8B · ${esc(i.device)} · ${esc(i.checkpoint)}</span>`
                   : `<span class="pill err">server not running: python server.py</span>`;
}
