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
  el.innerHTML = i ? `<span class="pill">${esc(i.model || "FERMI")} · ${esc(i.device)} · ${esc(i.checkpoint)}</span>`
                   : `<span class="pill err">server not running: python server.py</span>`;
}

// A "choice" question asked with its options in several orders and averaged: this cancels most of the position
// bias of a small classifier. Up to 4 options: every order; more options: as given and reversed.
// Returns {pr: {option: p}, best, ms}.
function orders(keys) {
  if (keys.length > 4) return [keys, [...keys].reverse()];
  const perms = a => a.length <= 1 ? [a] : a.flatMap((x, i) => perms([...a.slice(0, i), ...a.slice(i + 1)]).map(p => [x, ...p]));
  return perms(keys);
}
async function fermiChoice(state, q, average = true) {
  const keys = Object.keys(q.criteria);
  const os = average ? orders(keys) : [keys];
  const qs = {};
  os.forEach((o, i) => qs["o" + i] = {...q, criteria: Object.fromEntries(o.map(k => [k, q.criteria[k]]))});
  const r = await fermi(state, qs);
  const pr = Object.fromEntries(keys.map(k => [k, os.reduce((t, _, i) => t + r.answers["o" + i].probabilities[k], 0) / os.length]));
  return {pr, best: keys.reduce((a, b) => pr[b] > pr[a] ? b : a), ms: r.ms};
}
