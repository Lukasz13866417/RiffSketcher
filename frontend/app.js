/* Riff Sketcher frontend: tap recorder, pitch graph, gesture tagging, candidates, preview.
 * Plain JS, no build step. All music logic lives on the Python side; this file only
 * maintains the Sketch object (same JSON shape as core/representation.py) and draws it.
 */
'use strict';

// ----------------------------------------------------------------------------- state
const state = {
  sketch: { observations: [], gestures: [], constraints: [], tempo_hint: null, meta: {} },
  passes: [],            // array of arrays of onset seconds (raw, per recording pass)
  holds: [],             // per pass: how long each tap was held (seconds, null when not released cleanly)
  recording: null,       // { start: number|null, taps: [] } while recording
  selection: new Set(),  // indices into sketch.observations
  candidates: [],
  selectedCandidate: -1,
  audio: null,
  exposure: { count: 0, seconds: 0 },
  feedbackNote: -1,
  library: null,
  paint: false,
  step: 0,
  engine: 'auto',
};

const $ = (id) => document.getElementById(id);
const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const midiName = (p) => `${NOTE_NAMES[p % 12]}${Math.floor(p / 12) - 1}`;
const setStatus = (msg) => { $('status').textContent = msg; };

async function api(path, body, raw = false) {
  const res = await fetch(path, body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
  if (!res.ok) { const t = await res.text(); throw new Error(`${path}: ${res.status} ${t}`); }
  return raw ? res : res.json();
}

// ----------------------------------------------------------------------------- step 1: tapping
function startRecording() {
  state.recording = { start: null, taps: [], holds: [] };
  $('btn-record').disabled = true; $('btn-stop').disabled = false;
  $('tap-pad').classList.add('recording'); $('tap-pad').textContent = 'recording… tap now';
  $('tap-pad').focus();
  setStatus('recording: first tap starts the clock');
}

function tap() {
  if (!state.recording) return;
  const now = performance.now() / 1000;
  if (state.recording.start === null) state.recording.start = now;
  state.recording.taps.push(now - state.recording.start);
  state.recording.holds.push(null);
  const pad = $('tap-pad');
  pad.classList.add('flash'); setTimeout(() => pad.classList.remove('flash'), 60);
  pad.textContent = `${state.recording.taps.length} taps`;
}

// a key or pointer release ends the hold of the latest tap: the note's length, roughly
function releaseTap() {
  const r = state.recording;
  if (!r || r.start === null) return;
  const i = r.taps.length - 1;
  if (i >= 0 && r.holds[i] === null) r.holds[i] = Math.max(0.02, performance.now() / 1000 - r.start - r.taps[i]);
}

function stopRecording() {
  if (!state.recording) return;
  releaseTap();
  const taps = state.recording.taps;
  const holds = state.recording.holds;
  state.recording = null;
  $('btn-record').disabled = false; $('btn-stop').disabled = true;
  $('tap-pad').classList.remove('recording'); $('tap-pad').textContent = 'tap here';
  if (taps.length === 0) { setStatus('no taps recorded'); return; }
  state.passes.push(taps);
  state.holds.push(holds);
  renderPasses();
  usePass(state.passes.length - 1); // the newest pass becomes the timeline; drawn heights are kept by index
  setStatus(`pass ${state.passes.length}: ${taps.length} taps, hold times recorded`);
}

function usePass(i) {
  setOnsets(state.passes[i], state.holds[i] || null);
  renderPasses(i);
}

function setOnsets(onsets, holds = null) {
  // keep heights where indices line up, so re-recording rhythm keeps drawn contour;
  // hold times belong to the taps, so they come from the pass (merged passes have none)
  const old = state.sketch.observations;
  state.sketch.observations = onsets.map((t, i) => ({
    t, y: old[i] ? old[i].y : 0, confidence: old[i] ? old[i].confidence : 0.5,
    hold: holds && holds[i] != null ? holds[i] : null,
  }));
  state.sketch.gestures = state.sketch.gestures.filter((g) => g.end < onsets.length);
  state.sketch.constraints = state.sketch.constraints.filter((c) => c.i < onsets.length && (c.j == null || c.j < onsets.length));
  state.selection.clear(); state.candidates = []; state.selectedCandidate = -1;
  renderAll(); updateRhythmInfo(); updateGraphPanel(); if (!$('graph-panel').hidden) renderGraph();
}

async function mergePasses() {
  if (state.passes.length < 2) return;
  const r = await api('/api/merge-passes', { passes: state.passes, tolerance_s: 0.08 });
  setOnsets(r.onsets);
  renderPasses(-1);
  setStatus(`merged ${state.passes.length} passes into ${r.onsets.length} notes`);
}

function renderPasses(active = -1) {
  const ul = $('pass-list'); ul.innerHTML = '';
  state.passes.forEach((p, i) => {
    const li = document.createElement('li');
    if (i === active) li.classList.add('active');
    li.innerHTML = `<span>pass ${i + 1}: ${p.length} taps, ${(p[p.length - 1] || 0).toFixed(2)} s</span>`;
    const use = document.createElement('button'); use.className = 'small'; use.textContent = 'use';
    use.onclick = () => usePass(i);
    const del = document.createElement('button'); del.className = 'small'; del.textContent = '×';
    del.onclick = () => { state.passes.splice(i, 1); state.holds.splice(i, 1); renderPasses(); $('btn-merge').disabled = state.passes.length < 2; };
    li.append(use, del); ul.appendChild(li);
  });
  $('btn-merge').disabled = state.passes.length < 2;
}

let rhythmReq = 0;
async function updateRhythmInfo() {
  const obs = state.sketch.observations;
  if (obs.length < 2) { $('rhythm-info').textContent = ''; return; }
  const req = ++rhythmReq;
  try {
    const q = await api('/api/quantise', { onsets: obs.map((o) => o.t), tempo_hint: state.sketch.tempo_hint });
    if (req !== rhythmReq) return; // a newer request superseded this one
    $('rhythm-info').textContent = `${obs.length} notes · estimated ${q.tempo.toFixed(0)} BPM · grid error ${(q.error_s * 1000).toFixed(0)} ms`;
  } catch (e) { $('rhythm-info').textContent = ''; }
}

// ----------------------------------------------------------------------------- step 2: graph
const HMIN = -24, HMAX = 24; // allowed height range for a note
const G = { padL: 44, padR: 16, padT: 12, padB: 24, yMin: -24, yMax: 24, zoomX: 1, span: 48, center: 0 };
const SPANS = [8, 12, 18, 24, 36, 48];
const ZOOMS_X = [1, 1.5, 2, 3, 4, 6, 8];

function applyZoom() {
  G.span = SPANS.includes(G.span) ? G.span : 48;
  const half = G.span / 2;
  G.center = Math.max(HMIN + half, Math.min(HMAX - half, G.center));
  G.yMin = G.center - half; G.yMax = G.center + half;
  svg.style.width = `${G.zoomX * 100}%`;
  $('zoom-info').textContent = `time ×${G.zoomX}, pitch ±${half}`;
}
function heightsCenter(all = false) {
  const sel = all ? [] : [...state.selection];
  const src = sel.length ? sel.map((i) => state.sketch.observations[i]) : state.sketch.observations;
  const ys = src.map((o) => o.y).filter((y) => y !== null);
  return ys.length ? Math.round((Math.min(...ys) + Math.max(...ys)) / 2) : 0;
}
function zoomPitch(dir) {
  const i = SPANS.indexOf(G.span);
  G.span = SPANS[Math.max(0, Math.min(SPANS.length - 1, i - dir))];
  G.center = heightsCenter();
  applyZoom(); renderGraph();
}
function zoomTime(dir) {
  const i = ZOOMS_X.indexOf(G.zoomX);
  G.zoomX = ZOOMS_X[Math.max(0, Math.min(ZOOMS_X.length - 1, i + dir))];
  applyZoom(); renderGraph();
}
function zoomFit() {
  const ys = state.sketch.observations.map((o) => o.y).filter((y) => y !== null);
  const range = ys.length ? Math.max(...ys) - Math.min(...ys) : 0;
  G.span = SPANS.find((sp) => sp >= range + 6) || 48;
  G.center = heightsCenter(true); G.zoomX = 1;
  applyZoom(); renderGraph();
}
const svg = $('graph');
const NS = 'http://www.w3.org/2000/svg';
const el = (tag, attrs = {}, text) => {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  return e;
};

function geom() {
  const W = svg.clientWidth || 900, H = svg.clientHeight || 360;
  const obs = state.sketch.observations;
  const tMax = Math.max(1, ...obs.map((o) => o.t)) * 1.05 + 0.2;
  const xOf = (t) => G.padL + (t / tMax) * (W - G.padL - G.padR);
  const tOf = (x) => ((x - G.padL) / (W - G.padL - G.padR)) * tMax;
  const yOf = (y) => G.padT + ((G.yMax - y) / (G.yMax - G.yMin)) * (H - G.padT - G.padB);
  const hOf = (py) => G.yMax - ((py - G.padT) / (H - G.padT - G.padB)) * (G.yMax - G.yMin);
  return { W, H, tMax, xOf, tOf, yOf, hOf };
}

function renderGraph() {
  const g = geom();
  svg.innerHTML = '';
  // grid
  const labelEvery = G.span >= 36 ? 6 : G.span >= 18 ? 3 : 1;
  for (let y = G.yMin; y <= G.yMax; y++) {
    const strong = y % 12 === 0;
    svg.appendChild(el('line', { x1: G.padL, x2: g.W - G.padR, y1: g.yOf(y), y2: g.yOf(y), class: `gridline${strong ? ' strong' : ''}` }));
    if (y % labelEvery === 0) svg.appendChild(el('text', { x: 6, y: g.yOf(y) + 4, class: 'label' }, y > 0 ? `+${y}` : `${y}`));
  }
  for (let t = 0; t < g.tMax; t += 0.5) {
    svg.appendChild(el('line', { x1: g.xOf(t), x2: g.xOf(t), y1: G.padT, y2: g.H - G.padB, class: `gridline${t % 2 === 0 ? ' strong' : ''}` }));
    if (t % 1 === 0) svg.appendChild(el('text', { x: g.xOf(t) + 2, y: g.H - 8, class: 'label' }, `${t}s`));
  }
  const obs = state.sketch.observations;
  // gesture regions
  const colors = { articulation: '#f0a93b', contour: '#4cc9f0', structure: '#b388ff' };
  state.sketch.gestures.forEach((tag, gi) => {
    const a = obs[tag.start], b = obs[tag.end];
    if (!a || !b) return;
    const x1 = g.xOf(a.t) - 6, x2 = g.xOf(b.t) + 6;
    const cat = state.library ? (state.library[tag.kind] || {}).category : 'contour';
    const lane = gi % 3;
    svg.appendChild(el('rect', { x: x1, y: G.padT + lane * 14, width: Math.max(12, x2 - x1), height: g.H - G.padT - G.padB - lane * 14, fill: colors[cat] || '#999', class: 'region' }));
    svg.appendChild(el('text', { x: x1 + 2, y: G.padT + 10 + lane * 14, class: 'region-label' }, tag.kind.replace('_', ' ')));
  });
  // facts
  state.sketch.constraints.forEach((c) => {
    const a = obs[c.i]; if (!a) return;
    if (c.kind === 'direction' && obs[c.i + 1]) {
      const label = c.value > 0 ? '↑' : c.value < 0 ? '↓' : '=';
      svg.appendChild(el('text', { x: (g.xOf(a.t) + g.xOf(obs[c.i + 1].t)) / 2 - 4, y: g.yOf((a.y ?? 0)) - 14, class: 'fact-label' }, label));
    } else if (c.kind === 'same' && obs[c.j]) {
      const y = g.yOf(a.y ?? 0) - 10;
      svg.appendChild(el('path', { d: `M${g.xOf(a.t)},${y} Q${(g.xOf(a.t) + g.xOf(obs[c.j].t)) / 2},${y - 24} ${g.xOf(obs[c.j].t)},${y}`, class: 'fact' }));
    } else if (c.kind === 'pitch') {
      svg.appendChild(el('text', { x: g.xOf(a.t) + 8, y: g.yOf(a.y ?? 0) + 4, class: 'fact-label' }, midiName(c.value)));
    }
  });
  // candidate overlay
  const cand = state.candidates[state.selectedCandidate];
  if (cand && cand.notes.length === obs.length) {
    const base = cand.notes[0].pitch - (obs[0].y ?? 0);
    const pts = cand.notes.map((n, i) => [g.xOf(obs[i].t), g.yOf(n.pitch - base)]);
    svg.appendChild(el('polyline', { points: pts.map((p) => p.join(',')).join(' '), class: 'candidate' }));
    pts.forEach(([x, y], i) => {
      const c = el('circle', { cx: x, cy: y, r: 4, class: 'cand-point' });
      c.addEventListener('click', (ev) => { ev.stopPropagation(); selectFeedbackNote(i); });
      c.appendChild(el('title', {}, `${cand.note_names[i]} (click to give feedback)`));
      svg.appendChild(c);
    });
  }
  // contour + points
  const known = obs.map((o, i) => [i, o]).filter(([, o]) => o.y !== null);
  if (known.length > 1) {
    svg.appendChild(el('polyline', { points: known.map(([, o]) => `${g.xOf(o.t)},${g.yOf(o.y)}`).join(' '), class: 'contour' }));
  }
  const pinned = new Set(state.sketch.constraints.filter((c) => c.kind === 'pitch').map((c) => c.i));
  obs.forEach((o, i) => {
    const cls = ['point'];
    if (state.selection.has(i)) cls.push('selected');
    if (o.y === null) cls.push('unknown');
    if (pinned.has(i)) cls.push('pinned');
    const c = el('circle', { cx: g.xOf(o.t), cy: g.yOf(o.y ?? 0), r: 6 + 3 * (o.confidence || 0), class: cls.join(' '), 'data-i': i });
    c.appendChild(el('title', {}, `note ${i + 1} @ ${o.t.toFixed(2)}s, height ${o.y === null ? '?' : o.y.toFixed(1)}, conf ${o.confidence.toFixed(2)}`));
    svg.appendChild(c);
  });
  $('sel-info').textContent = state.selection.size ? [...state.selection].sort((a, b) => a - b).map((i) => i + 1).join(', ') : 'none';
}

// pointer interaction on the graph
let drag = null;
function endDrag() { drag = null; svg.classList.remove('dragging'); }
svg.addEventListener('pointerdown', (ev) => {
  if (ev.button !== 0) return;
  ev.preventDefault(); // no text selection, no focus jump
  const target = ev.target;
  const g = geom();
  if (state.paint) {
    drag = { paint: true }; svg.setPointerCapture(ev.pointerId); svg.classList.add('dragging'); paintAt(ev); return;
  }
  if (target.classList && target.classList.contains('point')) {
    const i = Number(target.dataset.i);
    if (ev.shiftKey && state.selection.size) {
      const lo = Math.min(i, ...state.selection), hi = Math.max(i, ...state.selection);
      for (let k = lo; k <= hi; k++) state.selection.add(k);
    } else if (ev.ctrlKey || ev.metaKey) {
      state.selection.has(i) ? state.selection.delete(i) : state.selection.add(i);
    } else if (!state.selection.has(i)) {
      state.selection.clear(); state.selection.add(i);
    }
    const idx = [...state.selection];
    drag = { idx, startY: ev.clientY, orig: idx.map((k) => state.sketch.observations[k].y ?? 0), g };
    svg.setPointerCapture(ev.pointerId); svg.classList.add('dragging');
    renderGraph(); syncConfSlider();
  } else {
    state.selection.clear(); renderGraph();
  }
});
svg.addEventListener('pointermove', (ev) => {
  if (!drag) return;
  if (ev.buttons === 0) { endDrag(); return; } // the release happened somewhere we did not see
  ev.preventDefault();
  if (drag.paint) { paintAt(ev); return; }
  const dy = (ev.clientY - drag.startY) / ((drag.g.H - G.padT - G.padB) / (G.yMax - G.yMin));
  drag.idx.forEach((k, j) => {
    const o = state.sketch.observations[k];
    o.y = Math.max(HMIN, Math.min(HMAX, Math.round((drag.orig[j] - dy) * 2) / 2));
  });
  renderGraph();
});
svg.addEventListener('pointerup', endDrag);
svg.addEventListener('pointercancel', endDrag);
svg.addEventListener('lostpointercapture', endDrag);
window.addEventListener('pointerup', endDrag);
window.addEventListener('blur', endDrag);
svg.addEventListener('dblclick', (ev) => {
  if (ev.target.classList.contains('point')) return;
  const g = geom();
  const rect = svg.getBoundingClientRect();
  const t = g.tOf(ev.clientX - rect.left), y = Math.max(HMIN, Math.min(HMAX, Math.round(g.hOf(ev.clientY - rect.top))));
  const obs = state.sketch.observations;
  snapshot();
  obs.push({ t: Math.max(0, t), y, confidence: 0.8 });
  const order = obs.map((o, i) => i).sort((a, b) => obs[a].t - obs[b].t);
  const remap = new Map(order.map((old, nw) => [old, nw]));
  state.sketch.observations = order.map((i) => obs[i]);
  state.sketch.gestures.forEach((gt) => { gt.start = remap.get(gt.start); gt.end = remap.get(gt.end); });
  state.sketch.constraints.forEach((c) => { c.i = remap.get(c.i); if (c.j != null) c.j = remap.get(c.j); });
  state.selection.clear(); state.candidates = []; state.selectedCandidate = -1;
  renderAll(); updateRhythmInfo();
});
function paintAt(ev) {
  const g = geom(); const rect = svg.getBoundingClientRect();
  const t = g.tOf(ev.clientX - rect.left), y = Math.round(g.hOf(ev.clientY - rect.top) * 2) / 2;
  const obs = state.sketch.observations;
  let best = -1, bd = 0.08 * g.tMax;
  obs.forEach((o, i) => { const d = Math.abs(o.t - t); if (d < bd) { best = i; bd = d; } });
  if (best >= 0) { obs[best].y = Math.max(HMIN, Math.min(HMAX, y)); renderGraph(); }
}

let undoSnapshot = null;
function snapshot() {
  undoSnapshot = JSON.stringify({ sketch: state.sketch, passes: state.passes, holds: state.holds });
  $('btn-undo').disabled = false;
}
function undo() {
  if (!undoSnapshot) return;
  const obj = JSON.parse(undoSnapshot); undoSnapshot = null; $('btn-undo').disabled = true;
  load(obj); setStatus('undid the last tap edit');
}

function deleteSelection() {
  if (!state.selection.size) { setStatus('select notes on the timeline first'); return; }
  const n = state.selection.size;
  keepIndices(state.sketch.observations.map((o, i) => i).filter((i) => !state.selection.has(i)));
  setStatus(`removed ${n} note${n > 1 ? 's' : ''}`);
}

/** Keep only the given observation indices (sorted), re-basing time to the first kept note. */
function keepIndices(keep) {
  if (!keep.length) { setStatus('that would remove every note'); return; }
  snapshot();
  const remap = new Map(keep.map((old, nw) => [old, nw]));
  state.sketch.observations = keep.map((i) => state.sketch.observations[i]);
  state.sketch.gestures = state.sketch.gestures
    .filter((g) => remap.has(g.start) || remap.has(g.end))
    .map((g) => ({ ...g, start: remap.get(g.start) ?? remap.get([...remap.keys()].find((k) => k > g.start)), end: remap.get(g.end) ?? remap.get([...remap.keys()].reverse().find((k) => k < g.end)) }))
    .filter((g) => g.start != null && g.end != null && g.start <= g.end);
  state.sketch.constraints = state.sketch.constraints
    .filter((c) => remap.has(c.i) && (c.j == null || remap.has(c.j)))
    .map((c) => ({ ...c, i: remap.get(c.i), j: c.j == null ? null : remap.get(c.j) }));
  const t0 = state.sketch.observations[0].t;
  state.sketch.observations.forEach((o) => { o.t = Math.max(0, o.t - t0); });
  state.selection.clear(); state.candidates = []; state.selectedCandidate = -1;
  renderAll(); updateRhythmInfo(); updateGraphPanel();
}

function keepRange(from, to) { // 1-based, inclusive
  const n = state.sketch.observations.length;
  from = Math.max(1, Math.min(n, from)); to = Math.max(from, Math.min(n, to));
  keepIndices([...Array(to - from + 1).keys()].map((k) => k + from - 1));
  setStatus(`kept notes ${from}–${to}`);
}

function addNoteAtEnd() {
  const obs = state.sketch.observations;
  snapshot();
  const iois = obs.slice(1).map((o, i) => o.t - obs[i].t).sort((a, b) => a - b);
  const gap = iois.length ? iois[Math.floor(iois.length / 2)] : 0.5;
  const last = obs[obs.length - 1];
  obs.push({ t: last ? last.t + gap : 0, y: last ? last.y : 0, confidence: 0.8 });
  state.selection.clear(); state.selection.add(obs.length - 1);
  state.candidates = []; state.selectedCandidate = -1;
  renderAll(); updateRhythmInfo(); updateGraphPanel();
}

/** Smallest period p such that the inter-onset pattern repeats with period p. */
function detectRepetition() {
  const obs = state.sketch.observations;
  const n = obs.length;
  if (n < 4) return null;
  const iois = obs.slice(1).map((o, i) => o.t - obs[i].t);
  for (let p = 2; p <= Math.floor(n / 2); p++) {
    let ok = 0, total = 0;
    for (let i = 0; i + p < iois.length; i++) {
      const a = iois[i], b = iois[i + p];
      total++;
      if (Math.abs(a - b) <= Math.max(0.06, 0.25 * Math.max(a, b))) ok++;
    }
    if (total > 0 && ok / total >= 0.8) return p;
  }
  return null;
}

function keepOneRepetition() {
  const p = detectRepetition();
  if (!p) { setStatus('no repeated rhythm pattern found; use "Keep notes … to …" instead'); return; }
  keepRange(1, p);
  setStatus(`found a pattern of ${p} notes repeating; kept the first occurrence`);
}

function updateGraphPanel() {
  const has = state.sketch.observations.length > 0;
  $('graph-panel').hidden = state.step === 0 && !has;
  $('edit-taps').hidden = !has;
  const n = state.sketch.observations.length;
  if (has && n !== updateGraphPanel.lastN) { // note count changed: reset the range to "everything"
    $('keep-from').max = $('keep-to').max = n;
    $('keep-from').value = 1; $('keep-to').value = n;
  }
  updateGraphPanel.lastN = n;
}

function syncConfSlider() {
  const sel = [...state.selection];
  if (!sel.length) return;
  const c = state.sketch.observations[sel[0]].confidence;
  $('conf-slider').value = c; $('conf-val').textContent = c.toFixed(2);
}
$('conf-slider').addEventListener('input', (ev) => {
  const c = Number(ev.target.value); $('conf-val').textContent = c.toFixed(2);
  state.selection.forEach((i) => { state.sketch.observations[i].confidence = c; });
  renderGraph();
});
$('btn-mark-unknown').onclick = () => { state.selection.forEach((i) => { state.sketch.observations[i].y = null; state.sketch.observations[i].confidence = 0.1; }); renderGraph(); };

// ----------------------------------------------------------------------------- step 3: gestures and facts
async function loadLibrary() {
  const lib = await api('/api/gestures');
  state.library = {};
  const root = $('gesture-library'); root.innerHTML = '';
  for (const [catKey, cat] of Object.entries(lib)) {
    const div = document.createElement('div'); div.className = 'cat';
    div.innerHTML = `<div class="cat-name">${cat.label}</div>`;
    for (const [key, item] of Object.entries(cat.items)) {
      state.library[key] = { ...item, category: catKey };
      const b = document.createElement('button'); b.className = 'small'; b.textContent = item.label; b.title = item.desc;
      b.onclick = () => addGesture(key);
      div.appendChild(b);
    }
    root.appendChild(div);
  }
}

function addGesture(kind, start, end, confidence = 0.8, params) {
  const item = state.library[kind] || {};
  if (start === undefined) {
    if (!state.selection.size) { setStatus('select notes on the graph first'); return; }
    const sel = [...state.selection].sort((a, b) => a - b);
    start = sel[0]; end = item.scope === 'note' ? sel[0] : sel[sel.length - 1];
  }
  state.sketch.gestures.push({ kind, start, end, confidence, params: params || { ...(item.params || {}) } });
  renderTags(); renderGraph();
}

function addConstraint(c) {
  state.sketch.constraints = state.sketch.constraints.filter((x) => !(x.kind === c.kind && x.i === c.i && (x.j ?? null) === (c.j ?? null)));
  state.sketch.constraints.push(c);
  renderTags(); renderGraph();
}

function renderTags() {
  const ul = $('tag-list'); ul.innerHTML = '';
  state.sketch.gestures.forEach((g, i) => {
    const li = document.createElement('li');
    const label = state.library && state.library[g.kind] ? state.library[g.kind].label : g.kind;
    const p = Object.keys(g.params || {}).length ? ' ' + JSON.stringify(g.params) : '';
    li.innerHTML = `<span class="kind">${label}</span> notes ${g.start + 1}–${g.end + 1}${p} <span class="muted">conf ${g.confidence.toFixed(1)}</span>`;
    const x = document.createElement('button'); x.className = 'small'; x.textContent = '×';
    x.onclick = () => { state.sketch.gestures.splice(i, 1); renderTags(); renderGraph(); };
    li.appendChild(x); ul.appendChild(li);
  });
  state.sketch.constraints.forEach((c, i) => {
    const li = document.createElement('li');
    let txt = '';
    if (c.kind === 'pitch') txt = `note ${c.i + 1} is ${midiName(c.value)}`;
    else if (c.kind === 'same') txt = `notes ${c.i + 1} and ${c.j + 1} have the same pitch`;
    else if (c.kind === 'direction') txt = `note ${c.i + 2} is ${c.value > 0 ? 'higher than' : c.value < 0 ? 'lower than' : 'the same as'} note ${c.i + 1}`;
    else txt = JSON.stringify(c);
    li.innerHTML = `<span class="fact">fact</span> ${txt}`;
    const x = document.createElement('button'); x.className = 'small'; x.textContent = '×';
    x.onclick = () => { state.sketch.constraints.splice(i, 1); renderTags(); renderGraph(); };
    li.appendChild(x); ul.appendChild(li);
  });
}

async function suggestGestures() {
  const r = await api('/api/suggest-gestures', state.sketch);
  const ul = $('suggestions'); ul.innerHTML = '';
  if (!r.gestures.length) { ul.innerHTML = '<li class="muted">nothing obvious found</li>'; return; }
  r.gestures.forEach((g) => {
    const li = document.createElement('li');
    const label = state.library[g.kind] ? state.library[g.kind].label : g.kind;
    li.innerHTML = `<span>${label} on notes ${g.start + 1}–${g.end + 1}${g.params && g.params.period ? ` (period ${g.params.period})` : ''} <span class="muted">${Math.round(g.confidence * 100)}%</span></span>`;
    const ok = document.createElement('button'); ok.className = 'small'; ok.textContent = 'accept';
    ok.onclick = () => { addGesture(g.kind, g.start, g.end, 0.9, g.params); li.remove(); };
    const no = document.createElement('button'); no.className = 'small'; no.textContent = 'dismiss';
    no.onclick = () => li.remove();
    li.append(ok, no); ul.appendChild(li);
  });
}

$('btn-suggest').onclick = () => suggestGestures().catch((e) => setStatus(e.message));
async function fillGaps() {
  if (!state.sketch.gestures.some((g) => g.kind === 'tapping_run')) {
    setStatus('tag a region as "Tapping / fast run" first, then fill in the taps you missed'); return;
  }
  const r = await api('/api/fill-gaps', state.sketch);
  if (!r.inserted) { setStatus('no gaps looked like missed taps in the tagged runs'); return; }
  snapshot();
  state.sketch = r.sketch; state.sketch.meta ||= {};
  state.selection.clear(); state.candidates = []; state.selectedCandidate = -1;
  renderAll(); updateRhythmInfo(); updateGraphPanel();
  setStatus(`inserted ${r.inserted} placeholder note${r.inserted > 1 ? 's' : ''} (hollow points); delete any that are wrong, their pitch is left to the model`);
}
$('btn-fill-gaps').onclick = () => fillGaps().catch((e) => setStatus(e.message));
$('btn-set-pitch').onclick = () => {
  const sel = [...state.selection].sort((a, b) => a - b);
  if (!sel.length) { setStatus('select a note first'); return; }
  addConstraint({ kind: 'pitch', i: sel[0], j: null, value: Number($('pitch-select').value) });
};
document.querySelectorAll('button.dir').forEach((b) => {
  b.onclick = () => {
    const sel = [...state.selection].sort((a, b2) => a - b2);
    if (!sel.length) { setStatus('select a note first'); return; }
    sel.forEach((i) => { if (i + 1 < state.sketch.observations.length) addConstraint({ kind: 'direction', i, j: null, value: Number(b.dataset.dir) }); });
  };
});
$('btn-same').onclick = () => {
  const sel = [...state.selection].sort((a, b) => a - b);
  if (sel.length < 2) { setStatus('select at least two notes'); return; }
  addConstraint({ kind: 'same', i: sel[0], j: sel[sel.length - 1], value: null });
};
(function fillPitchSelect() {
  const s = $('pitch-select');
  for (let p = 36; p <= 84; p++) { const o = document.createElement('option'); o.value = p; o.textContent = midiName(p); if (p === 52) o.selected = true; s.appendChild(o); }
})();

// ----------------------------------------------------------------------------- step 4: reconstruct
async function reconstruct() {
  if (state.sketch.observations.length < 2) { setStatus('tap at least two notes first'); return; }
  $('recon-status').textContent = 'thinking…';
  try {
    state.sketch.meta.genre = $('genre-select').value || 'unknown';
    const r = await api('/api/reconstruct', { sketch: state.sketch, n_candidates: Number($('n-cand').value) || 5, engine: $('engine-select').value || 'auto' });
    state.candidates = r.candidates; state.selectedCandidate = r.candidates.length ? 0 : -1;
    renderCandidates(); renderGraph();
    $('recon-status').textContent = `${r.candidates.length} candidates from the ${r.engine} engine, ${r.candidates[0] ? r.candidates[0].tempo.toFixed(0) : '?'} BPM`;
    $('btn-play').disabled = state.selectedCandidate < 0;
  } catch (e) { $('recon-status').textContent = ''; setStatus(e.message); }
}

function renderCandidates() {
  const ol = $('candidates'); ol.innerHTML = '';
  state.candidates.forEach((c, i) => {
    const li = document.createElement('li');
    if (i === state.selectedCandidate) li.classList.add('selected');
    const method = (c.meta && c.meta.method) || '';
    const scoreTxt = method.startsWith('transformer')
      ? `model log-prob ${c.score.toFixed(2)} (≈${(100 * Math.exp(c.score)).toPrecision(2)}% of the model's probability mass)`
      : `search score ${c.score.toFixed(2)}`;
    const chips = (c.tropes || []).map((t) => `<span class="chip" title="${t.song_frac != null ? Math.round(t.song_frac * 100) + '% of songs in this genre use it' : 'not in the catalogue'}">${t.label}</span>`).join('');
    const fam = c.familiarity == null ? '' : `<span class="chip fam" title="share of this riff's 3-interval patterns that are common in the genre catalogue">${Math.round(c.familiarity * 100)}% familiar patterns</span>`;
    li.innerHTML = `<span class="key">${c.key}</span> <span class="muted">${scoreTxt} · ${method}</span><br>` +
      `<span class="notes">${c.note_names.join(' ')}</span>` +
      ((chips || fam) ? `<br><span class="chips">${fam}${chips}</span>` : '') +
      (c.violations.length ? `<br><span class="viol">violates: ${c.violations.join('; ')}</span>` : '');
    li.onclick = () => { state.selectedCandidate = i; renderCandidates(); renderGraph(); $('btn-play').disabled = false; state.feedbackNote = -1; $('feedback').hidden = true; };
    ol.appendChild(li);
  });
  const sel = state.candidates[state.selectedCandidate];
  $('listen-cand').textContent = sel ? `${state.selectedCandidate + 1}. ${sel.key} — ${sel.note_names.join(' ')}` : 'none (reconstruct first)';
  $('btn-play').disabled = !sel;
}
$('btn-reconstruct').onclick = reconstruct;

// ----------------------------------------------------------------------------- step 5: listen and refine
async function play() {
  const cand = state.candidates[state.selectedCandidate];
  if (!cand) return;
  stopAudio();
  const res = await api('/api/synthesize', cand, true);
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = new Audio(url); state.audio = a;
  $('btn-stop-audio').disabled = false;
  a.onloadedmetadata = () => { state.exposure.count += 1; state.exposure.seconds += a.duration; renderExposure(); };
  a.onended = () => { $('btn-stop-audio').disabled = true; URL.revokeObjectURL(url); };
  await a.play();
  setStatus(`playing candidate ${state.selectedCandidate + 1} (${cand.key})`);
}
function stopAudio() { if (state.audio) { state.audio.pause(); state.audio = null; $('btn-stop-audio').disabled = true; } }
function renderExposure() { $('exposure').textContent = `${state.exposure.count} previews, ${state.exposure.seconds.toFixed(1)} s`; }
$('btn-play').onclick = () => play().catch((e) => setStatus(e.message));
$('btn-stop-audio').onclick = stopAudio;

function selectFeedbackNote(i) {
  state.feedbackNote = i; $('feedback').hidden = false;
  const cand = state.candidates[state.selectedCandidate];
  $('fb-note').textContent = `${i + 1} (${cand.note_names[i]})`;
}
$('fb-right').onclick = () => {
  const cand = state.candidates[state.selectedCandidate]; const i = state.feedbackNote;
  if (!cand || i < 0) return;
  addConstraint({ kind: 'pitch', i, j: null, value: cand.notes[i].pitch });
  state.sketch.observations[i].confidence = 1.0; renderGraph();
  setStatus(`pinned note ${i + 1} to ${cand.note_names[i]}`);
};
function nudge(dir) {
  const cand = state.candidates[state.selectedCandidate]; const i = state.feedbackNote;
  if (!cand || i < 0) return;
  // move the drawn height in the opposite direction of the error and raise confidence;
  // also exclude the candidate pitch via a direction fact relative to the previous note when unambiguous
  const o = state.sketch.observations[i];
  o.y = (o.y ?? 0) + dir; o.confidence = Math.min(1, o.confidence + 0.15);
  if (i > 0) {
    const prev = cand.notes[i - 1].pitch, cur = cand.notes[i].pitch;
    const pinnedPrev = state.sketch.constraints.find((c) => c.kind === 'pitch' && c.i === i - 1);
    if (pinnedPrev && ((dir > 0 && cur <= prev) || (dir < 0 && cur >= prev))) addConstraint({ kind: 'direction', i: i - 1, j: null, value: dir });
  }
  renderGraph(); setStatus(`note ${i + 1} marked ${dir > 0 ? 'too low → raised' : 'too high → lowered'}; reconstruct again`);
}
$('fb-high').onclick = () => nudge(-1);
$('fb-low').onclick = () => nudge(+1);

// ----------------------------------------------------------------------------- persistence and global keys
function save() {
  const blob = new Blob([JSON.stringify({ sketch: state.sketch, passes: state.passes, holds: state.holds }, null, 2)], { type: 'application/json' });
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'riff-sketch.json'; a.click();
}
function load(obj) {
  state.sketch = obj.sketch || obj; state.passes = obj.passes || []; state.holds = obj.holds || [];
  state.sketch.gestures ||= []; state.sketch.constraints ||= []; state.sketch.meta ||= {};
  // sketches saved before the default confidence moved from 0.5 to 0.8: if nothing was ever
  // touched, lift them, otherwise the model treats every height as a semitone of noise
  const obs = state.sketch.observations;
  if (obs.length && obs.every((o) => o.confidence === 0.5)) obs.forEach((o) => { o.confidence = 0.8; });
  state.selection.clear(); state.candidates = []; state.selectedCandidate = -1;
  renderAll(); renderPasses(); updateRhythmInfo(); updateGraphPanel();
}
function persist() { try { localStorage.setItem('riff-sketcher', JSON.stringify({ sketch: state.sketch, passes: state.passes, holds: state.holds, step: state.step })); } catch (e) { /* ignore */ } }
$('btn-save').onclick = save;
$('file-load').onchange = (ev) => { const f = ev.target.files[0]; if (!f) return; f.text().then((t) => load(JSON.parse(t))).catch((e) => setStatus(e.message)); };
$('btn-clear').onclick = () => { if (confirm('Clear the whole sketch?')) { load({ sketch: { observations: [], gestures: [], constraints: [], tempo_hint: null, meta: {} }, passes: [] }); goTo(0); } };
$('tempo-hint').onchange = (ev) => { state.sketch.tempo_hint = ev.target.value ? Number(ev.target.value) : null; updateRhythmInfo(); };

$('btn-record').onclick = startRecording;
$('btn-stop').onclick = stopRecording;
$('btn-merge').onclick = () => mergePasses().catch((e) => setStatus(e.message));
$('tap-pad').addEventListener('pointerdown', (ev) => { ev.preventDefault(); if (!state.recording) startRecording(); else tap(); });
$('tap-pad').addEventListener('pointerup', releaseTap);
$('tap-pad').addEventListener('pointercancel', releaseTap);

document.addEventListener('keydown', (ev) => {
  const typing = ['INPUT', 'SELECT', 'TEXTAREA'].includes(ev.target.tagName);
  if (state.recording) {
    if (ev.key === 'Enter' || ev.key === 'Escape') { ev.preventDefault(); stopRecording(); return; }
    if (!typing && !ev.repeat && (ev.key === ' ' || /^[a-z0-9]$/i.test(ev.key))) { ev.preventDefault(); tap(); }
    return;
  }
  if (typing) return;
  if (ev.key === 'ArrowRight') { ev.preventDefault(); goTo(state.step + 1); return; }
  if (ev.key === 'ArrowLeft') { ev.preventDefault(); goTo(state.step - 1); return; }
  if ((ev.key === 'r' || ev.key === 'R') && state.step === 0) { ev.preventDefault(); startRecording(); }
  else if (ev.key === 'Delete' || ev.key === 'Backspace') { ev.preventDefault(); deleteSelection(); }
  else if (ev.key === 'p' || ev.key === 'P') { state.paint = true; svg.style.cursor = 'crosshair'; }
  else if (ev.key === 'ArrowUp' || ev.key === 'ArrowDown') {
    ev.preventDefault(); const d = ev.key === 'ArrowUp' ? 1 : -1;
    state.selection.forEach((i) => { const o = state.sketch.observations[i]; o.y = Math.max(HMIN, Math.min(HMAX, (o.y ?? 0) + d)); });
    renderGraph();
  }
});
document.addEventListener('keyup', (ev) => {
  if (state.recording) { releaseTap(); return; }
  if (ev.key === 'p' || ev.key === 'P') { state.paint = false; svg.style.cursor = ''; }
});
window.addEventListener('resize', renderGraph);
window.addEventListener('beforeunload', persist);
setInterval(persist, 5000);

function renderAll() { renderGraph(); renderTags(); renderCandidates(); renderExposure(); if (typeof renderDots === 'function' && $('step-dots')) renderDots(); }

// ----------------------------------------------------------------------------- wizard navigation
const STEPS = [...document.querySelectorAll('main .step')];

function stepDone(i) {
  const n = state.sketch.observations.length;
  if (i === 0) return n > 0;
  if (i === 1) return n > 0 && state.sketch.observations.some((o) => o.y !== null && o.y !== 0);
  if (i === 2) return state.sketch.gestures.length + state.sketch.constraints.length > 0;
  if (i === 3) return state.candidates.length > 0;
  if (i === 4) return state.exposure.count > 0;
  return false;
}

function renderDots() {
  const ol = $('step-dots'); ol.innerHTML = '';
  STEPS.forEach((sec, i) => {
    const li = document.createElement('li');
    if (i === state.step) li.classList.add('active');
    if (stepDone(i)) li.classList.add('done');
    li.innerHTML = `<span class="dot"></span><span class="lbl">${i + 1} ${sec.dataset.title}</span>`;
    li.onclick = () => goTo(i);
    ol.appendChild(li);
  });
  $('nav-prev').disabled = state.step === 0;
  $('nav-next').disabled = state.step === STEPS.length - 1;
}

function goTo(i) {
  if (state.recording) return; // finish the pass first
  i = Math.max(0, Math.min(STEPS.length - 1, i));
  if (i > 0 && state.sketch.observations.length === 0) { setStatus('tap a rhythm first'); i = 0; }
  state.step = i;
  STEPS.forEach((sec, k) => sec.classList.toggle('active', k === i));
  updateGraphPanel();
  renderDots();
  if (!$('graph-panel').hidden) renderGraph(); // the svg has a size only once it is visible
  if (i === 3 && !state.candidates.length) $('recon-status').textContent = 'press Reconstruct when the sketch says everything you know';
  window.scrollTo({ top: 0 });
  persist();
}
$('btn-keep-rep').onclick = keepOneRepetition;
$('btn-keep-range').onclick = () => keepRange(Number($('keep-from').value) || 1, Number($('keep-to').value) || 1);
$('btn-remove-sel').onclick = deleteSelection;
$('btn-add-note').onclick = addNoteAtEnd;
$('btn-undo').onclick = undo;
$('zoom-time-in').onclick = () => zoomTime(1);
$('zoom-time-out').onclick = () => zoomTime(-1);
$('zoom-pitch-in').onclick = () => zoomPitch(1);
$('zoom-pitch-out').onclick = () => zoomPitch(-1);
$('zoom-fit').onclick = zoomFit;
$('nav-prev').onclick = () => goTo(state.step - 1);
$('nav-next').onclick = () => goTo(state.step + 1);

async function loadEngines() {
  const sel = $('engine-select'); sel.innerHTML = '';
  const r = await api('/api/engines');
  r.engines.forEach((e) => {
    const o = document.createElement('option'); o.value = e.name;
    o.textContent = e.name === 'model' ? `model (${e.trained_on || 'trained'}, ${e.device})` : 'deterministic search';
    if (e.default) o.selected = true;
    sel.appendChild(o);
  });
  const model = r.engines.find((e) => e.name === 'model');
  const gsel = $('genre-select'); gsel.innerHTML = '';
  const genres = (model && model.genres && model.genres.length) ? model.genres : ['unknown'];
  genres.forEach((g) => {
    const o = document.createElement('option'); o.value = g; o.textContent = g === 'unknown' ? 'any' : g;
    gsel.appendChild(o);
  });
  gsel.disabled = genres.length <= 1;
  gsel.title = genres.length <= 1 ? 'the loaded model has no genre conditioning' : gsel.title;
  if (state.sketch.meta && genres.includes(state.sketch.meta.genre)) gsel.value = state.sketch.meta.genre;
}

// ----------------------------------------------------------------------------- boot
(async function boot() {
  let step = 0;
  try { await loadLibrary(); await loadEngines(); } catch (e) { setStatus('backend unreachable: ' + e.message); }
  try {
    const saved = localStorage.getItem('riff-sketcher');
    if (saved) { const obj = JSON.parse(saved); load(obj); step = obj.step || 0; setStatus('restored previous sketch'); }
  } catch (e) { /* ignore */ }
  applyZoom();
  renderAll();
  goTo(step);
})();
