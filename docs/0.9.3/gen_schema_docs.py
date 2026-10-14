#!/usr/bin/env python3
"""Generate the schema reference assets for the docs site.

Builds the schema + plugin catalog (see
:func:`pixel_patrol_base.core.schema_catalog.build_catalog`) and writes three
things into ``docs/``:

  * ``assets/schema.json`` - the full catalog, a linkable machine-readable asset.
  * ``assets/schema.html`` - a standalone interactive table map: producers (base
    processing, loaders, processors) -> columns -> widgets. Linked from the docs,
    not embedded.
  * the generated columns table inside ``schema.md`` (between marker comments),
    listing every table column and where it comes from.

Run from the repo root (or anywhere - paths are resolved relative to this file)::

    python docs/gen_schema_docs.py

Invoked by the docs deploy workflow so the assets stay in sync with the plugins.
"""

from __future__ import annotations

import json
from pathlib import Path

from pixel_patrol_base.core.schema_catalog import build_catalog, render_json_schema

DOCS = Path(__file__).resolve().parent
HTML_OUT        = DOCS / "assets" / "schema.html"
JSON_OUT        = DOCS / "assets" / "schema.json"
ROW_SCHEMA_OUT  = DOCS / "assets" / "row-schema.json"
PAGE = DOCS / "schema.md"

COLUMNS_BEGIN = "<!-- BEGIN GENERATED COLUMNS (docs/gen_schema_docs.py) -->"
COLUMNS_END = "<!-- END GENERATED COLUMNS -->"

# Dependency-free vanilla JS so the standalone page works offline.
# The catalog JSON replaces __CATALOG_JSON__ (replace, not str.format, so the JS
# braces don't need escaping).
_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PixelPatrol - Report Map</title>
<style>
  :root { --bg:#fff; --fg:#1b1b1f; --muted:#6a6a73; --line:#e3e3e8; --chip:#eef1f6;
          --accent:#4051b5; --panel:#f7f8fb; --edge:#c8c8c8;
          --c-file:#3b6ea5; --c-image:#2e7d6b; --c-derived:#6a8040; --c-agg:#a03868; --c-metric:#b8742a; --c-widget:#6a4bb5; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#1b1b1f; --fg:#e6e6ea; --muted:#9b9ba4; --line:#33343a; --chip:#2a2b31;
            --accent:#8a9bff; --panel:#212229; --edge:#4a4a4a;
            --c-file:#69a8e0; --c-image:#5fcdb4; --c-derived:#a0ba70; --c-agg:#d080a8; --c-metric:#e2a35a; --c-widget:#b39bff; }
  }
  * { box-sizing:border-box; }
  html, body { margin:0; }
  body { font-family:Inter,system-ui,sans-serif; color:var(--fg); background:var(--bg); font-size:14px; }
  #app { display:flex; flex-direction:column; min-height:100vh; }
  header { padding:.7rem 1rem; border-bottom:1px solid var(--line); }
  header h1 { margin:0 0 .25rem; font-size:1.15rem; }
  header p { margin:0; color:var(--muted); font-size:.85rem; }
  .legend { margin-top:.4rem; display:flex; gap:1.1rem; flex-wrap:wrap; font-size:.8rem; align-items:center; }
  .legend span { display:inline-flex; align-items:center; gap:.35rem; }
  .dot { width:.7rem; height:.7rem; border-radius:50%; display:inline-block; }
  .main { display:flex; flex:1; min-height:380px; border-bottom:1px solid var(--line); }
  #graph-wrap { position:relative; flex:1; min-width:0; overflow:auto; padding:.3rem .8rem 1rem; }
  #edges { position:absolute; top:0; left:0; pointer-events:none; z-index:0; }
  #graph { position:relative; z-index:1; display:grid; grid-template-columns:1fr 1fr 1fr; gap:1.6rem; align-items:start; }
  .colhead { font-size:.82rem; text-transform:uppercase; letter-spacing:.06em; color:var(--muted);
             font-weight:700; margin:.5rem 0 .2rem; position:sticky; top:0; background:var(--bg); padding:.2rem 0; z-index:2; }
  .colhead small { display:block; font-weight:400; text-transform:none; letter-spacing:0; font-size:.74rem; }
  .group-label { font-size:.72rem; text-transform:uppercase; letter-spacing:.05em; color:var(--muted);
                 margin:.7rem 0 .1rem; font-weight:700; }
  .node { border:1px solid var(--line); border-left-width:4px; border-radius:7px; padding:.4rem .6rem;
          margin:.32rem 0; background:var(--bg); cursor:pointer; transition:box-shadow .12s, opacity .12s; }
  .node:hover { box-shadow:0 2px 9px rgba(0,0,0,.12); }
  .node .title { font-weight:600; font-size:.92rem; }
  .node .sub { color:var(--muted); font-size:.74rem; margin-top:.05rem; }
  .node.indent { margin-left:1rem; border-left-style:dotted; }
  .node.big .title { font-size:1rem; }
  .dimmed { opacity:.16; }
  .node.selected { box-shadow:0 0 0 2px var(--accent); }
  aside { width:420px; min-width:420px; border-left:1px solid var(--line); background:var(--panel);
          overflow:auto; padding:1rem 1.1rem; }
  aside .placeholder { color:var(--muted); }
  aside h3 { margin:.1rem 0 .3rem; }
  aside .kind { font-size:.72rem; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); }
  .chip { display:inline-block; background:var(--chip); border-radius:11px; padding:.08rem .5rem;
          margin:.12rem .25rem .12rem 0; font-size:.76rem; }
  .chip.link { cursor:pointer; }
  .chip.link:hover { outline:1px solid var(--accent); }
  code { background:var(--chip); padding:.03rem .3rem; border-radius:4px;
         font-family:'JetBrains Mono',monospace; font-size:.82em; }
  table { border-collapse:collapse; width:100%; font-size:.82rem; }
  th,td { text-align:left; padding:.3rem .5rem; border-bottom:1px solid var(--line); vertical-align:top; }
  th { position:sticky; top:0; background:var(--bg); }
  .sec { margin-top:.8rem; }
  .sec h4 { margin:0 0 .25rem; font-size:.74rem; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); }
  button.reset { float:right; font:inherit; font-size:.8rem; background:var(--chip); color:var(--fg);
                 border:1px solid var(--line); border-radius:6px; padding:.2rem .55rem; cursor:pointer; }
  .col-divider { border:none; border-top:1px solid var(--line); margin:1.1rem 0 .6rem; }
  .col-section { border:1px dashed var(--line); border-radius:7px; padding:.4rem .5rem .5rem; margin:.5rem 0; }
  .col-section .node { background:var(--panel); margin:.25rem 0; }
  .col-section-label { font-size:.7rem; text-transform:uppercase; letter-spacing:.05em; color:var(--muted);
                       font-weight:700; margin-bottom:.3rem; }
  /* Bottom-sheet handle — hidden on desktop, shown on mobile */
  .sheet-handle { display: none; }
  @media (max-width: 700px) {
    #graph { grid-template-columns: 1fr; gap: .8rem; }
    #edges { display: none; }
    #graph-wrap { overflow: visible; }
    .colhead { position: relative; }
    body { padding-bottom: 68px; }
    aside {
      position: fixed; bottom: 0; left: 0; right: 0;
      width: 100%; min-width: unset; border-left: none;
      border-radius: 14px 14px 0 0;
      box-shadow: 0 -2px 16px rgba(0,0,0,.15);
      display: flex; flex-direction: column;
      overflow: hidden; padding: 0;
      height: 56px; max-height: 64vh;
      transition: height .28s ease; z-index: 200;
    }
    aside.sheet-open { height: 64vh; }
    .sheet-handle {
      display: flex; align-items: center; gap: .6rem;
      height: 56px; min-height: 56px; padding: 0 1rem;
      background: var(--panel); border-bottom: 1px solid var(--line);
      cursor: pointer; flex-shrink: 0;
    }
    .sheet-bar { width: 36px; height: 4px; background: var(--muted); border-radius: 2px; opacity: .4; flex-shrink: 0; }
    .sheet-label { flex: 1; font-size: .88rem; font-weight: 600; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    aside.sheet-open .sheet-label { color: var(--fg); }
    .sheet-close { display: none; background: none; border: none; font-size: 1.1rem; color: var(--muted); cursor: pointer; padding: 2px 6px; font-family: inherit; }
    aside.sheet-open .sheet-close { display: block; }
    #details-body { flex: 1; overflow-y: auto; padding: .75rem 1rem 1.5rem; }
  }
</style>
</head>
<body>
<div id="app">
  <header>
    <button class="reset" id="reset">Clear selection</button>
    <h1>Report Map</h1>
    <p>PixelPatrol creates a <code>.parquet</code> table. <b>File data</b> columns are always created. PixelPatrol plugins: <b>Loaders</b> create columns with image metadata and feed the image data to the <b>Processors</b>, which compute metrics and add metric columns to the table. <b>Widgets</b> offer different visualizations of the table and its columns. Click a node for details; hover to trace its links.</p>
    <div class="legend">
      <span><i class="dot" style="background:var(--c-image)"></i> Loader columns</span>
      <span><i class="dot" style="background:var(--c-metric)"></i> Processor metrics</span>
      <span><i class="dot" style="background:var(--c-widget)"></i> Widgets</span>
      <span><i class="dot" style="background:var(--c-file)"></i> File data</span>
      <span><i class="dot" style="background:var(--c-derived)"></i> Image derived columns</span>
      <span><i class="dot" style="background:var(--c-agg)"></i> Aggregation</span>
    </div>
  </header>
  <div class="main">
    <div id="graph-wrap">
      <svg id="edges"></svg>
      <div id="graph">
        <div class="col" id="col-loaders"><div class="colhead">Loaders<small>read images &amp; extract image metadata</small></div></div>
        <div class="col" id="col-mid"><div class="colhead">Processors<small>compute metrics</small></div></div>
        <div class="col" id="col-right"><div class="colhead">Widgets<small>shown in the interactive report</small></div></div>
      </div>
    </div>
    <aside id="details">
      <div class="sheet-handle" id="sheet-handle">
        <div class="sheet-bar"></div>
        <span class="sheet-label" id="sheet-label">Tap a node for details</span>
        <button class="sheet-close" id="sheet-close" aria-label="Close">&#x2715;</button>
      </div>
      <div id="details-body"><p class="placeholder">Select a node to see its description and columns.</p></div>
    </aside>
  </div>
</div>

<script id="catalog" type="application/json">__CATALOG_JSON__</script>
<script>
const CAT = JSON.parse(document.getElementById('catalog').textContent);
const SVGNS = 'http://www.w3.org/2000/svg';
const slug = s => String(s).replace(/[^a-z0-9]+/gi, '-').toLowerCase();
const nodeDomId = id => 'n-' + slug(id);

const rasterSchema = (CAT.loader_schemas || [])[0] || null;
const colCount = producer => CAT.columns.filter(c => c.producer === producer).length;

function el(tag, attrs={}, ...kids) {
  const n = document.createElement(tag);
  for (const [k,v] of Object.entries(attrs)) {
    if (k === 'class') n.className = v; else if (k === 'html') n.innerHTML = v; else n.setAttribute(k, v);
  }
  for (const kid of kids) if (kid != null) n.append(kid.nodeType ? kid : document.createTextNode(kid));
  return n;
}
function globToRe(t) { return new RegExp('^' + t.replace(/[-[\]{}()+?.,\\^$|#\s]/g, '\\$&').replace(/\*/g, '.*') + '$'); }

// Resolve a widget's declared inputs to concrete columns, driven by the catalog's
// column list. Input tokens: '*' (every column), '!<col>' (minus one), '<col>'
// (one column), a glob like '*_size' (a per-axis family), or the 'any metric
// column' sentinel.
function widgetColumns(w) {
  const out = new Map();
  const allInputs = [...(w.required_inputs || []), ...(w.inputs || [])];
  const excluded = new Set(allInputs.filter(t => t.startsWith('!')).map(t => t.slice(1)));
  const add = c => { if (!excluded.has(c.name)) out.set(c.name, c); };
  if (allInputs.includes('*')) CAT.columns.forEach(add);
  if (allInputs.includes('any metric column')) CAT.columns.filter(c => c.category === 'metric').forEach(add);
  for (const t of allInputs) {
    if (t === '*' || t === 'any metric column' || t.startsWith('!')) continue;
    const ex = CAT.columns.find(c => c.name === t);
    if (ex) { add(ex); continue; }
    for (const c of CAT.columns) {
      let m = false;
      if (c.regex) { try { m = new RegExp(c.regex).test(t); } catch {} }
      if (!m && t.includes('*')) { try { m = globToRe(t).test(c.name); } catch {} }
      if (m) add(c);
    }
  }
  return [...out.values()];
}
const widgetCols = Object.fromEntries(CAT.widgets.map(w => ['widget:'+w.id, widgetColumns(w)]));

// ---- node registry ---------------------------------------------------------
const byNode = {};
byNode['file'] = {kind:'producer', label:'File data', sub:'file-system scan',
  desc:'Scans the input directory and records each file: paths, sizes, extensions and timestamps.'};
byNode['derived'] = {kind:'producer', label:'Derived metadata', sub:'computed from image data',
  desc:'The pipeline computes these from the image data: ndim from dim_order, num_pixels from shape, and <axis>_size columns from the actual array shape per dimension. Loaders do not need to provide these.'};
byNode['agg']  = {kind:'producer', label:'Aggregation', sub:'adds per-dimension rows',
  desc:'PixelPatrol offers per-dimension slice statistics in the table. Aggregation of those statistics creates multiple rows per image - one per dimension slice (Z, T, C, ...) - and adds obs_level and dim_* coordinate columns to identify each slice.'};
if (rasterSchema) byNode['schema:'+rasterSchema.id] = {kind:'schema', label:rasterSchema.label, data:rasterSchema};
for (const l of CAT.loaders) byNode['loader:'+l.name] = {kind:'loader', label:l.name, data:l};
for (const p of CAT.processors) byNode['processor:'+p.name] = {kind:'processor', label:p.name, data:p};
for (const w of CAT.widgets) byNode['widget:'+w.id] = {kind:'widget', label:w.label||w.id, data:w};

// ---- edges -----------------------------------------------------------------
// Drawn straight from the catalog's connection list - the real data flow:
// file / aggregation / loaders -> processors -> widgets (loaders and the base
// producers also feed some widgets directly). There is no group hub, so a hovered
// node highlights only the nodes it actually connects to.
const edges = []; const seenEdge = new Set();
function addEdge(source, target, via) {
  const k = source+'>'+target;
  if (source && target && byNode[source] && byNode[target] && !seenEdge.has(k)) {
    seenEdge.add(k); edges.push({source, target, via});
  }
}
for (const c of CAT.connections || []) addEdge(c.source, c.target, c.via);

const outE = {}, inE = {};
edges.forEach((e,i) => {
  (outE[e.source] = outE[e.source] || []).push(i);
  (inE[e.target] = inE[e.target] || []).push(i);
});
const SCHEMA_ID = rasterSchema ? 'schema:'+rasterSchema.id : null;

// transitive reach along a direction, so a producer reaches its widgets (and back)
function reach(starts, adj, pick) {
  const nodes = new Set(starts), eset = new Set(), stack = [...starts];
  while (stack.length) {
    const n = stack.pop();
    for (const i of adj[n] || []) {
      eset.add(i);
      const nx = pick(edges[i]);
      if (!nodes.has(nx)) { nodes.add(nx); stack.push(nx); }
    }
  }
  return {nodes, eset};
}
function connected(id) {
  const d = reach([id], outE, e => e.target);
  const u = reach([id], inE, e => e.source);
  return {nodes: new Set([...d.nodes, ...u.nodes, id]), eset: new Set([...d.eset, ...u.eset])};
}

// ---- build nodes -----------------------------------------------------------
function nodeEl(id, title, sub, color, opts={}) {
  const cls = 'node' + (opts.indent ? ' indent' : '') + (opts.big ? ' big' : '');
  const node = el('div', {class:cls, id:nodeDomId(id)}, el('div', {class:'title'}, title));
  if (color) node.style.borderLeftColor = color;
  if (sub) node.append(el('div', {class:'sub'}, sub));
  node.dataset.node = id;
  node.addEventListener('mouseenter', () => highlight(id));
  node.addEventListener('mouseleave', () => highlight(selected));
  node.addEventListener('click', () => select(id));
  return node;
}

const left = document.getElementById('col-loaders');
for (const l of CAT.loaders) {
  const exts = l.extensions || [];
  const extPreview = exts.slice(0,3).join(', ') + (exts.length > 3 ? ` (+${exts.length - 3} more)` : '');
  const sub = 'Extensions: ' + (extPreview || 'none');
  left.append(nodeEl('loader:'+l.name, l.name, sub, 'var(--c-image)'));
}
left.append(el('hr', {class:'col-divider'}));
const fileSection = el('div', {class:'col-section'});
fileSection.append(el('div', {class:'col-section-label'}, 'Always created columns'));
fileSection.append(nodeEl('file', 'File data', colCount('file') + ' columns', 'var(--c-file)'));
left.append(fileSection);
const derivedSection = el('div', {class:'col-section'});
derivedSection.append(el('div', {class:'col-section-label'}, 'Image derived columns'));
derivedSection.append(nodeEl('derived', 'Derived metadata', colCount('derived') + ' columns', 'var(--c-derived)'));
left.append(derivedSection);
const aggSection = el('div', {class:'col-section'});
aggSection.append(el('div', {class:'col-section-label'}, 'Per-Dim Statistics'));
aggSection.append(nodeEl('agg', 'Aggregation', colCount('agg') + ' columns', 'var(--c-agg)'));
left.append(aggSection);

const mid = document.getElementById('col-mid');
for (const p of CAT.processors)
  mid.append(nodeEl('processor:'+p.name, p.name, p.columns.length + ' metrics', 'var(--c-metric)', {big:true}));

const right = document.getElementById('col-right');
for (const w of CAT.widgets)
  right.append(nodeEl('widget:'+w.id, w.label||w.id, w.group||'', 'var(--c-widget)'));

// ---- edges (SVG) -----------------------------------------------------------
const svg = document.getElementById('edges');
const wrap = document.getElementById('graph-wrap');
let paths = [];
function layoutEdges() {
  svg.innerHTML = ''; paths = [];
  const wr = wrap.getBoundingClientRect();
  const W = wrap.scrollWidth, H = wrap.scrollHeight;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.setAttribute('width', W); svg.setAttribute('height', H);
  edges.forEach(e => {
    const a = document.getElementById(nodeDomId(e.source)), b = document.getElementById(nodeDomId(e.target));
    if (!a || !b) { paths.push(null); return; }
    const ra = a.getBoundingClientRect(), rb = b.getBoundingClientRect();
    const x1 = ra.right - wr.left + wrap.scrollLeft, y1 = ra.top - wr.top + wrap.scrollTop + ra.height/2;
    const x2 = rb.left - wr.left + wrap.scrollLeft, y2 = rb.top - wr.top + wrap.scrollTop + rb.height/2;
    const dx = Math.max(36, (x2 - x1) * 0.45);
    const p = document.createElementNS(SVGNS, 'path');
    p.setAttribute('d', `M ${x1} ${y1} C ${x1+dx} ${y1}, ${x2-dx} ${y2}, ${x2} ${y2}`);
    p.setAttribute('fill', 'none'); p.setAttribute('stroke', 'var(--edge)'); p.setAttribute('stroke-width', '1.3');
    svg.append(p); paths.push(p);
  });
}

// ---- highlight + selection -------------------------------------------------
let selected = null;
function highlight(id) {
  const conn = id ? connected(id) : null;
  paths.forEach((p, i) => {
    if (!p) return;
    const on = id && conn.eset.has(i);
    p.setAttribute('stroke', on ? '#666' : 'var(--edge)');
    p.setAttribute('stroke-width', on ? '2.2' : '1.3');
    p.setAttribute('opacity', id ? (on ? '1' : '.1') : '.55');
  });
  document.querySelectorAll('.node').forEach(n => {
    n.classList.toggle('dimmed', !!id && !conn.nodes.has(n.dataset.node));
    n.classList.toggle('selected', n.dataset.node === selected);
  });
}
function sheetClose() {
  selected = null; highlight(null); renderDetails(null);
  document.getElementById('details').classList.remove('sheet-open');
  document.getElementById('sheet-label').textContent = 'Tap a node for details';
}
function select(id) {
  selected = (selected === id) ? null : id;
  highlight(selected); renderDetails(selected);
  if (window.innerWidth <= 700) {
    const aside = document.getElementById('details');
    const lbl = document.getElementById('sheet-label');
    if (selected) { aside.classList.add('sheet-open'); lbl.textContent = byNode[selected]?.label || selected; }
    else { aside.classList.remove('sheet-open'); lbl.textContent = 'Tap a node for details'; }
  }
}
document.getElementById('reset').addEventListener('click', sheetClose);
document.getElementById('sheet-handle').addEventListener('click', sheetClose);

// ---- details panel ---------------------------------------------------------
function colTable(cols) {
  const t = el('table');
  t.append(el('thead', {}, el('tr', {}, el('th',{},'Column'), el('th',{},'Type'), el('th',{},'Description'))));
  const b = el('tbody');
  for (const c of cols) b.append(el('tr', {},
    el('td', {}, el('code', {}, c.name)), el('td', {}, el('code', {}, c.dtype)), el('td', {}, c.description || '')));
  t.append(b); return t;
}
function chipLink(nid) {
  const nd = byNode[nid];
  const c = el('span', {class:'chip link'}, nd ? nd.label : nid);
  c.addEventListener('click', () => select(nid));
  return c;
}
function connList(panel, id) {
  const up = reach([id], inE, e => e.source).nodes;
  const down = reach([id], outE, e => e.target).nodes;
  up.delete(id); down.delete(id);
  if (up.size)   { const s = el('div', {class:'sec'}, el('h4', {}, 'Receives from')); [...up].forEach(n => s.append(chipLink(n))); panel.append(s); }
  if (down.size) { const s = el('div', {class:'sec'}, el('h4', {}, 'Feeds into')); [...down].forEach(n => s.append(chipLink(n))); panel.append(s); }
}

function renderDetails(id) {
  const panel = document.getElementById('details-body');
  panel.innerHTML = '';
  if (!id || !byNode[id]) {
    panel.append(el('p', {class:'placeholder'}, 'Select a node to see its description and columns.'));
    return;
  }
  const node = byNode[id], {kind, data} = node;
  panel.append(el('div', {class:'kind'}, kind));
  panel.append(el('h3', {}, node.label));

  if (kind === 'producer') {
    if (node.desc) panel.append(el('p', {}, node.desc));
    panel.append(el('div', {class:'sec'}, el('h4', {}, 'Columns'),
      colTable(CAT.columns.filter(c => c.producer === id))));
  } else if (kind === 'schema') {
    panel.append(el('p', {}, 'Common columns every raster-image loader produces (inherited, not redefined per loader).'));
    const s = el('div', {class:'sec'}, el('h4', {}, 'Inherited by'));
    data.loaders.forEach(l => s.append(chipLink('loader:'+l)));
    panel.append(s);
    panel.append(el('div', {class:'sec'}, el('h4', {}, 'Shared columns'), colTable(data.columns)));
    if (data.patterns?.length) panel.append(el('div', {class:'sec'}, el('h4', {}, 'Per-axis columns'),
      colTable(data.patterns.map(p => ({name:p.pattern, dtype:p.dtype, description:p.description})))));
  } else if (kind === 'loader') {
    if (data.description) panel.append(el('p', {}, data.description));
    if (data.extensions?.length) {
      const e = el('div', {class:'sec'}, el('h4', {}, 'Extensions'));
      data.extensions.forEach(x => e.append(el('span', {class:'chip'}, x)));
      panel.append(e);
    }
    if (rasterSchema) {
      const req = rasterSchema.columns.filter(c => c.required);
      if (req.length) {
        const s = el('div', {class:'sec'}, el('h4', {}, 'Required columns'), colTable(req));
        s.append(el('p', {style:'font-size:.8rem;color:var(--muted);margin:.4rem 0 0'}, 'Loaders may also add additional image metadata fields (e.g. channel_names, dim_names).'));
        panel.append(s);
      }
    }
    if (data.extra_columns?.length)
      panel.append(el('div', {class:'sec'}, el('h4', {}, 'Loader-specific columns'), colTable(data.extra_columns)));
  } else if (kind === 'processor') {
    if (data.description) panel.append(el('p', {}, data.description));
    const inp = data.input || {}; const bits = [];
    if (inp.axes?.length) bits.push('axes ' + inp.axes.join(','));
    if (inp.kinds?.length) bits.push('kinds ' + inp.kinds.join(','));
    if (inp.capabilities?.length) bits.push('caps ' + inp.capabilities.join(','));
    panel.append(el('p', {class:'kind'}, `input: ${bits.join(' · ') || 'any'} → output: ${data.output||''}`));
    panel.append(el('div', {class:'sec'}, el('h4', {}, 'Creates columns'), colTable(data.columns)));
  } else if (kind === 'widget') {
    if (data.info) panel.append(el('p', {}, (data.info.split('\n')[0]||'').replace(/[*_`#]/g,'')));
    const meta = el('div', {class:'sec'});
    if (data.group) meta.append(el('span', {class:'chip'}, data.group));
    if (data.scope) meta.append(el('span', {class:'chip'}, data.scope));
    panel.append(meta);
    const reqTokens = data.required_inputs || [];
    if (reqTokens.length) {
      const byName = Object.fromEntries(CAT.columns.map(c => [c.name, c]));
      const reqCols = reqTokens.map(t => byName[t] || {name: t, dtype: '', description: ''});
      panel.append(el('div', {class:'sec'}, el('h4', {}, 'Required column' + (reqCols.length > 1 ? 's' : '')), colTable(reqCols)));
    }
    const reqSet = new Set(reqTokens);
    const cols = (widgetCols[id] || []).filter(c => !reqSet.has(c.name));
    if (cols.length) panel.append(el('div', {class:'sec'}, el('h4', {}, 'Columns it uses'), colTable(cols)));
  }
  connList(panel, id);
}

// ---- go --------------------------------------------------------------------
function relayout() { layoutEdges(); highlight(selected); }
window.addEventListener('resize', relayout);
window.addEventListener('load', relayout);
requestAnimationFrame(relayout);
</script>
</body>
</html>
"""


_SOURCE_LABELS = {"file": "File system", "agg": "Aggregation", "image": "Loaders", "derived": "Pipeline"}


def _source_label(producer: str) -> str:
    """Friendly name for the producer that creates a column."""
    if producer.startswith("processor:"):
        return producer.split(":", 1)[1]
    return _SOURCE_LABELS.get(producer, producer)


def _cell(text: str) -> str:
    return str(text or "").replace("|", r"\|").replace("\n", " ")


def render_columns_table(catalog: dict) -> str:
    """Markdown table of every table column and where it comes from."""
    rows = [
        "| Column | Type | Description | Source | Created by | Package |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for col in catalog["columns"]:
        dtype = f"`{col['dtype']}`" if col.get("dtype") else ""
        rows.append(
            f"| `{col['name']}` | {dtype} | {_cell(col.get('description'))} "
            f"| {_source_label(col['producer'])} "
            f"| {_cell(', '.join(col.get('creators', [])))} "
            f"| {_cell(', '.join(col.get('packages', [])))} |"
        )
    return "\n".join(rows)


def _write_columns_block(table: str) -> None:
    """Replace the generated columns table in schema.md, in place."""
    md = PAGE.read_text(encoding="utf-8")
    if COLUMNS_BEGIN not in md or COLUMNS_END not in md:
        raise RuntimeError(
            f"Marker comments not found in {PAGE}. "
            f"Expected '{COLUMNS_BEGIN}' and '{COLUMNS_END}'."
        )
    start, end = md.index(COLUMNS_BEGIN), md.index(COLUMNS_END) + len(COLUMNS_END)
    block = f"{COLUMNS_BEGIN}\n{table}\n{COLUMNS_END}"
    PAGE.write_text(md[:start] + block + md[end:], encoding="utf-8")


def main() -> None:
    catalog = build_catalog()
    if not catalog["widgets"]:
        print("WARNING: Node not found or no widgets loaded — widget nodes omitted from schema map.")
    HTML_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    ROW_SCHEMA_OUT.write_text(json.dumps(render_json_schema(catalog), indent=2), encoding="utf-8")
    HTML_OUT.write_text(_TEMPLATE.replace("__CATALOG_JSON__", json.dumps(catalog)), encoding="utf-8")
    _write_columns_block(render_columns_table(catalog))
    print(f"Wrote {JSON_OUT.name}, {ROW_SCHEMA_OUT.name}, {HTML_OUT.name} and the columns table in {PAGE.name} "
          f"({len(catalog['columns'])} columns, {len(catalog['processors'])} processors, "
          f"{len(catalog.get('widgets', []))} widgets).")


if __name__ == "__main__":
    main()
