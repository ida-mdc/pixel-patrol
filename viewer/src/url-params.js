/**
 * URL parameter serialisation / deserialisation for viewer state.
 * Uses history.replaceState — no page reloads, no extra history entries.
 *
 * Supported params:
 *   palette   — color palette name
 *   group     — group-by column
 *   fc/fo/fv  — filter column / operator / value
 *   dims      — active dimensions, e.g. "c0.t1"  (dot-separated, no encoding needed)
 *   sig       — "1" when significance brackets enabled
 *   hidden    — dot-separated hidden widget IDs  (dot-separated, no encoding needed)
 *   extension — URL of a JSON extension manifest (repeatable); manifest format:
 *               { "plugins": ["./a.js", "./b.js"] }  (relative URLs resolved from manifest)
 */

export function writeUrlParams(state) {
  const params = new URLSearchParams(window.location.search);

  setOrDelete(params, 'palette', state.palette !== 'tab10' ? state.palette : null);
  setOrDelete(params, 'group',   state.groupCol || null);

  const { col, op, val } = state.filter ?? {};
  if (col && op && val !== '') {
    params.set('fc', col); params.set('fo', op); params.set('fv', val);
  } else {
    params.delete('fc'); params.delete('fo'); params.delete('fv');
  }

  const dimStr = Object.entries(state.dimensions ?? {}).map(([l, i]) => `${l}${i}`).join('.');
  setOrDelete(params, 'dims',   dimStr || null);
  setOrDelete(params, 'sig',    state.showSignificance ? '1' : null);
  setOrDelete(params, 'hidden', state.hiddenWidgets?.size > 0
    ? [...state.hiddenWidgets].sort().join('.') : null);

  const qs = params.toString();
  history.replaceState(null, '', qs ? `?${qs}` : window.location.pathname);
}

export function readUrlParams() {
  const params = new URLSearchParams(window.location.search);
  const out = {};

  if (params.has('palette')) out.palette = params.get('palette');
  if (params.has('group'))   out.groupCol = params.get('group') || null;

  const fc = params.get('fc'), fo = params.get('fo'), fv = params.get('fv');
  if (fc && fo && fv != null) out.filter = { col: fc, op: fo, val: fv };

  if (params.has('dims')) {
    const dims = {};
    for (const part of params.get('dims').split('.')) {
      const m = /^([a-z])(\d+)$/.exec(part.trim());
      if (m) dims[m[1]] = m[2];
    }
    if (Object.keys(dims).length) out.dimensions = dims;
  }

  if (params.has('sig'))    out.showSignificance = params.get('sig') === '1';
  if (params.has('hidden')) out.hiddenWidgets = new Set(params.get('hidden').split('.').filter(Boolean));

  return out;
}

function setOrDelete(params, key, value) {
  if (value != null) params.set(key, value);
  else params.delete(key);
}
