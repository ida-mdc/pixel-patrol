import { WIDGET_CONTAINER_ID, ID_WELCOME_SCREEN, ID_MAIN_APP, ID_LOADING_OVERLAY, ID_SIDEBAR_BACKDROP } from './constants.js';

const FILTER_OP_LABEL = {
  contains: 'contains',
  not_contains: 'does not contain',
  eq: '=',
  gt: '>',
  ge: '≥',
  lt: '<',
  le: '≤',
  in: 'in',
};

export function buildFrozenSidebarPayload(state, schema, plugins) {
  const dims = Object.entries(state.dimensions ?? {})
    .map(([letter, idx]) => `${letter.toUpperCase()}=${idx}`)
    .join(', ');
  const { col, op, val } = state.filter ?? {};
  let filterLine = null;
  if (col && op && val !== '') {
    const ol = FILTER_OP_LABEL[op] ?? op;
    filterLine = `${col} ${ol} ${val}`;
  }

  const visible = plugins.filter(p => {
    try {
      return p.requires(schema) && !state.hiddenWidgets.has(p.id);
    } catch {
      return false;
    }
  });

  return {
    palette:        state.palette,
    groupBy:        state.groupCol,
    dimensionsLine: dims || null,
    filterLine,
    significance:   !!state.showSignificance,
    widgetsLine:    visible.map(p => p.label).join(', ') || '(none)',
  };
}

export function formatFrozenSidebarHtml(frozen) {
  const rows = [
    ['Palette',     frozen.palette],
    ['Group by',    frozen.groupBy ?? 'None'],
    ['Dimensions',  frozen.dimensionsLine ?? 'All'],
    ['Row filter',  frozen.filterLine ?? 'None'],
    ['Significance', frozen.significance ? 'On' : 'Off'],
    ['Widgets',     frozen.widgetsLine],
  ];
  return rows.map(([k, v]) => `<div class="mb-1"><span class="text-muted">${k}:</span> ${escapeHtml(String(v))}</div>`).join('');
}

function escapeHtml(s) {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── Baked static HTML export ──────────────────────────────────────────────────

async function fetchText(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url}: ${res.status}`);
  return res.text();
}

async function fetchBase64(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url}: ${res.status}`);
  const buf = await res.arrayBuffer();
  const bytes = new Uint8Array(buf);
  let bin = '';
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return { b64: btoa(bin), mime: res.headers.get('content-type') || 'application/octet-stream' };
}

function guessMime(url) {
  const ext = url.split('?')[0].split('.').pop().toLowerCase();
  return { woff2: 'font/woff2', woff: 'font/woff', ttf: 'font/ttf',
           eot: 'application/vnd.ms-fontobject', svg: 'image/svg+xml',
           png: 'image/png', jpg: 'image/jpeg', jpeg: 'image/jpeg',
           gif: 'image/gif', webp: 'image/webp' }[ext] || 'application/octet-stream';
}

async function inlineFontsInCss(css, cssUrl) {
  const base = new URL(cssUrl).href;
  const re = /url\((['"]?)([^)'"]+)\1\)/g;
  const matches = [];
  let m;
  while ((m = re.exec(css)) !== null) {
    if (!m[2].startsWith('data:')) matches.push({ match: m[0], raw: m[2] });
  }
  for (const { match, raw } of matches) {
    try {
      const abs = new URL(raw, base).href;
      const { b64 } = await fetchBase64(abs);
      css = css.replace(match, `url("data:${guessMime(abs)};base64,${b64}")`);
    } catch { /* leave as-is */ }
  }
  return css;
}

/**
 * Capture the current rendered page as a fully self-contained static HTML file.
 * Inlines all CSS (including fonts), images, and canvas content as data URIs.
 * Strips interactive scripts; shows a frozen settings summary in the sidebar.
 *
 * @param {object} state
 * @param {object} schema
 * @param {object[]} plugins
 * @returns {Promise<string>}
 */
export async function exportBakedHtml(state, schema, plugins) {
  // DOMParser base is about:blank — resolve all URLs against the live page instead.
  const pageBase = window.location.href;

  // Canvas pixel data is not in outerHTML — capture from live DOM first.
  const liveCanvasDataUrls = [...document.querySelectorAll('canvas')].map(c => {
    try { return c.toDataURL(); } catch { return null; }
  });

  const doc = new DOMParser().parseFromString(document.documentElement.outerHTML, 'text/html');

  // Replace <canvas> with <img> data URIs (order matches live DOM)
  [...doc.querySelectorAll('canvas')].forEach((canvas, i) => {
    const dataUrl = liveCanvasDataUrls[i];
    if (!dataUrl) return;
    const img = doc.createElement('img');
    img.src = dataUrl;
    img.setAttribute('style', canvas.getAttribute('style') || '');
    img.style.imageRendering = 'pixelated';
    img.style.maxWidth = '100%';
    if (canvas.width)  img.width  = canvas.width;
    if (canvas.height) img.height = canvas.height;
    canvas.replaceWith(img);
  });

  // Inline stylesheets
  for (const link of [...doc.querySelectorAll('link[rel="stylesheet"]')]) {
    const rawHref = link.getAttribute('href');
    if (!rawHref) continue;
    try {
      const absHref = new URL(rawHref, pageBase).href;
      const style = doc.createElement('style');
      style.textContent = await inlineFontsInCss(await fetchText(absHref), absHref);
      link.replaceWith(style);
    } catch { /* leave external link */ }
  }

  // Inline images
  for (const img of [...doc.querySelectorAll('img[src]')]) {
    const rawSrc = img.getAttribute('src');
    if (!rawSrc || rawSrc.startsWith('data:')) continue;
    try {
      const { b64, mime } = await fetchBase64(new URL(rawSrc, pageBase).href);
      img.setAttribute('src', `data:${mime};base64,${b64}`);
    } catch { /* leave as-is */ }
  }

  // Strip scripts and non-content chrome
  doc.querySelectorAll('script[type="module"], script[src]').forEach(s => s.remove());
  [ID_WELCOME_SCREEN, ID_LOADING_OVERLAY, ID_SIDEBAR_BACKDROP].forEach(id => doc.getElementById(id)?.remove());
  doc.querySelectorAll('input[type="file"]').forEach(el => el.remove());

  // Ensure main app is visible
  const mainApp = doc.getElementById(ID_MAIN_APP);
  if (mainApp) mainApp.style.display = 'flex';

  // Populate and show the frozen settings banner
  const banner     = doc.getElementById('sidebar-frozen-banner');
  const bannerBody = doc.getElementById('sidebar-frozen-body');
  if (banner && bannerBody) {
    banner.classList.remove('d-none');
    bannerBody.innerHTML = formatFrozenSidebarHtml(buildFrozenSidebarPayload(state, schema, plugins));
  }

  // Replace sidebar with only the frozen banner
  const sidebarCardBody = doc.querySelector('#sidebar .card-body');
  if (sidebarCardBody && banner) {
    sidebarCardBody.innerHTML = '';
    sidebarCardBody.appendChild(banner);
  }

  // Disable interactive controls inside widgets
  doc.querySelectorAll(`#${WIDGET_CONTAINER_ID} select, #${WIDGET_CONTAINER_ID} input, #${WIDGET_CONTAINER_ID} button`)
    .forEach(el => { el.disabled = true; });

  // Fix Plotly SVG stacking and hide modebar
  const staticFixes = doc.createElement('style');
  staticFixes.textContent = [
    '.js-plotly-plot .svg-container { position: relative !important; }',
    '.js-plotly-plot .svg-container > svg { position: absolute !important; top: 0 !important; left: 0 !important; }',
    '.modebar { display: none !important; }',
  ].join('\n');
  doc.head.appendChild(staticFixes);

  doc.title = `Pixel Patrol Snapshot – ${new Date().toLocaleDateString()}`;

  return '<!DOCTYPE html>\n' + doc.documentElement.outerHTML;
}
