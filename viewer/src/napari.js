/**
 * napari point-inspector contributor.
 *
 * Registered into the point-inspector drawer (see registry.registerInspector).
 * Only available when the viewer is served from Python (window.__PP_SERVER),
 * where the backend can service /api/open-napari - which reuses the pipeline's
 * loader to open the exact slice/tile/chunk the clicked row represents.
 *
 * This is the reference "hook something into the point detail" plugin: an
 * external viewer extension could register an equivalent contributor via
 * window.PixelPatrol.registerInspector(...).
 */

import { SERVER_MODE } from './query.js';

/** True when the backend can service /api/open-napari (Python-launched viewer). */
export const NAPARI_ENABLED = SERVER_MODE;

let toastEl = null;

/** Show a brief, self-dismissing status message in the corner. */
function toast(message, isError = false) {
  if (!toastEl) {
    toastEl = document.createElement('div');
    toastEl.style.cssText =
      'position:fixed;bottom:20px;right:20px;z-index:9999;max-width:340px;' +
      'padding:10px 14px;border-radius:6px;font-size:13px;color:#fff;' +
      'box-shadow:0 2px 8px rgba(0,0,0,0.25);transition:opacity 0.3s;';
    document.body.appendChild(toastEl);
  }
  toastEl.style.background = isError ? '#b23b3b' : '#2f7d4f';
  toastEl.textContent = message;
  toastEl.style.opacity = '1';
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => { toastEl.style.opacity = '0'; }, isError ? 6000 : 3500);
}

/** Ask the server to open the given row's image (or slice) in napari. */
async function openInNapari(fileRowNumber) {
  toast('Opening in napari…');
  try {
    const res = await fetch('/api/open-napari', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file_row_number: fileRowNumber }),
    });
    if (!res.ok) {
      toast(`napari: ${(await res.text()) || res.statusText}`, true);
      return;
    }
    toast('napari window opening…');
  } catch (err) {
    toast(`napari: ${err.message}`, true);
  }
}

/** The point-inspector contributor object. */
export const napariInspector = {
  id: 'napari',
  label: 'napari',
  order: 20,  // below the data table (thumbnail uses a negative order to sit on top)
  requires: (row) => NAPARI_ENABLED && row.file_row_number != null && row.type !== 'folder',
  render(container, row) {
    const btn = document.createElement('button');
    btn.textContent = 'Open in napari';
    btn.style.cssText =
      'padding:6px 12px;border:1px solid rgba(128,128,128,0.4);border-radius:6px;' +
      'cursor:pointer;background:var(--card-bg,#fff);color:inherit;font-size:13px;';
    btn.addEventListener('click', () => openInNapari(Number(row.file_row_number)));
    container.appendChild(btn);
  },
};
