/**
 * Minimal pub/sub state store.
 *
 * Events:
 *   'render'  — re-render all plugins with current state (palette changed, etc.)
 *   'query'   — full re-query + re-render (filter/groupby/dimensions changed)
 */

export const state = {
  palette:          'tab10',
  groupCol:         null,     // string|null
  filter:           { col: '', op: '', val: '' },
  dimensions:       {},       // { dimLetter: idx }   e.g. { t: '0', c: '1' }
  showSignificance: false,
  hiddenWidgets:    new Set(), // set of plugin IDs to hide
  /** Offline snapshot bundle: sidebar is read-only and URL sync is disabled */
  sidebarLocked:    false,
};

const listeners = {};

export function on(event, fn) {
  (listeners[event] ??= new Set()).add(fn);
}

export function off(event, fn) {
  listeners[event]?.delete(fn);
}

export function emit(event) {
  listeners[event]?.forEach(fn => fn());
}

/** Merge patch into state and optionally fire an event. */
export function setState(patch, event = null) {
  Object.assign(state, patch);
  if (event) emit(event);
}

export function resetState(defaultGroupCol) {
  state.palette          = 'tab10';
  state.groupCol         = defaultGroupCol;
  state.filter           = { col: '', op: '', val: '' };
  state.dimensions       = {};
  state.showSignificance = false;
  state.hiddenWidgets    = new Set();
  state.sidebarLocked    = false;
  emit('query');
}
