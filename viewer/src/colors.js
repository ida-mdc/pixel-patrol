import chroma from 'chroma-js';
import { DEFAULT_PALETTE } from './constants.js';

// ── matplotlib/D3 tab10 & tab20 ───────────────────────────────────────────────
// Same palette as matplotlib's default (tab10/tab20) and D3 schemeCategory10/20.
// Designed for categorical data on white backgrounds: saturated, distinct, no near-white tones.
const TAB10 = [
  '#9467bd', '#ff7f0e','#17becf', '#2ca02c','#8c564b', '#7f7f7f','#bcbd22', '#d62728', '#1f77b4',
  , '#e377c2'
];
const TAB20 = [
  '#9467bd', '#c5b0d5', '#2ca02c', '#98df8a', '#e377c2',
  '#f7b6d2', '#17becf', '#9edae5', '#d62728', '#ff9896',
  '#bcbd22', '#dbdb8d', '#1f77b4', '#aec7e8', '#ff7f0e',
  '#ffbb78', '#8c564b', '#c49c94', '#7f7f7f', '#c7c7c7',
];

const FIXED_PALETTES = { tab10: TAB10, tab20: TAB20 };

function _fixedColors(palette, n) {
  const base = FIXED_PALETTES[palette];
  if (n <= base.length) return base.slice(0, n);
  // Cycle if more groups than palette entries.
  return Array.from({ length: n }, (_, i) => base[i % base.length]);
}

// ── ColorBrewer qualitative palettes ─────────────────────────────────────────
// Only qualitative sets — no sequential or diverging palettes, which include
// near-white tones that are invisible on a white background.
const QUALITATIVE_BREWER = new Set([
  'Accent', 'Dark2', 'Paired', 'Set1', 'Set2', 'Set3',
]);

// ── matplotlib `rainbow` colormap ────────────────────────────────────────────
// Kept for backward compatibility with Dash default (DEFAULT_CMAP = 'rainbow').
const _RAINBOW_R = [[0.000, 0.5], [0.125, 0.0], [0.375, 0.0], [0.625, 1.0], [0.875, 1.0], [1.000, 0.5]];
const _RAINBOW_G = [[0.000, 0.0], [0.125, 0.0], [0.375, 1.0], [0.625, 1.0], [0.875, 0.0], [1.000, 0.0]];
const _RAINBOW_B = [[0.000, 1.0], [0.125, 1.0], [0.375, 0.0], [0.625, 0.0], [0.875, 0.0], [1.000, 0.0]];

function _lerp(pts, t) {
  t = Math.max(0, Math.min(1, t));
  for (let i = 1; i < pts.length; i++) {
    if (t <= pts[i][0]) {
      const [x0, y0] = pts[i - 1];
      const [x1, y1] = pts[i];
      return y0 + ((t - x0) / (x1 - x0)) * (y1 - y0);
    }
  }
  return pts[pts.length - 1][1];
}

function _rainbowHex(t) {
  const r = Math.round(_lerp(_RAINBOW_R, t) * 255);
  const g = Math.round(_lerp(_RAINBOW_G, t) * 255);
  const b = Math.round(_lerp(_RAINBOW_B, t) * 255);
  return `#${r.toString(16).padStart(2,'0')}${g.toString(16).padStart(2,'0')}${b.toString(16).padStart(2,'0')}`;
}

function _rainbowColors(n) {
  if (n === 1) return [_rainbowHex(0.5)];
  return Array.from({ length: n }, (_, i) => _rainbowHex(i / (n - 1)));
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Available palette names — tab10 first (matplotlib/D3 default), then tab20,
 * then ColorBrewer qualitative palettes, then rainbow for backward compatibility.
 */
export function getPaletteNames() {
  const brewer = Object.keys(chroma.brewer)
    .filter(n => QUALITATIVE_BREWER.has(n))
    .sort();
  return ['tab10', 'tab20', ...brewer, 'rainbow'];
}

export function getColors(paletteName, n) {
  if (paletteName in FIXED_PALETTES) return _fixedColors(paletteName, n);
  if (paletteName === 'rainbow') return _rainbowColors(n);
  try {
    return chroma.scale(paletteName).colors(n);
  } catch {
    return _fixedColors(DEFAULT_PALETTE, n);
  }
}

export function buildColorMap(groups, palette = DEFAULT_PALETTE) {
  const colors = getColors(palette, groups.length);
  const map = {};
  groups.forEach((g, i) => { map[String(g)] = colors[i]; });
  return map;
}

export function groupColor(colorMap, g) {
  return colorMap[String(g)] ?? '#888';
}

/** Convert a hex color string to an rgba() CSS value. */
export function hexToRgba(hex, alpha) {
  const [r, g, b] = chroma(hex).rgb();
  return `rgba(${r},${g},${b},${alpha})`;
}
