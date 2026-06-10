/**
 * Data "resolution" a widget operates on - what one datapoint in the
 * widget represents. Purely descriptive metadata, used to render a small
 * badge with an explanatory tooltip on each widget card.
 */
export const SCOPES = {
  file: {
    label: 'per file',
    icon: '📄',
    color: '#6f42c1',
    desc: 'Each datapoint here is a file.',
  },
  image: {
    label: 'per image',
    icon: '🖼️',
    color: '#0d6efd',
    desc: 'Each datapoint here is an image. A file containing multiple images (e.g. a stack or container) contributes one datapoint per image.',
  },
  slice: {
    label: 'per slice',
    icon: '🧩',
    color: '#fd7e14',
    desc: 'Each datapoint here is a slice within an image (e.g. a channel, Z-plane, or timepoint).',
  },
};

/** Inline badge shown in a widget card header, or '' if scope is unset/unknown. */
export function scopeBadgeHtml(scope) {
  const s = SCOPES[scope];
  if (!s) return '';
  return `<span class="widget-scope-badge" style="--scope-color:${s.color}" title="${s.desc}">${s.icon} ${s.label}</span>`;
}
