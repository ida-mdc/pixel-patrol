const LUMA = [0.299, 0.587, 0.114];
const CHANNELS = [0, 1, 2];
const MAX_SCATTER_POINTS = 2000;
const WHEEL_PIXELS = 260;
const WHEEL_VALUE = 0.85;
const WHEEL_HEIGHT = 340;
const LABEL_RADIUS = 1.08;
const AXIS_PADDING = 1.16;
const HUE_TARGETS = [['red', 0], ['yellow', 60], ['green', 120], ['cyan', 180], ['blue', 240], ['magenta', 300]];
const WHEEL_SWATCH_LUMA = 0.6;
const NEUTRAL_CAST = 0.05;
const STRONG_CAST = 0.15;
const RING_SATURATIONS = [NEUTRAL_CAST, STRONG_CAST, 0.5, 1];
const LOW_SATURATION = 0.1;
const ROUNDING_DIGITS = 9;
const MIXED_HUES = 0.5;

const HUE_NAMES = ['red', 'orange', 'yellow', 'yellow-green', 'green', 'cyan-green',
  'cyan', 'azure', 'blue', 'violet', 'magenta', 'rose'];


// ---- colour maths (pure) ----

export function hueName(degrees) {
  return HUE_NAMES[Math.round((((degrees % 360) + 360) % 360) / 30) % HUE_NAMES.length];
}

export function hueDegrees(means) {
  const grey = (means[0] + means[1] + means[2]) / 3;
  if (!(grey > 0)) return null;
  const [dr, dg, db] = means.map(m => m - grey);
  return ((Math.atan2(Math.sqrt(3) * (dg - db), 2 * dr - dg - db) * 180 / Math.PI) + 360) % 360;
}

const finite = (values) => values.filter(Number.isFinite).sort((a, b) => a - b);

export function quantile(values, p) {
  const sorted = finite(values);
  if (!sorted.length) return null;
  const at = p * (sorted.length - 1);
  const lo = Math.floor(at);
  return sorted[lo] + (sorted[Math.min(lo + 1, sorted.length - 1)] - sorted[lo]) * (at - lo);
}



// ---- colour space (pure) ----

export function hsvSaturation(rgb) {
  const high = Math.max(...rgb);
  return high > 0 ? (high - Math.min(...rgb)) / high : 0;
}

export function hsvToRgb(hue, saturation, value) {
  const channel = (n) => {
    const k = (n + hue / 60) % 6;
    return value * (1 - saturation * Math.max(0, Math.min(k, 4 - k, 1)));
  };
  return [channel(5), channel(3), channel(1)];
}

// The wheel is a hue/saturation wheel: the angle is the hue and the radius the saturation of the average colour.
// The radius grows with the square root of the saturation, so weak casts stay clear of the centre while the
// full range, out to pure colours, is still on show.
export function wheelRadius(saturation) {
  return Math.sqrt(Math.min(1, Math.max(0, saturation)));
}

export function wheelPoint(rgb) {
  if (!rgb) return null;
  const hue = hueDegrees(rgb);
  const saturation = hsvSaturation(rgb);
  if (hue === null) return saturation === 0 ? { x: 0, y: 0, hue: 0, saturation } : null;
  const radius = wheelRadius(saturation);
  const angle = (hue * Math.PI) / 180;
  return { x: radius * Math.cos(angle), y: radius * Math.sin(angle), hue, saturation };
}

function wheelImage() {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = WHEEL_PIXELS;
  const context = canvas.getContext?.('2d');
  if (!context) return null;
  const image = context.createImageData(WHEEL_PIXELS, WHEEL_PIXELS);
  for (let row = 0; row < WHEEL_PIXELS; row++) {
    for (let col = 0; col < WHEEL_PIXELS; col++) {
      const x = -1 + (2 * (col + 0.5)) / WHEEL_PIXELS;
      const y = 1 - (2 * (row + 0.5)) / WHEEL_PIXELS;
      const radius = Math.hypot(x, y);
      const hue = ((Math.atan2(y, x) * 180) / Math.PI + 360) % 360;
      const rgb = hsvToRgb(hue, Math.min(1, radius * radius), WHEEL_VALUE).map(c => Math.round(255 * c));
      image.data.set([...rgb, radius <= 1 ? 255 : 0], 4 * (row * WHEEL_PIXELS + col));
    }
  }
  context.putImageData(image, 0, 0);
  return canvas.toDataURL();
}


// ---- assessment (pure) ----

const median = (images, id) => quantile(images.map(i => i[id]), 0.5);

// Where a group's average colour sits on the wheel: the median position of its images. The table reads its
// verdict from the same point the diamond is drawn at.
export function groupCentre(images) {
  const points = wheelPoints(images);
  if (!points.length) return null;
  const x = quantile(points.map(p => p.x), 0.5);
  const y = quantile(points.map(p => p.y), 0.5);
  return { x, y, saturation: Math.hypot(x, y) ** 2, hue: ((Math.atan2(y, x) * 180) / Math.PI + 360) % 360 };
}

const CAST_SCALE = `neutral below ${NEUTRAL_CAST}, slight below ${STRONG_CAST}, strong above`;
const SHARED_SHARE = 0.75;
const BALANCE_ORDER = ['neutral', 'slight cast', 'strong cast', 'mixed hues'];
const SATURATION_WORDS = [[LOW_SATURATION, 'grey'], [0.3, 'muted'], [0.6, 'moderate'], [Infinity, 'vivid']];
const SATURATION_ORDER = SATURATION_WORDS.map(([, word]) => word);
const SATURATION_SCALE = `grey below ${LOW_SATURATION}, muted below 0.3, moderate below 0.6, vivid above`;

// Every image is labelled on its own and a group is described by what most of its images are. When no label
// covers enough of them the cell lists the counts, because a median would hide the mix.
function describeGroup(labels, order) {
  const counts = order.map(label => [label, labels.filter(l => l === label).length]).filter(([, n]) => n);
  const [dominant, top] = counts.reduce((best, entry) => (entry[1] > best[1] ? entry : best));
  return { counts, dominant, shared: top / labels.length >= SHARED_SHARE };
}

const countList = (counts) => counts.map(([label, n]) => `${n} ${label}`).join(', ');

function imageBalance(image) {
  const point = wheelPoint(image.rgb);
  if (!point) return null;
  if (Number.isFinite(image.coherence) && image.coherence < MIXED_HUES && image.saturation >= LOW_SATURATION) return 'mixed hues';
  if (point.saturation < NEUTRAL_CAST) return 'neutral';
  return point.saturation < STRONG_CAST ? 'slight cast' : 'strong cast';
}

function balanceText(kind, centre) {
  const hue = hueName(centre.hue);
  if (kind === 'neutral') return 'neutral';
  if (kind === 'mixed hues') return `mixed hues, average ${centre.saturation < NEUTRAL_CAST ? 'is neutral' : `leans ${hue}`}`;
  return `${kind === 'slight cast' ? 'slight' : 'strong'} ${hue} cast`;
}

function balanceAssessment(images) {
  const labelled = images.map(image => [image, imageBalance(image)]).filter(([, kind]) => kind);
  if (!labelled.length) return null;
  const { counts, dominant, shared } = describeGroup(labelled.map(([, kind]) => kind), BALANCE_ORDER);
  const rule = `Each image is judged on its own: ${CAST_SCALE}; mixed hues when its coloured pixels disagree on their hue. The small dots on the colour wheel are these images.`;
  if (!shared) return { text: `varies: ${countList(counts)}`, varied: true, parts: counts, hint: `The images of this group differ in colour. ${rule}` };
  const centre = groupCentre(labelled.filter(([, kind]) => kind === dominant).map(([image]) => image));
  const text = balanceText(dominant, centre);
  const where = `Their average colour sits at saturation ${centre.saturation.toFixed(2)} towards ${hueName(centre.hue)}.`;
  return { text: counts.length > 1 ? `mostly ${text}` : text, hint: `${countList(counts)}. ${where} ${rule}` };
}

const saturationWord = (value) => SATURATION_WORDS.find(([limit]) => value < limit)[1];

function saturationAssessment(images) {
  const values = images.map(i => i.saturation).filter(Number.isFinite);
  if (!values.length) return null;
  const { counts, dominant, shared } = describeGroup(values.map(saturationWord), SATURATION_ORDER);
  const hint = `Saturation of the pixels of each image (0 is grey, 1 is a pure colour): ${SATURATION_SCALE}. ${countList(counts)}.`;
  if (!shared) return { text: `varies: ${countList(counts)}`, varied: true, parts: counts, hint };
  return { text: counts.length > 1 ? `mostly ${dominant}` : dominant, hint };
}

export function colourAssessment(images) {
  return { balance: balanceAssessment(images), saturation: saturationAssessment(images) };
}

// ---- data ----

const hasColumn = (ctx, column) => ctx.schema.allCols.includes(column);

const CHANNEL_NAMES = ['r', 'g', 'b'];
const PLOT_COLUMNS = { brightness: 'colour_brightness', saturation: 'colour_saturation', cast: 'colour_cast' };

const channelMeansSql = () => CHANNEL_NAMES.map((name, i) => `MAX(CASE WHEN "dim_s" = ${i} THEN "mean_intensity" END) AS "ch_${name}_mean"`).join(', ');

// One row per RGB image: the channel means pivoted out of the slice rows next to the processor's columns, plus the
// measures the plots draw. The table, the wheel and the plots all read this, so they cannot disagree. Rounding
// leaves 1e-16 where a value is really 0, which would stretch a plot axis.
export function imageTable(ctx) {
  const { andWhere, dimSubsetWhere } = ctx.sql;
  const pinned = Object.fromEntries(
    Object.entries(ctx.state.dimensions ?? {}).map(([letter, index]) => [letter, Number(index)]).filter(([, index]) => Number.isFinite(index)));
  const slices = dimSubsetWhere({ fixed: pinned, split: new Set(['s']) });
  const channels = CHANNEL_NAMES.map(name => `"ch_${name}_mean"`);
  const peak = `GREATEST(${channels.join(', ')})`;
  const lowest = `LEAST(${channels.join(', ')})`;
  const averageSaturation = `(${peak} - ${lowest}) / NULLIF(${peak}, 0)`;
  const complete = channels.map(c => `${c} IS NOT NULL`).join(' AND ');
  return `(SELECT *,
      ${channels.map((c, i) => `${LUMA[i]} * ${c}`).join(' + ')} AS ${PLOT_COLUMNS.brightness},
      ROUND("rgb_saturation_mean", ${ROUNDING_DIGITS}) AS ${PLOT_COLUMNS.saturation},
      ROUND(${averageSaturation}, ${ROUNDING_DIGITS}) AS ${PLOT_COLUMNS.cast}
    FROM (SELECT d.*, ch.* EXCLUDE (ch_path) FROM pp_data d
      JOIN (SELECT "path" AS ch_path, ${channelMeansSql()} FROM pp_all ${slices.length ? `WHERE ${slices.join(' AND ')}` : ''} GROUP BY "path") ch
      ON ch.ch_path = d."path") ${andWhere(ctx.where, complete)})`;
}

export function imageFromRow(row) {
  const number = (value) => (value == null ? null : Number(value));
  return {
    group: String(row.__group__), key: String(row.__key__), name: String(row.__name__ ?? row.__key__),
    frn: row.__frn__ == null ? null : Number(row.__frn__),
    rgb: [row.r, row.g, row.b].map(Number),
    brightness: number(row.brightness), contrast: number(row.contrast), saturation: number(row.saturation),
    coherence: number(row.coherence),
  };
}

async function loadImages(ctx) {
  const coherence = hasColumn(ctx, 'rgb_hue_coherence') ? '"rgb_hue_coherence"' : 'NULL';
  const rows = await ctx.queryRows(`
    SELECT ${ctx.sql.groupExpr()}, "path" AS __key__, ${hasColumn(ctx, 'name') ? '"name"' : '"path"'} AS __name__, "file_row_number" AS __frn__, "ch_r_mean" AS r, "ch_g_mean" AS g, "ch_b_mean" AS b,
           ${PLOT_COLUMNS.brightness} AS brightness, "rgb_luma_std" AS contrast,
           ${PLOT_COLUMNS.saturation} AS saturation, ${coherence} AS coherence
    FROM ${imageTable(ctx)}`);
  return rows.map(imageFromRow);
}


// ---- rendering ----

function groupBy(items, key) {
  const out = new Map();
  for (const item of items) out.set(key(item), [...(out.get(key(item)) ?? []), item]);
  return out;
}

// A group whose images differ says so, with how many fall in each class underneath.
function variesCell(parts, escapeHtml) {
  const list = parts.map(([label, count]) => `${count} ${escapeHtml(label)}`).join(' · ');
  return `varies<div style="font-size:0.8rem;color:#6c757d">${list}</div>`;
}

function assessmentCell(assessment, escapeHtml, { swatch = '' } = {}) {
  const content = assessment
    ? `<span title="${escapeHtml(assessment.hint)}">${assessment.varied ? variesCell(assessment.parts, escapeHtml) : `${swatch}${escapeHtml(assessment.text)}`}</span>`
    : '<span style="color:#868e96">n/a</span>';
  return `<td style="padding:6px 12px">${content}</td>`;
}

const dot = (colour, size = 12) =>
  `<span style="display:inline-block;width:${size}px;height:${size}px;border-radius:50%;background:${colour};margin-right:8px;vertical-align:middle;border:1px solid rgba(0,0,0,0.15)"></span>`;

function swatchColour(rgb) {
  const luma = LUMA[0] * rgb[0] + LUMA[1] * rgb[1] + LUMA[2] * rgb[2];
  const [r, g, b] = rgb.map(c => Math.round(255 * Math.min(1, (c / luma) * WHEEL_SWATCH_LUMA)));
  return `rgb(${r},${g},${b})`;
}

function averageColourSwatch(images) {
  const colours = images.map(i => i.rgb).filter(Boolean);
  if (!colours.length) return '';
  const typical = CHANNELS.map(c => quantile(colours.map(rgb => rgb[c]), 0.5));
  return dot(swatchColour(typical), 14);
}

function assessmentTable(ctx, byGroup) {
  const { escapeHtml } = ctx.plot;
  const body = [...byGroup].map(([group, images]) => {
    const found = colourAssessment(images);
    const label = escapeHtml(ctx.groupLabel?.(group) ?? group);
    return `<tr><td style="padding:6px 12px;white-space:nowrap">${dot(ctx.color.group(group), 10)}${label}</td>
      ${assessmentCell(found.balance, escapeHtml, { swatch: averageColourSwatch(images) })}${assessmentCell(found.saturation, escapeHtml)}</tr>`;
  }).join('');
  const table = document.createElement('table');
  table.className = 'table table-sm';
  table.style.cssText = 'font-size:0.9rem;width:100%';
  table.innerHTML = `<thead><tr><th style="padding:6px 12px">Group</th><th style="padding:6px 12px">Average colour</th>
    <th style="padding:6px 12px">Saturation</th></tr></thead><tbody>${body}</tbody>`;
  return table;
}

function sample(items, limit) {
  const step = Math.max(1, Math.ceil(items.length / limit));
  return items.filter((_, i) => i % step === 0);
}

function wheelPoints(images) {
  return images.flatMap(image => {
    const point = wheelPoint(image.rgb);
    return point ? [{ ...point, name: image.name, frn: image.frn }] : [];
  });
}

const satText = (saturation) => `saturation of the average colour ${saturation.toFixed(2)}`;

// Each dot is one image: its file name on hover, and its file_row_number in customdata, which makes it click-to-inspect
// and rings it in every linked plot.
function wheelTraces(ctx, byGroup) {
  const colour = (group) => ctx.color.group(group);
  const name = (group) => ctx.groupLabel?.(group) ?? group;
  const clouds = [...byGroup].map(([group, images]) => {
    const shown = sample(wheelPoints(images), MAX_SCATTER_POINTS);
    return { type: 'scatter', mode: 'markers', name: name(group), legendgroup: group, showlegend: false,
      x: shown.map(p => p.x), y: shown.map(p => p.y), customdata: shown.map(p => p.frn),
      text: shown.map(p => `<b>${p.name}</b><br>${satText(p.saturation)}`),
      marker: { size: 8, color: colour(group), opacity: 0.85, line: { width: 1, color: '#fff' } },
      hovertemplate: `%{text}<extra>${name(group)}</extra>` };
  });
  const centres = [...byGroup].map(([group, images]) => {
    const centre = groupCentre(images);
    return { type: 'scatter', mode: 'markers', name: name(group), legendgroup: group,
      x: [centre?.x ?? null], y: [centre?.y ?? null],
      text: [centre ? `median of the group<br>${satText(centre.saturation)}` : ''],
      marker: { size: 16, color: colour(group), symbol: 'diamond', line: { width: 2, color: '#fff' } },
      hovertemplate: `%{text}<extra>${name(group)}</extra>` };
  });
  const neutral = { type: 'scatter', mode: 'markers', x: [0], y: [0], hoverinfo: 'skip', showlegend: false,
    marker: { symbol: 'cross-thin', size: 16, color: '#212529', line: { width: 2, color: '#212529' } } };
  return [...clouds, neutral, ...centres];
}

const hueDirection = ([, degrees]) => ({ cos: Math.cos((degrees * Math.PI) / 180), sin: Math.sin((degrees * Math.PI) / 180) });

function graticule() {
  const rings = RING_SATURATIONS.map(s => {
    const r = wheelRadius(s);
    return { type: 'circle', xref: 'x', yref: 'y', x0: -r, x1: r, y0: -r, y1: r,
      line: { width: 1, color: 'rgba(255,255,255,0.8)', dash: s === 1 ? 'solid' : 'dot' } };
  });
  const spokes = HUE_TARGETS.map(target => ({ type: 'line', xref: 'x', yref: 'y', x0: 0, y0: 0, x1: hueDirection(target).cos, y1: hueDirection(target).sin,
    line: { width: 1, color: 'rgba(255,255,255,0.55)' } }));
  return [...rings, ...spokes];
}

function graticuleLabels() {
  const font = { size: 11, color: '#495057' };
  const hues = HUE_TARGETS.map(target => ({ x: hueDirection(target).cos * LABEL_RADIUS, y: hueDirection(target).sin * LABEL_RADIUS, text: target[0], showarrow: false, font }));
  const rings = RING_SATURATIONS.map(s => ({ x: 0, y: -wheelRadius(s), text: String(s), showarrow: false, xanchor: 'center', yanchor: 'top',
    font: { size: 10, color: '#6c757d' }, bgcolor: 'rgba(255,255,255,0.6)' }));
  return [...hues, ...rings];
}

const HIDDEN_AXIS = { showticklabels: false, ticks: '', zeroline: false, showgrid: false, title: { text: '' } };

// The table next to the wheel carries the group colours, so the wheel needs no legend.
function wheelLayout({ mini = false } = {}) {
  const source = wheelImage();
  return {
    showlegend: false,
    margin: { l: 4, r: 4, t: 4, b: 4 },
    xaxis: { ...HIDDEN_AXIS, range: mini ? [-1, 1] : [-AXIS_PADDING, AXIS_PADDING], constrain: 'domain' },
    yaxis: { ...HIDDEN_AXIS, range: mini ? [-1, 1] : [-AXIS_PADDING, AXIS_PADDING], scaleanchor: 'x', scaleratio: 1 },
    shapes: graticule(),
    annotations: mini ? [] : graticuleLabels(),
    images: source ? [{ source, xref: 'x', yref: 'y', x: -1, y: 1, sizex: 2, sizey: 2,
      xanchor: 'left', yanchor: 'top', sizing: 'stretch', layer: 'below' }] : [],
  };
}

function column(parent, css) {
  const el = document.createElement('div');
  el.style.cssText = css;
  parent.appendChild(el);
  return el;
}

function caption(parent, title, detail) {
  const el = document.createElement('div');
  el.innerHTML = `<div style="font-weight:600;font-size:0.9rem">${title}</div><div style="font-size:0.8rem;color:#6c757d">${detail}</div>`;
  parent.appendChild(el);
}

// [column, title, y axis]: the title names the measure, the y axis says how it is measured.
const PLOTS = [
  [PLOT_COLUMNS.brightness, 'Brightness', 'Brightness (luma)'],
  ['rgb_luma_std', 'Contrast', 'Contrast'],
  [PLOT_COLUMNS.saturation, 'Saturation', 'Saturation'],
  [PLOT_COLUMNS.cast, 'Colour cast', 'Colour cast (saturation of the average colour)'],
];

// The same distribution plots as Intensity and Image Quality: one per measure in a flex grid, drawn by the shared plot engine.
async function drawDistributions(parent, ctx, groups, count) {
  const { flexGrid, groupingLabel, engine } = ctx.plot;
  const perRow = groups.length <= 2 ? 3 : groups.length === 3 ? 2 : 1;
  const { wrap, flexBasisPct } = flexGrid(parent, perRow);
  const table = imageTable(ctx);
  for (const [numCol, label, axis] of PLOTS) {
    const cell = document.createElement('div');
    cell.style.cssText = `flex:0 0 ${flexBasisPct}%;min-width:300px;margin-bottom:20px;box-sizing:border-box`;
    wrap.appendChild(cell);
    await engine.renderDistribution(cell, ctx, {
      numCol,
      source: { table, where: '' },
      catSql: ctx.sql.groupCol(),
      catLabel: groupingLabel(''),
      yLabel: axis,
      title: `Distribution of ${label}<br><sup>one point per image; n=${count.toLocaleString()}</sup>`,
      showSignificance: !!ctx.state.showSignificance,
      maxRawPoints: engine.CONSTANTS.MAX_VIOLIN_POINTS,
      series: { isCategory: true },
      categoriesOrder: groups,
      catLabelFn: ctx.groupLabel,
      layout: { yaxis: { title: axis, rangemode: 'nonnegative' } },
    });
  }
}

// Everything is attached before Plotly draws into it, so the plots are sized to their final width.
async function drawWidget(container, ctx, images) {
  const byGroup = groupBy(images, i => i.group);
  const groups = ctx.groups.filter(g => byGroup.has(g));
  const top = column(container, 'display:flex;flex-wrap:wrap;gap:24px;align-items:center;margin-bottom:20px');
  const wheel = column(top, `flex:0 0 ${WHEEL_HEIGHT}px;max-width:100%`);
  caption(wheel, 'Average colour of each image', 'hue is the angle, saturation the distance from neutral');
  const plot = ctx.plot.append(wheel, wheelTraces(ctx, byGroup), { ...ctx.plot.LAYOUT, height: WHEEL_HEIGHT, ...wheelLayout() });
  ctx.plot.registerPointPlot?.(plot, ctx, PLOT_COLUMNS.cast);
  column(top, 'flex:1 1 480px;min-width:0').appendChild(assessmentTable(ctx, byGroup));
  await drawDistributions(container, ctx, groups, images.length);
}

const NO_COLOUR_MESSAGE = '<div class="no-data">No RGB images found. This widget needs a three-sample colour axis (S) ' +
  'with per-channel statistics.</div>';

const INFO = [
  'Assesses the **colour** of RGB images per group. Nothing here is good or bad: a stain is meant to be coloured.',
  '',
  '- **Table:** each image is judged on its own, then the group is summarised. If at least ¾ of its images agree you see that class ("mostly …" if some differ), otherwise the counts, e.g. "5 grey, 2 vivid". **Average colour** is neutral, a slight or a strong cast by the saturation of the average colour (below 0.05, below 0.15, above), or mixed hues when the coloured pixels disagree on their hue. **Saturation** of the pixels: grey < 0.1, muted < 0.3, moderate < 0.6, vivid.',
  '- **Colour wheel:** the angle is the hue, the distance from the grey centre the saturation of the average colour (square-root scale; rings at the 0.05 and 0.15 limits, 0.5 and 1). Dots are images (hover for the name, click to inspect and highlight it in the plots) and diamonds group medians, in the colours of the table.',
  '- **Plots:** brightness (luma), contrast (spread of luma), saturation, and colour cast (the distance on the wheel).',
].join('\n');

// ---- overview tile ----

export function overviewSummary(byGroup) {
  const images = [...byGroup.values()].reduce((n, list) => n + list.length, 0);
  const lead = `${images.toLocaleString()} RGB ${images === 1 ? 'image' : 'images'}`;
  const verdicts = [...byGroup.values()].map(list => colourAssessment(list).balance).filter(Boolean);
  if (!verdicts.length) return `${lead}.`;
  const varied = verdicts.filter(v => v.varied).length;
  if (varied) return `${lead}, average colour <strong>varies between images</strong> in ${varied === verdicts.length ? 'every group' : `${varied} of ${verdicts.length} groups`}.`;
  const texts = new Set(verdicts.map(v => v.text));
  if (texts.size === 1) return `${lead}, average colour <strong>${verdicts[0].text}</strong> in every group.`;
  return `${lead}, average colour <strong>differs between groups</strong>.`;
}

const colourWidget = {
  id: 'rgb-colour-assessment',
  required_inputs: ['mean_intensity', 'rgb_luma_std', 'rgb_saturation_mean', 'dim_<axis>'],
  inputs: ['rgb_hue_coherence'],
  group: 'Dataset Stats',
  scope: 'image',
  label: 'Colour Assessment',
  shortLabel: 'Colour',
  info: INFO,

  requires(schema) {
    return ['mean_intensity', 'rgb_luma_std', 'rgb_saturation_mean'].every(c => schema.allCols.includes(c)) && (schema.dimCols ?? []).includes('dim_s');
  },

  async overviewMessage(ctx) {
    try {
      const images = await loadImages(ctx);
      return images.length ? overviewSummary(groupBy(images, i => i.group)) : null;
    } catch { return null; }
  },

  async overviewPlot(container, ctx) {
    const images = await loadImages(ctx);
    if (!wheelPoints(images).length) return false;
    ctx.plot.appendMini(container, wheelTraces(ctx, groupBy(images, i => i.group)), wheelLayout({ mini: true }));
    return true;
  },

  async render(container, ctx) {
    const images = await loadImages(ctx);
    if (!images.length) { container.innerHTML = NO_COLOUR_MESSAGE; return; }
    await drawWidget(container, ctx, images);
  },
};

export default [colourWidget];
