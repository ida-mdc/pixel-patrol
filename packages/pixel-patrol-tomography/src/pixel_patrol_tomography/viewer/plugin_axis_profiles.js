/**
 * Widget: row & column profiles (x_profile_mean / y_profile_mean from the axis-profile processor).
 *
 * A profile is the mean intensity of every column (X) or row (Y) of an image, averaged over all other
 * dimensions. Dead or hot sensor lines show up as dips or spikes. Up to MAX_PROFILE_LINES images are drawn as one
 * line each; above that, DuckDB reduces all profiles to a mean ± std band per group, so only the band is fetched.
 * The parked border check lives in flagged_borders.js.
 */

const PROFILE_AXES = [
  { axis: 'X', column: 'x_profile_mean', title: 'Column profile (X)', lineName: 'column' },
  { axis: 'Y', column: 'y_profile_mean', title: 'Row profile (Y)', lineName: 'row' },
];
const MAX_PROFILE_LINES = 50;
const MAX_MINI_LINES = 20;
const MAX_IMAGE_OPTIONS = 500;

const RELATIVE_ID = 'axprof-relative-cb';
const IMAGE_SEL_ID = 'axprof-image-select';

// The viewer re-renders every widget on each grouping/filter change; the controls are restored from here.
const controlState = { imageKey: '', relative: true };

export default {
  id: 'axis-profiles',
  label: 'Row & Column Profiles',
  shortLabel: 'Row/Col Profiles',
  group: 'Dataset Stats',
  scope: 'image',
  info: [
    'Mean intensity of every column (X) and row (Y) of each image, averaged over all other dimensions',
    '(e.g. all tilts of a tilt series). Dead sensor lines show up as dips, hot lines as spikes.',
    '',
    `Up to ${MAX_PROFILE_LINES} images: one line per image. More images: the mean of all images per group, `
      + 'with a band of ± one standard deviation. Pick an image to see it alone.',
    '',
    '**Relative to image median** – divides each profile by the median of that profile: '
      + '1 = typical line, 0 = dead line.',
    'Images with a median ≤ 0 (e.g. reconstructions centred on zero) are left out of relative values.',
    '',
    'A single dead pixel is averaged over a whole line and stays invisible; '
      + 'this view finds dead or hot lines and edges.',
  ].join('\n'),

  requires(schema) {
    return PROFILE_AXES.every(({ column }) => schema.allCols.includes(column));
  },

  async overviewPlot(container, ctx) {
    const rows = await fetchProfileRows(ctx, '', MAX_MINI_LINES);
    const traces = buildProfileTraces(ctx, rows, PROFILE_AXES[0], { relative: true, mini: true });
    if (!traces.length) return false;
    ctx.plot.appendMini(container, traces, {
      xaxis: { zeroline: false },
      yaxis: { title: 'col ÷ median' },
      shapes: [buildReferenceLine()],
    });
    return true;
  },

  async render(container, ctx) {
    const imageOptions = await fetchImageOptions(ctx);
    container.appendChild(buildControls(ctx, imageOptions, getShownImageKey(imageOptions)));
    const plots = container.appendChild(document.createElement('div'));
    const draw = async () => {
      const imageKey = getShownImageKey(imageOptions);
      const relative = controlState.relative;
      const counts = await fetchImageCounts(ctx, imageKey);
      const showBands = counts.total > MAX_PROFILE_LINES;
      const profiles = showBands
        ? await fetchProfileBands(ctx, relative)
        : await fetchProfileRows(ctx, imageKey, MAX_PROFILE_LINES);
      plots.querySelectorAll('.js-plotly-plot').forEach(el => Plotly.purge(el));
      plots.replaceChildren();
      renderProfiles(plots, ctx, { counts, showBands, profiles }, { relative, imageKey });
    };
    wireControls(container, draw);
    await draw();
  },
};

async function fetchImageOptions(ctx) {
  return ctx.queryRows(`
    SELECT ${buildImageKeySql(ctx)} AS key, ${buildImageLabelSql(ctx)} AS label
    FROM pp_data ${buildProfileWhere(ctx)}
    ORDER BY label LIMIT ${MAX_IMAGE_OPTIONS}
  `);
}

// A remembered image hidden by the current filter shows as "All images", but stays remembered for later renders.
function getShownImageKey(imageOptions) {
  const isListed = imageOptions.some(option => String(option.key) === controlState.imageKey);
  return isListed ? controlState.imageKey : '';
}

// total: images with profiles; excluded: of those, images with a profile median ≤ 0 (no relative values).
async function fetchImageCounts(ctx, imageKey) {
  const { q } = ctx.sql;
  const positiveMedians = PROFILE_AXES.map(({ column }) => `list_median(${q(column)}) > 0`).join(' AND ');
  const [row] = await ctx.queryRows(`
    SELECT COUNT(*) AS total, COUNT(*) FILTER (WHERE NOT (${positiveMedians})) AS excluded
    FROM pp_data ${buildImageWhere(ctx, imageKey)}
  `);
  return { total: Number(row?.total ?? 0), excluded: Number(row?.excluded ?? 0) };
}

// One row per image (hash-sampled up to `limit`): __group__ (from groupExpr), __label__, and per axis the profile
// values and their median.
async function fetchProfileRows(ctx, imageKey, limit) {
  const { q } = ctx.sql;
  const profileCols = PROFILE_AXES.map(
    ({ axis, column }) => `${q(column)} AS ${axis}_values, list_median(${q(column)}) AS ${axis}_median`,
  );
  const result = await ctx.query(`
    SELECT ${ctx.sql.groupExpr()}, ${buildImageLabelSql(ctx)} AS __label__, ${profileCols.join(', ')}
    FROM pp_data ${buildImageWhere(ctx, imageKey)}
    ORDER BY hash(${buildImageKeySql(ctx)}) LIMIT ${limit}
  `);
  return result.toArray();
}

// Per axis, one row per group and coordinate i: mean and std over all images, and n = number of images with a finite
// value there (images of different size have profiles of different length, so n drops towards the far end).
async function fetchProfileBands(ctx, relative) {
  const { q } = ctx.sql;
  const fetchAxisBand = ({ column }) => ctx.queryRows(`
    WITH profiles AS (
      SELECT ${ctx.sql.groupExpr()}, ${q(column)} AS profile, list_median(${q(column)}) AS median
      FROM pp_data ${buildProfileWhere(ctx)}
    ), points AS (
      SELECT __group__, unnest(range(len(profile))) AS i, unnest(profile) / ${relative ? 'median' : '1'} AS value
      FROM profiles ${relative ? 'WHERE median > 0' : ''}
    )
    SELECT __group__, i, avg(value) AS mean, stddev_samp(value) AS std, count(value) AS n
    FROM points WHERE isfinite(value)
    GROUP BY __group__, i ORDER BY __group__, i
  `);
  const bands = await Promise.all(PROFILE_AXES.map(fetchAxisBand));
  return Object.fromEntries(PROFILE_AXES.map(({ axis }, index) => [axis, bands[index]]));
}

function renderProfiles(plots, ctx, { counts, showBands, profiles }, { relative, imageKey }) {
  const { total, excluded } = counts;
  if (!total) {
    appendNote(plots, 'No row/column profiles in the filtered images.');
    return;
  }
  if (showBands) {
    appendNote(plots, `${total.toLocaleString()} images: showing the mean ± one standard deviation per group.`);
  }
  if (relative && excluded) {
    appendNote(plots, `${excluded} image${excluded === 1 ? ' has' : 's have'} a profile median ≤ 0 and cannot be `
      + 'shown relative to it - switch off "Relative to image median" to include them.');
  }
  const showLegend = !imageKey && ctx.groups.length > 1;
  for (const profileAxis of PROFILE_AXES) {
    const traces = showBands
      ? buildBandTraces(ctx, profiles[profileAxis.axis], profileAxis, { relative, showLegend })
      : buildProfileTraces(ctx, profiles, profileAxis, { relative, showLegend });
    if (!traces.length) continue;
    ctx.plot.append(plots, traces, {
      title: { text: profileAxis.title },
      xaxis: { title: { text: `${profileAxis.lineName} index` }, zeroline: false },
      yaxis: { title: { text: relative ? 'mean ÷ image median' : 'mean intensity' } },
      shapes: relative ? [buildReferenceLine()] : [],
      height: 340,
      showlegend: showLegend,
      ...(showLegend ? { legend: ctx.plot.plotlyLegendConfig } : {}),
    }, 'margin-bottom:16px');
  }
}

function buildProfileTraces(ctx, rows, { axis, lineName }, { relative, showLegend = false, mini = false }) {
  // Legend entries follow trace order: sort the sampled images by group so the legend matches other widgets.
  const getGroupIndex = row => ctx.groups.indexOf(String(row.__group__));
  const sortedRows = [...rows].sort((a, b) => getGroupIndex(a) - getGroupIndex(b));
  const groupsWithLegend = new Set();
  const traces = [];
  for (const row of sortedRows) {
    const median = Number(row[`${axis}_median`]);
    if (relative && !(median > 0)) continue;
    const values = toNumbers(ctx, row[`${axis}_values`]).map(value => (relative ? value / median : value));
    const group = String(row.__group__);
    const label = ctx.plot.escapeHtml(String(row.__label__));
    traces.push({
      type: 'scatter',
      mode: 'lines',
      x: values.map((_, i) => i),
      y: values,
      name: ctx.groupLabel(group),
      legendgroup: group,
      showlegend: showLegend && !groupsWithLegend.has(group),
      opacity: 0.7,
      line: { color: ctx.color.group(group), width: 1 },
      hovertemplate: `${label}<br>${lineName} %{x}: %{y:${relative ? '.3f' : '.4g'}}<extra></extra>`,
      ...(mini ? { hoverinfo: 'skip' } : {}),
    });
    groupsWithLegend.add(group);
  }
  return traces;
}

// Per group: a transparent lower edge, the upper edge filled down to it, and the mean line on top (only it hovers).
function buildBandTraces(ctx, bandRows, { lineName }, { relative, showLegend }) {
  const format = relative ? '.3f' : '.4g';
  const traces = [];
  for (const group of ctx.groups.map(String)) {
    const points = bandRows.filter(row => String(row.__group__) === group);
    if (!points.length) continue;
    const x = points.map(point => Number(point.i));
    const mean = points.map(point => Number(point.mean));
    // stddev_samp is NULL where only one image contributes.
    const std = points.map(point => Number(point.std ?? 0));
    const color = ctx.color.group(group);
    const name = ctx.groupLabel(group);
    const edge = { type: 'scatter', mode: 'lines', x, line: { width: 0, color }, legendgroup: group };
    traces.push({ ...edge, y: mean.map((value, i) => value - std[i]), showlegend: false, hoverinfo: 'skip' });
    traces.push({
      ...edge,
      y: mean.map((value, i) => value + std[i]),
      fill: 'tonexty',
      fillcolor: ctx.color.hexToRgba(color, 0.2),
      showlegend: false,
      hoverinfo: 'skip',
    });
    traces.push({
      type: 'scatter',
      mode: 'lines',
      x,
      y: mean,
      name,
      legendgroup: group,
      showlegend: showLegend,
      line: { color, width: 1.5 },
      customdata: points.map((point, i) => [std[i], Number(point.n)]),
      hovertemplate: `${ctx.plot.escapeHtml(name)}<br>${lineName} %{x}: %{y:${format}} ± %{customdata[0]:${format}}`
        + ' (%{customdata[1]} images)<extra></extra>',
    });
  }
  return traces;
}

function appendNote(container, text) {
  const note = document.createElement('div');
  note.style.cssText = 'font-size:0.88em;color:#555;margin-bottom:8px';
  note.textContent = text;
  container.appendChild(note);
}

function buildReferenceLine() {
  return { type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 1, y1: 1, line: { color: '#999', width: 1, dash: 'dot' } };
}

function buildControls(ctx, imageOptions, shownImageKey) {
  const { escapeHtml } = ctx.plot;
  const optionsHtml = imageOptions
    .map(option => {
      const key = String(option.key);
      const selected = key === shownImageKey ? ' selected' : '';
      return `<option value="${escapeHtml(key)}"${selected}>${escapeHtml(String(option.label))}</option>`;
    })
    .join('');
  const truncationNote = imageOptions.length === MAX_IMAGE_OPTIONS
    ? `<small class="text-muted">Showing first ${MAX_IMAGE_OPTIONS} images.</small>`
    : '';
  const el = document.createElement('div');
  el.style.cssText = 'display:flex;flex-wrap:wrap;gap:20px;align-items:flex-end;margin-bottom:16px';
  el.innerHTML = `
    <div style="max-width:400px;flex:1 1 240px">
      <div style="font-weight:600;margin-bottom:6px">Image</div>
      <select id="${IMAGE_SEL_ID}" class="form-select form-select-sm">
        <option value="">All images</option>
        ${optionsHtml}
      </select>
      ${truncationNote}
    </div>
    <div class="form-check" style="margin-bottom:4px">
      <input class="form-check-input" type="checkbox" id="${RELATIVE_ID}"${controlState.relative ? ' checked' : ''}>
      <label class="form-check-label" for="${RELATIVE_ID}" style="font-size:0.9em">Relative to image median</label>
    </div>
  `;
  return el;
}

function wireControls(container, onDraw) {
  container.querySelector(`#${IMAGE_SEL_ID}`).addEventListener('change', event => {
    controlState.imageKey = event.target.value;
    onDraw();
  });
  container.querySelector(`#${RELATIVE_ID}`).addEventListener('change', event => {
    controlState.relative = event.target.checked;
    onDraw();
  });
}

function buildProfileWhere(ctx) {
  const { q } = ctx.sql;
  return ctx.sql.andWhere(ctx.where, PROFILE_AXES.map(({ column }) => `${q(column)} IS NOT NULL`).join(' AND '));
}

function buildImageWhere(ctx, imageKey) {
  const where = buildProfileWhere(ctx);
  if (!imageKey) return where;
  return ctx.sql.andWhere(where, `${buildImageKeySql(ctx)} = '${imageKey.replace(/'/g, "''")}'`);
}

// Unique key per image: path + child_id when present, path alone otherwise (same as the histogram widget).
function buildImageKeySql(ctx) {
  return ctx.schema.allCols.includes('child_id')
    ? `("path" || COALESCE(' • ' || "child_id", ''))`
    : '"path"';
}

function buildImageLabelSql(ctx) {
  const base = ctx.schema.allCols.includes('name') ? '"name"' : '"path"';
  return ctx.schema.allCols.includes('child_id')
    ? `(${base} || COALESCE(' • ' || "child_id", ''))`
    : base;
}

function toNumbers(ctx, value) {
  const arr = ctx.data.extractBinary(value);
  return arr ? Array.from(arr, Number) : [];
}
