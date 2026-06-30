import { buildColorMap, groupColor as _groupColor, hexToRgba, getColors, getPaletteNames } from './colors.js';
import { GROUP_ALL, GROUP_COL_ALIAS, WIDGET_CONTAINER_ID } from './constants.js';
import { buildWhere, q as _q, sample, andWhere, groupCol as _groupCol, groupExpr as _groupExpr, dimSubsetWhere, stripWhere } from './sql.js';
import { buildScopedWhere } from './cohort-sql.js';
import { appendPlot, appendPlots, appendMiniPlot, niceName, escapeHtml, bargap, createFlexGrid, sliceToggles, appendGroupLegend, prependWarning, dtypeRange, dataAvailabilityWarning, groupingLabel, legendWithGrouping, formatBytes, humanList, statTable, appendInvariantTable, tilePreviewTable, LEGEND, LAYOUT } from './plot-utils.js';
import * as plotEngine from './plot-engine.js';
import { META_COLS } from './schema.js';
import { updateFilteredInfo } from './controls.js';
import { state } from './state.js';
import { writeUrlParams } from './url-params.js';
import { pluginGroup, orderedGroupNames } from './plugin-groups.js';
import { buildGroupLabels } from './group-labels.js';
import { scopeBadgeHtml, setScopeBadge } from './scopes.js';

/**
 * Build a plugin context object.
 *
 * ctx is passed to every plugin's render() call. Plugins should only interact
 * with DuckDB through ctx.query / ctx.queryRows to keep the sampling strategy
 * centralised.
 *
 * @param {object} conn       DuckDB async connection
 * @param {object} schema     detected schema
 * @param {object} state      current UI state
 * @param {object} colorMap   {group: hex}
 * @param {string} where      File-scope SQL WHERE fragment (or '')
 * @param {string} userWhere  User filter SQL WHERE fragment (or '')
 * @param {string[]} groups   distinct group values (already fetched)
 * @param {number} filteredCount
 * @param {number} totalRows
 *
 * Additional fields available on ctx:
 * @property {Record<string,string>} groupLabels  maps each original group value to a
 *   shortened display label (e.g. ".../ dataset_A" instead of a long file path).
 *   Use for Plotly trace names and axis tick labels - never for SQL.
 * @property {(g: string) => string} groupLabel  convenience wrapper:
 *   returns ctx.groupLabels[g] ?? String(g). Falls back to the raw value
 *   if no mapping exists (e.g. for groups discovered after ctx was built).
 * @property {(palette: string, n: number) => string[]} color.getColors
 *   returns n colors from the named palette - use for ad-hoc groupings
 *   (e.g. a column other than the active group-by) not covered by colorMap.
 * @property {() => string[]} color.getPaletteNames  lists the available
 *   palette names accepted by color.getColors.
 */
function buildCtx(conn, schema, state, colorMap, where, userWhere, groups, filteredCount, totalRows) {
  const legend = legendWithGrouping(LEGEND, state, '');
  const groupLabels = buildGroupLabels(groups);

  return {
    schema,
    state,
    colorMap,
    where,
    userWhere,
    groups,
    groupLabels,
    groupLabel: (g) => groupLabels[g] ?? String(g),
    filteredCount,
    totalRows,

    /**
     * Run arbitrary SQL against pp_data.
     * Returns a raw Apache Arrow Table - use when you need binary columns
     * (thumbnails, histograms) or want Arrow-native aggregation.
     */
    query(sql) {
      return conn.query(sql);
    },

    /**
     * Run SQL and convert each row to a plain JS object.
     * Binary columns become Uint8Array; numeric columns become numbers.
     * Convenient for most non-binary queries.
     */
    async queryRows(sql) {
      const result = await conn.query(sql);
      return result.toArray().map(r => r.toJSON());
    },

    /**
     * Shorthand: SELECT <cols> FROM pp_data <where> USING SAMPLE <n>
     * groupCol (if set) is always included as __group__.
     */
    async querySample(cols, n = 5000) {
      const gcExpr  = state.groupCol ? `${_q(state.groupCol)} AS ${GROUP_COL_ALIAS}, ` : `'${GROUP_ALL}' AS ${GROUP_COL_ALIAS}, `;
      const colList = cols.map(_q).join(', ');
      const sql     = `SELECT ${gcExpr}${colList} FROM pp_data ${where} ${sample(n)}`;
      return this.queryRows(sql);
    },

    /**
     * Columns holding a single non-null value across every filtered row, found
     * in one SQL pass. Returns [{ col, value }] (value as string); columns with
     * NULLs or more than one distinct value are omitted. Used to surface
     * "shared by all files" invariant properties.
     */
    async constantValues(cols) {
      const present = (cols ?? []).filter(Boolean);
      if (!present.length) return [];
      const exprs = present.map(c => {
        const col = _q(c);
        return `CASE WHEN COUNT(*) = COUNT(${col}) AND COUNT(DISTINCT ${col}) = 1
                     THEN CAST(MIN(${col}) AS VARCHAR) END AS ${col}`;
      }).join(', ');
      const [row = {}] = await this.queryRows(`SELECT ${exprs} FROM pp_data ${where}`);
      return present.flatMap(c => {
        const v = row[c];
        return v != null && String(v) !== '' ? [{ col: c, value: String(v) }] : [];
      });
    },

    /** SQL helper: safely double-quote an identifier. */
    sql: {
      q:         _q,
      andWhere,
      sample,
      groupCol:  () => _groupCol(state),
      groupExpr: () => _groupExpr(state),

      /**
       * WHERE parts pinning one long-format aggregation subset. `q`, `dimCols`
       * and `baseWhere` (the active user filter) are injected; callers pass only
       * `{ fixed, split, obsLevel }`. See sql.js dimSubsetWhere.
       */
      dimSubsetWhere: (opts = {}) => dimSubsetWhere({
        q: _q,
        dimCols:   schema.dimCols ?? [],
        baseWhere: stripWhere(userWhere),
        ...opts,
      }),
    },

    /** Color helpers pre-bound to the current colorMap. */
    color: {
      group:         (g) => _groupColor(colorMap, g),
      hexToRgba,
      getColors,
      getPaletteNames,
    },

    /** Plotly plot helpers (mirror of plot-utils.js). */
    plot: {
      append:      appendPlot,
      appendMany:  appendPlots,
      appendMini:  appendMiniPlot,
      niceName,
      escapeHtml,
      bargap,
      flexGrid:    createFlexGrid,
      sliceToggles,
      setScopeBadge,
      prependWarning,
      dtypeRange,
      dataAvailabilityWarning,
      formatBytes,
      humanList,
      statTable,
      invariantTable: appendInvariantTable,
      tilePreviewTable,

      /**
       * One bar series per group over a shared categorical X axis - the
       * "grouped categorical bars" idiom used across the metadata/file-stats
       * family. `value(cat, group)` returns the bar height (0 when missing).
       * The same traces feed a full card (ctx.plot.append) or a tile preview
       * (ctx.plot.appendMini, pass { mini: true } to drop hover).
       */
      groupedBarTraces: (cats, value, { mini = false } = {}) =>
        groups.map(g => ({
          type: 'bar',
          name: groupLabels[g] ?? String(g),
          x: cats,
          y: cats.map(c => value(c, g)),
          marker: { color: _groupColor(colorMap, g) },
          ...(mini ? { hoverinfo: 'skip' } : {}),
        })),
      renderDomGroupLegend: (container, opts = {}) => appendGroupLegend(
        container,
        opts.groups ?? groups,
        opts.colorFn ?? (g => _groupColor(colorMap, g)),
        { state, minGroups: opts.minGroups ?? 2, labelFn: opts.labelFn ?? ((g) => groupLabels[g] ?? String(g)) },
      ),
      groupingLabel: (fallback = '') => groupingLabel(state, fallback),
      plotlyLegendConfig: legend,
      LAYOUT,

      /**
       * Shared plotting engine - choosePlotKind decides how to plot a column
       * pair; renderDistribution + render* draw it with consistent large-data
       * scaling and significance. See plot-engine.js.
       */
      engine: plotEngine,
    },

    /** Schema constants (column lists, patterns). */
    META_COLS,

    /** Data utilities shared across plugins. */
    data: { extractBinary },
  };
}

/**
 * Decode an Arrow binary/list cell into a JS numeric array.
 * Handles typed arrays, BigInt arrays, Arrow list vectors, and plain arrays.
 */
function extractBinary(val) {
  if (!val) return null;
  if (val instanceof Uint8Array)    return val;
  if (val instanceof Int32Array)    return val;
  if (val instanceof Float32Array)  return val;
  if (val instanceof Float64Array)  return val;
  if (val instanceof BigInt64Array)  return Array.from(val, v => Number(v));
  if (val instanceof BigUint64Array) return Array.from(val, v => Number(v));
  if (Array.isArray(val)) return val;
  if (typeof val.toArray === 'function') {
    const arr = val.toArray();
    if (arr instanceof BigInt64Array || arr instanceof BigUint64Array) return Array.from(arr, v => Number(v));
    return arr;
  }
  if (val.values) return val.values;
  return null;
}

/**
 * Discover active plugins, query group/count metadata, then render each plugin.
 * Safe to call multiple times - clears the widget container each time.
 *
 * @param {object[]} plugins  registered plugin objects
 * @param {object}   conn     DuckDB async connection
 * @param {object}   schema
 * @param {object}   state
 * @param {number}   totalRows
 */
/** Prefer first occurrence if the registry ever lists the same id twice. */
function dedupePluginsById(plugins) {
  const out = [];
  const seen = new Set();
  for (const p of plugins) {
    if (!p?.id || seen.has(p.id)) continue;
    seen.add(p.id);
    out.push(p);
  }
  return out;
}

export async function renderAll(plugins, conn, schema, state, totalRows) {
  plugins = dedupePluginsById(plugins);

  const userWhere = buildWhere(state.filter);
  const where = buildScopedWhere(schema, state);

  // Fetch distinct groups and filtered count in parallel.
  const gcExpr = state.groupCol ? _q(state.groupCol) : `'${GROUP_ALL}'`;
  const [groupResult, countResult] = await Promise.all([
    conn.query(
      `SELECT DISTINCT ${gcExpr} AS g FROM pp_data ${where} ORDER BY 1 LIMIT 50`,
    ),
    conn.query(`SELECT COUNT(*) AS n FROM pp_data ${where}`),
  ]);

  const groups = sortGroups(groupResult.toArray().map(r => String(r.g)));
  const filteredCount = Number(countResult.toArray()[0].n);
  const colorMap     = buildColorMap(groups, state.palette);

  updateFilteredInfo(filteredCount, totalRows);

  const ctx = buildCtx(
    conn, schema, state, colorMap, where, userWhere, groups, filteredCount, totalRows,
  );

  const container = document.getElementById(WIDGET_CONTAINER_ID);
  container.innerHTML = '';

  const activePlugins = plugins.filter(p => {
    try { return p.requires(schema) && !state.hiddenWidgets.has(p.id); }
    catch { return false; }
  });

  // Always reset the collapse-all button before rendering; renderCondensedGallery
  // will re-wire it if tiles are expanded.
  const collapseAllBtn = document.getElementById('collapse-all-btn');
  if (collapseAllBtn) { collapseAllBtn.hidden = true; collapseAllBtn.onclick = null; }

  if (state.condensedMode) {
    await renderCondensedGallery(container, activePlugins, ctx, collapseAllBtn);
  } else {
    await renderClassic(container, activePlugins, ctx);
  }
}

/** Render the full plugin body into `body`, surfacing errors inline. */
async function renderPluginBody(plugin, body, ctx) {
  try {
    await plugin.render(body, ctx);
  } catch (err) {
    body.innerHTML = `<div class="text-danger small p-2">
      <strong>${plugin.id}</strong>: ${err.message}
    </div>`;
    console.error(`[viewer] plugin "${plugin.id}" error:`, err);
  }
}

/**
 * Classic (condensed-mode off) layout: grouped section headers, every widget
 * rendered as a full, always-expanded card.
 */
async function renderClassic(container, activePlugins, ctx) {
  const grouped = new Map();
  for (const plugin of activePlugins) {
    const grp = pluginGroup(plugin);
    if (!grouped.has(grp)) grouped.set(grp, []);
    grouped.get(grp).push(plugin);
  }
  const orderedGroups = orderedGroupNames(activePlugins);

  for (const groupName of orderedGroups) {
    const title = document.createElement('h3');
    title.className = 'my-3 text-primary';
    title.textContent = groupName;
    container.appendChild(title);

    for (const plugin of grouped.get(groupName)) {
      const card = createCard(plugin.label, plugin.info, plugin.scope);
      container.appendChild(card);
      await renderPluginBody(plugin, card.querySelector('.widget-card-body'), ctx);
    }
  }
}

// Plugins ordered by their group's canonical order, so related tiles cluster
// in the gallery even though the section headers are dropped.
function orderActivePlugins(activePlugins) {
  const grouped = new Map();
  for (const p of activePlugins) {
    const g = pluginGroup(p);
    if (!grouped.has(g)) grouped.set(g, []);
    grouped.get(g).push(p);
  }
  const ordered = [];
  for (const g of orderedGroupNames(activePlugins)) ordered.push(...grouped.get(g));
  return ordered;
}

/**
 * Condensed-mode "gallery" layout.
 *
 * Summary widgets stay pinned on top as full-width expanded cards. Every other
 * widget becomes a square, clickable tile showing a short title, one simplified
 * "hero" plot (plugin.condensedPlot) and a one-line plain-language message
 * (plugin.condensedSummary). Clicking a tile expands it *in place*: the tile
 * grows into a full-width card with the widget's normal render() (all controls)
 * and a × to collapse back to the preview. Several tiles can be open at once and
 * the open set is persisted in state.openedWidgets (and the URL), so a filter
 * change or reload restores the same expanded widgets.
 */
async function renderCondensedGallery(container, activePlugins, ctx, collapseAllBtn) {
  const ordered      = orderActivePlugins(activePlugins);
  const summaryPlugs = ordered.filter(p => pluginGroup(p) === 'Summary' && p.id === 'summary');
  const tilePlugs    = ordered.filter(p => !summaryPlugs.includes(p));

  // Pinned summary on top - rendered bare (no card header / narrative line) so
  // the overview opens on a clean title → KPIs → breakdown block.
  for (const plugin of summaryPlugs) {
    const panel = document.createElement('div');
    panel.className = 'gallery-summary';
    container.appendChild(panel);
    await renderPluginBody(plugin, panel, ctx);
  }

  // Tile gallery for everything else.
  const grid = document.createElement('div');
  grid.className = 'widget-grid';
  container.appendChild(grid);

  // Registry: plugin id → that tile's collapse function. Tiles register on expand
  // and unregister on collapse; the topbar button syncs its visibility to the registry size.
  const collapseRegistry = new Map();
  const syncBar = () => {
    if (collapseAllBtn) collapseAllBtn.hidden = collapseRegistry.size === 0;
  };

  if (collapseAllBtn) {
    collapseAllBtn.onclick = () => {
      for (const collapse of [...collapseRegistry.values()]) collapse({ focus: false });
    };
  }

  // Pre-create the cells in order so tiles keep their grouped ordering, then
  // fill them in parallel: each tile's condensedSummary/condensedPlot queries are
  // independent, so DuckDB can pipeline them instead of running one tile at a time.
  const cells = tilePlugs.map(() => {
    const cell = document.createElement('div');
    cell.className = 'widget-cell';
    grid.appendChild(cell);
    return cell;
  });
  await Promise.all(tilePlugs.map((plugin, i) => buildTile(cells[i], plugin, ctx, collapseRegistry, syncBar)));
}

/** Persist the opened-widget set to the URL (skipped for locked snapshots). */
function persistOpened() {
  if (!state.sidebarLocked) writeUrlParams(state);
}

/** Group → emoji used as a placeholder when a tile has no hero plot. */
const GROUP_GLYPH = {
  'File Stats':   '📁',
  'Metadata':     '🏷️',
  'Dataset Stats':'📊',
  'Visualization':'🖼️',
  'Explore':      '🔍',
};

/**
 * Fill one (pre-created, already-appended) gallery `cell`. A cell holds two views
 * of the same widget: a collapsed preview tile (title + hero plot + message) and, once
 * opened, a full-width card with the widget's normal render(). Clicking the tile
 * expands the cell in place; the card's × collapses it back. The open state is
 * mirrored into state.openedWidgets so it survives re-renders.
 */
async function buildTile(cell, plugin, ctx, collapseRegistry, syncBar) {
  let summary = null;
  if (plugin.condensedSummary) {
    try { summary = await plugin.condensedSummary(ctx); } catch { /* best-effort */ }
  }
  const { text, warning } = summary ? normalizeSummary(summary) : { text: '', warning: false };

  const tile = document.createElement('button');
  tile.type = 'button';
  tile.className = 'widget-tile';

  const head = document.createElement('div');
  head.className = 'widget-tile-head';
  const title = document.createElement('span');
  title.className = 'widget-tile-title';
  title.textContent = plugin.shortLabel ?? plugin.label;
  head.appendChild(title);
  tile.appendChild(head);

  const plotArea = document.createElement('div');
  plotArea.className = 'widget-tile-plot';
  tile.appendChild(plotArea);

  // No per-tile colour legend: the group → colour mapping is shown once, as a
  // coloured dot per group in the summary table at the top of the report.
  // The band is always rendered (even with no text) so the plot area, message,
  // and "view all plots" bar stay the same size on every tile.
  const msg = document.createElement('div');
  msg.className = warning ? 'widget-tile-msg widget-tile-msg-warning' : 'widget-tile-msg';
  if (text) {
    msg.innerHTML = `${warning ? '<span class="widget-tile-msg-icon" aria-hidden="true">⚠️</span>' : ''}<span>${text}</span>`;
  }
  tile.appendChild(msg);

  const open = document.createElement('span');
  open.className = 'widget-tile-open';
  // "View all plots" (rather than just "Open") signals the tile is a preview and
  // the full widget may hold more than the single hero plot shown here.
  open.textContent = 'View all plots ▸';
  tile.appendChild(open);
  tile.title = 'View all plots';

  cell.appendChild(tile);

  // The hero plot needs a laid-out, visible host so Plotly can size it, so draw
  // it lazily the first time the tile is actually shown (never while expanded).
  let heroDrawn = false;
  const ensureHero = async () => {
    if (heroDrawn) return;
    heroDrawn = true;
    let drew = false;
    if (plugin.condensedPlot) {
      try { drew = (await plugin.condensedPlot(plotArea, ctx)) !== false && plotArea.childElementCount > 0; }
      catch (err) { console.error(`[viewer] plugin "${plugin.id}" condensedPlot error:`, err); }
    }
    if (!drew) {
      plotArea.classList.add('widget-tile-plot-empty');
      plotArea.textContent = GROUP_GLYPH[pluginGroup(plugin)] ?? '📈';
    }
  };

  let card = null; // full widget card, built on first expand

  const collapse = async ({ focus = true } = {}) => {
    cell.classList.remove('widget-cell-expanded');
    if (card) card.hidden = true;
    tile.hidden = false;
    if (state.openedWidgets.delete(plugin.id)) persistOpened();
    collapseRegistry.delete(plugin.id);
    syncBar();
    await ensureHero();
    if (focus) { tile.focus(); cell.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); }
  };

  const expand = async ({ focus = true } = {}) => {
    tile.hidden = true;
    cell.classList.add('widget-cell-expanded');
    if (!card) {
      card = createCard(plugin.shortLabel ?? plugin.label, plugin.info, plugin.scope);
      card.classList.add('widget-cell-card');

      const close = document.createElement('button');
      close.type = 'button';
      close.className = 'widget-card-close';
      close.title = 'Collapse';
      close.setAttribute('aria-label', `Collapse ${plugin.label}`);
      close.innerHTML = '<span aria-hidden="true">×</span>';
      close.addEventListener('click', (e) => { e.stopPropagation(); collapse(); });

      // Clicking the header (anywhere but the ⓘ info toggle) collapses the card.
      const header = card.querySelector('.widget-card-header');
      header.classList.add('widget-card-header-collapsible');
      header.title = 'Collapse';
      header.addEventListener('click', (e) => {
        if (e.target.closest('.widget-info-btn')) return;
        collapse();
      });
      header.appendChild(close);

      cell.appendChild(card);
      await renderPluginBody(plugin, card.querySelector('.widget-card-body'), ctx);
    } else {
      card.hidden = false;
    }
    collapseRegistry.set(plugin.id, collapse);
    syncBar();
    if (!state.openedWidgets.has(plugin.id)) { state.openedWidgets.add(plugin.id); persistOpened(); }
    if (focus) card.querySelector('.widget-card-close')?.focus();
  };

  tile.addEventListener('click', () => expand());

  // Restore persisted expanded state without stealing focus or re-persisting.
  if (state.openedWidgets?.has(plugin.id)) await expand({ focus: false });
  else await ensureHero();
}

/** Normalize a plugin's condensedSummary() return value to {text, warning}. */
function normalizeSummary(summary) {
  if (typeof summary === 'string') return { text: summary, warning: false };
  return { text: summary.text, warning: !!summary.warning };
}

function createCard(label, info, scope) {
  const div = document.createElement('div');
  div.className = 'widget-card';

  const header = document.createElement('div');
  header.className = 'widget-card-header';

  // Title and its info toggle stay together on the left; the scope badge (and,
  // in expanded mode, the × close button) sit on the right.
  const titleWrap = document.createElement('div');
  titleWrap.className = 'widget-card-titlewrap';

  const title = document.createElement('span');
  title.className = 'widget-card-title';
  title.textContent = label;
  titleWrap.appendChild(title);

  let panel = null;
  if (info) {
    panel = document.createElement('div');
    panel.className = 'widget-info-panel';
    panel.hidden = true;
    panel.innerHTML = renderInfoHtml(info);

    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'widget-info-btn';
    btn.title = 'About this widget';
    btn.setAttribute('aria-label', 'About this widget');
    btn.innerHTML = '<span class="widget-info-icon" aria-hidden="true">ⓘ</span>';
    btn.addEventListener('click', () => { panel.hidden = !panel.hidden; });
    titleWrap.appendChild(btn);
  }

  header.appendChild(titleWrap);

  const badge = scopeBadgeHtml(scope);
  if (badge) header.insertAdjacentHTML('beforeend', badge);

  div.appendChild(header);
  if (panel) div.appendChild(panel);

  const body = document.createElement('div');
  body.className = 'widget-card-body';
  div.appendChild(body);

  return div;
}

/** Minimal markdown → HTML converter for info panel text. */
function renderInfoHtml(text) {
  let html = '';
  for (const para of text.trim().split(/\n\n+/)) {
    const lines = para.split('\n');
    const bullets = lines.filter(l => /^\s*-\s/.test(l));
    const prose   = lines.filter(l => !/^\s*-\s/.test(l));
    if (prose.length)   html += `<p>${prose.map(mdInline).join('<br>')}</p>`;
    if (bullets.length) html += `<ul>${bullets.map(l => `<li>${mdInline(l.replace(/^\s*-\s*/, ''))}</li>`).join('')}</ul>`;
  }
  return html;
}

function sortGroups(groups) {
  const allNumeric = groups.length > 0 && groups.every(g => g !== '' && !isNaN(Number(g)));
  if (allNumeric) return [...groups].sort((a, b) => Number(a) - Number(b));
  return [...groups].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
}

function mdInline(t) {
  return t
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/`(.+?)`/g, '<code>$1</code>');
}
