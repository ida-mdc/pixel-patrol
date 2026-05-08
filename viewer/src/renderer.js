import { buildColorMap, groupColor as _groupColor, hexToRgba } from './colors.js';
import { GROUP_ALL, GROUP_COL_ALIAS, WIDGET_CONTAINER_ID } from './constants.js';
import { buildWhere, q as _q, sample, andWhere, groupCol as _groupCol, groupExpr as _groupExpr } from './sql.js';
import { buildScopedWhere } from './cohort-sql.js';
import { appendPlot, appendPlots, niceName, escapeHtml, bargap, createFlexGrid, appendGroupLegend, groupingLabel, legendWithGrouping, LEGEND, LAYOUT } from './plot-utils.js';
import { META_COLS } from './schema.js';
import { updateFilteredInfo } from './controls.js';
import { state } from './state.js';
import { pluginGroup, orderedGroupNames } from './plugin-groups.js';
import { buildGroupLabels } from './group-labels.js';

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
 *   Use for Plotly trace names and axis tick labels — never for SQL.
 * @property {(g: string) => string} groupLabel  convenience wrapper:
 *   returns ctx.groupLabels[g] ?? String(g). Falls back to the raw value
 *   if no mapping exists (e.g. for groups discovered after ctx was built).
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
     * Returns a raw Apache Arrow Table — use when you need binary columns
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

    /** SQL helper: safely double-quote an identifier. */
    sql: {
      q:         _q,
      andWhere,
      sample,
      groupCol:  () => _groupCol(state),
      groupExpr: () => _groupExpr(state),
    },

    /** Color helpers pre-bound to the current colorMap. */
    color: {
      group:    (g) => _groupColor(colorMap, g),
      hexToRgba,
    },

    /** Plotly plot helpers (mirror of plot-utils.js). */
    plot: {
      append:      appendPlot,
      appendMany:  appendPlots,
      niceName,
      escapeHtml,
      bargap,
      flexGrid:    createFlexGrid,
      renderDomGroupLegend: (container, opts = {}) => appendGroupLegend(
        container,
        opts.groups ?? groups,
        opts.colorFn ?? (g => _groupColor(colorMap, g)),
        { state, minGroups: opts.minGroups ?? 2, labelFn: opts.labelFn ?? ((g) => groupLabels[g] ?? String(g)) },
      ),
      groupingLabel: (fallback = '') => groupingLabel(state, fallback),
      plotlyLegendConfig: legend,
      LAYOUT,
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
 * Safe to call multiple times — clears the widget container each time.
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
      const card = createCard(plugin.label, plugin.info);
      container.appendChild(card);
      const body = card.querySelector('.widget-card-body');

      try {
        await plugin.render(body, ctx);
      } catch (err) {
        body.innerHTML = `<div class="text-danger small p-2">
          <strong>${plugin.id}</strong>: ${err.message}
        </div>`;
        console.error(`[viewer] plugin "${plugin.id}" error:`, err);
      }
    }
  }
}

function createCard(label, info) {
  const div = document.createElement('div');
  div.className = 'widget-card';

  const header = document.createElement('div');
  header.className = 'widget-card-header';

  const title = document.createElement('span');
  title.className = 'widget-card-title';
  title.textContent = label;
  header.appendChild(title);

  if (info) {
    const panel = document.createElement('div');
    panel.className = 'widget-info-panel';
    panel.hidden = true;
    panel.innerHTML = renderInfoHtml(info);

    const btn = document.createElement('button');
    btn.className = 'widget-info-btn';
    btn.title = 'About this widget';
    btn.textContent = 'ⓘ';
    btn.addEventListener('click', () => { panel.hidden = !panel.hidden; });
    header.appendChild(btn);

    div.appendChild(header);
    div.appendChild(panel);
  } else {
    div.appendChild(header);
  }

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
