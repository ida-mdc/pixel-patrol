const SKIP_COLS = new Set([
  'thumbnail', 'histogram_counts', 'channel_names', 'obs_level',
  'histogram_min', 'histogram_max', 'histogram_nan_count', 'file_row_number',
]);
const PAGE_SIZE   = 10;
const LARGE_TABLE = 10_000;

// Priority columns shown first; rest sorted alphabetically
const PRIORITY_COLS = ['path', 'child_id'];

// Escape a string for use inside `ILIKE '%...%' ESCAPE '\'` - quotes for SQL,
// and %/_ (LIKE wildcards) so a literal search term matches literally.
function escSql(str) {
  return String(str).replace(/'/g, "''").replace(/[\\%_]/g, c => '\\' + c);
}

function orderBy(col, dir) {
  return `ORDER BY "${col.replaceAll('"', '""')}" ${dir} ${dir === 'ASC' ? 'NULLS FIRST' : 'NULLS LAST'}`;
}

export default {
  id: 'image-table',
  label: 'Image Table',
  group: 'Summary',
  scope: 'image',
  info: 'Full-image statistics — one row per image file, no per-slice or per-channel rows. ' +
        'Click a column header to sort. ' +
        'Search (Enter): substring match across all columns for small datasets, ' +
        'path and child_id only for large ones (≥10k images).',

  requires(schema) {
    return schema.isLongFormat;
  },

  async render(container, ctx) {
    const { q, andWhere } = ctx.sql;

    const available   = ctx.schema.allCols.filter(c => !SKIP_COLS.has(c));
    const front       = PRIORITY_COLS.filter(c => available.includes(c));
    const rest        = available.filter(c => !front.includes(c)).sort();
    const displayCols = [...front, ...rest];

    if (!displayCols.length) {
      container.innerHTML = '<div class="no-data">No displayable columns found.</div>';
      return;
    }

    // Fetch total once — used for browse pagination and to pick search strategy
    const [[{ n: totalCount }]] = await Promise.all([
      ctx.queryRows(`SELECT COUNT(*) AS n FROM pp_data ${ctx.where}`),
    ]);
    const total0 = Number(totalCount ?? 0);

    const idCols     = PRIORITY_COLS.filter(c => available.includes(c));
    const searchCols = total0 < LARGE_TABLE ? displayCols : idCols;
    const searchHint = total0 >= LARGE_TABLE
      ? `Search sub-string (Enter) — ${idCols.join(', ')} only`
      : 'Search sub-string (Enter)';

    let page    = 0;
    let sortCol = available.includes('path') ? 'path' : displayCols[0];
    let sortDir = 'ASC';
    let search  = '';

    function formatVal(val) {
      if (val == null)          return { text: '', style: 'color:#aaa' };
      if (typeof val === 'bigint') {
        return { text: val.toLocaleString(), style: 'text-align:right;font-variant-numeric:tabular-nums' };
      }
      if (typeof val === 'number') {
        const text = Number.isInteger(val) ? val.toLocaleString() : Number(val.toPrecision(4)).toString();
        return { text, style: 'text-align:right;font-variant-numeric:tabular-nums' };
      }
      const str = String(val);
      return str.length > 60
        ? { text: '…' + str.slice(-57), title: str, style: '' }
        : { text: str, style: '' };
    }

    async function doRender() {
      container.innerHTML = '<div class="no-data">Loading…</div>';
      try {
        const colExprs    = displayCols.map(c => q(c)).join(', ');
        const isSearching = Boolean(search.trim() && searchCols.length);
        const likes       = isSearching
          ? searchCols.map(c => `CAST(${q(c)} AS VARCHAR) ILIKE '%${escSql(search)}%' ESCAPE '\\'`).join(' OR ')
          : null;
        const wh = isSearching ? andWhere(ctx.where, `(${likes})`) : ctx.where;

        const [total, rows] = await Promise.all([
          isSearching
            ? ctx.queryRows(`SELECT COUNT(*) AS n FROM pp_data ${wh}`)
                .then(r => Number(r[0]?.n ?? 0))
            : Promise.resolve(total0),
          ctx.queryRows(`
            SELECT ${colExprs} FROM pp_data ${wh}
            ${orderBy(sortCol, sortDir)}
            LIMIT ${PAGE_SIZE} OFFSET ${page * PAGE_SIZE}
          `),
        ]);

        const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
        if (page >= totalPages) page = totalPages - 1;

        container.innerHTML = '';

        // ── Toolbar ──────────────────────────────────────────────────────────
        const searchInput = Object.assign(document.createElement('input'), {
          type:        'text',
          placeholder: searchCols.length ? searchHint : 'No text columns to search',
          value:       search,
          disabled:    !searchCols.length,
        });
        searchInput.style.cssText =
          'padding:4px 8px;border:1px solid #dee2e6;border-radius:4px;font-size:13px;min-width:320px';
        searchInput.addEventListener('keydown', e => {
          if (e.key === 'Enter')  { search = searchInput.value; page = 0; doRender(); }
          if (e.key === 'Escape') { search = ''; searchInput.value = ''; page = 0; doRender(); }
        });

        const clearBtn = Object.assign(document.createElement('button'), { textContent: '✕', title: 'Clear search' });
        clearBtn.style.cssText  = 'padding:3px 8px;border:1px solid #dee2e6;border-radius:4px;background:#fff;cursor:pointer;font-size:12px';
        clearBtn.style.display  = search ? 'inline-block' : 'none';
        clearBtn.addEventListener('click', () => { search = ''; page = 0; doRender(); });

        const countLabel = Object.assign(document.createElement('span'), {
          textContent: `${total.toLocaleString()} ${isSearching ? 'match' : 'image'}${total !== 1 ? 'es' : ''}`,
        });
        countLabel.style.cssText = 'font-size:13px;color:#6c757d';

        const toolbar = document.createElement('div');
        toolbar.style.cssText = 'display:flex;align-items:center;gap:8px;margin-bottom:10px;flex-wrap:wrap';
        toolbar.append(searchInput, clearBtn, countLabel);
        container.appendChild(toolbar);

        if (!rows.length) {
          container.insertAdjacentHTML('beforeend',
            `<div class="no-data">${isSearching ? 'No matching images.' : 'No data.'}</div>`);
          return;
        }

        // ── Table ─────────────────────────────────────────────────────────────
        const thead = document.createElement('thead');
        thead.appendChild(displayCols.reduce((tr, col) => {
          const th = Object.assign(document.createElement('th'), {
            textContent: col.replace(/_/g, ' ') + (col === sortCol ? (sortDir === 'ASC' ? ' ▲' : ' ▼') : ''),
            title: `Sort by ${col}`,
          });
          th.style.cssText = 'cursor:pointer;user-select:none';
          th.addEventListener('click', () => {
            if (sortCol === col) sortDir = sortDir === 'ASC' ? 'DESC' : 'ASC';
            else { sortCol = col; sortDir = 'ASC'; }
            page = 0;
            doRender();
          });
          tr.appendChild(th);
          return tr;
        }, document.createElement('tr')));

        const tbody = document.createElement('tbody');
        for (const row of rows) {
          const tr = document.createElement('tr');
          for (const col of displayCols) {
            const td = document.createElement('td');
            const { text, title, style } = formatVal(row[col]);
            td.textContent = text;
            if (style) td.style.cssText = style;
            if (title) td.title = title;
            tr.appendChild(td);
          }
          tbody.appendChild(tr);
        }

        const table = Object.assign(document.createElement('table'), { className: 'stat-table' });
        table.style.cssText = 'font-size:13px;white-space:nowrap';
        table.append(thead, tbody);

        const wrapper = document.createElement('div');
        wrapper.style.cssText = 'overflow-x:auto;margin-bottom:10px';
        wrapper.appendChild(table);
        container.appendChild(wrapper);

        // ── Pagination ────────────────────────────────────────────────────────
        if (totalPages > 1) {
          const mkBtn = (label, disabled, cb) => {
            const b = Object.assign(document.createElement('button'), { textContent: label, disabled });
            b.style.cssText = `padding:3px 10px;border:1px solid #dee2e6;border-radius:4px;background:#fff;` +
                              `cursor:${disabled ? 'default' : 'pointer'};font-size:12px;opacity:${disabled ? 0.4 : 1}`;
            b.addEventListener('click', cb);
            return b;
          };
          const pageLabel = Object.assign(document.createElement('span'), {
            textContent: `Page ${page + 1} of ${totalPages}`,
          });
          const pagRow = document.createElement('div');
          pagRow.style.cssText = 'display:flex;align-items:center;gap:8px;font-size:13px;margin-top:4px';
          pagRow.append(
            mkBtn('‹ Prev', page === 0,             () => { page--; doRender(); }),
            pageLabel,
            mkBtn('Next ›', page >= totalPages - 1, () => { page++; doRender(); }),
          );
          container.appendChild(pagRow);
        }
      } catch (e) {
        container.innerHTML = '<div class="no-data">Failed to load table data.</div>';
        console.error('[image-table]', e);
      }
    }

    await doRender();
  },
};
