// Columns considered metadata (shared with schema.js META_COLS constant).
const META_COLS = [
  'dim_order', 'dtype', 'ndim', 'pixel_size_X', 'pixel_size_Y', 'pixel_size_Z',
];

const DIST_COLS = ['dtype', 'dim_order'];

export default {
  id: 'metadata',
  group: 'Metadata',
  scope: 'image',
  label: 'Metadata',
  info: 'Shows the distribution of **pixel data types** and **dimension ordering** across groupings.\n\nAlso lists properties shared by all files, and available dimension ranges.',

  requires(schema) {
    const hasDist = DIST_COLS.some(c => schema.allCols.includes(c));
    const hasMeta = META_COLS.some(c => schema.allCols.includes(c));
    return hasDist || hasMeta || Object.keys(schema.dimensionInfo).length > 0;
  },

  async render(container, ctx) {
    try {
      const { q, groupCol: gcFn, andWhere } = ctx.sql;
      const { append: appendPlot, niceName, escapeHtml, prependWarning, dataAvailabilityWarning } = ctx.plot;
      const invariants = [];  // collects single-value DIST_COLS + META_COLS shared properties
      const varying    = [];  // DIST_COLS with more than one distinct value across the dataset

      for (const col of DIST_COLS) {
        if (!ctx.schema.allCols.includes(col)) continue;
  
        const gcExpr = gcFn();
        const rows   = await ctx.queryRows(`
          SELECT ${q(col)} AS __cat__,
                 ${gcExpr} AS __group__,
                 COUNT(*)  AS count
          FROM pp_data ${andWhere(ctx.where, `${q(col)} IS NOT NULL`)}
          GROUP BY 1, 2
          ORDER BY 1, 2
        `);
  
        const totalRows = await ctx.queryRows(`SELECT COUNT(*) AS n FROM pp_data ${ctx.where}`);
        const total     = Number(totalRows[0]?.n ?? 0);
        const withData  = rows.reduce((s, r) => s + Number(r.count), 0);
        const label     = niceName(col);

        dataAvailabilityWarning(container, [{ label, present: withData }], total, { unit: 'images' });

        if (!rows.length) continue;

        const cats   = [...new Set(rows.map(r => String(r.__cat__)))].sort();

        // Single unique value - no plot needed; show in shared properties table instead.
        if (cats.length === 1) {
          invariants.push({ Property: niceName(col), Value: cats[0] });
          continue;
        }

        varying.push({ col, cats });

        const groups = ctx.groups;

        const traces = groups.map(g => ({
          type:   'bar',
          name:   ctx.groupLabel(g),
          x:      cats,
          y:      cats.map(cat => {
            const r = rows.find(r => String(r.__cat__) === cat && String(r.__group__) === g);
            return r ? Number(r.count) : 0;
          }),
          marker: { color: ctx.color.group(g) },
        }));
  
        appendPlot(container, traces, {
          title:      { text: `${label} Distribution` },
          barmode:    'stack',
          height:     500,
          xaxis:      { title: label, type: 'category' },
          yaxis:      { title: 'Count' },
          showlegend: groups.length > 1,
          ...(groups.length > 1 ? { legend: ctx.plot.plotlyLegendConfig } : {}),
        }, 'margin-bottom:24px');
      }
  
      if (varying.length) {
        const parts = varying.map(({ col, cats }) =>
          `<code>${col}</code> (${cats.map(c => `<code>${escapeHtml(c)}</code>`).join(', ')})`);
        const list  = parts.length === 1
          ? parts[0]
          : parts.slice(0, -1).join(', ') + ' and ' + parts[parts.length - 1];
        const dtypeNote = varying.some(v => v.col === 'dtype')
          ? ' Mixed <code>dtype</code> is worth a particularly close look: a pixel value of 100 sits near the bottom of the <code>uint16</code> range but the middle of the <code>uint8</code> range, so the same number can mean very different things.'
          : '';
        prependWarning(container, {
          level: 'red',
          html: `This report contains more than one value for ${list}. ` +
            `That can point to inconsistent acquisition - different instruments, software, or settings - ` +
            `and is worth checking before comparing groups.${dtypeNote}`,
        });
      }

      const available  = META_COLS.filter(c => ctx.schema.allCols.includes(c) && !DIST_COLS.includes(c));
  
      if (available.length) {
        const exprs = available.map(c => {
          const col           = q(c);
          const baseInvariant = `(COUNT(*) = COUNT(${col}) AND COUNT(DISTINCT ${col}) = 1)`;
          const invariant     = c.endsWith('_size')
            ? `(${baseInvariant} AND MIN(${col}) > 1)`
            : baseInvariant;
          return `CASE WHEN ${invariant} THEN CAST(MIN(${col}) AS VARCHAR) ELSE NULL END AS ${col}`;
        }).join(',\n');
  
        const rows = await ctx.queryRows(`SELECT ${exprs} FROM pp_data ${ctx.where}`);
        const row  = rows[0] ?? {};
        for (const col of available) {
          const val = row[col];
          if (val != null && String(val) !== '') {
            invariants.push({ Property: col.replace(/_/g, ' '), Value: String(val) });
          }
        }
      }
  
  
      if (invariants.length) {
        const h = document.createElement('h6');
        h.style.cssText = 'margin-top:16px;margin-bottom:10px';
        h.textContent = 'Properties shared by all files that report it';
        container.appendChild(h);
        const table = document.createElement('table');
        table.className = 'stat-table';
        table.innerHTML = `
          <thead><tr><th>Property</th><th>Value</th></tr></thead>
          <tbody>${invariants.map(r => `<tr><td>${r.Property}</td><td>${r.Value}</td></tr>`).join('')}</tbody>
        `;
        container.appendChild(table);
      }
  
      if (!container.firstChild) {
        container.innerHTML = '<div class="no-data">No metadata columns found.</div>';
      }
    
    } catch {
      container.innerHTML = '<div class="no-data">Failed to load data.</div>';
    }
  },
};
