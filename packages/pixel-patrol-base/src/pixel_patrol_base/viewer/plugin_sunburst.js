const MAX_FILES_FOR_SUNBURST = 500;
const MIXED_COLOR = '#aaaaaa';

export default {
  id: 'sunburst',
  required_inputs: ['path', 'size_bytes'],
  inputs: [],
  group: 'Summary',
  scope: 'file',
  info: 'Sunburst view of the **file and folder hierarchy**.\n\nClick a slice to zoom in; click the center to zoom out.',
  label: 'File Structure Sunburst',
  shortLabel: 'Folder Structure',

  requires(schema) {
    return schema.allCols.includes('path') && schema.allCols.includes('size_bytes');
  },

  // A single file is not a hierarchy.
  async requiresData(ctx) {
    return await countFiles(ctx) > 1;
  },

  async overviewMessage(ctx) {
    try {
      const count = await countFiles(ctx);
      return `Folder structure of <strong>${count.toLocaleString()} files</strong>.`;
    } catch { return null; }
  },

  async overviewPlot(container, ctx) {
    const pathWhere = ctx.sql.andWhere(ctx.where, '"path" IS NOT NULL');

    // Match the full view's depth: show individual files for small datasets,
    // roll up to folders only when there are too many to draw individually.
    const foldersOnly = await countFiles(ctx) > MAX_FILES_FOR_SUNBURST;
    const rows = await fetchSunburstRows(ctx, { foldersOnly, pathWhere });
    if (!rows.length) return false;

    const { ids, labels, parents, values, colors } =
      buildHierarchy(rows, ctx.colorMap, { foldersOnly, sizeMode: false });
    if (!ids.length) return false;

    ctx.plot.appendMini(container, [{
      type: 'sunburst', ids, labels, parents, values,
      marker: { colors }, branchvalues: 'total', hoverinfo: 'skip',
    }], { margin: { l: 0, r: 0, t: 0, b: 0 } });
    return true;
  },

  async render(container, ctx) {
    try {
      const sizeMode = container.dataset.sizeMode === 'true';
      const { andWhere } = ctx.sql;
      const pathWhere = andWhere(ctx.where, '"path" IS NOT NULL');

      // Above the threshold, roll files up to their folders so the chart stays legible.
      const foldersOnly = await countFiles(ctx) > MAX_FILES_FOR_SUNBURST;
      const rows = await fetchSunburstRows(ctx, { foldersOnly, pathWhere });

      if (!rows.length) {
        container.innerHTML = '<div class="no-data">No path data available.</div>';
        return;
      }

      const { ids, labels, parents, values, colors } =
        buildHierarchy(rows, ctx.colorMap, { foldersOnly, sizeMode });

      if (!ids.length) {
        container.innerHTML = '<div class="no-data">Could not build file hierarchy.</div>';
        return;
      }

      const hoverTemplate = sizeMode
        ? '<b>%{label}</b><br>Size: %{value:.3s}B<extra></extra>'
        : '<b>%{label}</b><br>Files: %{value}<extra></extra>';

      container.innerHTML = `
        <div class="d-flex justify-content-end mb-2">
          <div class="btn-group btn-group-sm" role="group">
            <button type="button" class="btn btn-sm sunburst-mode-btn ${!sizeMode ? 'btn-secondary' : 'btn-outline-secondary'}" data-mode="count">File count</button>
            <button type="button" class="btn btn-sm sunburst-mode-btn ${sizeMode ? 'btn-secondary' : 'btn-outline-secondary'}" data-mode="size">File size</button>
          </div>
        </div>
        <div class="sunburst-plot-area"></div>
      `;

      for (const btn of container.querySelectorAll('.sunburst-mode-btn')) {
        btn.onclick = () => {
          container.dataset.sizeMode = String(btn.dataset.mode === 'size');
          this.render(container, ctx);
        };
      }

      ctx.plot.append(container.querySelector('.sunburst-plot-area'), [{
        type:          'sunburst',
        ids,
        labels,
        parents,
        values,
        marker:        { colors },
        branchvalues:  'total',
        hovertemplate: hoverTemplate,
      }], {
        margin: { l: 0, r: 0, t: 30, b: 0 },
        height: 550,
      });
      ctx.plot.renderDomGroupLegend?.(container);

    } catch {
      container.innerHTML = '<div class="no-data">Failed to load data.</div>';
    }
  },
};

async function countFiles(ctx) {
  const [{ n }] = await ctx.queryRows(
    `SELECT COUNT(DISTINCT "path")::BIGINT AS n
     FROM pp_data ${ctx.sql.andWhere(ctx.where, '"path" IS NOT NULL')}`);
  return Number(n ?? 0);
}

// One row per folder (with file count + size) when rolled up, else one per file.
// Both read the per-file relation, so a container counts and sizes once.
function fetchSunburstRows(ctx, { foldersOnly, pathWhere }) {
  const gcExpr = ctx.sql.groupCol();
  const src    = ctx.sql.perFile(pathWhere);
  return foldersOnly
    ? ctx.queryRows(`
        SELECT regexp_extract("path"::VARCHAR, '^(.*)/[^/]+$', 1) AS path,
               ${gcExpr} AS __group__,
               COUNT(DISTINCT "path")::INTEGER AS __n__,
               SUM("size_bytes")::BIGINT AS __size__
        FROM ${src}
        GROUP BY 1, 2
      `)
    : ctx.queryRows(`
        SELECT "path", ${gcExpr} AS __group__, "size_bytes"::BIGINT AS __size__
        FROM ${src}
      `);
}

function buildHierarchy(rows, colorMap, { foldersOnly, sizeMode }) {
  const groupColor  = (g) => colorMap[String(g)] ?? '#888';
  const allPaths    = rows.map(r => String(r.path ?? ''));
  const sep         = detectSeparator(allPaths);
  const commonRoot  = findCommonRoot(allPaths, sep);

  const normalised = rows.map(r => {
    let rel = String(r.path ?? '');
    if (commonRoot && rel.startsWith(commonRoot)) rel = rel.slice(commonRoot.length);
    rel = rel.replace(/^[/\\]+/, '');
    let n;
    if (sizeMode) {
      n = Number(r.__size__ ?? 0);
    } else {
      n = foldersOnly ? Number(r.__n__ ?? 0) : 1;
    }
    return { path: rel, group: String(r.__group__), n: Number.isFinite(n) ? n : 0 };
  });

  const rootName = commonRoot ? commonRoot.split(/[/\\]/).filter(Boolean).pop() ?? 'Root' : 'Root';

  const nodeCount  = { [rootName]: 0 };
  const nodeGroups = { [rootName]: new Set() };
  const nodeParent = { [rootName]: '' };
  const nodeLabel  = { [rootName]: rootName };

  for (const { path, group, n } of normalised) {
    if (!n) continue;
    const parts = path.split(/[/\\]/).filter(Boolean);

    let fileId;
    if (foldersOnly) {
      fileId = rootName
        ? (path ? `${rootName}${sep}${path}` : rootName)
        : (path || rootName || '');
    } else {
      fileId = rootName ? `${rootName}${sep}${path}` : path;
      const parentId = getParentId(fileId, sep, rootName);
      nodeCount[fileId]  = n;
      nodeGroups[fileId] = new Set([group]);
      nodeParent[fileId] = parentId;
      nodeLabel[fileId]  = parts[parts.length - 1] || fileId;
    }

    let cur = foldersOnly ? fileId : getParentId(fileId, sep, rootName);
    while (true) {
      if (cur in nodeCount) {
        let p = cur;
        while (true) {
          nodeCount[p] += n;
          nodeGroups[p].add(group);
          if (p === rootName) break;
          p = nodeParent[p];
        }
        break;
      } else {
        const parentId = getParentId(cur, sep, rootName);
        nodeCount[cur]  = n;
        nodeGroups[cur] = new Set([group]);
        nodeParent[cur] = parentId;
        const nameParts = cur.split(sep);
        nodeLabel[cur]  = nameParts[nameParts.length - 1] || cur;
        if (cur === rootName) break;
        cur = parentId;
      }
    }
  }

  const ids = [], labels = [], parents = [], values = [], colors = [];
  for (const [id, count] of Object.entries(nodeCount)) {
    if (count === 0) continue;
    ids.push(id);
    labels.push(nodeLabel[id] ?? id);
    parents.push(nodeParent[id] ?? '');
    values.push(count);
    const uniqueGroups = nodeGroups[id] ?? new Set();
    colors.push(uniqueGroups.size === 1 ? groupColor([...uniqueGroups][0]) : MIXED_COLOR);
  }
  return { ids, labels, parents, values, colors };
}

function getParentId(id, sep, rootName) {
  const idx = id.lastIndexOf(sep);
  if (idx === -1) return rootName || '';
  return id.slice(0, idx) || rootName || '';
}

function detectSeparator(paths) {
  const hasForward  = paths.some(p => p.includes('/'));
  const hasBackward = paths.some(p => p.includes('\\'));
  return hasBackward && !hasForward ? '\\' : '/';
}

function findCommonRoot(paths, sep) {
  if (!paths.length) return '';
  const hasLeadingSlash = paths[0].startsWith('/') || paths[0].startsWith('\\');
  const parts = paths.map(p => p.split(/[/\\]/).filter(Boolean));
  const min   = Math.min(...parts.map(p => p.length));
  let common  = [];
  for (let i = 0; i < min; i++) {
    const val = parts[0][i];
    if (parts.every(p => p[i] === val)) common.push(val);
    else break;
  }
  if (!common.length) return '';
  return (hasLeadingSlash ? sep : '') + common.join(sep);
}