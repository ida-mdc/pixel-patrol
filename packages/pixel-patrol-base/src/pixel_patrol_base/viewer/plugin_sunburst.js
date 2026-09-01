const MAX_FILES_FOR_SUNBURST = 500;
const MIXED_COLOR = '#aaaaaa';

// Slice size metric. Image count is what reveals container files: they are the
// slices worth more than one file.
const MODES = [
  { id: 'files',  label: 'File count',  pick: s => s.files  },
  { id: 'images', label: 'Image count', pick: s => s.images },
  { id: 'size',   label: 'File size',   pick: s => s.bytes  },
];

const HOVER = '<b>%{label}</b><br>Files: %{customdata[0]:,}<br>' +
              'Images: %{customdata[1]:,}<br>Size: %{customdata[2]:.3s}B<extra></extra>';

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
      buildHierarchy(rows, ctx.colorMap, { foldersOnly, mode: 'files' });
    if (!ids.length) return false;

    ctx.plot.appendMini(container, [{
      type: 'sunburst', ids, labels, parents, values,
      marker: { colors }, branchvalues: 'total', hoverinfo: 'skip',
    }], { margin: { l: 0, r: 0, t: 0, b: 0 } });
    return true;
  },

  async render(container, ctx) {
    try {
      const mode = MODES.some(m => m.id === container.dataset.mode) ? container.dataset.mode : 'files';
      const { andWhere } = ctx.sql;
      const pathWhere = andWhere(ctx.where, '"path" IS NOT NULL');

      // Above the threshold, roll files up to their folders so the chart stays legible.
      const foldersOnly = await countFiles(ctx) > MAX_FILES_FOR_SUNBURST;
      const rows = await fetchSunburstRows(ctx, { foldersOnly, pathWhere });

      if (!rows.length) {
        container.innerHTML = '<div class="no-data">No path data available.</div>';
        return;
      }

      const { ids, labels, parents, values, colors, customdata } =
        buildHierarchy(rows, ctx.colorMap, { foldersOnly, mode });

      if (!ids.length) {
        container.innerHTML = '<div class="no-data">Could not build file hierarchy.</div>';
        return;
      }

      const buttons = MODES.map(m =>
        `<button type="button" class="btn btn-sm sunburst-mode-btn ${m.id === mode ? 'btn-secondary' : 'btn-outline-secondary'}" data-mode="${m.id}">${m.label}</button>`).join('');
      container.innerHTML = `
        <div class="d-flex justify-content-end mb-2">
          <div class="btn-group btn-group-sm" role="group">${buttons}</div>
        </div>
        <div class="sunburst-plot-area"></div>
      `;

      for (const btn of container.querySelectorAll('.sunburst-mode-btn')) {
        btn.onclick = () => {
          container.dataset.mode = btn.dataset.mode;
          this.render(container, ctx);
        };
      }

      // Sizing slices by image count makes each datapoint an image, not a file.
      ctx.plot.setScopeBadge?.(
        container.closest('.widget-card')?.querySelector('.widget-scope-badge'),
        mode === 'images' ? 'image' : 'file');

      ctx.plot.append(container.querySelector('.sunburst-plot-area'), [{
        type:          'sunburst',
        ids,
        labels,
        parents,
        values,
        customdata,
        marker:        { colors },
        branchvalues:  'total',
        hovertemplate: HOVER,
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
               SUM(__n_images__)::INTEGER AS __images__,
               SUM("size_bytes")::BIGINT AS __size__
        FROM ${src}
        GROUP BY 1, 2
      `)
    : ctx.queryRows(`
        SELECT "path", ${gcExpr} AS __group__,
               __n_images__::INTEGER AS __images__, "size_bytes"::BIGINT AS __size__
        FROM ${src}
      `);
}

function buildHierarchy(rows, colorMap, { foldersOnly, mode }) {
  const groupColor  = (g) => colorMap[String(g)] ?? '#888';
  const allPaths    = rows.map(r => String(r.path ?? ''));
  const sep         = detectSeparator(allPaths);
  const commonRoot  = findCommonRoot(allPaths, sep);

  const normalised = rows.map(r => {
    let rel = String(r.path ?? '');
    if (commonRoot && rel.startsWith(commonRoot)) rel = rel.slice(commonRoot.length);
    rel = rel.replace(/^[/\\]+/, '');
    return {
      path:   rel,
      group:  String(r.__group__),
      files:  foldersOnly ? num(r.__n__) : 1,
      images: num(r.__images__),
      bytes:  num(r.__size__),
    };
  });

  const rootName = commonRoot ? commonRoot.split(/[/\\]/).filter(Boolean).pop() ?? 'Root' : 'Root';

  const nodeStats  = { [rootName]: { files: 0, images: 0, bytes: 0 } };
  const nodeGroups = { [rootName]: new Set() };
  const nodeParent = { [rootName]: '' };
  const nodeLabel  = { [rootName]: rootName };

  for (const rec of normalised) {
    const { path, group } = rec;
    if (!rec.files && !rec.images && !rec.bytes) continue;
    const parts = path.split(/[/\\]/).filter(Boolean);

    let fileId;
    if (foldersOnly) {
      fileId = rootName
        ? (path ? `${rootName}${sep}${path}` : rootName)
        : (path || rootName || '');
    } else {
      fileId = rootName ? `${rootName}${sep}${path}` : path;
      const parentId = getParentId(fileId, sep, rootName);
      nodeStats[fileId]  = { files: rec.files, images: rec.images, bytes: rec.bytes };
      nodeGroups[fileId] = new Set([group]);
      nodeParent[fileId] = parentId;
      nodeLabel[fileId]  = parts[parts.length - 1] || fileId;
    }

    let cur = foldersOnly ? fileId : getParentId(fileId, sep, rootName);
    while (true) {
      if (cur in nodeStats) {
        let p = cur;
        while (true) {
          addStats(nodeStats[p], rec);
          nodeGroups[p].add(group);
          if (p === rootName) break;
          p = nodeParent[p];
        }
        break;
      } else {
        const parentId = getParentId(cur, sep, rootName);
        nodeStats[cur]  = { files: rec.files, images: rec.images, bytes: rec.bytes };
        nodeGroups[cur] = new Set([group]);
        nodeParent[cur] = parentId;
        const nameParts = cur.split(sep);
        nodeLabel[cur]  = nameParts[nameParts.length - 1] || cur;
        if (cur === rootName) break;
        cur = parentId;
      }
    }
  }

  const pick = (MODES.find(m => m.id === mode) ?? MODES[0]).pick;
  const ids = [], labels = [], parents = [], values = [], colors = [], customdata = [];
  for (const [id, s] of Object.entries(nodeStats)) {
    const value = pick(s);
    if (!value) continue;
    ids.push(id);
    labels.push(nodeLabel[id] ?? id);
    parents.push(nodeParent[id] ?? '');
    values.push(value);
    customdata.push([s.files, s.images, s.bytes]);
    const uniqueGroups = nodeGroups[id] ?? new Set();
    colors.push(uniqueGroups.size === 1 ? groupColor([...uniqueGroups][0]) : MIXED_COLOR);
  }
  return { ids, labels, parents, values, colors, customdata };
}

function num(v) {
  const n = Number(v ?? 0);
  return Number.isFinite(n) ? n : 0;
}

function addStats(target, rec) {
  target.files  += rec.files;
  target.images += rec.images;
  target.bytes  += rec.bytes;
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