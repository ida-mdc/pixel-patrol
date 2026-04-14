import { q, groupCol, andWhere } from '../sql.js';
import { groupColor } from '../colors.js';
import { appendPlot } from '../plot-utils.js';

const MAX_FILES_FOR_SUNBURST = 500;
const MIXED_COLOR = '#aaaaaa';

/**
 * File Structure Sunburst — matches Dash FileSunburstWidget:
 *   - Drilldown hierarchy of file/folder paths
 *   - Colours by group; mixed-group folders shown in grey
 *   - Switches to folders-only view when > MAX_FILES_FOR_SUNBURST files
 */
export default {
  id: 'sunburst',
  info: 'Sunburst view of the **file and folder hierarchy**.\n\nClick a slice to zoom in; click the center to zoom out.',
  label: 'File Structure Sunburst',

  requires(schema) {
    return schema.allCols.includes('path') && schema.allCols.includes('size_bytes');
  },

  async render(container, ctx) {
    const gcExpr = groupCol(ctx.state);

    // Dash widget switches to folders-only view when file count is large.
    // In the viewer we *also* avoid fetching millions of paths by aggregating
    // to per-folder counts in SQL when large.
    const countRow = (await ctx.queryRows(`
      SELECT COUNT(*)::BIGINT AS n
      FROM pp_data ${andWhere(ctx.where, '"path" IS NOT NULL')}
    `))[0];
    const numFiles = Number(countRow?.n ?? 0);

    const foldersOnly = numFiles > MAX_FILES_FOR_SUNBURST;

    const rows = foldersOnly
      ? await ctx.queryRows(`
          SELECT
            -- Parent folder of the file path ('' when no slash)
            regexp_extract("path"::VARCHAR, '^(.*)/[^/]+$', 1) AS path,
            ${gcExpr} AS __group__,
            COUNT(*)::INTEGER AS __n__
          FROM pp_data ${andWhere(ctx.where, '"path" IS NOT NULL')}
          GROUP BY 1, 2
        `)
      : await ctx.queryRows(`
          SELECT "path" AS path, ${gcExpr} AS __group__
          FROM pp_data ${andWhere(ctx.where, '"path" IS NOT NULL')}
        `);

    if (!rows.length) {
      container.innerHTML = '<div class="no-data">No path data available.</div>';
      return;
    }

    const { ids, labels, parents, values, colors } =
      buildHierarchy(rows, ctx.colorMap, { foldersOnly });

    if (!ids.length) {
      container.innerHTML = '<div class="no-data">Could not build file hierarchy.</div>';
      return;
    }

    appendPlot(container, [{
      type:          'sunburst',
      ids,
      labels,
      parents,
      values,
      marker:        { colors },
      branchvalues:  'total',
      hovertemplate: '<b>%{label}</b><br>Files: %{value}<extra></extra>',
    }], {
      margin: { l: 0, r: 0, t: 30, b: 0 },
      height: 550,
    });
  },
};

// ── Hierarchy builder (port of Dash _build_hierarchy_and_colors) ─────────────

function buildHierarchy(rows, colorMap, { foldersOnly }) {
  const allPaths = rows.map(r => String(r.path ?? ''));

  const sep = detectSeparator(allPaths);
  const commonRoot = findCommonRoot(allPaths, sep);

  // Strip common root prefix and normalise to forward slashes.
  const normalised = rows.map(r => {
    let rel = String(r.path ?? '');
    if (commonRoot && rel.startsWith(commonRoot)) rel = rel.slice(commonRoot.length);
    rel = rel.replace(/^[/\\]+/, '');
    const n = foldersOnly ? Number(r.__n__ ?? 0) : 1;
    return { path: rel, group: String(r.__group__), n: Number.isFinite(n) ? n : 0 };
  });

  const rootName    = commonRoot ? commonRoot.split(/[/\\]/).filter(Boolean).pop() ?? 'Root' : 'Root';
  const displayRoot = 'Root';

  // Tree maps
  const nodeCount   = {};   // id → count
  const nodeGroups  = {};   // id → Set<group>
  const nodeParent  = {};   // id → parent id
  const nodeLabel   = {};   // id → display label

  // Initialise root
  nodeCount['']  = 0;
  nodeGroups[''] = new Set();
  nodeParent[''] = '';
  nodeLabel['']  = displayRoot;

  if (rootName) {
    nodeCount[rootName]  = 0;
    nodeGroups[rootName] = new Set();
    nodeParent[rootName] = '';
    nodeLabel[rootName]  = rootName;
  }

  for (const { path, group, n } of normalised) {
    if (!n) continue;
    const parts = path.split(/[/\\]/).filter(Boolean);

    let fileId;
    if (foldersOnly) {
      // Folders-only mode: each row already represents a folder with aggregated count.
      // Count that folder (and its ancestors), not its parent (otherwise we stop 1 level early).
      fileId = rootName
        ? (path ? `${rootName}${sep}${path}` : rootName)
        : (path || rootName || '');
    } else {
      fileId = rootName ? `${rootName}${sep}${path}` : path;
      const parentId = getParentId(fileId, sep, rootName);
      nodeCount[fileId]  = 1;
      nodeGroups[fileId] = new Set([group]);
      nodeParent[fileId] = parentId;
      nodeLabel[fileId]  = parts[parts.length - 1] || fileId;
    }

    // Walk ancestors and bubble up counts.
    let cur = foldersOnly ? fileId : getParentId(fileId, sep, rootName);
    while (true) {
      if (cur in nodeCount) {
        // Existing node — just increment all the way to root.
        let p = cur;
        while (true) {
          nodeCount[p] += n;
          nodeGroups[p].add(group);
          if (p === '' || p === rootName) break;
          p = nodeParent[p];
        }
        break;
      } else {
        // New folder node
        const parentId = getParentId(cur, sep, rootName);
        nodeCount[cur]  = n;
        nodeGroups[cur] = new Set([group]);
        nodeParent[cur] = parentId;
        const nameParts = cur.split(sep);
        nodeLabel[cur]  = nameParts[nameParts.length - 1] || cur;
        if (cur === '' || cur === rootName) break;
        cur = parentId;
      }
    }
  }

  // Build output arrays
  const ids     = [];
  const labels  = [];
  const parents = [];
  const values  = [];
  const colors  = [];

  for (const [id, count] of Object.entries(nodeCount)) {
    if (count === 0 && id !== '') continue;
    ids.push(id);
    labels.push(nodeLabel[id] ?? id);
    parents.push(id === '' ? '' : (nodeParent[id] ?? ''));
    values.push(count);

    const uniqueGroups = nodeGroups[id] ?? new Set();
    if (uniqueGroups.size === 1) {
      const g = [...uniqueGroups][0];
      colors.push(groupColor(colorMap, g));
    } else {
      colors.push(MIXED_COLOR);
    }
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
  // Preserve whether paths are absolute (start with / or \) so we can reconstruct correctly.
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
  // Re-attach the leading separator so startsWith() comparisons against raw paths work.
  return (hasLeadingSlash ? sep : '') + common.join(sep);
}
