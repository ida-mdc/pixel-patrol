/**
 * Map Points plugin - shows locations on an interactive MapLibre map.
 *
 * Reads `latitude`, `longitude`, and `footprint` (GeoJSON string).
 *
 * The images' locations are shown as cluster badges: one pie per area, colored by group
 * (a plain circle without a grouping), with the summed number of images.
 * Hovering a badge shows a popup with the
 * image path(s) and, if a grouping is selected, a per-group tally.
 *
 * Note: MapLibre's clustering stores coordinates as 32-bit floats, so badge
 * positions can be off by up to ~1 m.
 */

const MAPLIBRE_SCRIPT_URL = 'https://unpkg.com/maplibre-gl@6.2.0/dist/maplibre-gl.mjs';
// map licence: CC BY 4.0 (https://sgx.geodatenzentrum.de/web_public/gdz/lizenz/deu/Nutzungsbedingungen_basemapworld.pdf)
const MAP_STYLE_URL = 'https://sgx.geodatenzentrum.de/gdz_basemapworld_vektor/styles/bm_web_wld_col.json';
const MAX_POPUP_PATHS  = 5;  // up to this many overlapping images, list their paths; beyond it, a summary
const MAX_POPUP_GROUPS = 3;  // for how many groups display the number of images when they overlap
const POINT_COLOR      = '#4a90d9';
const PIE_PIXEL_RATIO  = 2;  // pie images are drawn at 2x so they stay sharp on high-dpi screens

async function loadMaplibre() {
  // cached, so condensedView and render don't load it twice
  return await import(MAPLIBRE_SCRIPT_URL);
}

// needed for the static html snapshot. idle WebGL map returns
// blank buffer, so redraw when the snapshot is generated.
function captureableInSnapshot(map, container) {
  const redraw = () => {
    if (!document.contains(container)) {
      window.removeEventListener('pp:before-snapshot', redraw);
      return;
    }
    map.redraw();
  };
  window.addEventListener('pp:before-snapshot', redraw);
}

// Group the query rows by coordinate and build one GeoJSON
// feature (geometry + arbitrary `properties`) per distinct position.
// The images' {path, group, fileRowNumber frn} list is stored as a JSON string because MapLibre
// only preserves flat properties when it hands features back in hover events.
function groupedPointFeatures(rows, groups) {
  const rowsPerCoord = new Map();  // "lat,lon" -> rows at exactly that position
  for (const row of rows) {
    const key = `${row.lat},${row.lon}`;
    if (!rowsPerCoord.has(key)) rowsPerCoord.set(key, []);
    rowsPerCoord.get(key).push(row);
  }

  const features = [];
  for (const rowsHere of rowsPerCoord.values()) {
    const members = rowsHere.map(row => ({ path: row.path, group: row.__group__, frn: Number(row.frn) }));
    const properties = { count: rowsHere.length, members: JSON.stringify(members) };
    for (const group of groups) properties[groupCountKey(group)] = 0;
    for (const member of members) properties[groupCountKey(member.group)] += 1;
    features.push({
      type: 'Feature',
      geometry: { type: 'Point', coordinates: [Number(rowsHere[0].lon), Number(rowsHere[0].lat)] },
      properties,
    });
  }
  return features;
}

// Name of the feature property holding the number of images of a group. One flat property per
// group, so that clusterProperties can sum them up per cluster.
function groupCountKey(group) {
  return `count_of_${group}`;
}

// The {path, group, fileRowNumber frn} members of every feature stacked at the hovered pixel.
function membersOf(features) {
  const members = [];
  for (const feature of features) {
    members.push(...JSON.parse(feature.properties.members));
  }
  return members;
}

function abbreviatedCount(valueExpr) {
  // shorten numbers: 2000 → "2k", 2345 → "2.3k", 3200000 → "3.2M"
  return ['case',
    ['>=', valueExpr, 1e6], ['concat', ['/', ['round', ['/', valueExpr, 1e5]], 10], 'M'],
    ['>=', valueExpr, 1e3], ['concat', ['/', ['round', ['/', valueExpr, 1e2]], 10], 'k'],
    ['to-string', valueExpr],
  ];
}

// Draw a pie as an image: one slice per group (proportional to its share) and, with `withHub`, a dark
// hub in the middle that the count text sits on. `shares` and `colors` have one entry per group.
function drawPieImage(shares, colors, radius, withHub) {
  const size = 2 * radius * PIE_PIXEL_RATIO;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const canvasContext = canvas.getContext('2d');
  canvasContext.scale(PIE_PIXEL_RATIO, PIE_PIXEL_RATIO);

  const total = shares.reduce((sum, share) => sum + share, 0);
  let startAngle = -Math.PI / 2;  // start at 12 o'clock
  shares.forEach((share, groupIndex) => {
    const endAngle = startAngle + 2 * Math.PI * share / total;
    canvasContext.beginPath();
    canvasContext.moveTo(radius, radius);
    canvasContext.arc(radius, radius, radius, startAngle, endAngle);
    canvasContext.fillStyle = colors[groupIndex];
    canvasContext.fill();
    startAngle = endAngle;
  });

  canvasContext.beginPath();
  canvasContext.arc(radius, radius, radius - 0.5, 0, 2 * Math.PI);
  canvasContext.strokeStyle = '#fff';
  canvasContext.stroke();

  if (withHub) {
    canvasContext.beginPath();
    canvasContext.arc(radius, radius, radius * 0.62, 0, 2 * Math.PI);
    canvasContext.fillStyle = 'rgba(0,0,0,0.6)';
    canvasContext.fill();
  }
  return canvasContext.getImageData(0, 0, size, size);
}

// Pie markers are map images named "pie:<share of group 0>:<share of group 1>:..." (shares in 1/16ths,
// so only few distinct images exist however many images the map has). MapLibre asks for an image it
// doesn't have yet via the missing-image resolver; we draw and add it right then.
// (The 'styleimagemissing' event fires too late: the icon is already dropped from that layout pass.)
function addPieImageGenerator(map, { colors, radius, withHub }) {
  map.setMissingStyleImageResolver((imageId) => {
    if (!imageId.startsWith('pie:')) return;  // the basemap style asks for its own icons here too
    const shares = imageId.split(':').slice(1).map(Number);
    map.addImage(imageId, drawPieImage(shares, colors, radius, withHub), { pixelRatio: PIE_PIXEL_RATIO });
  });
}

// Expression building the pie image name of a cluster from its summed per-group counts.
function pieImageNameExpression(groups, totalExpr) {
  const shareParts = groups.flatMap(group => {
    const share = ['round', ['*', 16, ['/', ['get', groupCountKey(group)], totalExpr]]];
    return [':', share];
  });
  return ['concat', 'pie', ...shareParts];
}

// Add the image positions to the map as cluster badges, aggregated by on-screen proximity.
// Each badge is a pie showing how its images split over the groups; `bounds` grows to contain all points.
function addPointsLayers(map, rows, ctx, { radius, bounds, showCount = false }) {
  const groups = ctx.groups;
  const features = groupedPointFeatures(rows, groups);
  for (const feature of features) bounds.extend(feature.geometry.coordinates);

  map.addSource('points-clustered', {
    type: 'geojson',
    data: { type: 'FeatureCollection', features },
    cluster:          true,
    clusterRadius:    20,  // merge features within 20 screen pixels
    clusterMinPoints: 1,   // cluster even single points (but see findImagesUnderCursor: not always)
    clusterMaxZoom:   24,  // never un-cluster (features would lose cluster_id/total)
    clusterProperties: {
      total: ['+', ['get', 'count']],  // sum image counts, not feature counts
      ...Object.fromEntries(groups.map(group => [groupCountKey(group), ['+', ['get', groupCountKey(group)]]])),
    },
  });

  // Clustered features carry the summed 'total'; a feature the clustering
  // left unwrapped only has its own 'count' — hence the coalesce.
  const totalExpr = ['coalesce', ['get', 'total'], ['get', 'count']];
  const withHub = showCount && groups.length > 1;  // the count text needs a dark background on a multi-color pie
  addPieImageGenerator(map, { colors: groupColorsOf(ctx), radius: radius + 6, withHub });
  // The id 'points-cluster' is what the hover/click code looks up.
  map.addLayer({
    id:     'points-cluster',
    type:   'symbol',  // 'symbol' layers draw text and/or icons
    source: 'points-clustered',
    layout: {
      'icon-image':            pieImageNameExpression(groups, totalExpr),
      'icon-allow-overlap':    true,  // show every pie, don't hide ones that collide
      'icon-ignore-placement': true,
    },
  });

  if (showCount) {  // don't show numbers in condensed view
    map.addLayer({
      id:     'points-cluster-count',
      type:   'symbol',
      source: 'points-clustered',
      layout: { 'text-field': abbreviatedCount(totalExpr), 'text-size': 12 },
      paint:  { 'text-color': '#fff' },
    });
  }
}

// One color per ctx.groups entry; without a grouping selected, everything gets the plain point color.
function groupColorsOf(ctx) {
  const grouped = ctx.plot.groupingLabel() !== '';
  return ctx.groups.map(group => (grouped ? ctx.color.group(group) : POINT_COLOR));
}

// Group color legend in the top-left corner of the map (the shared legend hides itself for a single group).
function addGroupLegend(container, ctx) {
  const legend = document.createElement('div');
  legend.style.cssText = 'position:absolute;top:10px;left:10px;z-index:1;max-width:60%;padding:0 8px;' +
                         'background:rgba(255,255,255,0.9);color:#222;border-radius:4px';
  ctx.plot.renderDomGroupLegend(legend);
  if (legend.children.length) container.appendChild(legend);
}

// ---------------------------------------------------------------------------
// Hover popups
// ---------------------------------------------------------------------------

// Popup text for the images at one hovered spot. `members` is a list of
// {path, group, fileRowNumber frn} objects.
function popupHtml(ctx, members, kindLabel) {
  const escapeHtml = ctx.plot.escapeHtml;
  const groupingLabel = ctx.plot.groupingLabel();  // '' when no grouping is selected

  // Few images: list their paths.
  if (members.length <= MAX_POPUP_PATHS) {
    return members.map((member) => {
      let line = escapeHtml(String(member.path ?? ''));
      if (groupingLabel) line += ` <i>(${escapeHtml(ctx.groupLabel(member.group))})</i>`;
      return line;
    }).join('<br>');
  }

  // Many images: a count plus a per-group tally.
  let html = `${members.length} ${kindLabel}`;
  if (groupingLabel) {
    // Tally members per group, largest group first (≈ Counter.most_common).
    const countPerGroup = new Map();
    for (const member of members) {
      countPerGroup.set(member.group, (countPerGroup.get(member.group) ?? 0) + 1);
    }
    const entries = [...countPerGroup.entries()];
    entries.sort((entryA, entryB) => entryB[1] - entryA[1]);

    const parts = entries.slice(0, MAX_POPUP_GROUPS)
      .map(([group, count]) => `${count}× ${escapeHtml(ctx.groupLabel(group))}`);
    const hidden = entries.length - parts.length;
    if (hidden > 0) parts.push(`+${hidden} more`);
    html += `<br><b>${escapeHtml(groupingLabel)}:</b> ${parts.join(', ')}`;
  }
  return html;
}

// The images under a mouse position. A point badge (possibly a cluster) wins over the footprints beneath it.
// `anchor` is where a popup should sit: the badge, or the mouse for polygons (no single meaningful anchor point).
async function findImagesUnderCursor(map, event, layers) {
  const hits = map.queryRenderedFeatures(event.point, { layers });
  if (!hits.length) return null;

  const badge = hits.find(hit => hit.layer.id === 'points-cluster');
  if (badge) {
    const clusterId = badge.properties.cluster_id;
    const members = clusterId !== undefined
      ? membersOf(await map.getSource('points-clustered').getClusterLeaves(clusterId, 1e6, 0))
      : membersOf([badge]);  // feature the clustering left unwrapped
    return { members, anchor: badge.geometry.coordinates, kindLabel: 'points' };
  }

  // a polygon spanning tiles can be returned more than once
  const membersPerFrn = new Map(hits.map(hit => [hit.properties.frn, hit.properties]));
  return { members: [...membersPerFrn.values()], anchor: event.lngLat, kindLabel: 'footprints' };
}

// Clickable list of images; each link carries the file row number for the inspector.
function buildImagePickerListHtml(ctx, members) {
  const escapeHtml = ctx.plot.escapeHtml;
  const grouped = ctx.plot.groupingLabel();
  const links = members.map(member => {
    const dot = grouped ? `<span style="color:${ctx.color.group(member.group)}">●</span> ` : '';
    return `${dot}<a href="#" data-frn="${Number(member.frn)}">${escapeHtml(String(member.path ?? ''))}</a>`;
  });
  return `<div style="max-height:200px;overflow:auto;white-space:nowrap">${links.join('<br>')}</div>`;
}

// Hover shows a popup for points and polygons. Click opens the inspector: directly for one image,
// via a list to pick from when several are stacked at one location or overlap.
function addMapInteractions(maplibregl, map, ctx) {
  const popup = new maplibregl.Popup({ offset: 25, closeOnClick: false, closeButton: false });
  const picker = new maplibregl.Popup({ offset: 25, maxWidth: '400px' });
  const interactiveLayers = ['points-cluster', 'footprints-fill'];

  const hide = () => {
    map.getCanvas().style.cursor = '';
    popup.remove();
  };

  map.on('mousemove', async (event) => {
    const hit = await findImagesUnderCursor(map, event, interactiveLayers);
    if (!hit) return hide();
    map.getCanvas().style.cursor = 'pointer';
    if (picker.isOpen()) return popup.remove();  // don't stack the hover popup on the picker
    popup.setLngLat(hit.anchor).setHTML(popupHtml(ctx, hit.members, hit.kindLabel)).addTo(map);
  });

  map.on('click', async (event) => {
    const hit = await findImagesUnderCursor(map, event, interactiveLayers);
    if (!hit) return;
    if (hit.members.length === 1) return ctx.openInspector(Number(hit.members[0].frn));
    picker.setLngLat(hit.anchor).setHTML(buildImagePickerListHtml(ctx, hit.members)).addTo(map);
  });

  picker.on('open', () => {
    picker.getElement().addEventListener('click', (clickEvent) => {
      const link = clickEvent.target.closest('a[data-frn]');
      if (!link) return;
      clickEvent.preventDefault();
      ctx.openInspector(Number(link.dataset.frn));
    });
  });

  // mousemove stops firing once the cursor leaves the canvas entirely.
  map.on('mouseout', hide);
}

export default {
  id:    'map-points',
  label: 'Locations',
  group: 'Geospatial extension',
  scope: 'image',

  requires(schema) {
    const cols = ['latitude', 'longitude', 'footprint'];
    return cols.every(c => schema.allCols.includes(c));
  },

  async overviewPlot(container, ctx) {
    const rows = await ctx.queryRows(`
      SELECT
        "latitude" AS lat,
        "longitude" AS lon,
        ${ctx.sql.groupCol()} AS __group__
      FROM pp_data
      ${ctx.sql.andWhere(ctx.where, '"latitude" IS NOT NULL AND "longitude" IS NOT NULL')}
      ORDER BY random()
      LIMIT 100
    `);
    if (!rows.length) return false;  // placeholder icon will be displayed

    container.style.height = '100%';
    container.style.width = '100%';

    const maplibregl = await loadMaplibre();
    const map = new maplibregl.Map({
      container,
      style: MAP_STYLE_URL,
      center: [0, 0],
      zoom: 1,
      attributionControl: false,
      interactive: false,  // so that map can be clicked to open the condensed view
      preserveDrawingBuffer: true, // needed for the static html snapshot
    });
    captureableInSnapshot(map, container);

    // add points when map has loaded
    map.on('load', () => {
      const bounds = new maplibregl.LngLatBounds();
      addPointsLayers(map, rows, ctx, { radius: 3, bounds });
      map.fitBounds(bounds, { padding: 20, maxZoom: 8, animate: false });
    });

    return true;
  },

  async render(container, ctx) {
    const rows = await ctx.queryRows(`
      SELECT
        "latitude"  AS lat,
        "longitude" AS lon,
        "path" AS path,
        "file_row_number" AS frn,
        "footprint" as footprint,
        ${ctx.sql.groupCol()} AS __group__
      FROM pp_data
      ${ctx.sql.andWhere(ctx.where, '"latitude" IS NOT NULL AND "longitude" IS NOT NULL')}
    `);

    if (!rows.length) {
      container.textContent = 'No geographic data available.';
      return;
    }

    // The card body has no fixed size of its own — without an explicit
    // height the map would collapse to 0 pixels.
    container.style.height = '600px';
    container.style.minHeight = '200px';
    container.style.width = '100%';
    container.style.position = 'relative';

    const maplibregl = await loadMaplibre();

    const map = new maplibregl.Map({
      container,
      style: MAP_STYLE_URL,
      center: [0, 0],
      zoom: 2,
      attributionControl: true,
      pitch: 0,
      preserveDrawingBuffer: true,  // needed for the static html snapshot
    });
    captureableInSnapshot(map, container);
    map.addControl(new maplibregl.ScaleControl());
    map.addControl(new maplibregl.NavigationControl());

    addGroupLegend(container, ctx);

    map.on('load', () => {
      const bounds = new maplibregl.LngLatBounds();

      // One polygon feature per image that has a footprint; grow `bounds`
      // by every polygon corner.
      const footprintFeatures = [];
      for (const row of rows) {
        if (row.footprint == null) continue;
        const geometry = JSON.parse(row.footprint);
        for (const [lon, lat] of geometry.coordinates[0]) {
          bounds.extend([lon, lat]);
        }
        footprintFeatures.push({
          type: 'Feature',
          geometry,
          properties: { path: row.path, group: row.__group__, frn: Number(row.frn) },
        });
      }

      map.addSource('footprints', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: footprintFeatures },
      });

      // add footprint filling and outline first
      map.addLayer({
        id:     'footprints-fill',
        type:   'fill',
        source: 'footprints',
        paint: { 'fill-color': POINT_COLOR, 'fill-opacity': 0.15 },
      });
      map.addLayer({
        id:     'footprints-outline',
        type:   'line',
        source: 'footprints',
        paint: { 'line-color': POINT_COLOR, 'line-width': 1.5 },
      });
      // on top, draw points
      addPointsLayers(map, rows, ctx, { radius: 6, bounds, showCount: true });

      addMapInteractions(maplibregl, map, ctx);

      map.fitBounds(bounds, { padding: 50, maxZoom: 10 });
    });

    // Redraw the map whenever the container is resized
    const observer = new ResizeObserver(() => map.resize());
    observer.observe(container);
  },
};
