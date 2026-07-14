/**
 * Map Points plugin - shows locations on an interactive MapLibre map.
 *
 * Reads `latitude`, `longitude`, and `footprint` (GeoJSON string).
 *
 * The images' locations are shown as cluster badges: one circle per area
 * with the summed number of images.
 * Hovering a badge shows a popup with the
 * image name(s) and, if a grouping is selected, a per-group tally.
 *
 * Note: MapLibre's clustering stores coordinates as 32-bit floats, so badge
 * positions can be off by up to ~1 m.
 */

const MAPLIBRE_SCRIPT_URL = 'https://unpkg.com/maplibre-gl@latest/dist/maplibre-gl.js';
// map licence: CC BY 4.0 (https://sgx.geodatenzentrum.de/web_public/gdz/lizenz/deu/Nutzungsbedingungen_basemapworld.pdf)
const MAP_STYLE_URL = 'https://sgx.geodatenzentrum.de/gdz_basemapworld_vektor/styles/bm_web_wld_col.json';
const MAX_POPUP_GROUPS = 3;  // for how many groups display the number of images when they overlap

// Fetch and run the MapLibre library from the CDN (cached after the first
// call). It registers itself as the global variable `maplibregl`.
async function loadMaplibre() {
  await import(MAPLIBRE_SCRIPT_URL);
  return maplibregl;
}

// Group the query rows by coordinate and build one GeoJSON
// feature (geometry + arbitrary `properties`) per distinct position.
// The images' {name, group} list is stored as a JSON string because MapLibre
// only preserves flat properties when it hands features back in hover events.
function groupedPointFeatures(rows) {
  const rowsPerCoord = new Map();  // "lat,lon" -> rows at exactly that position
  for (const row of rows) {
    const key = `${row.lat},${row.lon}`;
    if (!rowsPerCoord.has(key)) rowsPerCoord.set(key, []);
    rowsPerCoord.get(key).push(row);
  }

  const features = [];
  for (const rowsHere of rowsPerCoord.values()) {
    const members = rowsHere.map(row => ({ name: row.name, group: row.__group__ }));
    features.push({
      type: 'Feature',
      geometry: { type: 'Point', coordinates: [Number(rowsHere[0].lon), Number(rowsHere[0].lat)] },
      properties: { count: rowsHere.length, members: JSON.stringify(members) },
    });
  }
  return features;
}

// The {name, group} members of every feature stacked at the hovered pixel.
function membersOf(features) {
  const members = [];
  for (const feature of features) {
    members.push(...JSON.parse(feature.properties.members));
  }
  return members;
}

function abbreviatedCount(valueExpr) {
  // shorten numberss: 2000 → "2k", 2345 → "2.3k", 3200000 → "3.2M"
  return ['case',
    ['>=', valueExpr, 1e6], ['concat', ['/', ['round', ['/', valueExpr, 1e5]], 10], 'M'],
    ['>=', valueExpr, 1e3], ['concat', ['/', ['round', ['/', valueExpr, 1e2]], 10], 'k'],
    ['to-string', valueExpr],
  ];
}

// Add the image positions to the map as cluster badges, aggregated by
// on-screen proximity.
function addPointsLayers(map, rows, { radius, bounds, showCount = false }) {
  const features = groupedPointFeatures(rows);
  if (bounds) {
    for (const feature of features) bounds.extend(feature.geometry.coordinates);
  }

  map.addSource('points-clustered', {
    type: 'geojson',
    data: { type: 'FeatureCollection', features },
    cluster:          true,
    clusterRadius:    20,  // merge features within 20 screen pixels
    clusterMinPoints: 1,   // even a lone coordinate becomes a (1-member) cluster
    clusterMaxZoom:   24,  // never un-cluster (features would lose cluster_id/total)
    clusterProperties: { total: ['+', ['get', 'count']] },  // sum image counts, not feature counts
  });

  map.addLayer({
    id:     'points-cluster',
    type:   'circle',
    source: 'points-clustered',
    paint: {
      'circle-radius':       radius + 6,
      'circle-color':        '#4a90d9',
      'circle-stroke-color': '#fff',
      'circle-stroke-width': 1,
    },
  });

  if (showCount) {  // don't show numbers in condensed view
    // Clustered features carry the summed 'total'; a feature the clustering
    // left unwrapped only has its own 'count' — hence the coalesce.
    const countExpr = ['coalesce', ['get', 'total'], ['get', 'count']];
    map.addLayer({
      id:     'points-cluster-count',
      type:   'symbol',  // 'symbol' layers draw text (and/or icons)
      source: 'points-clustered',
      layout: { 'text-field': abbreviatedCount(countExpr), 'text-size': 12 },
      paint:  { 'text-color': '#fff' },
    });
  }
}

// ---------------------------------------------------------------------------
// Hover popups
// ---------------------------------------------------------------------------

// Popup text for the images at one hovered spot. A single image shows its
// name; several show a count plus a per-group tally capped at
// MAX_POPUP_GROUPS entries. `members` is a list of {name, group} objects.
function popupHtml(ctx, members, kindLabel) {
  const escapeHtml = ctx.plot.escapeHtml;
  const groupingLabel = ctx.plot.groupingLabel();  // '' when no grouping is selected

  if (members.length === 1) {
    let html = escapeHtml(String(members[0].name ?? ''));
    if (groupingLabel) {
      html += `<br><b>${escapeHtml(groupingLabel)}:</b> ${escapeHtml(ctx.groupLabel(members[0].group))}`;
    }
    return html;
  }

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

function addHoverPopup(map, popup, layerId, anchor, htmlForEvent) {
  map.on('mouseenter', layerId, async (event) => {
    map.getCanvas().style.cursor = 'pointer';
    const lngLat = anchor === 'mouse' ? event.lngLat : event.features[0].geometry.coordinates;
    popup.setLngLat(lngLat);
    popup.setHTML(await htmlForEvent(event));
    popup.addTo(map);
  });
  map.on('mouseleave', layerId, () => {
    map.getCanvas().style.cursor = '';
    popup.remove();
  });
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

  async condensedPlot(container, ctx) {
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
    });

    // add points when map has loaded
    map.on('load', () => {
      const bounds = new maplibregl.LngLatBounds();
      addPointsLayers(map, rows, { radius: 3, bounds });
      map.fitBounds(bounds, { padding: 20, maxZoom: 8, animate: false });
    });

    return true;
  },

  async render(container, ctx) {
    const rows = await ctx.queryRows(`
      SELECT
        "latitude"  AS lat,
        "longitude" AS lon,
        "name" AS name,
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
    });
    map.addControl(new maplibregl.ScaleControl());
    map.addControl(new maplibregl.NavigationControl());

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
          properties: { name: row.name, group: row.__group__ },
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
        paint: { 'fill-color': '#4a90d9', 'fill-opacity': 0.15 },
      });
      map.addLayer({
        id:     'footprints-outline',
        type:   'line',
        source: 'footprints',
        paint: { 'line-color': '#4a90d9', 'line-width': 1.5 },
      });
      // on top, draw points
      addPointsLayers(map, rows, { radius: 6, bounds, showCount: true });

      // One shared popup instance, moved and refilled on each hover.
      const popup = new maplibregl.Popup({ offset: 25, closeOnClick: false });

      // Cluster badges: a cluster feature only carries the aggregate total,
      // so ask the source for the member features behind it.
      addHoverPopup(map, popup, 'points-cluster', 'feature', async (event) => {
        const clusterId = event.features[0].properties.cluster_id;
        let members;
        if (clusterId !== undefined) {
          const leaves = await map.getSource('points-clustered').getClusterLeaves(clusterId, 1e6, 0);
          members = membersOf(leaves);
        } else {
          members = membersOf(event.features);  // feature the clustering left unwrapped
        }
        return popupHtml(ctx, members, 'points');
      });

      // Footprint polygons (anchor at the mouse, not the polygon center).
      addHoverPopup(map, popup, 'footprints-fill', 'mouse',
        (event) => popupHtml(ctx, event.features.map(f => f.properties), 'footprints'));

      map.fitBounds(bounds, { padding: 50, maxZoom: 10 });
    });

    // Redraw the map whenever the container is resized
    const observer = new ResizeObserver(() => {
      if (map) map.resize();
    });
    observer.observe(container);
  },
};
