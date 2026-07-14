/**
 * Map Points plugin - shows locations on an interactive MapLibre map.
 *
 * Reads `latitude`, `longitude`, and `footprint` (GeoJSON string).
 */

const MAPLIBRE_SCRIPT_URL = 'https://unpkg.com/maplibre-gl@latest/dist/maplibre-gl.js';
// map licence: CC BY 4.0 (https://sgx.geodatenzentrum.de/web_public/gdz/lizenz/deu/Nutzungsbedingungen_basemapworld.pdf)
const MAP_STYLE_URL = 'https://sgx.geodatenzentrum.de/gdz_basemapworld_vektor/styles/bm_web_wld_col.json';

// Dynamic import of the same URL resolves from the module cache after the
async function loadMaplibre() {
  await import(MAPLIBRE_SCRIPT_URL);
  return maplibregl;
}

function pointFeature(r) {
  return {
    type: 'Feature',
    geometry: { type: 'Point', coordinates: [Number(r.lon), Number(r.lat)] },
    properties: { name: r.name, lat: r.lat, lon: r.lon, group: r.__group__ },
  };
}

// Same group -> color mapping used by the bar/histogram widgets
// (ctx.color.group), expressed as a MapLibre 'match' expression on the
// feature's 'group' property so it follows the active group-by column.
function groupColorMatch(ctx) {
  return [
    'match', ['get', 'group'],
    ...ctx.groups.flatMap(g => [String(g), ctx.color.group(g)]),
    '#888', // fallback for a group value not in ctx.groups (shouldn't normally happen)
  ];
}

// Popup text: name, plus group value if group is selected
function popupHtml(ctx, { name, group }) {
  const { escapeHtml } = ctx.plot;
  const groupingLabel = ctx.plot.groupingLabel();
  let html = escapeHtml(String(name ?? ''));
  if (groupingLabel) html += `<br>${escapeHtml(groupingLabel)}: ${escapeHtml(ctx.groupLabel(group))}`;
  return html;
}

// Adds a GeoJSON point source + circle layer, extending `bounds` (if given)
// with every point's coordinates.
function addPointsLayer(map, rows, { radius, color, bounds }) {
  const features = rows.map(pointFeature);
  if (bounds) features.forEach(f => bounds.extend(f.geometry.coordinates));
  map.addSource('points', { type: 'geojson', data: { type: 'FeatureCollection', features } });
  map.addLayer({
    id:     'points-layer',
    type:   'circle',
    source: 'points',
    paint: {
      'circle-radius':       radius,
      'circle-color':        color,
      'circle-stroke-color': '#fff',
      'circle-stroke-width': radius > 4 ? 1 : 0.5,
    },
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
    const { andWhere, groupCol: gcFn } = ctx.sql;

    // Points only (no footprints/names), capped at 100 rows — a full render
    // could be backed by 100k+ images and a real MapLibre instance isn't cheap.
    const rows = await ctx.queryRows(`
      SELECT
        "latitude" AS lat,
        "longitude" AS lon,
        ${gcFn()} AS __group__
      FROM pp_data
      ${andWhere(ctx.where, '"latitude" IS NOT NULL AND "longitude" IS NOT NULL')}
      ORDER BY random()
      LIMIT 100
    `);
    if (!rows.length) return false;

    container.style.height = '100%';
    container.style.width = '100%';


    const maplibregl = await loadMaplibre();
    const map = new maplibregl.Map({
      container,
      style: MAP_STYLE_URL,
      center: [0, 0],
      zoom: 1,
      attributionControl: false,
      interactive: false,
    });

    map.on('load', () => {
      const bounds = new maplibregl.LngLatBounds();
      addPointsLayer(map, rows, { radius: 3, color: groupColorMatch(ctx), bounds });
      map.fitBounds(bounds, { padding: 20, maxZoom: 8, animate: false });
    });

    return true;
  },

  async render(container, ctx) {
    const { andWhere, groupCol: gcFn } = ctx.sql;

    const rows = await ctx.queryRows(`
      SELECT
        "latitude"  AS lat,
        "longitude" AS lon,
        "name" AS name,
        "footprint" as footprint,
        ${gcFn()} AS __group__
      FROM pp_data
      ${andWhere(ctx.where, '"latitude" IS NOT NULL AND "longitude" IS NOT NULL')}
    `);

    if (!rows.length) {
      container.textContent = 'No geographic data available.';
      return;
    }

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

    map.addControl(new maplibregl.NavigationControl());

    map.on('load', () => {
      const bounds = new maplibregl.LngLatBounds();

      const footprintFeatures = rows
        .filter(r => r.footprint != null)
        .map(r => {
          const geometry = JSON.parse(r.footprint);
          // Extend bounds with all polygon corners
          geometry.coordinates[0].forEach(([lon, lat]) => bounds.extend([lon, lat]));
          return { type: 'Feature', geometry, properties: { name: r.name, group: r.__group__ } };
        });

      map.addSource('footprints', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: footprintFeatures },
      });

      // Fill — add before points so circles render on top
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

      addPointsLayer(map, rows, { radius: 6, color: groupColorMatch(ctx), bounds });

      // Hover tooltip (points)
      const popup = new maplibregl.Popup({ offset: 25, closeOnClick: false });
      map.on('mouseenter', 'points-layer', e => {
        map.getCanvas().style.cursor = 'pointer';
        const { name, group } = e.features[0].properties;
        popup.setLngLat(e.features[0].geometry.coordinates).setHTML(popupHtml(ctx, { name, group })).addTo(map);
      });
      map.on('mouseleave', 'points-layer', () => {
        map.getCanvas().style.cursor = '';
        popup.remove();
      });

      // Hover tooltip (footprints)
      map.on('mouseenter', 'footprints-fill', e => {
        map.getCanvas().style.cursor = 'pointer';
        const { name, group } = e.features[0].properties;
        // Use mouse position rather than geometry center for polygons
        popup.setLngLat(e.lngLat).setHTML(popupHtml(ctx, { name, group })).addTo(map);
      });
      map.on('mouseleave', 'footprints-fill', () => {
        map.getCanvas().style.cursor = '';
        popup.remove();
      });

      map.fitBounds(bounds, { padding: 50, maxZoom: 10 });
    });

    // Optional: handle resize (if container changes)
    const observer = new ResizeObserver(() => {
      if (map) map.resize();
    });
    observer.observe(container);
  },
};
