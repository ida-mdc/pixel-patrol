/**
 * Map Points plugin - shows locations on an interactive MapLibre map.
 *
 * Reads `latitude` and `longitude` from `pp_data`, filters out nulls.
 */

export default {
  id:    'map-points',
  label: 'Locations',
  group: 'Geospatial extension',
  scope: 'image',

  requires(schema) {
    const cols = ['latitude', 'longitude'];
    return cols.every(c => schema.allCols.includes(c));
  },

  async render(container, ctx) {
    const { andWhere } = ctx.sql;

    const rows = await ctx.queryRows(`
      SELECT 
        "latitude"  AS lat,
        "longitude" AS lon,
        "name" AS name
      FROM pp_data
      ${andWhere(ctx.where, '"latitude" IS NOT NULL AND "longitude" IS NOT NULL')}
    `);

    if (!rows.length) {
      container.textContent = 'No geographic data available.';
      return;
    }

    container.style.height = '600px';      // fixed height — simple & reliable
    container.style.minHeight = '200px';   // fallback min-height
    container.style.width = '100%';
    container.style.position = 'relative'; // helps with map positioning

    await import('https://unpkg.com/maplibre-gl@latest/dist/maplibre-gl.js');

    // Initialize map (lightweight dark style)
    const map = new maplibregl.Map({
      container: container,
      style: 'https://demotiles.maplibre.org/style.json',
      center: [0, 0],
      zoom: 2,
      attributionControl: true,
      pitch: 0,
    });

    // Add basic controls
    map.addControl(new maplibregl.NavigationControl());

    // Once map loads, add data & markers
    map.on('load', () => {
      // Add GeoJSON source
      const features = rows.map(r => ({
        type: 'Feature',
        geometry: {
          type: 'Point',
          coordinates: [Number(r.lon), Number(r.lat)], // [lon, lat] GeoJSON order
        },
        properties: {
          name: r.name,
          lat: r.lat,
          lon: r.lon,
        },
      }));

      map.addSource('points', {
        type: 'geojson',
        data: {
          type: 'FeatureCollection',
          features,
        },
      });

      // Add circle layer for points
      map.addLayer({
        id: 'points-layer',
        type: 'circle',
        source: 'points',
        paint: {
          'circle-radius': 6,
          'circle-color': [
            'match',
            ['get', 'site'],
            ...Object.entries(ctx.colorMap)
              .flatMap(([site, color]) => [site, color])
              .concat(['#888']), // default
          ],
          'circle-stroke-color': '#fff',
          'circle-stroke-width': 1,
        },
      });

      // Add hover interaction (tooltip)
      const popup = new maplibregl.Popup({ offset: 25, closeOnClick: false });

      map.on('mouseenter', 'points-layer', e => {
        map.getCanvas().style.cursor = 'pointer';
        const { name, lat, lon } = e.features[0].properties;
        popup.setLngLat(e.features[0].geometry.coordinates)
             .setText(`${name}`)
             .addTo(map);
      });

      map.on('mouseleave', 'points-layer', () => {
        map.getCanvas().style.cursor = '';
        popup.remove();
      });

      // Zoom to extent (simple fitBounds via bounds)
      const bounds = new maplibregl.LngLatBounds();
      features.forEach(f => bounds.extend(f.geometry.coordinates));
      map.fitBounds(bounds, { padding: 50, maxZoom: 10 });
    });

    // Optional: handle resize (if container changes)
    const observer = new ResizeObserver(() => {
      if (map) map.resize();
    });
    observer.observe(container);
  },
};