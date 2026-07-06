/**
 * Map Points plugin - shows locations on an interactive MapLibre map.
 *
 * Reads `latitude`, `longitude`, and `footprint` (GeoJSON string) from `pp_data`.
 */

export default {
  id:    'map-points',
  label: 'Locations',
  group: 'Geospatial extension',
  scope: 'image',

  requires(schema) {
    const cols = ['latitude', 'longitude', 'footprint'];
    return cols.every(c => schema.allCols.includes(c));
  },

  async render(container, ctx) {
    const { andWhere } = ctx.sql;

    const rows = await ctx.queryRows(`
      SELECT 
        "latitude"  AS lat,
        "longitude" AS lon,
        "name" AS name,
        "footprint" as footprint
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
      // map licence: CC BY 4.0 (https://sgx.geodatenzentrum.de/web_public/gdz/lizenz/deu/Nutzungsbedingungen_basemapworld.pdf)
      style: 'https://sgx.geodatenzentrum.de/gdz_basemapworld_vektor/styles/bm_web_wld_col.json',
      center: [0, 0],
      zoom: 2,
      attributionControl: true,
      pitch: 0,
    });

    // Add basic controls
    map.addControl(new maplibregl.NavigationControl());

    // Once map loads, add data & markers
    map.on('load', () => {
      const bounds = new maplibregl.LngLatBounds();

      const footprintFeatures = rows
          .filter(r => r.footprint != null)
          .map(r => {
            const geometry = JSON.parse(r.footprint);
            // Extend bounds with all polygon corners
            geometry.coordinates[0].forEach(([lon, lat]) => bounds.extend([lon, lat]));
            return {
              type: 'Feature',
              geometry,
              properties: { name: r.name },
            };
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
          paint: {
            'fill-color':   '#4a90d9',
            'fill-opacity': 0.15,
          },
        });

        // Outline
        map.addLayer({
          id:     'footprints-outline',
          type:   'line',
          source: 'footprints',
          paint: {
            'line-color': '#4a90d9',
            'line-width': 1.5,
          },
        });

      // Add GeoJSON source
      const pointFeatures = rows.map(r => ({
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
        data: { type: 'FeatureCollection', features: pointFeatures },
      });

      // Add circle layer for points
      map.addLayer({
        id:     'points-layer',
        type:   'circle',
        source: 'points',
        paint: {
          'circle-radius': 6,
          'circle-color': [
            'match', ['get', 'site'],
            ...Object.entries(ctx.colorMap)
              .flatMap(([site, color]) => [site, color])
              .concat(['#888']),
          ],
          'circle-stroke-color': '#fff',
          'circle-stroke-width': 1,
        },
      });


      // Hover tooltip (points)
      const popup = new maplibregl.Popup({ offset: 25, closeOnClick: false });

      map.on('mouseenter', 'points-layer', e => {
        map.getCanvas().style.cursor = 'pointer';
        const { name } = e.features[0].properties;
        popup.setLngLat(e.features[0].geometry.coordinates)
             .setText(`${name}`)
             .addTo(map);
      });
      map.on('mouseleave', 'points-layer', () => {
        map.getCanvas().style.cursor = '';
        popup.remove();
      });


      // Hover tooltip (footprints)
      map.on('mouseenter', 'footprints-fill', e => {
        map.getCanvas().style.cursor = 'pointer';
        const { name } = e.features[0].properties;
        // Use mouse position rather than geometry center for polygons
        popup.setLngLat(e.lngLat)
             .setText(name)
             .addTo(map);
      });

      map.on('mouseleave', 'footprints-fill', () => {
        map.getCanvas().style.cursor = '';
        popup.remove();
      });

      // Zoom to extent (simple fitBounds via bounds)
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