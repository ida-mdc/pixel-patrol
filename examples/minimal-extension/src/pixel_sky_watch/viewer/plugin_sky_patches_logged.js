/**
 * Sky Patches Logged plugin — part of the Pixel Sky Watch Extension.
 *
 * A widget about the *fake image metadata* itself (not anything derived from
 * pixel data): it tallies how many sky patches were logged at each time of
 * day, split by how cloudy it was — both fields read straight out of the
 * parquet schema metadata by SkyPatchLoader, exactly the way a real loader
 * would surface OME-XML/EXIF acquisition info.
 *
 * Load via the extension manifest (recommended):
 *   ?extension=https://your-host/extension.json
 *
 * Or standalone:
 *   ?plugin=https://your-host/plugin_sky_patches_logged.js
 */

const TIME_ORDER  = ['dawn', 'day', 'dusk', 'night'];
const COVER_ORDER = ['clear', 'cloudy'];
const COVER_COLOR = { clear: '#ffd54f', cloudy: '#90a4ae' };

export default {
  id:    'sky-patches-logged',
  label: 'Sky Patches Logged',
  group: 'Pixel Sky Watch',

  requires(schema) {
    return ['time_of_day', 'cloud_cover'].every(c => schema.allCols.includes(c));
  },

  async render(container, ctx) {
    const rows = await ctx.queryRows(`
      SELECT "time_of_day" AS time_of_day, "cloud_cover" AS cloud_cover, COUNT(*) AS cnt
      FROM pp_data
      WHERE "time_of_day" IS NOT NULL AND "cloud_cover" IS NOT NULL
        ${ctx.where ? 'AND ' + ctx.where.replace(/^WHERE\s+/i, '') : ''}
      GROUP BY 1, 2
    `);

    if (!rows.length) {
      container.textContent = 'No sky patch metadata available.';
      return;
    }

    const times = TIME_ORDER.filter(t => rows.some(r => r.time_of_day === t));
    const covers = COVER_ORDER.filter(c => rows.some(r => r.cloud_cover === c));

    Plotly.newPlot(container, covers.map(cover => ({
      type: 'bar',
      name: cover,
      x: times,
      y: times.map(t => {
        const match = rows.find(r => r.time_of_day === t && r.cloud_cover === cover);
        return match ? Number(match.cnt) : 0;
      }),
      marker: { color: COVER_COLOR[cover] ?? '#888' },
    })), {
      title: { text: 'How many sky patches were logged, and how cloudy was it?' },
      barmode: 'stack',
      height: 380,
      margin: { l: 50, r: 20, t: 50, b: 60 },
      xaxis: { title: 'Time of day', categoryarray: times, categoryorder: 'array' },
      yaxis: { title: 'Patches logged', rangemode: 'tozero', dtick: 1 },
    }, { responsive: true });
  },
};
