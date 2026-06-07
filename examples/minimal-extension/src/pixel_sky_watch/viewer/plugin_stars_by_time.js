/**
 * Stars by Time of Day plugin — part of the Pixel Sky Watch Extension.
 *
 * Plots the star_count computed by StarSpotterProcessor (from the actual
 * pixel data) against the time_of_day fake-metadata field read straight from
 * the parquet "sky patch" files, split and colored by group (folder) just
 * like the built-in widgets. A jittered scatter is used (rather than a
 * box/violin) because each category only holds a handful of points — exactly
 * the kind of small sample where distributional summaries would mislead.
 *
 * Load via the extension manifest (recommended):
 *   ?extension=https://your-host/extension.json
 *
 * Or standalone:
 *   ?plugin=https://your-host/plugin_stars_by_time.js
 */

const TIME_ORDER = ['dawn', 'day', 'dusk', 'night'];

export default {
  id:    'stars-by-time-of-day',
  label: 'Stars Spotted by Time of Day',
  group: 'Pixel Sky Watch',

  requires(schema) {
    return ['time_of_day', 'star_count'].every(c => schema.allCols.includes(c));
  },

  async render(container, ctx) {
    const rows = await ctx.queryRows(`
      SELECT
        "time_of_day"         AS time_of_day,
        "imported_path_short" AS folder,
        "star_count"          AS stars
      FROM pp_data
      WHERE "time_of_day" IS NOT NULL AND "star_count" IS NOT NULL
        ${ctx.where ? 'AND ' + ctx.where.replace(/^WHERE\s+/i, '') : ''}
    `);

    if (!rows.length) {
      container.textContent = 'No sky patch data available.';
      return;
    }

    const times   = TIME_ORDER.filter(t => rows.some(r => r.time_of_day === t));
    const folders = [...new Set(rows.map(r => r.folder))].sort();

    // Jitter the x position within each time-of-day category so overlapping
    // points (e.g. two patches both logged at "night") stay visible.
    const xJitter = () => (Math.random() - 0.5) * 0.5;

    Plotly.newPlot(container, folders.map(folder => {
      const sub = rows.filter(r => r.folder === folder);
      return {
        type: 'scatter',
        mode: 'markers',
        name: folder,
        x: sub.map(r => times.indexOf(r.time_of_day) + xJitter()),
        y: sub.map(r => Number(r.stars)),
        text: sub.map(r => r.time_of_day),
        hovertemplate: '%{text}: %{y} stars<extra>%{fullData.name}</extra>',
        marker: { size: 12, color: ctx.colorMap[folder] ?? '#888' },
      };
    }), {
      title: { text: 'How many stars show up at each time of day?' },
      height: 420,
      margin: { l: 60, r: 20, t: 50, b: 60 },
      xaxis: {
        title: 'Time of day',
        tickmode: 'array',
        tickvals: times.map((_, i) => i),
        ticktext: times,
        range: [-0.5, times.length - 0.5],
      },
      yaxis: { title: 'Stars spotted', rangemode: 'tozero' },
    }, { responsive: true });
  },
};
