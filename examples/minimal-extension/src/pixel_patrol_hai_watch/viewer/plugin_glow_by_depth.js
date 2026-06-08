/**
 * Glow by Depth plugin - part of the Pixel HAI Watch Extension.
 *
 * Plots the glow_count computed by GlowSpotterProcessor (from the actual
 * pixel data) against the depth_zone fake-metadata field read straight from
 * the parquet "dive patch" files, split and colored by group (dive site)
 * just like the built-in widgets. A jittered scatter is used (rather than a
 * box/violin) because each category only holds a handful of points - exactly
 * the kind of small sample where distributional summaries would mislead.
 *
 * Load via the extension manifest (recommended):
 *   ?extension=https://your-host/extension.json
 *
 * Or standalone:
 *   ?plugin=https://your-host/plugin_glow_by_depth.js
 */

const DEPTH_ORDER = ['sunlit', 'twilight', 'midnight', 'abyss'];

export default {
  id:    'glow-by-depth',
  label: 'Glow Sightings by Depth',
  group: 'Pixel HAI Watch',

  requires(schema) {
    return ['depth_zone', 'glow_count'].every(c => schema.allCols.includes(c));
  },

  async render(container, ctx) {
    const { andWhere } = ctx.sql;
    const rows = await ctx.queryRows(`
      SELECT
        "depth_zone"          AS depth_zone,
        "imported_path_short" AS site,
        "glow_count"          AS glows
      FROM pp_data
      ${andWhere(ctx.where, '"depth_zone" IS NOT NULL AND "glow_count" IS NOT NULL')}
    `);

    if (!rows.length) {
      container.textContent = 'No dive patch data available.';
      return;
    }

    const zones = DEPTH_ORDER.filter(z => rows.some(r => r.depth_zone === z));
    const sites = [...new Set(rows.map(r => r.site))].sort();

    // Jitter the x position within each depth-zone category so overlapping
    // points (e.g. two patches both logged in the "abyss") stay visible.
    const xJitter = () => (Math.random() - 0.5) * 0.5;

    Plotly.newPlot(container, sites.map(site => {
      const sub = rows.filter(r => r.site === site);
      return {
        type: 'scatter',
        mode: 'markers',
        name: site,
        x: sub.map(r => zones.indexOf(r.depth_zone) + xJitter()),
        y: sub.map(r => Number(r.glows)),
        text: sub.map(r => r.depth_zone),
        hovertemplate: '%{text}: %{y} glows<extra>%{fullData.name}</extra>',
        marker: { size: 12, color: ctx.colorMap[site] ?? '#888' },
      };
    }), {
      title: { text: 'How much bioluminescent glow shows up at each depth?' },
      height: 420,
      margin: { l: 60, r: 20, t: 50, b: 60 },
      xaxis: {
        title: 'Depth zone (shallow → deep)',
        tickmode: 'array',
        tickvals: zones.map((_, i) => i),
        ticktext: zones,
        range: [-0.5, zones.length - 0.5],
      },
      yaxis: { title: 'Glows spotted', rangemode: 'tozero' },
    }, { responsive: true });
  },
};
