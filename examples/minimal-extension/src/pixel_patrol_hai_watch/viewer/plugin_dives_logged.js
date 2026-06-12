/**
 * Dives Logged plugin - part of the Pixel HAI Watch Extension.
 *
 * A widget about the *fake image metadata* itself (not anything derived from
 * pixel data): it tallies how many dive snapshots were logged in each ocean
 * layer, split by dive site - "depth_zone" is read straight out of the
 * parquet schema metadata by SharkCamLoader, exactly the way a real loader
 * would surface OME-XML/EXIF acquisition info; "site" is the
 * `imported_path_short` grouping every Pixel Patrol report already carries.
 *
 * Load via the extension manifest (recommended):
 *   ?extension=https://your-host/extension.json
 *
 * Or standalone:
 *   ?plugin=https://your-host/plugin_dives_logged.js
 */

const DEPTH_ORDER = ['sunlit', 'twilight', 'midnight', 'abyss'];

export default {
  id:    'dives-logged',
  label: 'Dives Logged',
  group: 'Pixel HAI Watch',
  scope: 'image',

  requires(schema) {
    return schema.allCols.includes('depth_zone');
  },

  async render(container, ctx) {
    const { andWhere } = ctx.sql;
    const rows = await ctx.queryRows(`
      SELECT "depth_zone" AS depth_zone, "imported_path_short" AS site, COUNT(*) AS cnt
      FROM pp_data
      ${andWhere(ctx.where, '"depth_zone" IS NOT NULL')}
      GROUP BY 1, 2
    `);

    if (!rows.length) {
      container.textContent = 'No dive metadata available.';
      return;
    }

    const zones = DEPTH_ORDER.filter(z => rows.some(r => r.depth_zone === z));
    const sites = [...new Set(rows.map(r => r.site))].sort();

    Plotly.newPlot(container, sites.map(site => ({
      type: 'bar',
      name: site,
      x: zones,
      y: zones.map(z => {
        const match = rows.find(r => r.depth_zone === z && r.site === site);
        return match ? Number(match.cnt) : 0;
      }),
      marker: { color: ctx.colorMap[site] ?? '#888' },
    })), {
      title: { text: 'How many dives were logged in each ocean layer?' },
      barmode: 'stack',
      height: 380,
      margin: { l: 50, r: 20, t: 50, b: 60 },
      xaxis: { title: 'Depth zone (shallow → deep)', categoryarray: zones, categoryorder: 'array' },
      yaxis: { title: 'Dives logged', rangemode: 'tozero', dtick: 1 },
    }, { responsive: true });
  },
};
