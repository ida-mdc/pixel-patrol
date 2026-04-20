/**
 * Mood Trend plugin — part of the Markdown Diary Extension.
 *
 * Shows a stacked bar chart of mood occurrences per folder and a
 * year-over-year positivity trend line chart.
 *
 * Load via the extension manifest (recommended):
 *   ?extension=https://your-host/extension.json
 *
 * Or standalone:
 *   ?plugin=https://your-host/plugin_mood_trend.js
 */

export default {
  id:    'diary-mood-trend',
  label: 'Mood Trend',

  requires(schema) {
    return ['moods', 'entry_date', 'positivity_factor', 'imported_path_short']
      .every(c => schema.allCols.includes(c));
  },

  async render(container, ctx) {
    const moodRows = await ctx.queryRows(`
      SELECT
        mood,
        "imported_path_short" AS folder,
        COUNT(*) AS cnt
      FROM pp_data, UNNEST("moods") AS t(mood)
      ${ctx.where}
      GROUP BY 1, 2
      ORDER BY 1, 2
    `);

    const trendRows = await ctx.queryRows(`
      SELECT
        "imported_path_short"                        AS folder,
        YEAR(TRY_CAST("entry_date" AS DATE))         AS yr,
        STRFTIME(TRY_CAST("entry_date" AS DATE), '%m-%d') AS md,
        AVG("positivity_factor")                     AS mean_pos
      FROM pp_data
      WHERE "entry_date" IS NOT NULL AND "positivity_factor" IS NOT NULL
        ${ctx.where ? 'AND ' + ctx.where.replace(/^WHERE\s+/i, '') : ''}
      GROUP BY 1, 2, 3
      ORDER BY 1, 2, 3
    `);

    const folders = [...new Set([
      ...moodRows.map(r => r.folder),
      ...trendRows.map(r => r.folder),
    ])].sort();

    container.style.cssText = 'display:flex;flex-wrap:wrap;gap:16px';

    // Append both divs before rendering so Plotly measures the correct flex width.
    const countsDiv = document.createElement('div');
    const trendDiv  = document.createElement('div');
    countsDiv.style.cssText = 'flex:1 1 320px;min-width:0';
    trendDiv.style.cssText  = 'flex:1 1 320px;min-width:0';
    container.appendChild(countsDiv);
    container.appendChild(trendDiv);

    const moods = [...new Set(moodRows.map(r => r.mood))].sort();
    Plotly.newPlot(countsDiv, folders.map(folder => {
      const byMood = Object.fromEntries(
        moodRows.filter(r => r.folder === folder).map(r => [r.mood, Number(r.cnt)])
      );
      return {
        type: 'bar', name: folder,
        x: moods, y: moods.map(m => byMood[m] ?? 0),
        marker: { color: ctx.colorMap[folder] ?? '#888' },
      };
    }), {
      title: { text: 'Mood Occurrences per Folder' }, barmode: 'stack',
      height: 420, margin: { l: 40, r: 20, t: 50, b: 80 },
      xaxis: { title: 'Mood' }, yaxis: { title: 'Count' },
    }, { responsive: true });

    Plotly.newPlot(trendDiv, folders.flatMap(folder => {
      const rows = trendRows.filter(r => r.folder === folder);
      return [...new Set(rows.map(r => r.yr))].sort().map(yr => {
        const sub = rows.filter(r => r.yr === yr).sort((a, b) => a.md < b.md ? -1 : 1);
        return {
          type: 'scatter', mode: 'lines+markers', name: String(folder),
          x: sub.map(r => `2000-${r.md}`), y: sub.map(r => Number(r.mean_pos)),
          line: { color: ctx.colorMap[folder] ?? '#333' },
          marker: { color: ctx.colorMap[folder] ?? '#333' },
          hovertemplate: '%{x|%d/%m} · %{y:.3f}<extra>' + folder + '</extra>',
        };
      });
    }), {
      title: { text: 'Mean Positivity Over the Year' },
      height: 420, margin: { l: 40, r: 20, t: 50, b: 60 },
      xaxis: { tickformat: '%d/%m', title: 'Date (day/month)', range: ['2000-01-01', '2000-12-31'] },
      yaxis: { title: 'Mean Positivity' },
    }, { responsive: true });
  },
};
