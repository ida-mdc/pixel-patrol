/**
 * Word Frequency plugin — part of the Markdown Diary Extension.
 *
 * Shows a bar chart of the most frequent words in the diary entries,
 * derived from the free_text column produced by MarkdownDiaryLoader.
 *
 * Load via the extension manifest (recommended):
 *   ?extension=https://your-host/extension.json
 *
 * Or standalone:
 *   ?plugin=https://your-host/plugin_word_frequency.js
 */

const STOPWORDS = new Set([
  'the','a','an','and','or','but','in','on','at','to','of','for',
  'is','it','its','was','are','be','been','have','has','had','with',
  'this','that','they','i','my','me','we','you','your','he','she',
  'his','her','not','no','so','if','as','by','do','did','from','up',
]);

export default {
  id:    'diary-word-frequency',
  label: 'Word Frequency',

  requires(schema) {
    return schema.allCols.includes('free_text');
  },

  async render(container, ctx) {
    const rows = await ctx.queryRows(`
      SELECT "free_text" FROM pp_data
      WHERE "free_text" IS NOT NULL
        ${ctx.where ? 'AND ' + ctx.where.replace(/^WHERE\s+/i, '') : ''}
    `);

    const freq = {};
    for (const { free_text } of rows) {
      for (const word of free_text.toLowerCase().match(/\b[a-z]{3,}\b/g) ?? []) {
        if (!STOPWORDS.has(word)) freq[word] = (freq[word] ?? 0) + 1;
      }
    }

    const top = Object.entries(freq)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 40);

    if (!top.length) {
      container.textContent = 'No text data available.';
      return;
    }

    Plotly.newPlot(container, [{
      type: 'bar',
      x: top.map(([w]) => w),
      y: top.map(([, c]) => c),
      marker: { color: Object.values(ctx.colorMap)[0] ?? '#4c72b0' },
    }], {
      title:  { text: 'Top 40 Words in Diary Entries' },
      height: 420,
      margin: { l: 40, r: 20, t: 50, b: 100 },
      xaxis:  { title: 'Word', tickangle: -45 },
      yaxis:  { title: 'Count' },
    }, { responsive: true });
  },
};
