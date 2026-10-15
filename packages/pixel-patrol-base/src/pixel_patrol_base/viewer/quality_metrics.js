// Quality-signal columns, matched by name regardless of which processor produced them
// (mostly QualityMetricsProcessor; nan_fraction comes from BasicMetricsProcessor).
// `hintUp`/`hintDown`/`goodDirection` are only set where the reading is a well-established, monotonic one.
export const QUALITY_METRIC_INFO = {
  laplacian_variance: {
    desc: 'High-frequency content measure. Values depend on both image quality and scene content: ' +
          'blur and heavy compression push values down; sharpness, noise, and fine texture push ' +
          'values up. A smooth, featureless scene will score low regardless ' +
          'of focus quality. Clipped pixels create artificial edges at clip boundaries and inflate ' +
          'the score.',
    hintUp: 'more high-frequency content (sharpness, noise, or fine texture)',
    hintDown: 'less high-frequency content (blur, compression, or smooth content)',
  },
  spectral_slope: {
    desc: 'Log-log slope of the radially averaged power spectrum, fit over the mid-frequency band ' +
          '(5–40% of Nyquist). Typical range: −2 to −4. Closer to 0 indicates noise or uniform ' +
          'content. More negative values indicate blur or stronger low-frequency dominance. ' +
          'Most interpretable when comparing images of similar content type.',
    hintUp: 'flatter spectrum (noise or uniform content)',
    hintDown: 'steeper spectrum (blur or low-frequency dominance)',
    nullReason: 'Requires a minimum spatial extent of 32 × 32 px.',
  },
  dark_clipping_fraction: {
    desc: 'Fraction of pixels at the dtype\'s minimum representable value (0 for unsigned integers; ' +
          'NaN for float, which has no fixed lower bound). Detects underexposure, background, and ' +
          'nodata at the dark end.',
    hintUp: 'more dark-clipped pixels', hintDown: 'fewer dark-clipped pixels', goodDirection: 'down',
  },
  bright_clipping_fraction: {
    desc: 'Fraction of pixels at the dtype\'s maximum representable value (integer types only; NaN for float).',
    hintUp: 'more bright-clipped pixels', hintDown: 'fewer bright-clipped pixels', goodDirection: 'down',
  },
  nan_fraction: {
    desc: 'Fraction of pixels excluded as NaN (missing/invalid data).',
    hintUp: 'more NaN pixels', hintDown: 'fewer NaN pixels', goodDirection: 'down',
  },
};

export const QUALITY_METRIC_BASES = new Set(Object.keys(QUALITY_METRIC_INFO));

/** Look up a quality metric's shared metadata by column name, stripping any per-channel suffix. */
export function describeQualityMetric(col) {
  for (const [base, info] of Object.entries(QUALITY_METRIC_INFO)) {
    if (col === base || col.startsWith(base + '_')) return info;
  }
  return null;
}

// QUALITY_METRIC_INFO's key order is curated by generalization/importance (sharpness,
// noise, exposure clipping, contrast/texture, then compression artifacts last - most
// niche/situational). Parquet column order is alphabetical (an internal processing
// detail unrelated to this), so widgets sort explicitly by this instead.
const QUALITY_METRIC_ORDER = Object.keys(QUALITY_METRIC_INFO);

/** Sort rank for a metric column by curated importance order; ties (e.g. non-quality
 * columns) sort last and preserve their original relative order (stable sort). */
export function qualityMetricRank(col) {
  const idx = QUALITY_METRIC_ORDER.findIndex(base => col === base || col.startsWith(base + '_'));
  return idx === -1 ? Number.MAX_SAFE_INTEGER : idx;
}
