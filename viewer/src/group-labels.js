/**
 * Build a mapping from original group values to short display labels.
 *
 * @param {string[]} groups  distinct group values
 * @returns {Record<string, string>}  object mapping each original value to its display label
 */
export function buildGroupLabels(groups) {
  if (groups.length <= 1) return Object.fromEntries(groups.map(g => [g, g]));

  const maxLen = Math.max(...groups.map(g => g.length));
  if (maxLen <= 20) return Object.fromEntries(groups.map(g => [g, g]));

  const pathCount = groups.filter(g => g.includes('/') || g.includes('\\')).length;
  if (pathCount > groups.length / 2) return buildPathLabels(groups);

  return buildPrefixSuffixLabels(groups);
}

function buildPathLabels(groups) {
  const sep = groups.some(g => g.includes('/')) ? '/' : '\\';
  const splitParts = groups.map(g => g.split(sep));
  const maxParts = Math.max(...splitParts.map(p => p.length));

  for (let n = 1; n <= maxParts; n++) {
    const labels = splitParts.map(parts => parts.slice(-n).join(sep));
    if (new Set(labels).size === labels.length) {
      const result = {};
      for (let i = 0; i < groups.length; i++) {
        const parts = splitParts[i];
        result[groups[i]] = parts.length <= n
          ? groups[i]
          : `...${sep}${parts.slice(-n).join(sep)}`;
      }
      return result;
    }
  }

  return Object.fromEntries(groups.map(g => [g, g]));
}

function buildPrefixSuffixLabels(groups) {
  let prefix = groups[0];
  for (const g of groups.slice(1)) {
    let i = 0;
    while (i < prefix.length && i < g.length && prefix[i] === g[i]) i++;
    prefix = prefix.slice(0, i);
  }

  const hasPrefix = prefix.length >= 10;
  if (!hasPrefix) return Object.fromEntries(groups.map(g => [g, g]));

  let suffix = groups[0];
  for (const g of groups.slice(1)) {
    let i = 0;
    while (i < suffix.length && i < g.length && suffix[suffix.length - 1 - i] === g[g.length - 1 - i]) i++;
    suffix = i > 0 ? suffix.slice(suffix.length - i) : '';
  }

  const hasSuffix = suffix.length >= 10;

  const shorts = groups.map(g => {
    let core = g.slice(prefix.length);
    if (hasSuffix) core = core.slice(0, core.length - suffix.length);
    return hasSuffix ? `...${core}...` : `...${core}`;
  });

  const unique = new Set(shorts).size === shorts.length;
  const nonEmpty = shorts.every(s => {
    const withoutLeading = s.startsWith('...') ? s.slice(3) : s;
    const core = withoutLeading.endsWith('...') ? withoutLeading.slice(0, -3) : withoutLeading;
    return core.length > 0;
  });

  if (unique && nonEmpty) return Object.fromEntries(groups.map((g, i) => [g, shorts[i]]));

  return Object.fromEntries(groups.map(g => [g, g]));
}