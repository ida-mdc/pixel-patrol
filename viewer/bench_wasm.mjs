/**
 * DuckDB WASM parquet benchmark — Node.js version.
 *
 * Must be run from the viewer/ directory (or any directory whose node_modules
 * contains @duckdb/duckdb-wasm) because ESM resolution walks up from this file.
 *
 * Usage:
 *   node bench_wasm.mjs '<json-config>'
 *
 * Config keys:
 *   files      { baseline, shuffled }   absolute local paths
 *   http_base  string                   http://host:port  (optional, activates HTTP mode)
 *   rg_files   { [rg_size]: path }      row-group sweep files
 *   max_imgs   number                   thumbnails to fetch (default 256)
 *   repeats    number                   benchmark repeats   (default 3)
 */

import * as duckdb from '@duckdb/duckdb-wasm';
import { Worker }   from 'worker_threads';
import { createRequire } from 'module';
import path  from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const require   = createRequire(import.meta.url);

// require.resolve returns .../dist/duckdb-node.cjs — dirname gives us the dist/ folder.
const DIST_DIR = path.dirname(require.resolve('@duckdb/duckdb-wasm'));

// ── DuckDB WASM init (same API as the viewer's loader.js) ─────────────────────

async function initDuckDB() {
  const worker = new Worker(path.join(DIST_DIR, 'duckdb-node-mvp.worker.cjs'));
  const logger = new duckdb.VoidLogger();
  const db     = new duckdb.AsyncDuckDB(logger, worker);
  await db.instantiate(path.join(DIST_DIR, 'duckdb-mvp.wasm'));
  const conn = await db.connect();
  return { db, conn };
}

// ── Benchmark helper ──────────────────────────────────────────────────────────

async function timed(label, section, fn, repeats) {
  const times = [];
  for (let i = 0; i < repeats; i++) {
    const t0 = performance.now();
    await fn();
    times.push(performance.now() - t0);
  }
  const mean = times.reduce((a, b) => a + b, 0) / times.length;
  return { label, section, mean_ms: mean, min_ms: Math.min(...times) };
}

// ── Parse config ──────────────────────────────────────────────────────────────

const config   = JSON.parse(process.argv[2]);
const MAX_IMGS = config.max_imgs ?? 256;
const REPEATS  = config.repeats  ?? 3;
const HTTP     = config.http_base ?? '';   // empty → local NODE_FS mode

const { db, conn } = await initDuckDB();
const results = [];

// ── File registration (local mode only) ───────────────────────────────────────

if (!HTTP) {
  for (const [alias, filePath] of Object.entries(config.files ?? {})) {
    if (filePath) {
      await db.registerFileHandle(
        `${alias}.parquet`, filePath,
        duckdb.DuckDBDataProtocol.NODE_FS, true,
      );
    }
  }
  for (const [rg, filePath] of Object.entries(config.rg_files ?? {})) {
    if (filePath) {
      await db.registerFileHandle(
        `rg_${rg}.parquet`, filePath,
        duckdb.DuckDBDataProtocol.NODE_FS, true,
      );
    }
  }
}

function ref(alias, localPath) {
  return HTTP
    ? `'${HTTP}/${path.basename(localPath)}'`
    : `'${alias}.parquet'`;
}

function rgRef(rg, filePath) {
  return HTTP
    ? `'${HTTP}/${path.basename(filePath)}'`
    : `'rg_${rg}.parquet'`;
}

// ── 1. Random sample benchmarks ───────────────────────────────────────────────

if (config.files?.baseline) {
  const r = ref('baseline', config.files.baseline);
  results.push(await timed(
    'A: Baseline + reservoir', 'random',
    () => conn.query(`
      SELECT row_index, thumbnail FROM read_parquet(${r})
      USING SAMPLE ${MAX_IMGS} ROWS (reservoir, 42)
    `),
    REPEATS,
  ));
}

if (config.files?.shuffled) {
  const r = ref('shuffled', config.files.shuffled);
  results.push(await timed(
    'B: Shuffled + LIMIT', 'random',
    () => conn.query(`
      SELECT row_index, thumbnail FROM read_parquet(${r})
      LIMIT ${MAX_IMGS}
    `),
    REPEATS,
  ));
}

// ── 2. Sorted top-N (two-step) ────────────────────────────────────────────────

for (const [alias, localPath, label] of [
  ['baseline', config.files?.baseline, 'B: Two-step on baseline'],
  ['shuffled',  config.files?.shuffled,  'D: Two-step on shuffled (rg=2048)'],
]) {
  if (!localPath) continue;
  const r = ref(alias, localPath);
  results.push(await timed(label, 'sorted', async () => {
    const idRes = await conn.query(`
      SELECT row_index FROM read_parquet(${r})
      ORDER BY mean_intensity DESC NULLS LAST LIMIT ${MAX_IMGS}
    `);
    const ids = idRes.toArray().map(row => Number(row.row_index));
    await conn.query(`
      WITH ids AS (SELECT UNNEST([${ids.join(',')}]) AS rid)
      SELECT t.row_index, t.thumbnail
      FROM read_parquet(${r}) t
      JOIN ids ON t.row_index = ids.rid
    `);
  }, REPEATS));
}

// ── 3. Scalar full-column reads ───────────────────────────────────────────────

if (config.files?.baseline) {
  const r = ref('baseline', config.files.baseline);

  results.push(await timed('Baseline: 1 col',     'scalar',
    () => conn.query(`SELECT mean_intensity FROM read_parquet(${r})`), REPEATS));

  results.push(await timed('Baseline: 3 cols',    'scalar',
    () => conn.query(`SELECT mean_intensity, sharpness, snr FROM read_parquet(${r})`), REPEATS));

  results.push(await timed('Baseline: GROUP BY',  'scalar',
    () => conn.query(`
      SELECT report_group, AVG(mean_intensity), STDDEV(mean_intensity)
      FROM read_parquet(${r}) GROUP BY report_group
    `), REPEATS));
}

// ── 4. Row-group size sweep ───────────────────────────────────────────────────

for (const [rg, filePath] of Object.entries(config.rg_files ?? {})) {
  if (!filePath) continue;
  const r = rgRef(rg, filePath);

  results.push(await timed(`rg=${rg}`, 'rg_random', () => conn.query(`
    SELECT row_index, thumbnail FROM read_parquet(${r}) LIMIT ${MAX_IMGS}
  `), REPEATS));

  results.push(await timed(`rg=${rg}`, 'rg_scalar', () => conn.query(`
    SELECT mean_intensity FROM read_parquet(${r})
  `), REPEATS));
}

// ── Output & exit ─────────────────────────────────────────────────────────────

process.stdout.write(JSON.stringify(results));
await db.terminate();
process.exit(0);
