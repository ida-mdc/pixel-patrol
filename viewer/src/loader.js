import * as duckdb from '@duckdb/duckdb-wasm';
import duckdbMvpWorkerUrl from '@duckdb/duckdb-wasm/dist/duckdb-browser-mvp.worker.js?url';
import duckdbMvpWasmUrl from '@duckdb/duckdb-wasm/dist/duckdb-mvp.wasm?url';
import { detectSchema, pickDefaultGroupCol } from './schema.js';
import { q } from './sql.js';
import { FILE_ROW_NUMBER } from './constants.js';

const MAX_UNIQUE_GROUP = 12; // Match Dash app (pixel-patrol-base)

/**
 * Stable row id for two-step mosaic queries (pick ids without blobs, then fetch thumbnails).
 * Prefer explicit columns from the export; else `file_row_number` from read_parquet (see view DDL).
 */
const ROW_ID_COL_CANDIDATES = [
  'row_index', '_row_index', 'record_idx', '__row_index',
  FILE_ROW_NUMBER,
];

function pickRowIdColumnFromSchema(allCols) {
  for (const c of ROW_ID_COL_CANDIDATES) {
    if (allCols.includes(c)) return c;
  }
  return null;
}

/** Shut down an existing DuckDB instance and its worker. */
export async function terminateDuckDB(db, conn) {
  try { await conn?.close(); } catch {}
  try { await db?.terminate(); } catch {}
}

/** Initialise DuckDB WASM using locally bundled files (avoids CDN version mismatches). */
export async function initDuckDB() {
  const absWorkerUrl = new URL(duckdbMvpWorkerUrl, location.href).href;
  const absWasmUrl   = new URL(duckdbMvpWasmUrl,   location.href).href;

  const workerUrl = URL.createObjectURL(
    new Blob([`importScripts("${absWorkerUrl}");`], { type: 'text/javascript' }),
  );
  const worker = new Worker(workerUrl);
  const logger = new duckdb.VoidLogger();
  const db     = new duckdb.AsyncDuckDB(logger, worker);
  await db.instantiate(absWasmUrl, null);
  URL.revokeObjectURL(workerUrl);

  const conn = await db.connect();
  return { db, conn };
}

/** Fetch a URL with optional progress callback: onProgress(loadedBytes, totalBytes). */
async function fetchWithProgress(url, onProgress) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching ${url}`);
  const total  = Number(resp.headers.get('Content-Length') ?? 0);
  const reader = resp.body.getReader();
  const chunks = [];
  let loaded   = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress?.(loaded, total);
  }
  const out = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) { out.set(chunk, offset); offset += chunk.byteLength; }
  return out;
}

/** Register a remote URL and create the pp_data view/table, downloading the file upfront. */
export async function loadFromUrl(db, conn, url, onProgress) {
  const buf = await fetchWithProgress(url, onProgress);
  if (isArrowFile(url)) {
    await conn.query(`DROP TABLE IF EXISTS pp_data`);
    await conn.insertArrowFromIPCStream(buf, { name: 'pp_data', create: true });
    return finishLoad(conn, null);
  } else {
    const fname = `remote_${++_loadSeq}.parquet`;
    await db.registerFileBuffer(fname, buf);
    await conn.query(
      `CREATE OR REPLACE VIEW pp_data AS SELECT * FROM read_parquet('${fname}', file_row_number = true)`,
    );
    return finishLoad(conn, fname);
  }
}

function isArrowFile(name) {
  return /\.(arrow|ipc|arrows)$/i.test(name);
}

// Monotonic counter so each load gets a unique virtual filename.
// DuckDB WASM caches row-group metadata keyed by filename; reusing the same
// name for a different File object causes stale-offset reads → TProtocolException.
let _loadSeq = 0;

/**
 * Register a local File object and create the pp_data view/table.
 *
 * For Arrow IPC files: reads the full buffer and inserts as an in-memory table.
 * For Parquet files: uses DuckDB's BROWSER_FILEREADER protocol for slice-based access.
 */
export async function loadFromFile(db, conn, file) {
  if (isArrowFile(file.name)) {
    const buf = await file.arrayBuffer();
    await conn.query(`DROP TABLE IF EXISTS pp_data`);
    await conn.insertArrowFromIPCStream(new Uint8Array(buf), { name: 'pp_data', create: true });
    return finishLoad(conn, null);
  } else {
    const fname = `upload_${++_loadSeq}.parquet`;
    await db.registerFileHandle(fname, file, duckdb.DuckDBDataProtocol.BROWSER_FILEREADER, true);
    await conn.query(
      `CREATE OR REPLACE VIEW pp_data AS SELECT * FROM read_parquet('${fname}', file_row_number = true)`,
    );
    return finishLoad(conn, fname);
  }
}

export async function finishLoad(conn, parquetPath = null) {
  // Step 1: get schema (before any view manipulation)
  const schemaResult = await conn.query(
    `SELECT column_name AS name, data_type AS type
     FROM information_schema.columns
     WHERE table_name = 'pp_data'
     ORDER BY ordinal_position`,
  );
  const columns = schemaResult.toArray().map(r => ({ name: String(r.name), type: String(r.type) }));

  const hasObsLevel = columns.some(c => c.name === 'obs_level');
  if (!hasObsLevel) {
    const err = new Error('This file was created with an older version of PixelPatrol and is no longer supported.');
    err.isOldFormat = true;
    throw err;
  }

  if (parquetPath) {
    // Browser WASM: create pp_all (all rows) and pp_data (image rows)
    const p = parquetPath.replace(/'/g, "''");
    await conn.query(`CREATE OR REPLACE VIEW pp_all AS SELECT * FROM read_parquet('${p}', file_row_number = true)`);
    // Materialize obs_level=0 rows as a TABLE so plugin queries hit an in-memory
    // table rather than re-scanning the full parquet on every render.
    await conn.query(`DROP VIEW IF EXISTS pp_data`);
    await conn.query(`CREATE OR REPLACE TABLE pp_data AS SELECT * FROM pp_all WHERE obs_level = 0`);
  }
  // Server mode (parquetPath=null): pp_all and pp_data are already set up by the Python server.

  const totalRows = Number((await conn.query(`SELECT COUNT(*) AS n FROM pp_data`)).toArray()[0].n);

  const schema = detectSchema(columns);
  schema.rowIdColumn = pickRowIdColumnFromSchema(schema.allCols);

  const sliceCheck = await conn.query(`SELECT (MAX(obs_level) > 0) AS v FROM pp_all`);
  schema.hasDimSlices = !!sliceCheck.toArray()[0].v;

  // Populate dimensionInfo by querying distinct values from pp_all.
  if (schema.dimCols.length > 0) {
    for (const dimCol of schema.dimCols) {
      try {
        const res = await conn.query(
          `SELECT DISTINCT ${q(dimCol)} AS v FROM pp_all WHERE ${q(dimCol)} IS NOT NULL ORDER BY v`,
        );
        const vals = res.toArray().map(r => Number(r.v));
        if (vals.length > 0) {
          const letter = dimCol.replace('dim_', '');
          schema.dimensionInfo[letter] = vals;
        }
      } catch (e) {
        console.warn(`[viewer] could not query distinct values for ${dimCol}:`, e);
      }
    }
  }

  // Cardinality-filter the schema-heuristic group candidates.
  schema.groupCols = await filterGroupColsByCardinality(conn, schema.groupCols);

  // Always include any URL-param group col if it exists in the parquet.
  const urlGroup = new URLSearchParams(window.location.search).get('group');
  if (urlGroup && schema.allCols.includes(urlGroup) && !schema.groupCols.includes(urlGroup)) {
    schema.groupCols.push(urlGroup);
  }

  schema.defaultGroupCol = pickDefaultGroupCol(schema.allCols, schema.groupCols, totalRows);

  // Always include the default group col too.
  if (schema.defaultGroupCol && !schema.groupCols.includes(schema.defaultGroupCol)) {
    schema.groupCols.push(schema.defaultGroupCol);
  }

  const reportMeta = parquetPath
    ? await _readParquetMeta(conn, parquetPath)
    : _emptyReportMeta();

  return { schema, totalRows, projectName: reportMeta.projectName, description: reportMeta.description, reportMeta };
}

/** Read all pp_* keys from the parquet file's KV metadata footer. */
async function _readParquetMeta(conn, path) {
  try {
    const p   = path.replace(/'/g, "''");
    const res = await conn.query(
      `SELECT decode(key)::VARCHAR AS k, decode(value)::VARCHAR AS v
       FROM parquet_kv_metadata('${p}')
       WHERE decode(key)::VARCHAR LIKE 'pp_%'`,
    );
    const raw = Object.fromEntries(res.toArray().map(r => [String(r.k), String(r.v)]));
    return _parseRawMeta(raw);
  } catch {
    return _emptyReportMeta();
  }
}

export function parseServerMeta(raw) {
  return raw ? _parseRawMeta(raw) : _emptyReportMeta();
}

function _parseRawMeta(raw) {
  let paths = [];
  try { paths = JSON.parse(raw.pp_paths || '[]'); } catch {}
  let processingStats = {};
  try { processingStats = JSON.parse(raw.pp_processing_stats || '{}'); } catch {}
  let privacySummary = [];
  try { privacySummary = JSON.parse(raw.pp_privacy_summary || '[]'); } catch {}
  return {
    projectName:     raw.pp_project_name || null,
    description:     raw.pp_description  || null,
    flavor:          raw.pp_flavor       || null,
    version:         raw.pp_version      || null,
    createdAt:       raw.pp_created_at   || null,
    loader:          raw.pp_loader       || null,
    baseDir:         raw.pp_base_dir     || null,
    paths,
    processingStats,
    privacySummary,
  };
}

function _emptyReportMeta() {
  return { projectName: null, description: null, flavor: null, version: null,
           createdAt: null, loader: null, baseDir: null, paths: [], processingStats: {}, privacySummary: [] };
}

async function filterGroupColsByCardinality(conn, cols) {
  if (!cols.length) return [];
  // Sample 10 000 rows - enough to reliably detect 2–12 unique values without
  // fetching every column chunk from a remote file.
  const exprs = cols.map(c => `COUNT(DISTINCT ${q(c)}) AS ${q(c)}`).join(', ');
  try {
    const res = await conn.query(`SELECT ${exprs} FROM (SELECT ${cols.map(q).join(', ')} FROM pp_data LIMIT 10000)`);
    const first = res.toArray()[0];
    if (!first) return [];

    const rowObj = typeof first.toJSON === 'function' ? first.toJSON() : first;
    if (!rowObj || typeof rowObj !== 'object') return [];

    return cols.filter(c => {
      const n = Number(rowObj[c]);
      return Number.isFinite(n) && n >= 2 && n <= MAX_UNIQUE_GROUP;
    });
  } catch (err) {
    console.warn('[viewer] group cardinality query failed', err);
    return [];
  }
}
