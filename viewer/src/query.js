import { tableFromIPC } from 'apache-arrow';

/**
 * True when served by the local Python viewer server (viewer_server.py).
 * The server injects <script>window.__PP_SERVER = true;</script> into index.html.
 */
export const SERVER_MODE = typeof window !== 'undefined' && !!window.__PP_SERVER;

/**
 * Create a mock DuckDB connection that routes all SQL to the local Python
 * server's /api/query endpoint, which executes it with native (multi-threaded)
 * DuckDB and returns an Arrow IPC stream.
 *
 * The returned object has the same {query, execute} interface as a real DuckDB
 * WASM connection so all existing code works without modification.
 */
export function makeServerConn() {
  // The report manager (launch_server.py) serves several reports on one port
  // and injects window.__PP_REPORT_ID so /api/query can be routed to the right
  // report. The standalone viewer server has no id and answers /api/query for
  // its single parquet.
  const reportId = (typeof window !== 'undefined' && window.__PP_REPORT_ID) || null;
  const queryUrl = reportId ? `/api/query?report=${encodeURIComponent(reportId)}` : '/api/query';

  async function query(sql) {
    const res = await fetch(queryUrl, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ sql }),
    });
    if (!res.ok) {
      const msg = await res.text().catch(() => res.statusText);
      throw new Error(`Server query failed (${res.status}): ${msg}`);
    }
    return tableFromIPC(await res.arrayBuffer());
  }
  // DuckDB WASM exposes both .query() and .execute() - alias them.
  return { query, execute: query };
}
