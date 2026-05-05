import { defineConfig } from 'vite';
import { cpSync, rmSync, existsSync, readFileSync } from 'fs';
import { resolve } from 'path';

// After a production build, sync the output into the Python package so that
// `pixel-patrol view` picks up the latest viewer without a manual copy step.
const VIEWER_DIST_IN_PKG = resolve(
  __dirname,
  '../packages/pixel-patrol-base/src/pixel_patrol_base/viewer_dist',
);

// DuckDB WASM npm packages reference .map files that are never included in the
// published package. Strip the sourceMappingURL comment in the load hook (before
// Vite tries to read the missing .map file) so the dev server doesn't warn.
function stripMissingSourcemaps() {
  return {
    name: 'strip-missing-sourcemaps',
    load(id) {
      const path = id.split('?')[0];
      if (path.includes('@duckdb') && path.endsWith('.js')) {
        try {
          const code = readFileSync(path, 'utf8');
          return { code: code.replace(/\/\/# sourceMappingURL=\S+\.map\s*$/m, ''), map: null };
        } catch {
          return null;
        }
      }
      return null;
    },
  };
}

function syncToPythonPackage() {
  return {
    name: 'sync-viewer-dist',
    closeBundle() {
      if (!existsSync(VIEWER_DIST_IN_PKG)) return;
      try {
        rmSync(VIEWER_DIST_IN_PKG,  { recursive: true, force: true });
        cpSync(resolve(__dirname, 'dist'), VIEWER_DIST_IN_PKG, { recursive: true });
        console.log(`\n✓ viewer_dist synced → ${VIEWER_DIST_IN_PKG}`);
      } catch (e) {
        console.warn(`\n⚠ viewer_dist sync failed: ${e.message}`);
      }
    },
  };
}

export default defineConfig({
  test: {
    environment: 'happy-dom',
    globals: true,
  },

  // Relative base so the built output works from any subdirectory or
  // file:// as well as GitHub Pages (which may serve from /repo-name/).
  base: './',

  optimizeDeps: {
    // DuckDB WASM uses dynamic worker creation internally; don't pre-bundle it.
    exclude: ['@duckdb/duckdb-wasm'],
  },

  plugins: [stripMissingSourcemaps(), syncToPythonPackage()],

  build: {
    target: 'es2022',
    // Keep DuckDB workers/WASM as separate files rather than inlining.
    rollupOptions: {
      output: {
        assetFileNames: 'assets/[name]-[hash][extname]',
      },
    },
  },

  // Dev server needs these headers for DuckDB's SharedArrayBuffer (multi-thread mode).
  // GitHub Pages doesn't send them, so DuckDB falls back to single-thread (mvp bundle)
  // automatically via selectBundle().
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
    // Allow Vitest to import plugin files from sibling packages directory.
    fs: { allow: ['..'] },
  },
});
