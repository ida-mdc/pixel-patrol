import { defineConfig } from 'vite';
import { cpSync, rmSync, readFileSync, existsSync } from 'fs';
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

function emitJsLicenses() {
  return {
    name: 'emit-js-licenses',
    generateBundle() {
      const read = (p) => (existsSync(p) ? readFileSync(p, 'utf8') : '');
      const arrowNotice = read(resolve(__dirname, 'node_modules/apache-arrow/NOTICE.txt'));
      const chromaLicense = read(resolve(__dirname, 'node_modules/chroma-js/LICENSE'));
      this.emitFile({
        type: 'asset',
        fileName: 'LICENSES.txt',
        source: [
          'Third-party JavaScript licenses',
          '================================',
          '',
          'The viewer bundle includes the following third-party libraries.',
          '',
          '--------------------------------------------------------------------------------',
          '',
          'apache-arrow  (Apache-2.0)',
          '',
          arrowNotice,
          '',
          '--------------------------------------------------------------------------------',
          '',
          'chroma-js  (BSD-3-Clause AND Apache-2.0)',
          '',
          chromaLicense,
        ].join('\n'),
      });
    },
  };
}

function syncToPythonPackage() {
  return {
    name: 'sync-viewer-dist',
    closeBundle() {
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
    // The viewer is the repo's only JS toolchain, so it also runs the unit tests
    // for any built-in widgets. Widgets ship inside Python packages (no
    // package.json of their own), and by convention each package keeps its
    // widget tests next to the widget source under tests/viewer/. The wildcard
    // picks up every such package - not just today's - so a new widget-bearing
    // package needs no change here. server.fs.allow below lets Vitest read those
    // sibling files.
    include: [
      'src/**/*.test.js',
      '../packages/*/tests/viewer/**/*.test.js',
    ],
  },

  // Relative base so the built output works from any subdirectory or
  // file:// as well as GitHub Pages (which may serve from /repo-name/).
  base: './',

  optimizeDeps: {
    // DuckDB WASM uses dynamic worker creation internally; don't pre-bundle it.
    exclude: ['@duckdb/duckdb-wasm'],
  },

  plugins: [stripMissingSourcemaps(), emitJsLicenses(), syncToPythonPackage()],

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
