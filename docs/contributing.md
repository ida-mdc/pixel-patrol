# Contributing

Pixel Patrol is an open project and we welcome contributions of all kinds - bug reports, feature suggestions, documentation improvements, new loaders or processors, and viewer plugins. If you want to get involved, [open an issue](https://github.com/ida-mdc/pixel-patrol/issues) or reach out at [support@helmholtz-imaging.de](mailto:support@helmholtz-imaging.de).

Fork the repository on GitHub and open a pull request with your changes.

---

## What's non-obvious

**Monorepo with multiple packages.** The repository contains several independently installable packages under `packages/`. Install the ones you need in editable mode:

```bash
uv pip install -e packages/pixel-patrol-base -e packages/pixel-patrol-image -e packages/pixel-patrol-loader-bio
```

**The viewer build syncs to the Python package.** The viewer is a separate Vite app in `viewer/`. Running `npm run build` in that directory compiles it and automatically copies the output into `pixel-patrol-base`, so `pixel-patrol view` always picks up your latest changes. For local development with hot reload use `npm run dev`.

**Previewing the full site locally.** Use `./tools/dev_serve.sh` to build and serve the landing page, docs, and viewer together at the correct URLs (`localhost:8000/` and `localhost:8000/docs/`). For docs-only editing with live reload, `uv run --with mkdocs-material mkdocs serve` is faster.

**CI runs a wider matrix on main.** Python tests run on Ubuntu only for feature branches, and additionally on Windows and macOS for pushes to `main` and PRs targeting it.
