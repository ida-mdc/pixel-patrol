#!/usr/bin/env bash
# Local dev server — mirrors the production URL structure:
#   http://localhost:8000/          → landing page
#   http://localhost:8000/docs/     → documentation
#
# For live doc editing use: uv run --with mkdocs-material mkdocs serve
# (docs only, at localhost:8001/pixel-patrol/docs/)

set -e
OUT=/tmp/pp-dev

uv run --with mkdocs-material mkdocs build --site-dir "$OUT/docs" --quiet
cp docs/home.html "$OUT/index.html"
cp -r docs/assets "$OUT/assets"

echo ""
echo "  Landing page: http://localhost:8000/"
echo "  Docs:         http://localhost:8000/docs/"
echo ""
python3 -m http.server 8000 --directory "$OUT"
