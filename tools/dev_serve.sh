#!/usr/bin/env bash
# Local dev server — mirrors the production URL structure:
#   http://localhost:8000/          → landing page
#   http://localhost:8000/docs/     → documentation
#
# For live doc editing use: uv run --with mkdocs-material mkdocs serve
# (docs only, at localhost:8001/pixel-patrol/docs/)

set -e
OUT=/tmp/pp-dev

# Clean root HTML files so deleted pages don't linger between rebuilds
rm -f "$OUT"/*.html

uv run --with mkdocs-material mkdocs build --site-dir "$OUT/docs" --quiet
uv run python3 -c "
from pixel_patrol_base.viewer_pages import build_github_pages_site
build_github_pages_site('$OUT')
"

echo ""
echo "  Landing page: http://localhost:8000/"
echo "  Docs:         http://localhost:8000/docs/"
echo ""
python3 -m http.server 8000 --directory "$OUT"
