#!/usr/bin/env bash
# publish_pypi.sh — build and upload all packages to PyPI
#
# Usage:
#   ./scripts/publish_pypi.sh
#
# Requires:
#   - PYPI_API_TOKEN env var set
#   - build and twine installed
#   - bump_toml_version.py already run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PACKAGES=(
    "pixel-patrol-base"
    "pixel-patrol"
    "pixel-patrol-image"
    "pixel-patrol-loader-bio"
    "pixel-patrol-aqqua"
)

[ -n "${PYPI_API_TOKEN:-}" ] || { echo "ERROR: PYPI_API_TOKEN is not set"; exit 1; }

# read version from pixel-patrol-base as the canonical source
VERSION=$(python - "$REPO_ROOT" <<'EOF'
import sys, tomlkit, pathlib
doc = tomlkit.parse((pathlib.Path(sys.argv[1]) / "packages/pixel-patrol-base/pyproject.toml").read_text())
print(doc["project"]["version"])
EOF
)
echo "=== Releasing v$VERSION ==="

# ── build viewer ──────────────────────────────────────────────────────────────
echo ""
echo "=== Building JS viewer ==="
cd "$REPO_ROOT/viewer"
npm install
npm run build
cd "$REPO_ROOT"

# ── build ─────────────────────────────────────────────────────────────────────
echo ""
echo "=== Building packages ==="
for pkg in "${PACKAGES[@]}"; do
    echo "--- building $pkg ---"
    cd "$REPO_ROOT/packages/$pkg"
    rm -rf -- build dist *.egg-info src/*.egg-info
    python -m build
    cd "$REPO_ROOT"
done

# ── upload ────────────────────────────────────────────────────────────────────
echo ""
echo "=== Uploading to PyPI ==="
for pkg in "${PACKAGES[@]}"; do
    echo "--- uploading $pkg ---"
    cd "$REPO_ROOT/packages/$pkg"
    python -m twine upload \
        --non-interactive \
        -u __token__ \
        -p "$PYPI_API_TOKEN" \
        --skip-existing \
        dist/*
    cd "$REPO_ROOT"
done

# ── commit, tag, push ─────────────────────────────────────────────────────────
echo ""
echo "=== Committing and tagging v$VERSION ==="
git add packages/*/pyproject.toml
git commit -m "release v$VERSION"
git tag -a "v$VERSION" -m "v$VERSION"
git push origin main
git push origin "v$VERSION"

echo ""
echo "✓ Released v$VERSION"

# ── github release tag bump ───────────────────────────────────────────────────

## Currently we do so manually:

# VERSION=0.5.0    ### change this
#gh release create "v$VERSION" \
#  --title "pixel-patrol $VERSION" \
#  --notes "## v$VERSION
#
### Packages
#* pixel-patrol: $VERSION — https://pypi.org/project/pixel-patrol/$VERSION/
#* pixel-patrol-base: $VERSION — https://pypi.org/project/pixel-patrol-base/$VERSION/
#* pixel-patrol-image: $VERSION — https://pypi.org/project/pixel-patrol-image/$VERSION/
#* pixel-patrol-loader-bio: $VERSION — https://pypi.org/project/pixel-patrol-loader-bio/$VERSION/"
