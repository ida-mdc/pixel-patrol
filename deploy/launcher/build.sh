#!/usr/bin/env bash
# Build the Pixel Patrol launcher binary.
#
# Usage:
#   cd deploy/launcher
#   ./build.sh
#
# Output: dist/pixel-patrol-launcher   (Linux / macOS)
#         dist/pixel-patrol-launcher.exe  (Windows, if built there)
#
# Requires the "pixel-patrol" micromamba environment.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="pixel-patrol"

echo "==> Installing build dependencies into the '${ENV_NAME}' micromamba env…"
micromamba run -n "${ENV_NAME}" pip install --quiet pyinstaller flask

ASSETS_DIR="${SCRIPT_DIR}/../../packages/pixel-patrol-base/src/pixel_patrol_base/report/assets"
ASSETS_DIR="$(realpath "${ASSETS_DIR}")"

echo "==> Building launcher binary…"
micromamba run -n "${ENV_NAME}" pyinstaller \
    --onefile \
    --name pixel-patrol-launcher \
    --distpath "${SCRIPT_DIR}/dist" \
    --workpath "${SCRIPT_DIR}/build" \
    --specpath "${SCRIPT_DIR}" \
    --hidden-import flask \
    --add-data "${ASSETS_DIR}/icon.png:." \
    --add-data "${ASSETS_DIR}/Helmholtz-Imaging_Mark.png:." \
    --icon "${SCRIPT_DIR}/icon.png" \
    --clean \
    "${SCRIPT_DIR}/launcher.py"

BINARY="${SCRIPT_DIR}/dist/pixel-patrol-launcher"
if [[ -f "${BINARY}" ]]; then
    chmod +x "${BINARY}"
    echo ""
    echo "==> Done!  Binary: ${BINARY}"
    echo ""
    echo "    Double-click it (or run it from a terminal) to start Pixel Patrol."
    echo "    On first launch it will ask which loader you want, then install"
    echo "    everything into ~/.pixel-patrol/ automatically."
else
    echo "ERROR: binary not found at expected path ${BINARY}" >&2
    exit 1
fi
