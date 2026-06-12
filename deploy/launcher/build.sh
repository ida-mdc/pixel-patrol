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

VIEWER_ASSETS="$(realpath "${SCRIPT_DIR}/../../viewer/public")"
LAUNCH_ASSETS="$(realpath "${SCRIPT_DIR}/../../packages/pixel-patrol-base/src/pixel_patrol_base/launch_assets")"

echo "==> Building launcher binary…"
micromamba run -n "${ENV_NAME}" pyinstaller \
    --onefile \
    --name pixel-patrol-launcher \
    --distpath "${SCRIPT_DIR}/dist" \
    --workpath "${SCRIPT_DIR}/build" \
    --specpath "${SCRIPT_DIR}" \
    --hidden-import flask \
    --add-data "${VIEWER_ASSETS}/icon.png:." \
    --add-data "${VIEWER_ASSETS}/Helmholtz-Imaging_Mark.png:." \
    --add-data "${LAUNCH_ASSETS}/prevalidation.png:." \
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
