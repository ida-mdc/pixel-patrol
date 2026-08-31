#!/usr/bin/env bash
# Run all tests: JS (viewer) + Python (per package, skipped if not installed).
# Collects all failures rather than stopping at the first one.
ROOT=$(cd "$(dirname "$0")/.." && pwd)
FAILED=()
SKIPPED=()

echo "=== JS tests ==="
(cd "$ROOT/viewer" && npm test) || FAILED+=("viewer (JS)")

echo ""
echo "=== Python tests ==="
for pkg in "$ROOT"/packages/*/; do
  name=$(basename "$pkg")
  if [ ! -d "$pkg/tests" ]; then continue; fi
  import="${name//-/_}"
  if ! python3 -c "import $import" 2>/dev/null; then
    echo "--- $name (skipped, not installed) ---"
    SKIPPED+=("$name (not installed)")
    continue
  fi
  echo "--- $name ---"
  output=$(uv run pytest "$pkg/tests" -rs 2>&1)
  code=$?
  echo "$output"
  if [ $code -eq 5 ]; then
    echo "(no tests found)"
  elif [ $code -ne 0 ]; then
    FAILED+=("$name (Python)")
  fi
  skipped=$(echo "$output" | grep -oP '\d+ skipped' | head -1)
  if [ -n "$skipped" ]; then
    SKIPPED+=("$name ($skipped)")
  fi
done

echo ""
echo "=== Example scripts ==="
for script in "$ROOT"/examples/0{1,2,3}_*.py; do
  name=$(basename "$script")
  echo "--- $name ---"
  output=$(cd "$ROOT/examples" && uv run python - "$(basename "$script")" 2>&1 <<'PYEOF'
import sys, runpy
import pixel_patrol_base.api as api
api.view = lambda *a, **kw: None
runpy.run_path(sys.argv[1], run_name="__main__")
PYEOF
  )
  code=$?
  echo "$output"
  if [ $code -ne 0 ]; then
    FAILED+=("$name (example)")
  fi
done

echo ""
if [ ${#SKIPPED[@]} -gt 0 ]; then
  echo "SKIPPED:"
  for s in "${SKIPPED[@]}"; do echo "  - $s"; done
fi
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "All tests passed."
else
  echo "FAILED:"
  for f in "${FAILED[@]}"; do echo "  - $f"; done
  exit 1
fi
