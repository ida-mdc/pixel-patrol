"""Quick smoke-run helper for CI or local checks."""
from pathlib import Path
import sys

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[3]
    bench = repo_root / "scripts" / "bench" / "benchmark_processors.py"
    if not bench.exists():
        print("benchmark script not found", file=sys.stderr)
        raise SystemExit(1)
    # Run a quick local run (single-process)
    rc = __import__("subprocess").run([sys.executable, str(bench), "--run-local", "--base-path", "examples/datasets/demo_data_2d", "--processing-max-workers", "1", "--out", "bench_out/smoke.json"]).returncode
    raise SystemExit(rc)
