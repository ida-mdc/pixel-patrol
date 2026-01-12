"""Benchmark processors across branches or locally.

Usage:
  - Compare branches: run from repository root and pass --branches branch1 branch2
  - The script will create temporary git worktrees for each branch and run a local benchmark in each worktree.
  - Local benchmarking (--run-local) instruments processor classes to record per-processor timings (accurate in single-process mode where workers=1), measures wall-clock, optional cProfile output, and writes a JSON result file.

Notes:
  - Requires `git` on PATH for branch comparison.
  - Optionally installs `psutil` to report memory usage (fall back gracefully if not available).

"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("bench")

try:
    import psutil
except Exception:
    psutil = None


@dataclass
class ProcessorStats:
    calls: int = 0
    total_time: float = 0.0


def _find_local_src_paths(repo_root: Path) -> List[str]:
    """Return list of package src directories under the repo to use via PYTHONPATH."""
    src_paths = []
    pkgs = repo_root.glob("packages/*/src")
    for p in pkgs:
        if p.is_dir():
            src_paths.append(str(p))
    # Also include top-level `src/` if present
    top_src = repo_root / "src"
    if top_src.exists() and top_src.is_dir():
        src_paths.append(str(top_src))
    return src_paths


def _run_subprocess_in_dir(cmd: List[str], cwd: Path, extra_env: Optional[Dict[str, str]] = None) -> int:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    logger.info("Running subprocess in %s: %s", cwd, " ".join(cmd))
    proc = subprocess.run([sys.executable, "-u"] + cmd, cwd=str(cwd), env=env)
    return proc.returncode


def _make_worktree(repo_root: Path, branch: str, allow_clone: bool = False) -> Path:
    """Return a path suitable for benchmarking the given branch.

    Behavior:
    - If the requested branch equals the currently checked-out branch in repo_root, return repo_root (use current worktree as-is).
    - Otherwise, try `git worktree add --detach <tmpdir> <branch>`.
    - If worktree add fails and `allow_clone` is True, fallback to `git clone --branch ...`.
    - If all fails, raise a RuntimeError and do not clone without explicit permission.
    """
    # If the repo root is already on the requested branch, use it directly (preserves unstaged changes)
    try:
        cur = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(repo_root), capture_output=True, text=True)
        if cur.returncode == 0:
            current_branch = (cur.stdout or "").strip()
            if current_branch == branch:
                logger.info("Using current worktree for branch %s (will not modify it). Repo root: %s", branch, repo_root)
                return repo_root
    except Exception:
        logger.debug("Could not determine current branch; proceeding to create an isolated worktree")

    tmpdir = Path(tempfile.mkdtemp(prefix=f"bench_{branch.replace('/', '_')}_"))
    # Try creating a git worktree first (non-destructive).
    res = subprocess.run(["git", "worktree", "add", "--detach", str(tmpdir), branch], cwd=str(repo_root), capture_output=True, text=True)
    if res.returncode == 0:
        logger.info("Created worktree for branch %s at %s (repo root: %s)", branch, tmpdir, repo_root)
        return tmpdir

    logger.warning("git worktree failed for branch %s (repo root %s): %s", branch, repo_root, (res.stderr or res.stdout).strip())

    if allow_clone:
        # Fallback: try cloning local repo (explicitly allowed)
        res2 = subprocess.run(["git", "clone", "--branch", branch, "--single-branch", str(repo_root), str(tmpdir)], cwd=str(repo_root), capture_output=True, text=True)
        if res2.returncode == 0:
            logger.info("Created clone for branch %s at %s", branch, tmpdir)
            return tmpdir
        logger.warning("git clone fallback failed for branch %s: %s", branch, (res2.stderr or res2.stdout).strip())

    # Cleanup tmpdir and raise informative error
    try:
        import shutil
        if tmpdir.exists():
            shutil.rmtree(tmpdir)
    except Exception:
        logger.exception("Failed to clean up temp dir %s", tmpdir)

    raise RuntimeError(
        f"Failed to prepare a worktree for branch {branch} (repo_root={repo_root}). To allow creating a cloned fallback, pass --allow-clone. Git worktree error: {(res.stderr or res.stdout).strip()!r}"
    )


def _remove_worktree(repo_root: Path, worktree_path: Path) -> None:
    # If the worktree_path is the repository root, do not remove it
    try:
        if Path(worktree_path).resolve() == Path(repo_root).resolve():
            logger.info("Not removing repo root worktree: %s", worktree_path)
            return
    except Exception:
        pass

    # Try to remove worktree; if that fails it's likely a clone so remove the directory directly.
    try:
        res = subprocess.run(["git", "worktree", "remove", "-f", str(worktree_path)], cwd=str(repo_root), capture_output=True, text=True)
        if res.returncode == 0:
            return
        logger.debug("git worktree remove failed (not a worktree?): %s", (res.stderr or res.stdout).strip())
    except Exception:
        logger.debug("git worktree remove raised an exception; falling back to directory removal")

    # Fallback: remove directory if it exists
    if worktree_path.exists():
        try:
            import shutil
            shutil.rmtree(worktree_path)
        except Exception:
            logger.exception("Failed to remove worktree/clone directory %s", worktree_path)


def compare_branches(repo_root: Path, branches: List[str], bench_args: List[str], out_dir: Path, snakeviz: bool = False, open_snakeviz: bool = False, allow_clone: bool = False, host_base_path: Optional[Path] = None, use_host_script: bool = True, use_host_data: bool = True) -> None:
    results = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for branch in branches:
        logger.info("Benchmarking branch %s", branch)
        try:
            wt = _make_worktree(repo_root, branch, allow_clone=allow_clone)
        except Exception as exc:
            logger.exception("Could not prepare worktree/clone for branch %s: %s", branch, exc)
            # skip this branch but continue with others
            continue

        try:
            # Build PYTHONPATH to prefer local src packages
            paths = _find_local_src_paths(wt)
            extra_env = {}
            if paths:
                extra_env["PYTHONPATH"] = os.pathsep.join(paths)

            # Ensure out/prof files are absolute and live in the central out_dir
            out_dir = out_dir.resolve()
            out_file = out_dir / f"bench_result_{branch.replace('/', '_')}.json"
            prof_file = out_dir / f"bench_profile_{branch.replace('/', '_')}.prof"

            # Make sure the bench script exists inside the worktree; if not, use the currently-running script
            script_path = Path(wt) / Path("scripts/bench/benchmark_processors.py")
            used_host_script = False
            if not script_path.exists():
                # fall back to the script file being executed right now (host path)
                script_path = Path(__file__).resolve()
                used_host_script = True
                logger.info("Benchmark script not found in worktree %s; using host script %s for branch %s",
                            wt, script_path, branch)

            # Build the command to run. We may need to override base-path to the host dataset when requested.
            # Filter out any --base-path entries in bench_args if use_host_data is True, so we can set the host path explicitly.
            bench_args_filtered = []
            i = 0
            while i < len(bench_args):
                a = bench_args[i]
                # skip host base-path entries when we override with host data
                if use_host_data and a == "--base-path":
                    i += 2
                    continue

                # Only forward processing parallelism to the branch that is intended to use it
                if a == "--processing-max-workers":
                    # value is the next token
                    value = bench_args[i + 1] if (i + 1) < len(bench_args) else None
                    branch_base = branch.split("/")[-1]
                    if branch_base == "chunking_and_multiprocessing":
                        bench_args_filtered.append(a)
                        if value is not None:
                            bench_args_filtered.append(value)
                    else:
                        logger.debug("Not forwarding %s to branch %s (only for chunking_and_multiprocessing)", a, branch)
                    i += 2
                    continue

                bench_args_filtered.append(a)
                i += 1

            if use_host_data and host_base_path is not None:
                bench_args_filtered += ["--base-path", str(host_base_path.resolve())]
                logger.info("Overriding base-path for branch %s to host dataset: %s", branch, host_base_path.resolve())

            cmd = [str(script_path), "--run-local", "--out", str(out_file), "--cprofile", str(prof_file)] + bench_args_filtered

            # If the host script is used, ensure the PYTHONPATH favors the worktree src directories so the run imports the branch's code
            if (use_host_script or used_host_script) and paths:
                # Prepend wt paths to PYTHONPATH so imports prefer branch sources
                extra_env["PYTHONPATH"] = os.pathsep.join(paths) + os.pathsep + extra_env.get("PYTHONPATH", "")

            rc = _run_subprocess_in_dir(cmd, wt, extra_env)
            if rc != 0:
                logger.warning("Benchmark for branch %s exited with code %s", branch, rc)
            else:
                try:
                    if out_file.exists():
                        with out_file.open("r", encoding="utf-8") as fh:
                            results[branch] = json.load(fh)
                    else:
                        logger.warning("Expected output file not found for branch %s: %s", branch, out_file)
                        results[branch] = {}
                    # attach profiler path if it exists
                    if prof_file.exists():
                        results[branch]["cprofile_file"] = str(prof_file)
                    # record whether we used the host script or host data
                    results[branch]["used_host_script"] = bool(used_host_script or use_host_script)
                    results[branch]["used_host_data"] = bool(use_host_data)
                    # record actual base path used for clarity
                    if use_host_data and host_base_path is not None:
                        results[branch]["actual_base_path"] = str(host_base_path.resolve())
                except Exception as exc:  # pragma: no cover - user will inspect logs
                    logger.exception("Failed to read result for branch %s: %s", branch, exc)
        finally:
            _remove_worktree(repo_root, wt)

    # Summarize comparison
    summary = {}
    for branch, data in results.items():
        summary[branch] = {
            "wall_time": data.get("wall_time_seconds"),
            "files": data.get("files_processed"),
            "memory_rss": data.get("memory_rss_bytes"),
        }
        # optionally harvest focused stats if present
        if data.get("focus_processor"):
            summary[branch]["focus_processor"] = data.get("focus_processor")
            summary[branch]["focus_stats"] = data.get("focus_stats")

    cmp_file = out_dir / "summary.json"
    cmp_file.write_text(json.dumps({"results": results, "summary": summary}, indent=2))
    print(json.dumps({"results": results, "summary": summary}, indent=2))

    # Optionally generate SnakeViz HTML reports for produced profiler files
    if snakeviz:
        for branch, data in results.items():
            prof = data.get("cprofile_file")
            if not prof:
                logger.info("No profile for branch %s; skipping SnakeViz", branch)
                continue
            prof_path = Path(prof)
            if not prof_path.exists():
                logger.warning("Profile file not found for branch %s: %s", branch, prof)
                continue
            try:
                # SnakeViz does not support exporting to a static HTML file via CLI. Instead, start SnakeViz
                # as a server. If `open_snakeviz` is True we start it which will usually open a browser.
                # We use Popen so the server runs in the background and does not block the main script.
                if open_snakeviz:
                    proc = subprocess.Popen([sys.executable, "-m", "snakeviz", str(prof_path)])
                    logger.info("Started SnakeViz server for branch %s (pid=%s) and opened browser", branch, proc.pid)
                    results[branch]["snakeviz_pid"] = proc.pid
                else:
                    # Start server-only (-s) without opening a browser; user can browse to the server manually
                    proc = subprocess.Popen([sys.executable, "-m", "snakeviz", "-s", str(prof_path)])
                    logger.info("Started SnakeViz server-only for branch %s (pid=%s). Use --open-snakeviz to open browser.", branch, proc.pid)
                    results[branch]["snakeviz_pid"] = proc.pid
            except FileNotFoundError:
                logger.warning("SnakeViz executable not found. Install with `pip install snakeviz` to view profiles.")
            except Exception:
                logger.exception("Failed to run SnakeViz for branch %s", branch)


def run_local_benchmark(
    base_path: Path,
    loader: Optional[str],
    selected_file_extensions: str | List[str],
    processing_max_workers: int,
    out_path: Optional[Path],
    enable_cprofile: Optional[Path],
    focus_processor: Optional[str] = None,
) -> int:
    """Run the benchmark in the current checkout. This function imports local packages from the repo.
    """
    # Import here to ensure we pick up local packages when run in a worktree
    import importlib
    import types

    # Local import to avoid importing at module import time
    try:
        from pixel_patrol_base import api
        from pixel_patrol_base import plugin_registry
    except Exception as exc:
        logger.exception("Failed to import pixel_patrol_base (ensure PYTHONPATH points to local src dirs): %s", exc)
        return 2

    # Prepare per-processor timing storage
    PER_PROCESSOR_STATS: Dict[str, ProcessorStats] = {}

    # Keep original register function and the classes it returns for name resolution
    original_register_func = plugin_registry.register_processor_plugins
    original_processor_classes = original_register_func()

    # Wrap the register_processor_plugins to return wrapped classes that record timings
    def _wrap_classes(classes):
        wrapped = []
        for cls in classes:
            name = cls.__name__

            def make_run(orig_cls, processor_name):
                def run_and_time(self, art):
                    import time
                    t0 = time.perf_counter()
                    try:
                        res = orig_cls.run(self, art)
                        return res
                    finally:
                        dt = time.perf_counter() - t0
                        stat = PER_PROCESSOR_STATS.setdefault(processor_name, ProcessorStats())
                        stat.calls += 1
                        stat.total_time += dt
                return run_and_time

            attrs = {"run": make_run(cls, name)}
            Wrapped = type(f"Timed_{name}", (cls,), attrs)
            # copy NAME if present
            if hasattr(cls, "NAME"):
                Wrapped.NAME = getattr(cls, "NAME")
            wrapped.append(Wrapped)
        return wrapped

    # Monkeypatch the registry function
    plugin_registry.register_processor_plugins = lambda: _wrap_classes(original_processor_classes)

    # Build/set project and settings
    try:
        # create project
        project = api.create_project("Benchmark Project", base_dir=base_path, loader=loader)
        # minimal settings optimized for benchmarking
        from pixel_patrol_base.core.project_settings import Settings

        # Construct Settings using only supported parameters to remain compatible with older branches
        from inspect import signature
        settings_kwargs = {"selected_file_extensions": selected_file_extensions}
        try:
            sig = signature(Settings)
            if "processing_max_workers" in sig.parameters:
                settings_kwargs["processing_max_workers"] = processing_max_workers
                logger.debug("Setting 'processing_max_workers' to %s", processing_max_workers)
            else:
                logger.debug("'processing_max_workers' not supported by Settings on this branch; skipping")
        except Exception:
            # If we can't inspect signature for any reason, attempt to set it and fallback on failure
            settings_kwargs["processing_max_workers"] = processing_max_workers

        settings = Settings(**settings_kwargs)
        api.set_settings(project, settings)

        # Optionally run cProfile to capture function-level stats
        if enable_cprofile:
            import cProfile
            pr = cProfile.Profile()
            pr.enable()

        t0 = time.perf_counter()
        api.process_files(project)
        wall = time.perf_counter() - t0

        if enable_cprofile:
            pr.disable()
            pr.dump_stats(str(enable_cprofile))

        # Get results
        try:
            records_df = api.get_records_df(project)
            num_files = 0 if records_df is None else records_df.height
        except Exception:
            num_files = 0

        # Memory usage
        rss = None
        if psutil is not None:
            rss = psutil.Process().memory_info().rss
        else:
            rss = None

        # locate focus processor stats if requested
        focus_stats = None
        focus_name = None
        if focus_processor:
            # try to match by class name
            if focus_processor in PER_PROCESSOR_STATS:
                focus_name = focus_processor
                focus_stats = asdict(PER_PROCESSOR_STATS[focus_processor])
            else:
                # try to match by plugin NAME attribute
                for cls in original_processor_classes:
                    cls_name = cls.__name__
                    plugin_name = getattr(cls, "NAME", None)
                    if plugin_name == focus_processor:
                        focus_name = cls_name
                        if cls_name in PER_PROCESSOR_STATS:
                            focus_stats = asdict(PER_PROCESSOR_STATS[cls_name])
                        break

        # Serialize stats
        out = {
            "wall_time_seconds": wall,
            "files_processed": num_files,
            "memory_rss_bytes": rss,
            "per_processor": {k: asdict(v) for k, v in PER_PROCESSOR_STATS.items()},
            "focus_processor": focus_name,
            "focus_stats": focus_stats,
        }

        if out_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=2)
        else:
            print(json.dumps(out, indent=2))

        return 0
    except Exception as exc:  # pragma: no cover - user will act on trace
        logger.exception("Benchmark run failed: %s", exc)
        return 3
    finally:
        # restore
        plugin_registry.register_processor_plugins = original_register_func


def parse_cli():
    p = argparse.ArgumentParser(description="Benchmark pixel-patrol processors across branches or locally")
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2], help="Repository root (used for worktree creation)")
    p.add_argument("--branches", nargs="*", help="Branches to compare (will create git worktrees and run benchmarks in each)")
    p.add_argument("--base-path", type=Path, default=Path("playground/benchmark/images_dims_tz_18"), help="Base path to dataset (default: playground/benchmark/images_dims_tz_18)")
    p.add_argument("--loader", type=str, default="bioio", help="Loader to use (default: bioio)")
    p.add_argument("--selected-file-extensions", type=str, default="all", help="File extensions to select, e.g. 'all' or 'tif,png'")
    p.add_argument("--processing-max-workers", type=int, default=1, help="Set processing parallelism; per-processor timings are only accurate with 1 worker")
    p.add_argument("--out", type=Path, default=None, help="Output JSON file for local run")
    p.add_argument("--run-local", action="store_true", help="Run the benchmark locally in current tree")
    p.add_argument("--cprofile", type=Path, default=None, help="If set, write cProfile stats to this file during the run")
    p.add_argument("--focus-processor", type=str, default=None, help="Name of processor to focus on (class name or plugin NAME, e.g., 'HistogramProcessor' or 'histogram')")
    p.add_argument("--snakeviz", action="store_true", help="Generate SnakeViz HTML reports for any cProfile files produced for branches (requires snakeviz).")
    p.add_argument("--open-snakeviz", action="store_true", help="Open generated SnakeViz HTML files in the default browser after creation.")
    p.add_argument("--postprocess-only", action="store_true", help="Skip running benchmarks and only run post-processing (SnakeViz) on existing profiler files in --out-dir")
    p.add_argument("--allow-clone", action="store_true", help="Allow fallback to 'git clone' when 'git worktree' fails. Disabled by default to avoid unexpected local clones.")
    p.add_argument("--use-host-script", dest="use_host_script", action="store_true", default=True, help="Always use the host benchmark script for branch runs (default: True). Use --no-use-host-script to allow branch-local script when available.")
    p.add_argument("--no-use-host-script", dest="use_host_script", action="store_false", help="Allow using a branch-local benchmark script when available.")
    p.add_argument("--use-host-data", dest="use_host_data", action="store_true", default=True, help="Always use the host dataset path for branch runs (default: True). Use --no-use-host-data to allow branch-local datasets when available.")
    p.add_argument("--no-use-host-data", dest="use_host_data", action="store_false", help="Allow using a branch-local dataset path when available.")
    p.add_argument("--out-dir", type=Path, default=Path("bench_out"), help="Directory for branch comparison outputs")
    return p.parse_args()


def postprocess_outdir(out_dir: Path, snakeviz: bool = False, open_snakeviz: bool = False) -> None:
    """Post-process an existing out_dir: locate .prof files and (optionally) start SnakeViz servers.

    Writes a `postprocess_summary.json` file to `out_dir` with discovered profiles and server PIDs.
    """
    out_dir = Path(out_dir).resolve()
    if not out_dir.exists():
        logger.error("Out dir does not exist: %s", out_dir)
        return

    results = {}
    for prof in sorted(out_dir.glob("bench_profile_*.prof")):
        branch = prof.stem.replace("bench_profile_", "")
        res_file = out_dir / f"bench_result_{branch}.json"
        res = None
        if res_file.exists():
            try:
                res = json.loads(res_file.read_text(encoding="utf-8"))
            except Exception:
                logger.exception("Failed to read result file for branch %s", branch)
        results[branch] = {"prof": str(prof), "result": res}

        if snakeviz:
            try:
                if open_snakeviz:
                    proc = subprocess.Popen([sys.executable, "-m", "snakeviz", str(prof)])
                    logger.info("Started SnakeViz server for branch %s (pid=%s) and opened browser", branch, proc.pid)
                    results[branch]["snakeviz_pid"] = proc.pid
                else:
                    proc = subprocess.Popen([sys.executable, "-m", "snakeviz", "-s", str(prof)])
                    logger.info("Started SnakeViz server-only for branch %s (pid=%s). Use --open-snakeviz to open browser.", branch, proc.pid)
                    results[branch]["snakeviz_pid"] = proc.pid
            except FileNotFoundError:
                logger.warning("SnakeViz not installed (pip install snakeviz) - skipping SnakeViz for branch %s", branch)
            except Exception:
                logger.exception("Failed to start SnakeViz for branch %s", branch)

    summary_file = out_dir / "postprocess_summary.json"
    summary_file.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_cli()
    # quick sanity check that repo_root is a git repository (helps catch default path mistakes)
    try:
        check = subprocess.run(["git", "rev-parse", "--git-dir"], cwd=str(args.repo_root), capture_output=True, text=True)
        if check.returncode != 0:
            logger.warning("Repository root %s does not look like a git repository: %s", args.repo_root, (check.stderr or check.stdout).strip())
    except Exception as exc:
        logger.debug("Could not run git to validate repo_root %s: %s", args.repo_root, exc)

    if args.postprocess_only:
        postprocess_outdir(args.out_dir, snakeviz=args.snakeviz, open_snakeviz=args.open_snakeviz)
        return 0

    if args.run_local:
        extensions = args.selected_file_extensions
        if isinstance(extensions, str) and extensions != "all":
            extensions = [e.strip() for e in extensions.split(",") if e.strip()]
        return run_local_benchmark(
            base_path=args.base_path,
            loader=args.loader,
            selected_file_extensions=extensions,
            processing_max_workers=args.processing_max_workers,
            out_path=args.out,
            enable_cprofile=args.cprofile,
            focus_processor=args.focus_processor,
        )

    if args.branches:
        bench_args = []
        # forward CLI args to local runs (except repo-root/branches)
        if args.base_path:
            bench_args += ["--base-path", str(args.base_path)]
        if args.loader:
            bench_args += ["--loader", args.loader]
        if args.selected_file_extensions:
            bench_args += ["--selected-file-extensions", str(args.selected_file_extensions)]

        bench_args += ["--processing-max-workers", str(args.processing_max_workers)]
        if args.focus_processor:
            bench_args += ["--focus-processor", args.focus_processor]

        host_base = None
        if args.base_path:
            host_base = Path(args.base_path).resolve()
            if not host_base.exists():
                logger.warning("Host base path does not exist: %s", host_base)

        # call compare (optionally generate SnakeViz HTML and open it)
        compare_branches(
            args.repo_root,
            args.branches,
            bench_args,
            args.out_dir,
            snakeviz=args.snakeviz,
            open_snakeviz=args.open_snakeviz,
            allow_clone=args.allow_clone,
            host_base_path=host_base,
            use_host_script=args.use_host_script,
            use_host_data=args.use_host_data,
        )
        return 0

    print("No action specified. Use --branches to compare or --run-local to run in this tree.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
