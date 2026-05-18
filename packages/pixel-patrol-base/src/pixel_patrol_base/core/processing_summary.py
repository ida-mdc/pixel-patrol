"""Compact processing summary: timing, throughput, per-stage breakdown."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_log = logging.getLogger(__name__)

_W = 62   # inner box width (between the ║ chars)
_BAR = 10  # bar chart width in characters


def _fmt_s(secs: float) -> str:
    if secs < 0:
        secs = 0.0
    if secs < 60:
        return f"{secs:.1f} s"
    m, s = divmod(int(secs), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m2 = divmod(m, 60)
    return f"{h}h {m2:02d}m"


def _bar(frac: float) -> str:
    filled = max(0, min(_BAR, round(frac * _BAR)))
    return "█" * filled + "░" * (_BAR - filled)


def _ln(content: str) -> str:
    return f"║{content:<{_W}}║"


def _sep() -> str:
    return f"╠{'═' * _W}╣"


@dataclass
class ProcessingSummary:
    n_files: int
    wall_s: float
    worker_count: int
    is_distributed: bool
    load_cpu_s: float
    processor_cpu_s: Dict[str, float]   # processor NAME -> total CPU-seconds
    project_name: str = ""
    date_str: str = ""
    n_tasks: int = 0                    # total tasks dispatched (slices + full-file + batch)
    n_workers_actual: int = 0           # peak workers connected during processing
    worker_nodes: List[str] = field(default_factory=list)   # unique node hostnames
    tasks_per_worker: Dict[str, int] = field(default_factory=dict)  # addr → task count

    def format(self) -> str:  # noqa: A003
        lines: list[str] = []
        lines.append(f"╔{'═' * _W}╗")
        lines.append(_ln("  Pixel Patrol  ·  Processing Summary".center(_W)))
        lines.append(_sep())

        # Meta: project · files (/ tasks)  ·  mode
        name = (self.project_name[:19] + "…") if len(self.project_name) > 20 else self.project_name
        n_w = self.n_workers_actual or self.worker_count
        if self.is_distributed:
            mode = f"distributed · {n_w} workers"
        else:
            mode = f"local · {self.worker_count} proc."

        files_part = f"{self.n_files} files"
        if self.n_tasks > self.n_files:
            files_part += f" / {self.n_tasks} tasks"
        meta = f"  {name}  ·  {files_part}  ·  {mode}"
        if len(meta) > _W:
            meta = meta[:_W - 1] + "…"
        lines.append(_ln(meta))

        # Timing overview
        throughput = self.n_files / self.wall_s if self.wall_s > 0 else 0.0
        total_cpu = self.load_cpu_s + sum(self.processor_cpu_s.values())
        lines.append(_ln(
            f"  {self.date_str}  ·  Wall {_fmt_s(self.wall_s)}"
            f"  ·  CPU {_fmt_s(total_cpu)}"
            f"  ·  {throughput:.2f} files/s"
        ))

        # Task throughput
        n_w = self.n_workers_actual or self.worker_count
        if self.n_tasks > 0 and self.wall_s > 0:
            tasks_per_min = self.n_tasks / (self.wall_s / 60.0)
            avg_per_w = self.n_tasks / n_w if n_w else 0
            lines.append(_ln(
                f"  {self.n_tasks} tasks  ·  {tasks_per_min:.1f} tasks/min"
                f"  ·  avg {avg_per_w:.0f} tasks/worker"
            ))

        # Node list (if distributed and multiple nodes)
        if self.worker_nodes and len(self.worker_nodes) > 1:
            prefix = "  Nodes  "
            shown, rest = [], list(self.worker_nodes)
            avail = _W - len(prefix)
            while rest:
                trailer = f"  … +{len(rest)} more" if rest else ""
                candidate = rest[0]
                if len("  ".join(shown + [candidate])) + len(trailer) <= avail:
                    shown.append(rest.pop(0))
                else:
                    break
            trailer = f"  … +{len(rest)} more" if rest else ""
            lines.append(_ln(prefix + "  ".join(shown) + trailer))

        lines.append(_sep())

        # Stage breakdown
        lines.append(_ln(f"  {'Stage':<18}{'CPU time':>9}  {'/ file':>7}  share"))
        lines.append(_ln(f"  {'─' * 56}"))

        stages: list[tuple[str, float]] = []
        if self.load_cpu_s > 0 or not self.processor_cpu_s:
            stages.append(("loading", self.load_cpu_s))
        for proc_name, cpu_s in self.processor_cpu_s.items():
            stages.append((proc_name, cpu_s))

        total_for_bars = sum(s for _, s in stages) or 1.0
        for stage_name, cpu_s in stages:
            frac = cpu_s / total_for_bars
            pct = round(frac * 100)
            per_file = cpu_s / self.n_files if self.n_files > 0 else 0.0
            lines.append(_ln(
                f"  {stage_name[:18]:<18}{_fmt_s(cpu_s):>9}"
                f"  {_fmt_s(per_file):>7}  {_bar(frac)}  {pct:>3}%"
            ))

        lines.append(f"╚{'═' * _W}╝")
        return "\n".join(lines)

    def write_next_to(self, parquet_path: Path) -> Path:
        summary_path = parquet_path.with_suffix(".summary.txt")
        try:
            summary_path.write_text(self.format(), encoding="utf-8")
        except OSError as exc:
            _log.warning("Could not write processing summary to %s: %s", summary_path, exc)
        return summary_path
