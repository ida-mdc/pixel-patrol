import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np

from pixel_patrol_base.core.contracts import FileInfo
from pixel_patrol_base.core.record import record_from, Record

FRONT_RE = re.compile(r"^---\s*(.*?)\s*---\s*(.*)$", re.DOTALL)
DATE_RE  = re.compile(r"(?m)^\s*date:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})\s*$")
MOOD_RE  = re.compile(r"(?m)^\s*moods?:\s*(.+?)\s*$")


def _parse_header(text: str) -> Tuple[Optional[str], List[str], str]:
    m = FRONT_RE.match(text)
    if not m:
        return None, [], text
    header, body = m.group(1), m.group(2)
    date  = DATE_RE.search(header)
    moods = MOOD_RE.search(header)
    date_str   = date.group(1) if date else None
    moods_list = []
    if moods:
        raw = moods.group(1)
        moods_list = [t.strip().lower() for t in raw.split(",") if t.strip()] if "," in raw else []
    return date_str, moods_list, body.strip()


class MarkdownDiaryLoader:
    NAME = "markdown-diary"

    SUPPORTED_EXTENSIONS: Set[str]  = {"md"}
    FOLDER_EXTENSIONS:    Set[str]  = set()
    CONTAINER_EXTENSIONS: Set[str]  = set()

    # No array-level columns — metadata columns come from the processor.
    OUTPUT_SCHEMA:          Dict[str, Any] = {}
    OUTPUT_SCHEMA_PATTERNS: List           = []

    def is_folder_supported(self, path: Path) -> bool:
        return False

    def read_header(self, file_path: Path) -> FileInfo:
        size = Path(file_path).stat().st_size
        return FileInfo(shape=(max(size, 1),), dtype=np.uint8, dim_order=("X",))

    def load(self, file_path: Path) -> Record:
        text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        date_str, moods_list, body = _parse_header(text)
        meta = {
            "entry_date": date_str,
            "moods":      moods_list,
            "free_text":  body,
            "dim_order":  "X",
        }
        # 1-element placeholder so the pipeline can extract dtype/shape from the record.
        return record_from(np.array([0], dtype=np.uint8), meta, kind="text/markdown")

    def load_range(self, file_path: Path, start: int, stop: int) -> Iterator[Tuple[str, Record]]:
        raise NotImplementedError("markdown-diary is not a container format")
