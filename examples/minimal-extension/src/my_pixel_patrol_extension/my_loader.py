from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Set

from pixel_patrol_base.core.artifact import Artifact

FRONT_RE = re.compile(r"^---\s*(.*?)\s*---\s*(.*)$", re.DOTALL)
DATE_RE  = re.compile(r"(?m)^\s*date:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})\s*$")
MOOD_RE  = re.compile(r"(?m)^\s*moods?:\s*(.+?)\s*$")

def _parse_header(text: str) -> tuple[Optional[str], List[str], str]:
    m = FRONT_RE.match(text)
    if not m:
        return None, [], text
    header, body = m.group(1), m.group(2)
    date = DATE_RE.search(header)
    moods = MOOD_RE.search(header)

    date_str = date.group(1) if date else None
    moods_list = []
    if moods:
        raw = moods.group(1)
        moods_list = [t.strip().lower() for t in raw.split(",") if t.strip()] if "," in raw else []
    return date_str, moods_list, body.strip()

class MarkdownDiaryLoader:
    NAME = "markdown-diary"

    SUPPORTED_EXTENSIONS: Set[str] = {"md"}

    def load(self, source: str) -> Artifact:
        p = Path(source)
        text = p.read_text(encoding="utf-8", errors="ignore")
        date_str, moods_list, body = _parse_header(text)

        meta = {
            "entry_date": date_str,          # e.g. "2025-08-01"
            "moods": moods_list,             # list[str]
            "free_text": body,               # markdown body (no header)
        }
        return Artifact(
            data=None,
            axes=set(),
            kind="text/markdown",
            meta=meta,
            capabilities=set(),
        )