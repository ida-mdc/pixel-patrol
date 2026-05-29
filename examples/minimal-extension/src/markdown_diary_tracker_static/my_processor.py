from __future__ import annotations

from typing import Dict, Optional

from pixel_patrol_base.core.contracts import ChunkKind
from pixel_patrol_base.core.record import Record
from pixel_patrol_base.core.specs import RecordSpec

POS = {"happy","joy","joyful","love","loved","great","good","calm","grateful","excited","relaxed","awesome"}
NEG = {"sad","angry","hate","hated","bad","awful","terrible","tired","sick","upset","anxious","stressed","frustrated"}


class MarkdownMoodProcessor:
    NAME       = "markdown-mood"
    CHUNK_KIND = ChunkKind.MEMORY          # process the whole record at once
    INPUT      = RecordSpec(kinds={"text/markdown"})
    OUTPUT     = "features"

    OUTPUT_SCHEMA          = {"positivity_factor": float}
    OUTPUT_SCHEMA_PATTERNS = []

    def run_chunk(self, record: Record) -> Dict:
        moods = [m.lower() for m in (record.meta.get("moods") or [])]
        pos   = sum(1 for m in moods if m in POS)
        neg   = sum(1 for m in moods if m in NEG)
        total = len(moods)
        score = 0.0 if total == 0 else (pos - neg) / total  # range [-1, 1]
        return {"positivity_factor": float(score)}

    def get_aggregation(self, col: str) -> Optional[callable]:
        if col not in self.OUTPUT_SCHEMA:
            return None
        # One chunk per file — return the value from the single chunk row.
        return lambda rows, g_dims: rows[0][col] if rows else None
