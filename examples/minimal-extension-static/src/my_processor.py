from __future__ import annotations

from typing import Dict

from pixel_patrol_base.core.record import Record
from pixel_patrol_base.core.contracts import ProcessResult
from pixel_patrol_base.core.specs import RecordSpec

POS = {"happy","joy","joyful","love","loved","great","good","calm","grateful","excited","relaxed","awesome"}
NEG = {"sad","angry","hate","hated","bad","awful","terrible","tired","sick","upset","anxious","stressed","frustrated"}

class MarkdownMoodProcessor:
    NAME   = "markdown-mood"
    INPUT  = RecordSpec(kinds={"text/markdown"})
    OUTPUT = "features"

    OUTPUT_SCHEMA = {
        "positivity_factor": float,
    }

    DESCRIPTION = "Computes positivity from YAML-like moods in Markdown front matter."

    def run(self, art: Record) -> ProcessResult:
        moods = [m.lower() for m in (art.meta.get("moods") or [])]
        pos = sum(1 for m in moods if m in POS)
        neg = sum(1 for m in moods if m in NEG)
        total = len(moods)
        score = 0.0 if total == 0 else (pos - neg) / total  # range [-1, 1]

        features: Dict[str, object] = {
            "positivity_factor": float(score),
        }
        return features
