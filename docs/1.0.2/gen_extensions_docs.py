#!/usr/bin/env python3
"""Inject the supported-extensions list into docs/home.html.

Scans SUPPORTED_EXTENSIONS literal assignments in all loader source files under
packages/, then writes the sorted, deduplicated list between the marker comments
in home.html.

Run from the repo root::

    python docs/gen_extensions_docs.py

Invoked by the docs deploy workflow so the list stays in sync with the repo.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

DOCS = Path(__file__).resolve().parent
PACKAGES = DOCS.parent / "packages"
HOME = DOCS / "home.html"

BEGIN = "<!-- BEGIN GENERATED EXTENSIONS (docs/gen_extensions_docs.py) -->"
END = "<!-- END GENERATED EXTENSIONS -->"

_PATTERN = re.compile(r"SUPPORTED_EXTENSIONS\s*(?::\s*\S+)?\s*=\s*(\{[^}]*\})", re.DOTALL)


def _collect_extensions() -> list[str]:
    exts: set[str] = set()
    for py_file in PACKAGES.rglob("*.py"):
        if ".venv" in py_file.parts or "tests" in py_file.parts:
            continue
        text = py_file.read_text(encoding="utf-8", errors="ignore")
        for match in _PATTERN.finditer(text):
            try:
                val = ast.literal_eval(match.group(1))
                if isinstance(val, (set, frozenset)):
                    exts.update(val)
            except Exception:
                pass
    return sorted(exts)


def main() -> None:
    exts = _collect_extensions()
    if not exts:
        print("No extensions found -- skipping home.html update.")
        return
    chips = "".join(f'<span class="ext-chip">{e}</span>' for e in exts)
    html = HOME.read_text(encoding="utf-8")
    if BEGIN not in html or END not in html:
        raise RuntimeError(f"Marker comments not found in {HOME}.")
    start = html.index(BEGIN)
    end = html.index(END) + len(END)
    HOME.write_text(html[:start] + BEGIN + chips + END + html[end:], encoding="utf-8")
    print(f"Wrote {len(exts)} extension chips to {HOME.name}: {', '.join(exts)}")


if __name__ == "__main__":
    main()
