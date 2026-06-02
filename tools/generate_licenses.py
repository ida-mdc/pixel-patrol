#!/usr/bin/env python3
"""Generate THIRD_PARTY_LICENSES.md for a package, run from the package directory.

Usage (from a package directory):
    uv run --with pip-licenses python ../../tools/generate_licenses.py > THIRD_PARTY_LICENSES.md
"""

import csv
import io
import re
import shutil
import subprocess
import sys

SPDX_MAP = {
    "Apache Software License":                                   "Apache-2.0",
    "Apache Software License; BSD License":                      "Apache-2.0 AND BSD",
    "BSD License":                                               "BSD",
    "BSD 3-Clause License":                                      "BSD-3-Clause",
    "BSD 2-Clause License":                                      "BSD-2-Clause",
    "GNU Lesser General Public License v2 or later (LGPLv2+)":  "LGPL-2.0-or-later",
    "GNU Lesser General Public License v2 (LGPLv2)":            "LGPL-2.0",
    "MIT License":                                               "MIT",
    "Mozilla Public License 2.0 (MPL 2.0)":                     "MPL-2.0",
    "Python Software Foundation License":                        "PSF-2.0",
    "ISC License (ISCL)":                                        "ISC",
    "Historical Permission Notice and Disclaimer (HPND)":        "HPND",
    "3-Clause BSD License":                                      "BSD-3-Clause",
}

FIRST_PARTY = [
    "pixel-patrol",
    "pixel-patrol-base",
    "pixel-patrol-aqqua",
    "pixel-patrol-image",
    "pixel-patrol-loader-bio",
    "pixel-patrol-tensorboard",
]

LOOKS_LIKE_TEXT = re.compile(
    r"(permission|software|liability|warranty|distribute|modification|"
    r"copyright notice|provided|furnished|sublicense)",
    re.IGNORECASE,
)


def normalize_license(raw: str) -> str:
    raw = raw.strip()
    if raw in SPDX_MAP:
        return SPDX_MAP[raw]
    if LOOKS_LIKE_TEXT.search(raw):
        return "UNKNOWN"
    return raw


def main():
    cli = shutil.which("pip-licenses")
    if not cli:
        print("pip-licenses not found. Run: uv run --with pip-licenses python tools/generate_licenses.py", file=sys.stderr)
        sys.exit(1)

    result = subprocess.run(
        [cli, "--format=csv", "--with-authors", "--with-urls", "--order=name",
         "--from=mixed", "--ignore-packages", *FIRST_PARTY],
        capture_output=True, text=True,
    )
    if not result.stdout.strip():
        print("No output from pip-licenses:", result.stderr, file=sys.stderr)
        sys.exit(1)

    reader = csv.DictReader(io.StringIO(result.stdout))
    rows = []
    for pkg in reader:
        rows.append((
            pkg.get("Name", ""),
            pkg.get("Version", ""),
            normalize_license(pkg.get("License", "")),
            pkg.get("Author", ""),
            pkg.get("URL", ""),
        ))

    headers = ("Name", "Version", "License", "Author", "URL")
    col_widths = [max(len(r[i]) for r in rows + [headers]) for i in range(5)]

    def fmt_row(r):
        return "| " + " | ".join(r[i].ljust(col_widths[i]) for i in range(5)) + " |"

    separator = "| " + " | ".join("-" * w for w in col_widths) + " |"
    print("\n".join([fmt_row(headers), separator] + [fmt_row(r) for r in rows]))


if __name__ == "__main__":
    main()
