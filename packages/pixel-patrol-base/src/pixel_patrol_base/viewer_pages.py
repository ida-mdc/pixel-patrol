from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from pixel_patrol_base.viewer_server import _discover_installed_extensions, find_viewer_dist


def _safe_name(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-")
    return safe or "extension"


def _inject_extension_urls(index_html: Path, urls: list[str]) -> None:
    html = index_html.read_text(encoding="utf-8")
    script = (
        "<script>\n"
        f"window.__PP_EXTENSION_URLS = {json.dumps(urls)};\n"
        "</script>\n"
    )
    if "</head>" in html:
        html = html.replace("</head>", script + "</head>", 1)
    else:
        html = script + html
    index_html.write_text(html, encoding="utf-8")


def build_github_pages_site(out_dir: str | Path = "gh-pages-site") -> Path:
    out_dir = Path(out_dir).resolve()
    dist_dir = find_viewer_dist()

    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(dist_dir, out_dir)

    extension_dirs = _discover_installed_extensions()
    ext_root = out_dir / "extensions"
    ext_root.mkdir(parents=True, exist_ok=True)

    urls: list[str] = []
    for idx, ext_dir in enumerate(extension_dirs):
        dst_name = f"{idx:02d}-{_safe_name(ext_dir.name)}"
        dst_dir = ext_root / dst_name
        shutil.copytree(ext_dir, dst_dir)
        urls.append(f"./extensions/{dst_name}/extension.json")

    (out_dir / "pp_extension_urls.json").write_text(
        json.dumps(urls, indent=2) + "\n",
        encoding="utf-8",
    )
    _inject_extension_urls(out_dir / "index.html", urls)
    return out_dir
