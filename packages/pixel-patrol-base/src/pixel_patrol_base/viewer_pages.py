from __future__ import annotations

import base64
import json
import mimetypes
import re
import shutil
from pathlib import Path

from pixel_patrol_base.viewer_server import _discover_installed_extensions, find_viewer_dist


def _safe_name(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-")
    return safe or "extension"


def _file_to_data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    if not mime:
        mime = "application/octet-stream"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _text_to_data_url(text: str, mime: str) -> str:
    b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return f"data:{mime};base64,{b64}"


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


def _inline_local_assets(html: str, dist_dir: Path) -> str:
    asset_data_url_map: dict[str, str] = {}
    assets_dir = dist_dir / "assets"
    if assets_dir.is_dir():
        for asset_path in assets_dir.rglob("*"):
            if not asset_path.is_file():
                continue
            rel = asset_path.relative_to(dist_dir).as_posix()
            name = asset_path.name
            data_url = _file_to_data_url(asset_path)
            # Support the common path forms emitted by bundlers/runtime code.
            asset_data_url_map[f"./{rel}"] = data_url
            asset_data_url_map[rel] = data_url
            asset_data_url_map[f"/{rel}"] = data_url
            # Some runtime chunks reference emitted assets by basename from
            # document root (e.g. "/duckdb-...worker.js"). Cover those too.
            asset_data_url_map[f"./{name}"] = data_url
            asset_data_url_map[name] = data_url
            asset_data_url_map[f"/{name}"] = data_url

    def inline_js_asset_urls(code: str) -> str:
        # Preserve import.meta.url behavior by replacing runtime-resolved asset
        # paths (worker/wasm/etc.) with direct data URLs before embedding.
        for rel, data_url in asset_data_url_map.items():
            code = code.replace(f'"{rel}"', f'"{data_url}"')
            code = code.replace(f"'{rel}'", f"'{data_url}'")
        return code

    def repl_script(match: re.Match[str]) -> str:
        attrs = match.group("attrs")
        src = match.group("src")
        if src.startswith("http://") or src.startswith("https://"):
            return match.group(0)
        path = (dist_dir / src.lstrip("./")).resolve()
        if not path.is_file():
            return match.group(0)
        code = path.read_text(encoding="utf-8")
        code = inline_js_asset_urls(code)
        return f"<script{attrs}>{code}</script>"

    def repl_style(match: re.Match[str]) -> str:
        href = match.group("href")
        if href.startswith("http://") or href.startswith("https://"):
            return match.group(0)
        path = (dist_dir / href.lstrip("./")).resolve()
        if not path.is_file():
            return match.group(0)
        css = path.read_text(encoding="utf-8")
        return f"<style>{css}</style>"

    def repl_media(match: re.Match[str]) -> str:
        attr = match.group("attr")
        url = match.group("url")
        if url.startswith("http://") or url.startswith("https://") or url.startswith("data:"):
            return match.group(0)
        path = (dist_dir / url.lstrip("./")).resolve()
        if not path.is_file():
            return match.group(0)
        return f'{attr}="{_file_to_data_url(path)}"'

    html = re.sub(
        r'<script(?P<attrs>[^>]*?)\s+src="(?P<src>[^"]+)"\s*>\s*</script>',
        repl_script,
        html,
    )
    html = re.sub(
        r'<link(?=[^>]*rel="stylesheet")[^>]*href="(?P<href>[^"]+)"[^>]*>',
        repl_style,
        html,
    )
    html = re.sub(r'(?P<attr>(?:src|href))="(?P<url>[^"]+)"', repl_media, html)
    return html


def _build_inline_extension_urls(extension_dirs: list[Path]) -> list[str]:
    urls: list[str] = []
    for ext_dir in extension_dirs:
        manifest_path = ext_dir / "extension.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        plugins = manifest.get("plugins", [])
        inline_plugins: list[str] = []
        for rel in plugins:
            plugin_path = (ext_dir / rel).resolve()
            if not plugin_path.is_file():
                continue
            inline_plugins.append(
                _text_to_data_url(plugin_path.read_text(encoding="utf-8"), "application/javascript")
            )
        manifest["plugins"] = inline_plugins
        urls.append(_text_to_data_url(json.dumps(manifest), "application/json"))
    return urls


def build_github_pages_site(out_dir: str | Path = "gh-pages-site") -> Path:
    out_dir = Path(out_dir).resolve()
    dist_dir = find_viewer_dist()

    # Promote hand-crafted site files from docs/ output to the site root.
    # mkdocs builds into out_dir/docs/ and copies non-markdown files verbatim.
    docs_out = out_dir / "docs"

    # Promote all root-level HTML files (home.html -> index.html, others keep their name)
    for html_file in docs_out.glob("*.html"):
        dst_name = "index.html" if html_file.name == "home.html" else html_file.name
        shutil.copy2(html_file, out_dir / dst_name)

    # Promote assets/ and example.parquet
    docs_assets = docs_out / "assets"
    if docs_assets.is_dir():
        dst_assets = out_dir / "assets"
        if dst_assets.exists():
            shutil.rmtree(dst_assets)
        shutil.copytree(docs_assets, dst_assets)
    example_parquet = docs_out / "example.parquet"
    if example_parquet.is_file():
        shutil.copy2(example_parquet, out_dir / "example.parquet")

    # Viewer lives at /viewer/ so the site root is free for the landing page.
    viewer_dir = out_dir / "viewer"
    if viewer_dir.exists():
        shutil.rmtree(viewer_dir)
    shutil.copytree(dist_dir, viewer_dir)

    extension_dirs = _discover_installed_extensions()
    ext_root = viewer_dir / "extensions"
    ext_root.mkdir(parents=True, exist_ok=True)

    urls: list[str] = []
    for idx, ext_dir in enumerate(extension_dirs):
        dst_name = f"{idx:02d}-{_safe_name(ext_dir.name)}"
        dst_dir = ext_root / dst_name
        shutil.copytree(ext_dir, dst_dir)
        urls.append(f"./extensions/{dst_name}/extension.json")

    (viewer_dir / "pp_extension_urls.json").write_text(
        json.dumps(urls, indent=2) + "\n",
        encoding="utf-8",
    )
    _inject_extension_urls(viewer_dir / "index.html", urls)

    # Tell the viewer that the logo should navigate back to the site landing page.
    viewer_index = viewer_dir / "index.html"
    html = viewer_index.read_text(encoding="utf-8")
    homepage_script = "<script>\nwindow.__PP_HOMEPAGE = '../';\n</script>\n"
    html = html.replace("</head>", homepage_script + "</head>", 1)
    viewer_index.write_text(html, encoding="utf-8")

    return out_dir


def build_single_file_viewer_html(output_html: str | Path) -> Path:
    output_html = Path(output_html).resolve()
    dist_dir = find_viewer_dist()
    index_html = dist_dir / "index.html"
    html = index_html.read_text(encoding="utf-8")

    html = _inline_local_assets(html, dist_dir)
    ext_urls = _build_inline_extension_urls(_discover_installed_extensions())
    script = (
        "<script>\n"
        f"window.__PP_EXTENSION_URLS = {json.dumps(ext_urls)};\n"
        "</script>\n"
    )
    if "</head>" in html:
        html = html.replace("</head>", script + "</head>", 1)
    else:
        html = script + html

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")
    return output_html
