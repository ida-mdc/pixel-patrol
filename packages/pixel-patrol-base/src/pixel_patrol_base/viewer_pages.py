from __future__ import annotations

import base64
import json
import mimetypes
import re
import shutil
from pathlib import Path

from pixel_patrol_base.viewer_server import _discover_installed_extensions, find_viewer_dist, resolve_extension_plugins

# Fallback used only if the build metadata is missing from the viewer bundle
# (e.g. an older bundle built before pp_build_meta.json was emitted).
_DEFAULT_DUCKDB_WASM_VERSION = "1.32.0"


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


def _inline_local_assets(html: str, dist_dir: Path, exclude_names: frozenset[str] = frozenset()) -> str:
    """Inline local script/style/media assets as data URLs.

    Assets whose file name is in exclude_names are left as-is (their references
    are not rewritten), so they can instead be served from a CDN - used to keep
    the large DuckDB WASM/worker out of the file while inlining everything else.
    """
    asset_data_url_map: dict[str, str] = {}
    assets_dir = dist_dir / "assets"
    if assets_dir.is_dir():
        for asset_path in assets_dir.rglob("*"):
            if not asset_path.is_file() or asset_path.name in exclude_names:
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
        # Rolldown (Vite 8's bundler) minifies these as template literals
        # (backticks) rather than quotes, so all three forms must be covered.
        for rel, data_url in asset_data_url_map.items():
            code = code.replace(f'"{rel}"', f'"{data_url}"')
            code = code.replace(f"'{rel}'", f"'{data_url}'")
            code = code.replace(f'`{rel}`', f'`{data_url}`')
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


def _read_duckdb_wasm_version(dist_dir: Path) -> str:
    meta_path = dist_dir / "pp_build_meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            version = meta.get("duckdbWasmVersion")
            if version:
                return str(version)
        except (ValueError, OSError):
            pass
    return _DEFAULT_DUCKDB_WASM_VERSION


def _duckdb_asset_names(dist_dir: Path) -> frozenset[str]:
    """File names of the bundled DuckDB WASM worker/module (hash-suffixed)."""
    assets_dir = dist_dir / "assets"
    if not assets_dir.is_dir():
        return frozenset()
    return frozenset(
        p.name for p in assets_dir.glob("*")
        if p.is_file() and "duckdb" in p.name.lower()
    )


def _duckdb_cdn_urls(dist_dir: Path) -> tuple[str, str]:
    """jsdelivr URLs for the (single-thread mvp) DuckDB WASM worker and module."""
    version = _read_duckdb_wasm_version(dist_dir)
    base = f"https://cdn.jsdelivr.net/npm/@duckdb/duckdb-wasm@{version}/dist"
    return (
        f"{base}/duckdb-browser-mvp.worker.js",
        f"{base}/duckdb-mvp.wasm",
    )


def _build_inline_extension_urls(extension_dirs: list[Path]) -> list[str]:
    urls: list[str] = []
    for ext_dir in extension_dirs:
        manifest_path = ext_dir / "extension.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        plugins = resolve_extension_plugins(ext_dir, manifest)
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


def _resolve_manifest_plugins(ext_dir: Path) -> None:
    """Bake an explicit ``plugins`` list into a copied extension's manifest.

    The client reads ``manifest.plugins`` directly and cannot resolve
    ``auto_detect`` over static HTTP (no directory listing). The dev server and
    single-file build resolve it server-side; for the site folder we resolve it
    here so the widgets are actually discovered.
    """
    manifest_path = ext_dir / "extension.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["plugins"] = resolve_extension_plugins(ext_dir, manifest)
    manifest.pop("auto_detect", None)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def build_github_pages_site(out_dir: str | Path = "gh-pages-site") -> Path:
    out_dir = Path(out_dir).resolve()
    dist_dir = find_viewer_dist()

    # Promote hand-crafted site files from docs/ output to the site root.
    # mkdocs builds into out_dir/docs/ and copies non-markdown files verbatim.
    docs_out = out_dir / "docs"

    # Promote root-level HTML files: home.html becomes the site index, others
    # keep their name. docs/index.html (the mkdocs Quickstart page) is skipped
    # here - it stays accessible at /docs/ and must not overwrite home.html.
    for html_file in docs_out.glob("*.html"):
        if html_file.name == "index.html":
            continue
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
        _resolve_manifest_plugins(dst_dir)
        urls.append(f"./extensions/{dst_name}/extension.json")

    (viewer_dir / "pp_extension_urls.json").write_text(
        json.dumps(urls, indent=2) + "\n",
        encoding="utf-8",
    )
    _inject_extension_urls(viewer_dir / "index.html", urls)

    return out_dir


def build_single_file_viewer_html(
    output_html: str | Path,
    offline: bool = False,
) -> Path:
    """Write a single-file HTML viewer.

    Default (light): inline the app's JS/CSS/images, but load the large DuckDB
    WASM from jsdelivr (and keep the existing Bootstrap CDN links). This is
    ~7 MB instead of ~56 MB and works for any locally-built/installed version,
    needing a network connection only for DuckDB. Extensions are inlined too.

    offline=True: a fully self-contained file with all JS/CSS/WASM inlined as
    data URLs - works without any network access but is large (~50 MB+).
    """
    output_html = Path(output_html).resolve()
    dist_dir = find_viewer_dist()
    index_html = dist_dir / "index.html"
    html = index_html.read_text(encoding="utf-8")

    head_scripts = []
    if offline:
        # Everything inlined, including the DuckDB worker/wasm - no network.
        html = _inline_local_assets(html, dist_dir)
    else:
        # Inline the app bundle but keep DuckDB out of the file.
        html = _inline_local_assets(html, dist_dir, exclude_names=_duckdb_asset_names(dist_dir))
        worker_url, wasm_url = _duckdb_cdn_urls(dist_dir)
        head_scripts.append(
            "<script>\n"
            f"window.__PP_DUCKDB_WORKER_URL = {json.dumps(worker_url)};\n"
            f"window.__PP_DUCKDB_WASM_URL = {json.dumps(wasm_url)};\n"
            "</script>\n"
        )

    ext_urls = _build_inline_extension_urls(_discover_installed_extensions())
    head_scripts.append(
        "<script>\n"
        f"window.__PP_EXTENSION_URLS = {json.dumps(ext_urls)};\n"
        "</script>\n"
    )

    head = "".join(head_scripts)
    if "</head>" in html:
        html = html.replace("</head>", head + "</head>", 1)
    else:
        html = head + html

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")
    return output_html
