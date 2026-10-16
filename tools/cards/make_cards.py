"""Render trading cards (63x88 mm, 600 dpi) for datasets profiled with PixelPatrol.

Usage: python make_cards.py specs/*.toml   ->  out/<slug>.png and out/sheet.pdf (A4, 3x3, crop marks)
"""
import io
import sys
import textwrap
import tomllib
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl
import segno
from PIL import Image, ImageDraw, ImageFont

DPI = 600
MM = DPI / 25.4
CARD_W, CARD_H = round(63 * MM), round(88 * MM)
BLEED = round(3 * MM)
INK, PAPER, MUTED = "#1c1f24", "#fbfaf6", "#6b7280"
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"


def font(size_mm: float, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(FONT_DIR / name), round(size_mm * MM))


def load_stats(parquet: Path) -> dict:
    df = pl.read_parquet(parquet)
    imgs = df.filter(pl.col("obs_level") == 0)
    sizes = imgs.select(["size_X", "size_Y"]).drop_nulls()
    return {
        "n_images": imgs.height,
        "total_mb": imgs["size_bytes"].sum() / 1e6,
        "dims": (int(sizes["size_X"].median()), int(sizes["size_Y"].median())) if sizes.height else None,
        "dtype": imgs["dtype"].drop_nulls().mode().first(),
        "ext": ", ".join(imgs["file_extension"].drop_nulls().unique().sort()),
        "thumbs": imgs.filter(pl.col("thumbnail").is_not_null())["thumbnail"].to_list(),
    }


def decode_thumb(raw: bytes) -> Image.Image:
    """PixelPatrol bakes thumbnails as a flat square RGBA uint8 canvas, padded transparent."""
    side = int(round((len(raw) / 4) ** 0.5))
    rgba = Image.fromarray(np.frombuffer(raw, np.uint8)[: side * side * 4].reshape(side, side, 4), "RGBA")
    box = rgba.getchannel("A").getbbox()
    return rgba.crop(box) if box else rgba


def cover(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    scale = max(size[0] / img.width, size[1] / img.height)
    big = img.resize((max(size[0], round(img.width * scale)), max(size[1], round(img.height * scale))), Image.LANCZOS)
    left, top = (big.width - size[0]) // 2, (big.height - size[1]) // 2
    return big.crop((left, top, left + size[0], top + size[1]))


def mosaic(thumbs: list[bytes], size_px: tuple[int, int], grid: int = 3) -> Image.Image:
    pick = np.linspace(0, len(thumbs) - 1, grid * grid).astype(int)
    gap = round(0.5 * MM)
    cell = ((size_px[0] - gap * (grid - 1)) // grid, (size_px[1] - gap * (grid - 1)) // grid)
    out = Image.new("RGB", size_px, PAPER)
    for i, idx in enumerate(pick):
        tile = Image.new("RGB", cell, INK)
        t = cover(decode_thumb(thumbs[idx]), cell)
        tile.paste(t, mask=t.getchannel("A"))
        out.paste(tile, ((i % grid) * (cell[0] + gap), (i // grid) * (cell[1] + gap)))
    return out


def qr_image(url: str, px: int) -> Image.Image:
    buf = io.BytesIO()
    segno.make(url, error="m").save(buf, kind="png", scale=1, border=0, dark=INK, light=PAPER)
    return Image.open(buf).convert("RGB").resize((px, px), Image.NEAREST)


def wrapped(draw, text, xy, fnt, width_px, fill, line_gap=1.25) -> int:
    chars = max(10, int(width_px / (fnt.size * 0.55)))
    y = xy[1]
    for line in textwrap.wrap(text, chars):
        draw.text((xy[0], y), line, font=fnt, fill=fill)
        y += round(fnt.size * line_gap)
    return y


REQUIRED = ("title", "tagline", "parquet", "citation", "doi", "license", "license_url", "source_url",
            "verified_on", "report_url", "accent")


def check_printable(spec: dict, name: str) -> None:
    missing = [k for k in REQUIRED if not spec.get(k)]
    if missing or spec.get("unverified"):
        raise SystemExit(f"{name}: refusing to render - " + (
            "marked unverified" if spec.get("unverified") else f"missing {', '.join(missing)}"))


def render_card(spec: dict, root: Path) -> Image.Image:
    s = load_stats((root / spec["parquet"]).resolve())
    card = Image.new("RGB", (CARD_W, CARD_H), PAPER)
    d = ImageDraw.Draw(card)
    m = round(4 * MM)
    inner = CARD_W - 2 * m
    d.rectangle([0, 0, CARD_W, round(13 * MM)], fill=spec["accent"])
    d.text((m, round(3 * MM)), spec["title"], font=font(5, True), fill="white")
    d.text((m, round(9 * MM)), spec["tagline"][:48], font=font(2.1), fill="white")

    art_h = round(inner * 0.70)
    card.paste(mosaic(s["thumbs"], (inner, art_h)), (m, round(15.5 * MM)))
    y = round(15.5 * MM) + art_h + round(2.5 * MM)

    dims = f'~{s["dims"][0]}×{s["dims"][1]}' if s["dims"] else "n/a"
    stats = [("IMAGES", f'{s["n_images"]:,}'), ("SIZE", f'{s["total_mb"]:.1f} MB'),
             ("PIXELS (MEDIAN)", dims), ("PIXEL TYPE", s["dtype"])]
    colw = (inner - round(19 * MM)) // 2
    for i, (k, v) in enumerate(stats):
        x, yy = m + (i % 2) * colw, y + (i // 2) * round(7 * MM)
        d.text((x, yy), k, font=font(1.7, True), fill=MUTED)
        d.text((x, yy + round(2 * MM)), v, font=font(2.4, True), fill=INK)
    y += round(14.5 * MM)

    qr_px = round(17 * MM)
    wrapped(d, spec["citation"], (m, y), font(1.6), inner - qr_px - round(2 * MM), INK, 1.3)
    d.text((m, CARD_H - round(5.5 * MM)), f'License: {spec["license"]}', font=font(1.8, True), fill=INK)
    card.paste(qr_image(spec["report_url"], qr_px), (CARD_W - m - qr_px, CARD_H - m - qr_px - round(3 * MM)))
    d.text((CARD_W - m - qr_px, CARD_H - m - round(2.2 * MM)), "scan: PixelPatrol report", font=font(1.3), fill=MUTED)
    d.rectangle([0, 0, CARD_W - 1, CARD_H - 1], outline=spec["accent"], width=round(0.6 * MM))
    return card


def sheet(cards: list[Image.Image], path: Path) -> None:
    a4 = (round(210 * MM), round(297 * MM))
    page = Image.new("RGB", a4, "white")
    d = ImageDraw.Draw(page)
    ox, oy = (a4[0] - 3 * CARD_W) // 2, (a4[1] - 3 * CARD_H) // 2
    for i in range(9):
        x, y = ox + (i % 3) * CARD_W, oy + (i // 3) * CARD_H
        page.paste(cards[i % len(cards)], (x, y))
    for i in range(4):  # crop marks
        x = ox + i * CARD_W
        d.line([x, oy - 5 * MM, x, oy - 1 * MM], fill="black", width=3)
        d.line([x, oy + 3 * CARD_H + 1 * MM, x, oy + 3 * CARD_H + 5 * MM], fill="black", width=3)
        y = oy + i * CARD_H
        d.line([ox - 5 * MM, y, ox - 1 * MM, y], fill="black", width=3)
        d.line([ox + 3 * CARD_W + 1 * MM, y, ox + 3 * CARD_W + 5 * MM, y], fill="black", width=3)
    page.save(path, "PDF", resolution=DPI)


def main(spec_paths: list[str]) -> None:
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    cards = []
    for sp in map(Path, spec_paths):
        spec = tomllib.loads(sp.read_text())
        check_printable(spec, sp.name)
        card = render_card(spec, sp.parent)
        card.save(out / f"{sp.stem}.png", dpi=(DPI, DPI))
        cards.append(card)
    sheet(cards, out / "sheet.pdf")
    print(f"wrote {len(cards)} card(s) + sheet.pdf to {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
