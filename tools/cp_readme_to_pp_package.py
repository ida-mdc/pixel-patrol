from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "README.md"
DST = ROOT / "packages/pixel-patrol/README.md"

# PyPI fetches images over HTTP so we need absolute GitHub raw URLs
GITHUB_RAW = "https://raw.githubusercontent.com/ida-mdc/pixel-patrol/main"

def rewrite_readme_assets_paths(lines):
    return [
        line
        .replace("packages/pixel-patrol/readme_assets/", f"{GITHUB_RAW}/packages/pixel-patrol/readme_assets/")
        .replace(
            "packages/pixel-patrol-base/src/pixel_patrol_base/processing_assets/",
            f"{GITHUB_RAW}/packages/pixel-patrol-base/src/pixel_patrol_base/processing_assets/",
        )
        for line in lines
    ]

def main():
    lines = SRC.read_text(encoding="utf8").splitlines()
    lines = rewrite_readme_assets_paths(lines)
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text("\n".join(lines) + "\n", encoding="utf8")

if __name__ == "__main__":
    main()