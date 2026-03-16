import sys
from pathlib import Path
from packaging.version import Version, InvalidVersion
import tomlkit

ROOT = Path(__file__).resolve().parents[1] / "packages"
PACKAGES = [
    "pixel-patrol-base",
    "pixel-patrol",
    "pixel-patrol-image",
    "pixel-patrol-loader-bio",
    "pixel-patrol-aqqua",
]

def load_toml(p: Path):
    return tomlkit.parse(p.read_text(encoding="utf8"))

def write_toml(p: Path, data):
    p.write_text(tomlkit.dumps(data), encoding="utf8")

def main(new_version: str):
    try:
        new = Version(new_version)
    except InvalidVersion:
        print(f"ERROR: '{new_version}' is not a valid PEP 440 version (e.g. 0.5.0)")
        raise SystemExit(1)

    for pkg in PACKAGES:
        py = ROOT / pkg / "pyproject.toml"
        if not py.exists():
            print(f"WARN: {py} not found")
            continue
        doc = load_toml(py)
        # check version before writing
        current_str = doc.get("project", {}).get("version")
        if current_str:
            current = Version(current_str)
            if new == current:
                print(f"ERROR: {pkg} is already at version {current_str}")
                raise SystemExit(1)
            if new < current:
                print(f"ERROR: {new_version} is older than {pkg}'s current version {current_str}")
                raise SystemExit(1)
        # set version
        if "project" in doc and "version" in doc["project"]:
            doc["project"]["version"] = new_version
            print(f"Set {pkg} -> {new_version}")
        # update inter-package pinned deps (exact match or ==)
        if "project" in doc and "dependencies" in doc["project"]:
            deps = doc["project"]["dependencies"]

            for i, d in enumerate(deps):
                updated = d
                for pkg_id in PACKAGES:
                    name = pkg_id.replace("_", "-")
                    if d.startswith(pkg_id) or d.startswith(name):
                        parts = d.split(";")
                        left = parts[0].strip()
                        rest = (";" + parts[1]) if len(parts) > 1 else ""
                        name_and_extras = left.split("[", 1)
                        pkg_name = name_and_extras[0].split("==")[0].split(">=")[0].split("~=")[0]
                        extras = "[" + name_and_extras[1] if len(name_and_extras) == 2 else ""
                        updated = f"{pkg_name}=={new_version}{extras}{rest}"
                        break

                if updated != d:
                    deps[i] = updated

        write_toml(py, doc)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: bump_versions.py NEW_VERSION (e.g. 0.2.0)")
        raise SystemExit(2)
    main(sys.argv[1])