"""Fetch a dataset from manifest.toml into $PP_CARDS_DATA/<id>/ (default: ./data).

  prepare.py <id>              full download + unpack
  prepare.py <id> --pp-args    print the pixel-patrol loader/extension/extra args (shell-quoted)
  prepare.py <id> --sample N   N evenly spaced files only (zip: range requests; hf: single-file downloads)
Downloaded archives are untrusted data: this script only writes files, it never executes them.
"""
import argparse
import fnmatch
import os
import shlex
import shutil
import tomllib
from pathlib import Path

import numpy as np

MANIFEST = Path(__file__).with_name("manifest.toml")


def data_root() -> Path:
    return Path(os.environ.get("PP_CARDS_DATA", "data")).resolve()


def fetch_zip_whole(entry: dict, out: Path) -> None:
    import urllib.request
    import zipfile
    archive = out.with_name(out.name + ".zip")
    with urllib.request.urlopen(entry["url"]) as r, open(archive, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    with zipfile.ZipFile(archive) as z:
        z.extractall(out)
    archive.unlink()


def fetch_zip(entry: dict, out: Path, sample: int | None) -> None:
    if not sample:
        return fetch_zip_whole(entry, out)
    try:
        from remotezip import RemoteZip
    except ModuleNotFoundError:
        raise SystemExit("--sample needs remotezip:  pip install remotezip")
    with RemoteZip(entry["url"]) as z:
        members = [i for i in z.infolist() if not i.is_dir()]
        if sample:
            matching = [i for i in members if fnmatch.fnmatch(i.filename, entry["sample"])]
            members = [matching[k] for k in np.linspace(0, len(matching) - 1, sample).astype(int)]
        for info in members:
            z.extract(info, out)
            print(f"  {info.filename} ({info.file_size / 1e6:.1f} MB)")


def fetch_hf(entry: dict, out: Path, sample: int | None) -> None:
    try:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    except ModuleNotFoundError:
        raise SystemExit("hf datasets need huggingface_hub:  pip install huggingface_hub")
    if not sample:
        snapshot_download(entry["repo"], repo_type="dataset", allow_patterns=entry["allow"], local_dir=out)
        return
    names = sorted(f for f in HfApi().list_repo_files(entry["repo"], repo_type="dataset")
                   if any(fnmatch.fnmatch(f, pat) for pat in entry["allow"]))
    for name in [names[k] for k in np.linspace(0, len(names) - 1, sample).astype(int)]:
        hf_hub_download(entry["repo"], name, repo_type="dataset", local_dir=out)
        print(f"  {name}")


def fetch_tar(entry: dict, out: Path, sample: int | None) -> None:
    """Stream tar archive(s) (plain or compressed; LILA's cct_images.tar.gz is uncompressed) from HTTP and extract as
    they arrive: archives are never stored. `url` may be one URL or `urls` a list.
    --sample N takes the first N image-like files of the first archive (cheap: the stream is abandoned after them)."""
    import tarfile
    import urllib.request
    done = 0
    for url in entry.get("urls", [entry.get("url")]):
        with urllib.request.urlopen(url) as r, tarfile.open(fileobj=r, mode="r|*") as tar:
            for member in tar:
                if member.isfile() and sample and done >= sample:
                    break
                if member.name.rsplit("/", 1)[-1].startswith("._"):  # macOS resource forks inside archives are not data
                    continue
                tar.extract(member, out, filter="data")  # 'data' filter rejects absolute paths, .. and special files
                if member.isfile():
                    done += 1
                    if done % 10000 == 0:
                        print(f"  {done:,} files", flush=True)
        if sample:
            break
    print(f"  {done:,} files extracted")


FETCHERS = {"zip": fetch_zip, "hf": fetch_hf, "tar": fetch_tar}


def pixel_patrol_args(entry: dict, only_existing_in: Path | None = None) -> list[str]:
    """only_existing_in: drop `-p <dir>` pairs whose dir is missing there (sample runs only; full runs stay strict)."""
    extra = entry.get("pp_args", [])
    if only_existing_in is not None:
        pairs = [extra[i:i + 2] for i in range(0, len(extra), 2)]  # every pp_args entry is a flag + value
        extra = [a for pair in pairs if pair[0] != "-p" or (only_existing_in / pair[1]).is_dir() for a in pair]
    exts = [a for ext in entry["extensions"] for a in ("-e", ext)]
    return ["--loader", entry["loader"], *exts, *extra]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--sample", type=int, metavar="N")
    ap.add_argument("--pp-args", action="store_true")
    ap.add_argument("--existing-in", type=Path, help="with --pp-args: drop -p paths that do not exist under this dir")
    args = ap.parse_args()
    entry = tomllib.loads(MANIFEST.read_text())[args.dataset]
    if args.pp_args:
        print(shlex.join(pixel_patrol_args(entry, args.existing_in)))
        return
    final = data_root() / args.dataset / ("sample" if args.sample else "full")
    partial = final.with_name(final.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    FETCHERS[entry["kind"]](entry, partial, args.sample)
    partial.rename(final)  # only a complete fetch ever appears under the final name
    print(final)


if __name__ == "__main__":
    main()
