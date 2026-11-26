"""Explicitly fetch or verify the pinned model inputs. No third-party dependencies."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.parse
import urllib.request


def check(path, item):
    if path.stat().st_size != item["size"]:
        raise ValueError(f"{path}: size mismatch")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if item["sha256"] is not None and digest != item["sha256"]:
        raise ValueError(f"{path}: SHA-256 mismatch")
    if item["sha256"] is None:
        # First audit of small, ordinary Git blobs. The frozen manifest records
        # the observed SHA-256 afterwards, like the LFS weights.
        with path.open("rb") as stream:
            git = hashlib.sha1(f"blob {item['size']}\0".encode(), usedforsecurity=False)
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                git.update(chunk)
        if git.hexdigest() != item["git_blob"]:
            raise ValueError(f"{path}: Git blob ID mismatch")
    return digest


def fetch(path, item, url):
    if path.exists():
        try:
            return check(path, item)
        except ValueError:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    # Verify before replacing an existing path. Readers never see a
    # partially downloaded file even when two processes fetch concurrently.
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                remaining = item["size"]
                while remaining:
                    chunk = response.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    stream.write(chunk)
                    remaining -= len(chunk)
                if response.read(1):
                    raise ValueError(f"{url}: download exceeds expected size")
            stream.close()
            digest = check(temporary, item)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, option in (("fetch", "--output"), ("verify", "--directory")):
        command = commands.add_parser(name)
        command.add_argument("--manifest", type=Path, default=Path("models/manifest.json"))
        command.add_argument(option, dest="directory", type=Path, required=True)
        command.add_argument("--repo", action="append", help="Limit to an exact repository ID")
        command.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    manifest = json.loads(args.manifest.read_text())
    sources = manifest["sources"]
    if args.repo and set(args.repo) - {source["repo"] for source in sources}:
        parser.error("--repo is absent from the manifest")

    def run(entry):
        source, item = entry
        relative = Path(source["repo"], source["revision"], item["filename"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe manifest path: {relative}")
        path = args.directory / relative
        url = "https://huggingface.co/" + urllib.parse.quote(
            f"{source['repo']}/resolve/{source['revision']}/{item['filename']}"
        )
        if args.command == "fetch":
            digest = fetch(path, item, url + "?download=true")
        else:
            digest = check(path, item)
        print(f"{digest}  {relative}", flush=True)

    entries = [
        (source, item)
        for source in sources
        if not args.repo or source["repo"] in args.repo
        for item in source["files"]
        if item["required"]
    ]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(run, entries))


if __name__ == "__main__":
    main()
