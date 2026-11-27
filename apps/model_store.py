"""Resolve and download Nathr's pinned checkpoints."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def default_cache():
    if models := os.environ.get("NATHR_MODELS"):
        return Path(models)
    cache = os.environ.get("XDG_CACHE_HOME")
    return (Path(cache) if cache else Path.home() / ".cache") / "nathr"


def checksum_matches(path, expected):
    if not path.is_file():
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected


def download(path, checksum, url):
    curl = shutil.which("curl")
    wget = shutil.which("wget") if not curl else None
    if not (curl or wget):
        raise RuntimeError("curl or wget is required to download a checkpoint")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".nathr-download-", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        print(f"Downloading {path.name}", file=sys.stderr, flush=True)
        if curl:
            progress = ["--progress-bar"] if sys.stderr.isatty() else ["--silent", "--show-error"]
            command = [curl, "--fail", "--location", "--retry", "3", *progress, "--output", str(temporary), url]
        else:
            progress = ["--show-progress"] if sys.stderr.isatty() else []
            command = [wget, "--quiet", "--tries=3", *progress, "-O", str(temporary), url]
        result = subprocess.run(command)
        if result.returncode:
            raise RuntimeError(f"could not download {path.name} (exit {result.returncode})")
        if not checksum_matches(temporary, checksum):
            raise RuntimeError(f"SHA-256 mismatch for {path.name}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class ModelStore:
    def __init__(self, directory, manifest=None):
        self.directory = Path(directory or default_cache()).expanduser().absolute()
        if manifest is None:
            base = Path(__file__).resolve().parent
            manifest = base / "models/manifest.json"
            if not manifest.is_file():
                manifest = base.parent / "models/manifest.json"
        data = json.loads(manifest.read_text(encoding="utf-8"))
        self.default = data["default_checkpoint"]
        self.profiles = data["checkpoints"]
        self.sources = {source["repository"]: source for source in data["sources"]}
        self.names = {name: profile for profile in self.profiles
                      for name in [profile["name"], *profile.get("aliases", [])]}

    def files(self, profile):
        source = self.sources[profile["repository"]]
        if filename := profile.get("file"):
            directory = self.directory
            model = self.directory / filename
            hashes = [item for item in source["files"] if item["filename"] == filename]
        else:
            directory = self.directory / source["repository"].rsplit("/", 1)[-1]
            model = directory
            hashes = source["files"]
        if not hashes:
            raise RuntimeError(f"no files in the manifest for {profile['name']}")
        return model, directory, source, hashes

    def pull(self, name):
        if name not in self.names:
            raise RuntimeError(f"unknown checkpoint {name!r}. Run nathr models to see available names")
        model, directory, source, hashes = self.files(self.names[name])
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / ".download.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            base = f"https://huggingface.co/{source['repository']}/resolve/{source['revision']}"
            for item in hashes:
                filename = item["filename"]
                path = directory / filename
                if not checksum_matches(path, item["sha256"]):
                    download(path, item["sha256"], f"{base}/{filename}?download=true")
        return model

    def locate(self, selection):
        explicit = selection is not None
        selection = selection if explicit else self.default
        path = Path(selection).expanduser()
        if explicit and path.exists():
            return path.absolute(), path.resolve().name, None
        if selection in self.names:
            profile = self.names[selection]
            model, _, _, _ = self.files(profile)
            return model, profile["name"], selection
        if "/" in selection or path.suffix:
            raise RuntimeError(f"checkpoint path does not exist: {path}")
        raise RuntimeError(f"unknown checkpoint {selection!r}. Run nathr models to see available names")

    def resolve(self, selection):
        path, name, source = self.locate(selection)
        return self.pull(source) if source else path, name

    def list(self):
        print(f"{'MODEL':<14} {'CACHE':<11} CHECKPOINT")
        for profile in self.profiles:
            _, directory, _, hashes = self.files(profile)
            cached = all((directory / item["filename"]).is_file() for item in hashes)
            name = profile["name"]
            label = profile["description"]
            if name == self.default:
                label += " (default)"
            print(f"{name:<14} {'cached' if cached else 'available':<11} {label}")


def command(kind, args):
    parser = argparse.ArgumentParser(prog=f"nathr {kind}",
                                     description="Download a pinned checkpoint." if kind == "pull"
                                     else "List named checkpoints and their cache status.")
    if kind == "pull":
        parser.add_argument("model", help="checkpoint name from nathr models")
    parser.add_argument("--models-dir", help="cache directory (overrides NATHR_MODELS)")
    options = parser.parse_args(args)
    store = ModelStore(options.models_dir)
    if kind == "pull":
        print(store.pull(options.model))
    else:
        store.list()


if __name__ == "__main__":
    try:
        command("pull", sys.argv[1:])
    except (OSError, RuntimeError, ValueError) as error:
        sys.exit(f"nathr: {error}")
