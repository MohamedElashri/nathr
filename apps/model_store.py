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
        self.names = {profile["name"]: (profile, None) for profile in self.profiles}
        for profile in self.profiles:
            profile.setdefault("default_variant", "original")
            profile.setdefault("variants", {"original": {}})
            self.names.update({name: (profile, variant) for name, variant in profile.get("aliases", {}).items()})

    def select(self, selection):
        name, separator, requested = selection.partition(":")
        if name not in self.names:
            raise RuntimeError(f"unknown checkpoint {name!r}. Run nathr models to see available names")
        profile, alias_variant = self.names[name]
        variant = requested.lower() if separator else alias_variant or profile["default_variant"]
        variants = profile["variants"]
        if variant not in variants:
            available = ", ".join(variants)
            raise RuntimeError(f"unknown variant {requested!r} for {profile['name']!r}. Available: {available}")
        model_id = profile["name"]
        if separator:
            model_id = f"{model_id}:{variant}"
        elif alias_variant is not None:
            model_id = name
        return profile, variant, model_id

    def files(self, profile, variant):
        artifact = profile["variants"][variant]
        source = self.sources[artifact.get("repository", profile["repository"])]
        if filename := artifact.get("file"):
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

    def pull(self, selection):
        profile, variant, _ = self.select(selection)
        model, directory, source, hashes = self.files(profile, variant)
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
        if selection.partition(":")[0] not in self.names and ("/" in selection or path.suffix):
            raise RuntimeError(f"checkpoint path does not exist: {path}")
        profile, variant, model_id = self.select(selection)
        model, _, _, _ = self.files(profile, variant)
        return model, model_id, selection

    def resolve(self, selection):
        path, name, source = self.locate(selection)
        return self.pull(source) if source else path, name

    def status(self, profile, variant):
        model, directory, _, hashes = self.files(profile, variant)
        cached = all((directory / item["filename"]).is_file() for item in hashes)
        return model, "cached" if cached else "available"

    def list(self, selection=None):
        if selection is not None:
            profile, _, _ = self.select(selection)
            print(f"{'VARIANT':<14} {'CACHE':<11} FILE")
            for variant in profile["variants"]:
                model, status = self.status(profile, variant)
                marker = " (default)" if variant == profile["default_variant"] else ""
                print(f"{variant:<14} {status:<11} {model.name}{marker}")
            return
        print(f"{'MODEL':<14} {'CACHE':<11} {'VARIANT':<10} CHECKPOINT")
        for profile in self.profiles:
            variant = profile["default_variant"]
            _, status = self.status(profile, variant)
            name = profile["name"]
            label = profile["description"]
            if name == self.default:
                label += " (default)"
            print(f"{name:<14} {status:<11} {variant:<10} {label}")


def command(kind, args):
    parser = argparse.ArgumentParser(prog=f"nathr {kind}",
                                     description="Download a pinned checkpoint." if kind == "pull"
                                     else "List named checkpoints and their cache status.")
    if kind == "pull":
        parser.add_argument("model", metavar="MODEL[:VARIANT]", help="checkpoint and optional quantization")
    else:
        parser.add_argument("model", nargs="?", metavar="MODEL", help="list available variants of one model")
    parser.add_argument("--models-dir", help="cache directory (overrides NATHR_MODELS)")
    options = parser.parse_args(args)
    store = ModelStore(options.models_dir)
    if kind == "pull":
        print(store.pull(options.model))
    else:
        store.list(options.model)


if __name__ == "__main__":
    try:
        command("pull", sys.argv[1:])
    except (OSError, RuntimeError, ValueError) as error:
        sys.exit(f"nathr: {error}")
