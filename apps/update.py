"""Check the latest Nathr release and update installer-managed launchers."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


LATEST_RELEASE = "https://api.github.com/repos/MohamedElashri/nathr/releases/latest"
VERSION = r"[0-9]{1,9}\.[0-9]{1,9}\.[0-9]{1,9}"
CHECK_INTERVAL = 24 * 60 * 60


def version_key(version):
    return tuple(map(int, version.split(".")))


def latest_version(timeout):
    if curl := shutil.which("curl"):
        command = [curl, "--disable", "--fail", "--silent", "--location", LATEST_RELEASE]
    elif wget := shutil.which("wget"):
        command = [wget, "--no-config", "--quiet", "--output-document=-", LATEST_RELEASE]
    else:
        raise RuntimeError("checking for updates requires curl or wget")
    try:
        result = subprocess.run(command, capture_output=True, timeout=timeout, check=True)
        release = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        raise RuntimeError("could not check GitHub for updates") from None
    tag = release.get("tag_name") if isinstance(release, dict) else None
    if not isinstance(tag, str) or not re.fullmatch("v" + VERSION, tag):
        raise RuntimeError("GitHub returned an unsupported release tag")
    return tag[1:]


def managed_install(executable, version):
    executable = Path(executable)
    release = executable.parent.parent
    if (executable == release / "bin/nathr" and release.name == f"nathr-{version}"
            and release.parent.name == "releases" and (release / ".archive-sha256").is_file()):
        return release.parent.parent
    return None


def managed_launcher(install, invoked):
    bin_dir = Path(os.environ.get("NATHR_BIN_DIR") or Path.home() / ".local/bin").expanduser()
    for path in (shutil.which(invoked), bin_dir / "nathr"):
        if path is None:
            continue
        launcher = Path(path).absolute()
        try:
            target = launcher.readlink()
        except OSError:
            continue
        if (launcher.name == "nathr" and target.parent.parent.parent == install / "releases"
                and target.parent.name == "bin" and target.name == "nathr"):
            match = re.fullmatch("nathr-(" + VERSION + ")", target.parent.parent.name)
            if match:
                return launcher, match[1]
    raise RuntimeError("could not find the managed launcher. Run nathr update through the installed nathr command")


def cached_version():
    cache_root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache").expanduser()
    path = cache_root / "nathr" / "release-check"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="ascii", errors="replace") as cache:
        fcntl.flock(cache, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cache.seek(0)
        saved = cache.read(64).strip()
        age = time.time() - os.fstat(cache.fileno()).st_mtime
        if 0 <= age < CHECK_INTERVAL and (saved == "-" or re.fullmatch(VERSION, saved)):
            return None if saved == "-" else saved
        try:
            latest = latest_version(timeout=1)
        except RuntimeError:
            latest = None
        cache.seek(0)
        cache.truncate()
        cache.write((latest or "-") + "\n")
        return latest


def startup_notice(executable, version):
    if os.environ.get("NATHR_NO_UPDATE_CHECK") == "1":
        return
    try:
        if managed_install(executable, version) is None:
            return
        latest = cached_version()
        if latest and version_key(latest) > version_key(version):
            print(f"Nathr {latest} is available. Run nathr update.", file=sys.stderr)
    except OSError:
        pass


def command(executable, directory, invoked, version, args):
    parser = argparse.ArgumentParser(prog="nathr update", description="Install the latest Nathr release.",
                                     allow_abbrev=False)
    parser.add_argument("--check", action="store_true", help="check for a newer release without installing")
    options = parser.parse_args(args)

    install = managed_install(executable, version)
    if not options.check:
        if install is None:
            raise RuntimeError("self-update requires an installation managed by install.sh. Source builds must be rebuilt")
        launcher, version = managed_launcher(install, invoked)

    latest = latest_version(timeout=10)
    if version_key(latest) <= version_key(version):
        print(f"Nathr {version} is up to date.")
        return
    if options.check:
        hint = " Run nathr update to install it." if install else ""
        print(f"Nathr {latest} is available. Current version is {version}.{hint}")
        return

    print(f"Updating Nathr {version} to {latest}.", flush=True)
    environment = dict(os.environ, NATHR_INSTALL_DIR=str(install), NATHR_BIN_DIR=str(launcher.parent),
                       NATHR_VERSION=latest)
    os.execvpe("bash", ["bash", str(Path(directory) / "install.sh"), "--update"], environment)
