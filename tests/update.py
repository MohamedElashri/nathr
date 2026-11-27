"""Exercise the packaged updater with deterministic release downloads."""

import hashlib
import io
import json
import os
from pathlib import Path
import pty
import select
import subprocess
import sys
import tarfile


def main(prefix, data, stage, model):
    executable = prefix / "bin/nathr"
    version = prefix.name.removeprefix("nathr-")
    major, minor, _ = map(int, version.split("."))
    newer = f"{major}.{minor + 1}.0"
    (prefix / ".archive-sha256").write_text("0" * 64 + "\n")
    launcher = stage / "custom bin/nathr"
    launcher.parent.mkdir()
    launcher.symlink_to(executable)

    downloads = stage / "release files"
    downloads.mkdir()
    metadata = downloads / "latest.json"
    calls = downloads / "calls"
    curl = launcher.with_name("curl")
    curl.write_text('''#!/usr/bin/env python3
import os
from pathlib import Path
import shutil
import sys
import time

root = Path(os.environ["NATHR_TEST_DOWNLOADS"])
url = next(arg for arg in sys.argv[1:] if arg.startswith("https://"))
with (root / "calls").open("a") as log:
    print(url, file=log)
if url == "https://api.github.com/repos/MohamedElashri/nathr/releases/latest":
    if (root / "slow").exists():
        time.sleep(30)
    sys.stdout.write((root / "latest.json").read_text())
elif url.startswith("https://github.com/MohamedElashri/nathr/releases/download/v"):
    shutil.copyfile(root / url.rsplit("/", 1)[-1], sys.argv[sys.argv.index("-o") + 1])
else:
    sys.exit(1)
''')
    curl.chmod(0o755)

    cache = stage / "cache"
    env = dict(os.environ, PATH=f"{launcher.parent}:{os.environ['PATH']}",
               XDG_CACHE_HOME=str(cache), NATHR_TEST_DOWNLOADS=str(downloads))
    for name in ("NATHR_INSTALL_DIR", "NATHR_BIN_DIR", "NATHR_VERSION", "NATHR_NO_UPDATE_CHECK"):
        env.pop(name, None)

    def set_latest(tag):
        metadata.write_text(json.dumps({"tag_name": tag}))

    def invoke(*args, **overrides):
        return subprocess.run(["nathr", *map(str, args)], stdin=subprocess.DEVNULL, capture_output=True,
                              text=True, env=dict(env, **overrides), timeout=10)

    def startup(**overrides):
        terminal, slave = pty.openpty()
        try:
            result = subprocess.run(["nathr", "run", model, "ab", "--max-new-tokens", "1"], stdin=slave,
                                    stdout=subprocess.PIPE, stderr=slave, env=dict(env, **overrides), timeout=10)
            notice = os.read(terminal, 4096).decode() if select.select([terminal], [], [], 0)[0] else ""
            assert result.returncode == 0, notice
            return notice
        finally:
            os.close(slave)
            os.close(terminal)

    set_latest(f"v{version}")
    result = invoke("update", "--check")
    assert result.returncode == 0 and "up to date" in result.stdout, result
    set_latest(f"v{newer}")
    result = invoke("update", "--check", NATHR_NO_UPDATE_CHECK="1")
    assert result.returncode == 0 and f"Nathr {newer} is available" in result.stdout, result
    assert launcher.readlink() == executable

    result = invoke("run", model, "ab", "--max-new-tokens", "1")
    assert result.returncode == 0, result
    assert len(calls.read_text().splitlines()) == 2
    assert f"Nathr {newer} is available" in startup()
    assert f"Nathr {newer} is available" in startup()
    assert "is available" not in startup(NATHR_NO_UPDATE_CHECK="1")
    assert len(calls.read_text().splitlines()) == 3

    os.utime(cache / "nathr/release-check", (0, 0))
    set_latest("invalid")
    assert "is available" not in startup()
    assert "is available" not in startup()
    assert len(calls.read_text().splitlines()) == 4
    result = invoke("update", "--check")
    assert result.returncode != 0 and "unsupported release tag" in result.stderr, result

    os.utime(cache / "nathr/release-check", (0, 0))
    set_latest(f"v{newer}")
    slow = downloads / "slow"
    slow.touch()
    assert "is available" not in startup()
    slow.unlink()
    assert "is available" not in startup()

    archive = downloads / "nathr-linux-x86_64.tar.gz"
    binary = f"#!/bin/sh\nprintf 'nathr {newer}\\n'\n".encode()
    with tarfile.open(archive, "w:gz") as package:
        entry = tarfile.TarInfo("nathr/bin/nathr")
        entry.mode = 0o755
        entry.size = len(binary)
        package.addfile(entry, io.BytesIO(binary))
    checksum = archive.with_name(archive.name + ".sha256")
    checksum.write_text("0" * 64 + "\n")
    set_latest(f"v{newer}")
    result = invoke("update")
    assert result.returncode != 0 and "SHA-256 verification" in result.stderr, result
    assert launcher.readlink() == executable

    checksum.write_text(hashlib.sha256(archive.read_bytes()).hexdigest() + "\n")
    result = invoke("update")
    assert result.returncode == 0 and f"Installed Nathr {newer}" in result.stdout, result
    expected = prefix.parent / f"nathr-{newer}/bin/nathr"
    assert launcher.readlink() == expected and executable.exists()
    assert invoke("--version").stdout.strip() == f"nathr {newer}"
    assert all(f"/download/v{newer}/" in url for url in calls.read_text().splitlines()
               if "/download/" in url)

    count = len(calls.read_text().splitlines())
    stale = subprocess.run(["bash", data / "install.sh", "--update"], capture_output=True, text=True, timeout=10,
                           env=dict(env, NATHR_INSTALL_DIR=str(prefix.parent.parent),
                                    NATHR_BIN_DIR=str(launcher.parent), NATHR_VERSION=version))
    assert stale.returncode == 0 and f"Nathr {newer} is up to date" in stale.stdout, stale
    assert launcher.readlink() == expected and len(calls.read_text().splitlines()) == count


if __name__ == "__main__":
    main(*map(Path, sys.argv[1:]))
