#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'HELP'
Install Nathr for the current user on x86-64 Linux.

Usage: bash install.sh [--archive FILE]

The default is the latest GitHub Release. NATHR_VERSION selects a release tag.
NATHR_INSTALL_DIR and NATHR_BIN_DIR override the application and command directories.
For an offline installation, place FILE.sha256 beside FILE.
HELP
}

archive_file=
case "${1-}" in
    -h|--help) usage; exit 0 ;;
    --archive)
        if (( $# != 2 )) || [[ ! -f $2 || ! -f $2.sha256 ]]; then
            printf 'Pass an archive with its adjacent .sha256 file.\n' >&2
            exit 2
        fi
        archive_file=$(realpath -s -- "$2")
        ;;
    '')
        if (( $# )); then usage >&2; exit 2; fi
        ;;
    *) usage >&2; exit 2 ;;
esac

if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
    printf 'The Nathr release requires x86-64 Linux.\n' >&2
    exit 1
fi

if [[ -z $archive_file ]] && ! command -v curl >/dev/null; then
    printf 'Install curl to download the Nathr release.\n' >&2
    exit 1
fi
if ! python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
    printf 'Nathr requires Python 3.10 or newer for the browser and model downloads.\n' >&2
    exit 1
fi

install_dir=$(realpath -m -- "${NATHR_INSTALL_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/nathr}")
bin_dir=$(realpath -m -- "${NATHR_BIN_DIR:-$HOME/.local/bin}")
releases="$install_dir/releases"
launcher="$bin_dir/nathr"

start_url=https://github.com/MohamedElashri/nathr/releases
version=${NATHR_VERSION:-}
version=${version#v}
if [[ -n $version ]]; then
    if [[ ! $version =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        printf 'NATHR_VERSION must be a release version such as 0.6.0.\n' >&2
        exit 2
    fi
    release="$start_url/download/v$version"
else
    release="$start_url/latest/download"
fi

mkdir -p -- "$releases" "$bin_dir"
exec 9>"$install_dir/.install.lock"
flock 9

if [[ -e $launcher || -L $launcher ]]; then
    target=$(readlink -- "$launcher" || true)
    if [[ $target != "$releases"/nathr-*/bin/nathr ]]; then
        printf 'A different installation already uses %s. Set NATHR_BIN_DIR to a different directory.\n' "$launcher" >&2
        exit 1
    fi
fi

work=$(mktemp -d "$install_dir/.install.XXXXXXXX")
link_stage=
trap 'rm -rf -- "$work"; if [[ -n $link_stage ]]; then rm -rf -- "$link_stage"; fi' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

archive=nathr-linux-x86_64.tar.gz
if [[ -n $archive_file ]]; then
    cp -- "$archive_file" "$work/$archive"
    cp -- "$archive_file.sha256" "$work/$archive.sha256"
else
    curl -fsSL --retry 3 "$release/$archive" -o "$work/$archive"
    curl -fsSL --retry 3 "$release/$archive.sha256" -o "$work/$archive.sha256"
fi

cd -- "$work"
read -r hash _ < "$archive.sha256"
if [[ ! $hash =~ ^[a-f0-9]{64}$ ]]; then
    printf 'Invalid Nathr release checksum.\n' >&2
    exit 1
fi
printf '%s  %s\n' "$hash" "$archive" | sha256sum --check --status || {
    printf 'The Nathr archive failed SHA-256 verification.\n' >&2
    exit 1
}

mkdir package
tar -xzf "$archive" --strip-components=1 -C package
if ! installed_version=$(package/bin/nathr --version); then
    printf 'On Ubuntu 24.04, install the runtime libraries with sudo apt install libicu74 libstdc++6.\n' >&2
    exit 1
fi
if [[ ! $installed_version =~ ^nathr\ ([0-9]+\.[0-9]+\.[0-9]+)$ ]]; then
    printf 'Invalid Nathr release version.\n' >&2
    exit 1
fi
actual_version=${BASH_REMATCH[1]}
if [[ -n $version && $version != "$actual_version" ]]; then
    printf 'Expected Nathr %s, got %s.\n' "$version" "$actual_version" >&2
    exit 1
fi
destination="$releases/nathr-$actual_version"
if [[ -e $destination || -L $destination ]]; then
    if [[ ! -f $destination/.archive-sha256 || $(<"$destination/.archive-sha256") != "$hash" ||
          ! -x $destination/bin/nathr ]]; then
        printf 'An existing package differs from the release at %s.\n' "$destination" >&2
        exit 1
    fi
else
    printf '%s\n' "$hash" > package/.archive-sha256
    mv -T -- package "$destination"
fi

link_stage=$(mktemp -d "$bin_dir/.nathr.XXXXXXXX")
ln -s -- "$destination/bin/nathr" "$link_stage/nathr"
mv -Tf -- "$link_stage/nathr" "$launcher"

printf 'Installed Nathr %s at %s\n' "$actual_version" "$launcher"
if [[ :$PATH: == *":$bin_dir:"* ]]; then
    printf 'Run nathr run to open the app.\n'
else
    printf 'Add the command directory to PATH, or run %q run.\n' "$launcher"
    printf "  export PATH=%q:\"\$PATH\"\n" "$bin_dir"
fi
