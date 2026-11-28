#!/usr/bin/env bash
usage() {
    cat <<'HELP'
Install Nathr for the current user on x86-64 Linux.

Usage: bash install.sh [--no-model] [--archive FILE] [--update]

The default installs the latest GitHub Release and fetches its chat model.
Use --no-model to defer downloading the model until nathr run.
NATHR_VERSION selects a release tag.
NATHR_INSTALL_DIR and NATHR_BIN_DIR override the application and command directories.
For an offline installation, place FILE.sha256 beside FILE. --archive skips the model download.
--update upgrades an existing installation to NATHR_VERSION and skips model downloads.
HELP
}

main() (
    set -euo pipefail

    archive_file=
    prefetch=true
    updating=false
    while (( $# )); do
        case "$1" in
            -h|--help) usage; exit 0 ;;
            --no-model) prefetch=false ;;
            --update) updating=true; prefetch=false ;;
            --archive)
                if (( $# < 2 )) || [[ -n $archive_file || ! -f $2 || ! -f $2.sha256 ]]; then
                    printf 'Pass one archive with its adjacent .sha256 file.\n' >&2
                    exit 2
                fi
                archive_file=$(realpath -s -- "$2")
                prefetch=false
                shift
                ;;
            *) usage >&2; exit 2 ;;
        esac
        shift
    done

    if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
        printf 'The Nathr release requires x86-64 Linux.\n' >&2
        exit 1
    fi

    if [[ -z $archive_file ]] && ! command -v curl >/dev/null && ! command -v wget >/dev/null; then
        printf 'Nathr needs curl or wget to download the release.\n' >&2
        exit 1
    fi
    if ! python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
        printf 'Nathr requires Python 3.10 or newer.\n' >&2
        exit 1
    fi

    install_dir=$(realpath -m -- "${NATHR_INSTALL_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/nathr}")
    bin_dir=$(realpath -m -- "${NATHR_BIN_DIR:-$HOME/.local/bin}")
    releases="$install_dir/releases"
    launcher="$bin_dir/nathr"

    fetch() {
        if command -v curl >/dev/null; then
            curl -fsSL --retry 3 "$1" -o "$2"
        else
            wget -q --tries=3 -O "$2" "$1"
        fi
    }

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
    if "$updating" && [[ -z $version ]]; then
        printf 'Set NATHR_VERSION when using --update.\n' >&2
        exit 2
    fi

    mkdir -p -- "$releases" "$bin_dir"
    exec 9>"$install_dir/.install.lock"
    flock 9

    target=
    if [[ -e $launcher || -L $launcher ]]; then
        target=$(readlink -- "$launcher" || true)
        if [[ $target != "$releases"/nathr-*/bin/nathr ]]; then
            printf 'A different installation already uses %s. Set NATHR_BIN_DIR to a different directory.\n' "$launcher" >&2
            exit 1
        fi
    fi
    if "$updating"; then
        current_version=${target#"$releases/nathr-"}
        current_version=${current_version%/bin/nathr}
        if [[ ! $current_version =~ ^[0-9]+\.[0-9]+\.[0-9]+$ || ! -x $launcher ||
              ! -f ${target%/bin/nathr}/.archive-sha256 ]]; then
            printf 'Nathr update needs an existing installation managed by install.sh.\n' >&2
            exit 1
        fi
        if [[ $(printf '%s\n' "$version" "$current_version" | sort -V | tail -n 1) == "$current_version" ]]; then
            printf 'Nathr %s is up to date.\n' "$current_version"
            exit 0
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
        fetch "$release/$archive" "$work/$archive"
        fetch "$release/$archive.sha256" "$work/$archive.sha256"
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
        printf 'The Ubuntu 24.04 release needs libicu74 and libstdc++6.\n' >&2
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

    if "$prefetch" && "$destination/bin/nathr" help pull >/dev/null 2>&1; then
        model=$(python3 -I -c 'import json, sys; print(json.load(sys.stdin)["default_checkpoint"])' \
            < "$destination/share/nathr/models/manifest.json")
        "$destination/bin/nathr" pull "$model" >/dev/null
    fi

    link_stage=$(mktemp -d "$bin_dir/.nathr.XXXXXXXX")
    ln -s -- "$destination/bin/nathr" "$link_stage/nathr"
    mv -Tf -- "$link_stage/nathr" "$launcher"

    printf 'Installed Nathr %s at %s\n' "$actual_version" "$launcher"
    if [[ :$PATH: == *":$bin_dir:"* ]]; then
        printf 'Try nathr run\n'
    else
        printf 'Add the command directory to PATH, or run %q run\n' "$launcher"
        printf "  export PATH=%q:\"\$PATH\"\n" "$bin_dir"
    fi
)

main "$@"
