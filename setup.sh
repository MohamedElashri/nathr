#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: ./setup.sh [gguf|safetensors|all]

Build and test Nathr, then verify generation with the selected SmolLM2-135M checkpoint.
  gguf         Download the Q4_0 GGUF checkpoint (default).
  safetensors  Download the original Safetensors checkpoint.
  all          Download and test both formats.
USAGE
}

if (( $# > 1 )); then
    usage >&2
    exit 2
fi
selection=${1-gguf}
case "$selection" in
    gguf|safetensors|all) ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

if [[ $(uname -s) != Linux ]] || [[ $(uname -m) != x86_64 && $(uname -m) != amd64 ]]; then
    printf 'Nathr requires x86-64 Linux.\n' >&2
    exit 1
fi

for tool in cmake ctest ninja curl sha256sum python3 flock; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'Missing required command: %s\n' "$tool" >&2
        exit 1
    fi
done

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mkdir -p build
exec 9>build/.setup.lock
flock 9

temporary_download=
trap '[[ -z $temporary_download ]] || rm -f -- "$temporary_download"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

checksum_matches() {
    local digest
    digest=$(sha256sum < "$1") || return 1
    [[ ${digest%% *} == "$2" ]]
}

download_file() {
    local filename=$1 expected=$2 url=$3
    if [[ -f $filename ]] && checksum_matches "$filename" "$expected"; then
        printf 'Verified cached file %s\n' "$filename"
        return
    fi

    printf 'Downloading %s\n' "$filename"
    temporary_download=$(mktemp -- "$filename.part.XXXXXXXX")
    curl --fail --location --retry 3 --output "$temporary_download" "$url"
    if ! checksum_matches "$temporary_download" "$expected"; then
        printf 'SHA-256 mismatch for %s\n' "$filename" >&2
        exit 1
    fi
    mv -fT -- "$temporary_download" "$filename"
    temporary_download=
}

download_checkpoint() {
    local repository=$1 destination=$2
    shift 2
    local downloads name checksum url
    downloads=$(python3 - "$repository" "$@" <<'PY'
import json
from pathlib import Path
import sys

repository, *names = sys.argv[1:]
manifest = json.loads(Path('models/manifest.json').read_text(encoding='utf-8'))
source = next((item for item in manifest['sources'] if item['repository'] == repository), None)
if source is None:
    sys.exit(f'Missing checkpoint repository in models/manifest.json: {repository}')
hashes = {item['filename']: item['sha256'] for item in source['files']}
for name in names:
    if name not in hashes:
        sys.exit(f'Missing checkpoint file in models/manifest.json: {name}')
    url = f"https://huggingface.co/{repository}/resolve/{source['revision']}/{name}?download=true"
    print(name, hashes[name], url, sep='\t')
PY
    )
    mkdir -p -- "$destination"
    while IFS=$'\t' read -r name checksum url; do
        download_file "$destination/$name" "$checksum" "$url"
    done <<< "$downloads"
}

cmake --preset release

models=()
if [[ $selection == gguf || $selection == all ]]; then
    gguf_name=SmolLM2-135M.Q4_0.gguf
    download_checkpoint QuantFactory/SmolLM2-135M-GGUF models/cache "$gguf_name"
    models+=("models/cache/$gguf_name")
fi
if [[ $selection == safetensors || $selection == all ]]; then
    safetensors_path=models/cache/SmolLM2-135M
    download_checkpoint HuggingFaceTB/SmolLM2-135M "$safetensors_path" \
        config.json model.safetensors tokenizer.json
    models+=("$safetensors_path")
fi

cmake --build --preset release --parallel 2
ctest --preset release

bin=./build/release/nathr
for model in "${models[@]}"; do
    printf 'Checking generation with %s\n' "$model"
    "$bin" generate "$model" 'Once upon a time' \
        --greedy --max-new-tokens 2 --context 64 --backend auto --json >/dev/null
done

printf '\nReady. Run from the repository root:\n'
for model in "${models[@]}"; do
    printf '  %s generate %s "Once upon a time" --greedy\n' "$bin" "$model"
done
