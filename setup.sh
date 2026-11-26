#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: ./setup.sh [--download-only] [--models-dir DIR] [CHECKPOINT]

Build and test Nathr, then run inference with the selected checkpoint.
  --download-only   Download verified model files without a C++ build.
  --models-dir DIR  Store models in DIR (default models/cache for builds,
                    or the user cache for --download-only).

Checkpoints:
  gguf         Download SmolLM2-135M Q4_0 GGUF (default).
  safetensors  Download the original SmolLM2-135M Safetensors checkpoint.
  all          Download and test both SmolLM2-135M formats.
  instruct     Download SmolLM2-135M-Instruct and its chat metadata.
  gpt2         Download the GPT-2 small Safetensors checkpoint.
  bert         Download the BERT base uncased Safetensors checkpoint.
  t5           Download the T5 small Safetensors checkpoint.
  minilm       Download the paraphrase-MiniLM-L3-v2 checkpoint.
  tinybert     Download the ms-marco-TinyBERT-L2-v2 checkpoint.
USAGE
}

download_only=false
models_dir=
selection=
while (( $# )); do
    case "$1" in
        --download-only) download_only=true ;;
        --models-dir)
            if (( $# < 2 )) || [[ -z $2 || $2 == --* ]]; then
                usage >&2
                exit 2
            fi
            shift
            models_dir=$1
            if [[ $models_dir != /* ]]; then
                models_dir="$PWD/$models_dir"
            fi
            ;;
        gguf|safetensors|all|instruct|gpt2|bert|t5|minilm|tinybert)
            if [[ -n $selection ]]; then
                usage >&2
                exit 2
            fi
            selection=$1
            ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option or checkpoint: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done
selection=${selection:-gguf}

if [[ $(uname -s) != Linux ]] || [[ $(uname -m) != x86_64 && $(uname -m) != amd64 ]]; then
    printf 'Nathr requires x86-64 Linux.\n' >&2
    exit 1
fi

tools=(curl sha256sum python3 flock)
if ! "$download_only"; then
    tools+=(cmake ctest ninja)
fi
for tool in "${tools[@]}"; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'Missing required command: %s\n' "$tool" >&2
        exit 1
    fi
done

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -z $models_dir ]]; then
    if "$download_only"; then
        models_dir="${XDG_CACHE_HOME:-$HOME/.cache}/nathr"
    else
        models_dir=models/cache
    fi
fi
if ! "$download_only"; then
    if [[ ! -f CMakePresets.json ]]; then
        printf 'Use --download-only with an installed copy of Nathr.\n' >&2
        exit 2
    fi
    mkdir -p build
    exec 9>build/.setup.lock
    flock 9
    cmake --preset release
fi
mkdir -p -- "$models_dir"
exec 8>"$models_dir/.download.lock"
flock 8

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
    curl --fail --location --retry 3 --progress-bar --output "$temporary_download" "$url"
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
        mkdir -p -- "$(dirname -- "$destination/$name")"
        download_file "$destination/$name" "$checksum" "$url"
    done <<< "$downloads"
}

models=()
if [[ $selection == gguf || $selection == all ]]; then
    gguf_name=SmolLM2-135M.Q4_0.gguf
    download_checkpoint QuantFactory/SmolLM2-135M-GGUF "$models_dir" "$gguf_name"
    models+=("$models_dir/$gguf_name")
fi
repository=
case "$selection" in
    safetensors|all) repository=HuggingFaceTB/SmolLM2-135M ;;
    instruct) repository=HuggingFaceTB/SmolLM2-135M-Instruct ;;
    gpt2) repository=openai-community/gpt2 ;;
    bert) repository=google-bert/bert-base-uncased ;;
    t5) repository=google-t5/t5-small ;;
    minilm) repository=sentence-transformers/paraphrase-MiniLM-L3-v2 ;;
    tinybert) repository=cross-encoder/ms-marco-TinyBERT-L2-v2 ;;
esac
if [[ -n $repository ]]; then
    destination="$models_dir/${repository##*/}"
    files=(config.json model.safetensors tokenizer.json)
    if [[ $selection == minilm ]]; then
        files+=(modules.json 1_Pooling/config.json sentence_bert_config.json)
    elif [[ $selection == instruct ]]; then
        files+=(tokenizer_config.json generation_config.json)
    fi
    download_checkpoint "$repository" "$destination" "${files[@]}"
    models+=("$destination")
fi
flock -u 8

if "$download_only"; then
    printf 'Model ready: %s\n' "${models[@]}"
    exit 0
fi

cmake --build --preset release --parallel 2
ctest --preset release

bin=./build/release/nathr
example_command() {
    local model=$1
    case "${model##*/}" in
        bert-base-uncased) example=("$bin" infer "$model" 'The sky is blue.' --output hidden_states) ;;
        t5-small) example=("$bin" generate "$model" 'translate English to German: The sky is blue.'
                           --greedy --max-new-tokens 12) ;;
        paraphrase-MiniLM-L3-v2) example=("$bin" infer "$model" 'The sky is blue.' --output sentence_embedding) ;;
        ms-marco-TinyBERT-L2-v2) example=("$bin" infer "$model" 'What color is the sky?' --text-pair
                                        'The sky is blue.' --output logits) ;;
        *) example=("$bin" generate "$model" 'Once upon a time' --greedy --max-new-tokens 16) ;;
    esac
    example+=(--backend auto)
}
for model in "${models[@]}"; do
    printf 'Checking inference with %s\n' "$model"
    if [[ $selection == instruct ]]; then
        printf '%s\n' '{"id":1,"messages":[{"role":"user","content":"What is the capital of France?"}],"max_new_tokens":8}' |
            "$bin" batch "$model" --backend auto |
            python3 -c 'import json, sys
events = list(map(json.loads, sys.stdin))
if not events or events[-1].get("reason") not in ("eos", "stop", "length"):
    sys.exit(f"Chat smoke test failed: {events}")'
        continue
    fi
    example_command "$model"
    "${example[@]}" >/dev/null
done

printf '\nReady. Run from the repository root:\n'
for model in "${models[@]}"; do
    if [[ $selection == instruct ]]; then
        printf ' %q' "$bin" serve "$model"
        printf '\n'
        continue
    fi
    example_command "$model"
    printf ' %q' "${example[@]}"
    printf '\n'
done
