#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: ./setup.sh [--download-only] [--models-dir DIR] [CHECKPOINT]

Build and test Nathr, then run inference with the selected checkpoint.
  --download-only   Download verified model files without a C++ build.
  --models-dir DIR  Store models in DIR (default models/cache for builds,
                    or the user cache for --download-only).

CHECKPOINT defaults to gguf. Use all for both base SmolLM2 formats.
Run nathr models to see the checkpoint catalog.
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
        -h|--help) usage; exit 0 ;;
        -*) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
        *)
            if [[ -n $selection ]]; then
                usage >&2
                exit 2
            fi
            selection=$1
            ;;
    esac
    shift
done
selection=${selection:-gguf}

if [[ $(uname -s) != Linux ]] || [[ $(uname -m) != x86_64 && $(uname -m) != amd64 ]]; then
    printf 'Nathr requires x86-64 Linux.\n' >&2
    exit 1
fi

tools=(python3)
if ! "$download_only"; then
    tools+=(cmake ctest ninja flock)
fi
for tool in "${tools[@]}"; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'Missing required command: %s\n' "$tool" >&2
        exit 1
    fi
done

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -z $models_dir ]] && ! "$download_only"; then
    models_dir=models/cache
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
store=apps/model_store.py
if [[ -f model_store.py ]]; then
    store=model_store.py
fi
options=()
if [[ -n $models_dir ]]; then
    options+=(--models-dir "$models_dir")
fi
selections=("$selection")
if [[ $selection == all ]]; then
    selections=(gguf safetensors)
fi
models=()
for name in "${selections[@]}"; do
    model=$(python3 -I "$store" "$name" "${options[@]}")
    models+=("$model")
done

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
declare -A chat_models
for model in "${models[@]}"; do
    printf 'Checking inference with %s\n' "$model"
    metadata=$("$bin" inspect "$model" --json)
    if python3 -c 'import json, sys; sys.exit(not json.load(sys.stdin).get("chat"))' <<<"$metadata"; then
        chat_models["$model"]=1
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
    if [[ ${chat_models[$model]:-0} == 1 ]]; then
        printf ' %q' "$bin" run "$model"
        printf '\n'
        continue
    fi
    example_command "$model"
    printf ' %q' "${example[@]}"
    printf '\n'
done
