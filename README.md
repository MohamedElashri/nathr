# Nathr

Nathr is an educational C++20 inference engine for small decoder-only language models. It runs GPT-2 and compatible Llama-style checkpoints, including SmolLM2-135M, offline on x86-64 Linux. [Example checkpoints and SHA-256 hashes](models/manifest.json)

The CLI can generate text, inspect checkpoints, tokenize prompts, and measure prefill and decode performance. The same engine is available as an embeddable C++ library. [Public API](include/nathr)

## Quick start

### 1. Set up Nathr

```sh
git clone https://github.com/MohamedElashri/nathr.git
cd nathr
./setup.sh
```

The script downloads and verifies the 92 MB GGUF checkpoint, builds Nathr, and runs the tests. [Prerequisites, other checkpoint options, and manual installation](#installation)

### 2. Generate text

Once setup prints `Ready`, run this from the `nathr` directory.

```sh
./build/release/nathr generate models/cache/SmolLM2-135M.Q4_0.gguf \
    "Once upon a time" --greedy --max-new-tokens 16
```

Nathr prints the continuation of the prompt.

```text
, the
people of the country were all very different. Some were farmers,
```

Replace the quoted prompt to try your own. [CLI options and examples](#cli-reference)

### Run GPT-2 small

From the repository root, run setup for GPT-2 small and pass the checkpoint directory to `generate`. The weights download is about 548 MB. [Pinned GPT-2 checkpoint](https://huggingface.co/openai-community/gpt2/tree/607a30d783dfa663caf39e06633721c8d4cfcd7e)

```sh
./setup.sh gpt2
./build/release/nathr generate models/cache/gpt2 "Once upon a time" --greedy --max-new-tokens 16
```

## How inference works

The inference engine guide covers the model equations and cache layout with worked examples. [Inference engine guide](docs/inference-engine.md)

```mermaid
flowchart LR
    Prompt --> Tokenizer
    Tokenizer -->|Token IDs| Transformer
    Transformer -->|Next-token scores| Sampler
    Sampler -->|Next input ID| Transformer
    Sampler -->|Sampled ID| Bytes["Token-to-byte lookup"]
    Bytes --> Text["UTF-8 output"]
```

1. The model loader checks tensor sizes and shapes and binds weights by name. Safetensors weights are read into a shared model buffer. GGUF files are memory-mapped, and quantized weights stay in their packed format. [Safetensors loader](src/formats/safetensors.cpp) · [GGUF loader](src/formats/gguf.cpp)
2. The byte-level BPE tokenizer looks up byte tokens and merges neighboring tokens in the order specified by the model's merge table. The resulting IDs index the embedding table. [Tokenizer](src/tokenizer/tokenizer.cpp)
3. The decoder runs causal attention and a feed-forward layer in each block. Llama-style models use RMSNorm, rotary position embeddings (RoPE), and SwiGLU. GPT-2 uses LayerNorm, learned position embeddings, and GELU. Both produce one score, or logit, per vocabulary token. [Llama decoder](src/models/llama/model.cpp) · [GPT-2 decoder](src/models/gpt2/model.cpp) · [Shared attention](src/backend/cpu/scalar.cpp)
4. **Prefill** processes the prompt in chunks and saves each layer's keys and values. During **decode**, the sampler takes the highest logit or draws a token using temperature, top-k, and top-p. Nathr emits the token and appends it when another generation step is needed. Attention reuses the saved keys and values for earlier tokens. Generation ends at the end-of-sequence token or the token limit. [Sessions](src/runtime/session.cpp) · [Sampling and generation](src/runtime/generation.cpp)

## CPU execution and memory

Nathr selects kernels by tensor encoding. The automatic backend uses AVX2 kernels when supported and falls back to scalar kernels where needed. During prefill, a weight row can serve several tokens in a small tile. Decoding a single token uses row-wise dot products. [CPU kernels](src/backend/cpu/kernels.cpp)

Sessions share model weights and own their key-value (KV) cache, working memory, and random state. KV cache size grows with the configured context capacity. FP16 cache storage uses half the memory of FP32. [Session allocation](src/runtime/session.cpp)

## Model compatibility

Nathr reads dimensions, attention head counts, context length, and boundary token IDs from the checkpoint. Checkpoints can be up to 1 GiB with vocabularies of up to 65,536 tokens. Session capacity is also subject to the KV cache memory limit. [Session limits](src/runtime/session.cpp)

- **Llama-style.** Single-file GGUF v3 or Safetensors with RMSNorm, SwiGLU, unscaled full-width RoPE, and bias-free projections. [Llama configuration](src/models/llama/config.cpp)
- **GPT-2.** Dense, single-file Safetensors with the GPT-2 attention and GELU configuration. [GPT-2 decoder](src/models/gpt2/model.cpp)

Text input supports byte-level BPE with either the ByteLevel pretokenizer alone or individual-digit splitting followed by ByteLevel. For GGUF, Nathr reads the vocabulary, merge ranks, and control tokens from metadata. You can pass `--tokenizer /path/to/tokenizer.json` to supply a compatible matching tokenizer. Models with other tokenizers can use `forward --tokens` or `generate --tokens --json` with IDs from an external tokenizer. [Tokenizer](src/tokenizer/tokenizer.cpp)

## CLI reference

Run these commands from the repository root. Use a GGUF file or a Safetensors directory as `MODEL`. The directory needs `config.json` and `model.safetensors`, plus `tokenizer.json` for text input and output. You can also write `--model PATH` and `--prompt TEXT` in place of the positional arguments.

```sh
nathr=./build/release/nathr
"$nathr" --help
"$nathr" --version
```

| Command | Options |
| --- | --- |
| `inspect MODEL` | Print checkpoint metadata and selected kernels as JSON. |
| `tokenize MODEL PROMPT` | Use `--stages` to see pretokenization or `--stdin` to read one JSON string per line. Use `--tokens IDS` in place of the prompt to decode IDs and `--skip-special` to omit special tokens. |
| `forward MODEL --tokens IDS` | Report the highest-scoring next token. Use `--cached` to exercise the KV cache, `--json` to include the final logits, or `--trace FILE` to save activations. `--logits-f32` streams one little-endian FP32 row per input token and is exclusive with `--json` and `--trace`. |
| `generate MODEL PROMPT` | Use `--tokens IDS` to supply IDs, `--bos` to prepend the beginning-of-sequence token, and `--max-new-tokens N` to limit output (default 32). Choose `--greedy` or tune sampling with `--temperature T`, `--top-k K`, `--top-p P`, and `--seed N`. Use `--json` for token IDs and generation metadata. |
| `bench MODEL` | Report prefill and decode timings as JSON. Set `--prompt-length N`, `--decode-length N`, `--warmups N`, and `--repetitions N` (defaults 128, 32, 2, and 7). |

Pass token IDs as comma-separated integers such as `1,7,2`. With a GGUF checkpoint, `tokenize` and `generate` accept `--tokenizer FILE` to supply a matching tokenizer.

`inspect`, `forward`, `generate`, and `bench` accept `--backend scalar|auto|avx2` and `--threads N` (1–64). The defaults are scalar execution and one thread. For `generate`, `bench`, and `forward --cached`, use `--context N` to set the session capacity, `--kv-cache f32|f16` for cache precision, and `--prefill-chunk N` (1–32) to set how many prompt tokens to process at once.

### Examples

In the same shell, use the default SmolLM2 checkpoint from `./setup.sh`. The forward command takes the IDs returned by `tokenize`.

```sh
model=models/cache/SmolLM2-135M.Q4_0.gguf
"$nathr" inspect "$model"
"$nathr" tokenize "$model" "Once upon a time"
"$nathr" forward "$model" --tokens 6403,1980,253,655 --cached
"$nathr" bench "$model" --prompt-length 16 --decode-length 4 \
    --warmups 1 --repetitions 2 --backend auto
```

After `./setup.sh gpt2`, try sampling from GPT-2 with a fixed seed.

```sh
"$nathr" generate models/cache/gpt2 "Once upon a time" \
    --temperature 0.8 --top-k 40 --top-p 0.95 --seed 7 --max-new-tokens 8
```

## Installation

### Prerequisites

Use x86-64 Linux with CMake 3.25+, Ninja, and GCC 12+ or Clang 16+. Cloning and downloading use Git, `curl`, and `sha256sum`. The setup script also uses Bash, Python 3 to read the checkpoint manifest, and `flock` from util-linux.

### Setup script options

Run `./setup.sh` or `./setup.sh gguf` for the quantized SmolLM2 checkpoint, `./setup.sh safetensors` for the original SmolLM2 weights, or `./setup.sh all` for both SmolLM2 formats. Use `./setup.sh gpt2` for GPT-2 small. Setup stores checkpoints under `models/cache`, reuses verified files, and prints a generation command for each checkpoint.

### 1. Build and test manually

From the repository root, run the build and tests. The tests use fixtures included in the repository.

```sh
cmake --preset release
cmake --build --preset release --parallel 2
ctest --preset release
./build/release/nathr --version
```

### 2. Download a model

Choose a SmolLM2-135M download below or use the GPT-2 setup command in Quick start. Each manual download block sets `model` to the path the CLI expects. Continue when every `sha256sum` check reports `OK`. [Supported downloads and SHA-256 hashes](models/manifest.json)

**GGUF (about 92 MB).** This single-file Q4_0 checkpoint includes the tokenizer and is the smaller download. [QuantFactory checkpoint](https://huggingface.co/QuantFactory/SmolLM2-135M-GGUF/blob/d948db3614be18259a175aafd7689a70f1cb4e2f/SmolLM2-135M.Q4_0.gguf)

```sh
model="$PWD/models/cache/SmolLM2-135M.Q4_0.gguf"
mkdir -p models/cache
curl --fail --location --retry 3 --output "$model" \
    'https://huggingface.co/QuantFactory/SmolLM2-135M-GGUF/resolve/d948db3614be18259a175aafd7689a70f1cb4e2f/SmolLM2-135M.Q4_0.gguf?download=true' &&
printf '%s  %s\n' '6f96e054a48f8fcdb5814abc1203ca12c0f9cce2791ca9a5153caad8c3b5479b' "$model" | sha256sum --check
```

**Safetensors (about 271 MB).** Download the original weights, configuration, and tokenizer into one directory. Pass that directory to Nathr. [Hugging Face checkpoint](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/tree/93efa2f097d58c2a74874c7e644dbc9b0cee75a2)

```sh
model="$PWD/models/cache/SmolLM2-135M"
(
    set -e
    mkdir -p "$model"
    cd "$model"
    base_url='https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/93efa2f097d58c2a74874c7e644dbc9b0cee75a2'
    for file in config.json model.safetensors tokenizer.json; do
        curl --fail --location --retry 3 --output "$file" "$base_url/$file?download=true"
    done
    sha256sum --check <<'SHA256'
1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843  config.json
80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1  model.safetensors
9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c  tokenizer.json
SHA256
)
```

If a download or checksum fails, rerun that block. To use an existing checkpoint, set `model` to its absolute path. Use the file path for GGUF and the directory path for Safetensors.

### 3. Test the downloaded model

Run this in the same shell as the download block, with `model` pointing to the verified checkpoint. [More CLI examples](#examples)

```sh
./build/release/nathr generate "$model" "Once upon a time" \
    --greedy --max-new-tokens 32 --context 512 --backend auto
```

### Install locally (optional)

The CLI also runs directly from the build directory. To install the CLI and C++ library under the checkout without administrator access, run the following.

```sh
cmake --install build/release --prefix "$PWD/build/install"
./build/install/bin/nathr --version
```

Use `./build/install/bin/nathr` in place of `./build/release/nathr` in the commands above. To use the library from another CMake project, set `CMAKE_PREFIX_PATH` to the absolute path of `build/install`, call `find_package(nathr CONFIG REQUIRED)`, and link your target against `nathr::nathr`. [C++ API](include/nathr)

The included consumer example checks that a separate C++ project can find and run the installed library. It exits successfully without printing output. [Consumer source](tests/consumer/main.cpp)

```sh
cmake -S tests/consumer -B build/consumer -G Ninja \
    -DCMAKE_PREFIX_PATH="$PWD/build/install"
cmake --build build/consumer
./build/consumer/nathr-consumer tests/fixtures/model
```
