# Nathr

Nathr runs small text Transformers locally. Chat, continue a prompt, or inspect model outputs in the browser. The same runtime is available from the CLI and as a C++ library. [How the engine works](docs/inference-engine.md)

## Install the release

The prebuilt package targets Ubuntu 24.04 on x86-64. Install the runtime tools if needed.

```sh
sudo apt update
sudo apt install ca-certificates curl libicu74 libstdc++6 python3
```

Download and verify the latest archive, then extract it into a new directory. [GitHub Releases](https://github.com/MohamedElashri/nathr/releases/latest)

```sh
mkdir -p nathr
cd nathr
release=https://github.com/MohamedElashri/nathr/releases/latest/download
curl -fLO "$release/nathr-linux-x86_64.tar.gz"
curl -fLO "$release/nathr-linux-x86_64.tar.gz.sha256"
sha256sum -c nathr-linux-x86_64.tar.gz.sha256 && \
  tar -xzf nathr-linux-x86_64.tar.gz --strip-components=1
```

## Start the browser app

From the extracted directory, download the chat checkpoint (about 270 MB) and start Nathr.

```sh
nathr=./bin/nathr
get_model=./share/nathr/setup.sh
"$get_model" --download-only --models-dir models/cache instruct
chat_model=models/cache/SmolLM2-135M-Instruct
"$nathr" serve "$chat_model"
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) and enter a message. The available modes depend on the loaded checkpoint. The server stays in this terminal until you press Ctrl+C.

## Choose a checkpoint

Pass a GGUF file or a directory containing `config.json` and `model.safetensors` as `MODEL`. Text input for Safetensors also needs `tokenizer.json`. The downloader verifies pinned files before use. [Pinned revisions and SHA-256 hashes](models/manifest.json) · [Supported configurations](docs/inference-engine.md#checkpoint-support-and-limits)

| Downloader selection | Path under `models/cache/` | Use |
| --- | --- | --- |
| `gguf` | `SmolLM2-135M.Q4_0.gguf` | Text continuation |
| `safetensors` | `SmolLM2-135M` | Text continuation with the original weights |
| `instruct` | `SmolLM2-135M-Instruct` | Chat |
| `gpt2` | `gpt2` | Text continuation |
| `t5` | `t5-small` | Text-to-text generation |
| `bert` | `bert-base-uncased` | Hidden states and pooler output |
| `minilm` | `paraphrase-MiniLM-L3-v2` | Sentence embeddings |
| `tinybert` | `ms-marco-TinyBERT-L2-v2` | Text-pair scores |

Use `"$get_model" --download-only --models-dir models/cache CHECKPOINT` to fetch another model. Restart the server with its path to use it in the browser. The `all` selection fetches both base SmolLM2 formats.

## CLI reference

```sh
"$nathr" --help
"$nathr" generate --help
"$nathr" help serve
```

| Command | Use |
| --- | --- |
| `inspect MODEL --summary` | See checkpoint capabilities and output names. The default output is JSON. |
| `generate MODEL PROMPT` | Generate from text, `--stdin`, `--tokens IDS`, or `--messages FILE`. Add `--json` for token IDs and the finish reason. |
| `infer MODEL PROMPT --output NAME` | Return a named tensor as JSON. Accepts plain text from `--stdin` or raw IDs from `--tokens IDS`. |
| `tokenize MODEL PROMPT` | Return token IDs. Use `--tokens IDS` to decode or `--stdin` to read one JSON string per line. |
| `batch MODEL` | Run JSONL generation and tensor requests with one loaded checkpoint. |
| `serve MODEL` | Start the browser interface and HTTP API on loopback. |
| `bench MODEL` | Measure decoder-only prefill and decode. |
| `encode MODEL PROMPT` | Return BERT hidden states as JSON rows. |
| `forward MODEL --tokens IDS` | Inspect decoder-only logits and intermediate activations. |

Use `--model PATH` and `--prompt TEXT` for named arguments. Token lists use commas, as in `--tokens '1, 7, 2'`. Place `--` before a positional prompt that starts with `--`.

Execution commands default to one CPU thread and the scalar backend. Use `--backend auto` to select available CPU kernels, `--threads N` to set the thread count, or `--backend avx2` to require AVX2. The server defaults to `auto`.

## CLI examples

### Generate and tokenize

Download the smaller SmolLM2 GGUF checkpoint for text continuation.

```sh
"$get_model" --download-only --models-dir models/cache gguf
model=models/cache/SmolLM2-135M.Q4_0.gguf
"$nathr" inspect "$model" --summary
"$nathr" tokenize "$model" "Once upon a time"
printf '%s' 'Once upon a time' |
    "$nathr" generate "$model" --stdin --sample --seed 7 --top-k 40 --max-new-tokens 8 --backend auto
```

Use `--greedy` for highest-score selection or `--sample` with `--temperature`, `--top-k`, and `--top-p` to control sampling. Add repeatable `--stop TEXT` or comma-separated `--stop-ids IDS` to end generation early. `--stdin` in `generate` and `infer` reads a single plain-text prompt through EOF and preserves line breaks.

For a checkpoint with an external tokenizer, use `generate MODEL --tokens IDS --json` and decode the returned IDs with that tokenizer.

The same command runs T5 with a source prompt.

```sh
"$get_model" --download-only --models-dir models/cache t5
"$nathr" generate models/cache/t5-small "translate English to German: The sky is blue." \
  --greedy --max-new-tokens 12 --backend auto
```

### Run chat in the terminal

```sh
"$nathr" generate "$chat_model" --messages - --greedy --max-new-tokens 16 --backend auto <<'JSON'
[
  {"role":"user","content":"What is the capital of France?"},
  {"role":"assistant","content":"Paris."},
  {"role":"user","content":"What country is it in?"}
]
JSON
```

Use `--messages FILE` to read the same JSON array from disk. Messages can begin with a system message, then alternate user and assistant turns and end with a user message. The checkpoint must have a supported chat template.

### Request a named output

```sh
"$get_model" --download-only --models-dir models/cache minilm
embedding_model=models/cache/paraphrase-MiniLM-L3-v2
"$nathr" inspect "$embedding_model" --summary
"$nathr" infer "$embedding_model" "The sky is blue." --output sentence_embedding --backend auto
```

`infer` returns `output`, `shape`, and flat row-major `values`. Use `--text-pair TEXT` for a BERT text pair, `--token-type-ids IDS` for segment IDs with raw tokens, or `--decoder-tokens IDS` for a T5 decoder prefix. Use `--tokenizer FILE` with a GGUF checkpoint that needs a matching external tokenizer.

### Batch requests

Each input line supplies an ID and one input. Generation lines emit `token` and `done` events. A line with `output` returns a `result` event with a named tensor.

```sh
"$nathr" batch "$model" --backend auto --stats <<'JSONL'
{"id":1,"prompt":"Once upon a time","max_new_tokens":8}
{"id":2,"tokens":[6403,1980,253,655],"max_new_tokens":4,"greedy":false,"top_k":10,"seed":7}
JSONL
```

Text input also emits decoded fragments. Send `{"cancel":2}` to cancel a pending request. Close stdin to let the batch finish. Use `batch --help` for queue, context, and memory limits.

### Benchmark a decoder

```sh
"$nathr" bench "$model" --prompt-length 16 --decode-length 4 --warmups 1 --repetitions 3 --backend auto
```

The JSON report includes prefill and decode throughput, inter-token timings, and KV memory.

## Serve over HTTP

With the server running, send a request from another terminal.

```sh
curl -fsS http://127.0.0.1:8000/model
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is the capital of France?"}],"temperature":0,"max_tokens":16}'
```

| Endpoint | Use |
| --- | --- |
| `GET /health` | Worker status and request counts |
| `GET /model` | Checkpoint capabilities, output shapes, and request limits |
| `GET /v1/models`, `GET /v1/models/{id}` | List and retrieve model IDs for OpenAI-compatible clients |
| `POST /infer` | A batch-style request without an ID, for generation or a named tensor |
| `POST /v1/completions` | Generate from one prompt or token array |
| `POST /v1/chat/completions` | Generate from text messages |
| `POST /v1/embeddings` | Embed text or token arrays with a checkpoint that exposes `sentence_embedding` |

Set `"stream":true` on a generation request for server-sent events. Embeddings accept up to 16 inputs and return fixed-width vectors as floats or base64. The server loads one checkpoint per process. Use `serve --help` to change the port, scheduler settings, or HTTP limits.

## Build from source

### Prerequisites

Build on x86-64 Linux with CMake 3.25+, Ninja, GCC 12+ or Clang 16+, and ICU 70+ development libraries (`libicu-dev` on Debian or Ubuntu). The setup script uses Git, Bash, Python 3, `curl`, `sha256sum`, and `flock`. Serving requires Python 3.10+.

```sh
git clone https://github.com/MohamedElashri/nathr.git
cd nathr
./setup.sh instruct
nathr=./build/release/nathr
get_model=./setup.sh
"$nathr" serve models/cache/SmolLM2-135M-Instruct
```

To build with an existing checkpoint, run `cmake --preset release`, `cmake --build --preset release`, and `ctest --preset release` from the source tree.

To install into a local prefix, run `cmake --install build/release --prefix "$PWD/build/install"` and use `./build/install/bin/nathr`. CMake consumers can use `find_package(nathr CONFIG REQUIRED)` and link `nathr::nathr`. [C++ consumer example](tests/consumer)

### Check model compatibility

From the source tree, run the checker after downloading the named checkpoints. It checks file hashes, token IDs, tensor values, and generation where supported.

```sh
python3 tests/compatibility.py gguf instruct
python3 tests/compatibility.py gguf --measure --threads 2 --repetitions 3
```

Use `all` to check every setup checkpoint. [Reference values and measurement details](docs/inference-engine.md#run-the-checks-with-nathr)
