# Nathr

Nathr runs small text Transformers locally. Chat, continue a prompt, and inspect model outputs from the terminal. The same runtime is available through an OpenAI-compatible local API and as a C++ library.

## Quick start

The prebuilt release targets Ubuntu 24.04 on x86-64. Have Bash, Python 3.10+, `ca-certificates`, `libicu74`, `libstdc++6`, `coreutils`, `tar`, and `flock` from `util-linux` available. Use `curl` or `wget` to fetch the installer. If you install the v0.6.0 release, have `curl` available for model downloads.

Run the installer as your regular user. It verifies the release and downloads the default chat model (about 270 MB). Have `$HOME/.local/bin` on PATH so you can run `nathr` in the same terminal. If needed, run `export PATH="$HOME/.local/bin:$PATH"` first.

```sh
curl -fsSL https://raw.githubusercontent.com/MohamedElashri/nathr/main/install.sh | bash
```

Or fetch the same script with `wget`.

```sh
wget -qO- https://raw.githubusercontent.com/MohamedElashri/nathr/main/install.sh | bash
```

Try a prompt from the same terminal.

```sh
nathr run smollm2 "What is the capital of France?"
```

The current source build also supports an interactive session with `nathr run`.

To update, rerun the install command. Nathr reuses cached checkpoints. Append `-s -- --no-model` after `bash` to defer downloading the model until `nathr run`. To install offline, download the archive and its checksum and pass the archive to `bash install.sh --archive FILE`.

## Run a model

```sh
nathr models
nathr run
nathr run minilm "The sky is blue."
nathr run ./model.gguf
```

Type one prompt per line in an interactive session. Chat models retain the conversation until you type `/reset`. Type `/exit` or press Ctrl+D to leave. A prompt on the command line runs once. Encoder checkpoints print a JSON tensor per prompt. Use `--output NAME` to select an output or `--max-new-tokens N` to limit a generated reply.

Named checkpoints download on first use and share a cache at `${XDG_CACHE_HOME:-$HOME/.cache}/nathr`. Use `NATHR_MODELS` or `--models-dir` to choose another cache. Use `nathr pull NAME` to prefetch a checkpoint and print its local path.

You can also pass a local GGUF file or a Safetensors directory containing `config.json` and `model.safetensors`. Built-in text input for Safetensors also uses `tokenizer.json`.

## CLI reference

```sh
nathr --help
nathr generate --help
nathr help serve
```

| Command | Use |
| --- | --- |
| `run [MODEL] [PROMPT]` | Start a terminal session or run one prompt. Uses `smollm2` by default. |
| `models` | List pinned checkpoint names and show which are cached. |
| `pull MODEL` | Download a pinned checkpoint and print its path. |
| `inspect MODEL --summary` | See checkpoint capabilities and output names. The default output is JSON. |
| `generate MODEL PROMPT` | Generate from text, `--stdin`, `--tokens IDS`, or `--messages FILE`. Add `--json` for token IDs and the finish reason. |
| `infer MODEL PROMPT --output NAME` | Return a named tensor as JSON. Accepts plain text from `--stdin` or raw IDs from `--tokens IDS`. |
| `tokenize MODEL PROMPT` | Return token IDs. Use `--tokens IDS` to decode or `--stdin` to read one JSON string per line. |
| `batch MODEL` | Run JSONL generation and tensor requests with one loaded checkpoint. |
| `serve [MODEL]` | Start the OpenAI-compatible API on loopback. |
| `bench MODEL` | Measure decoder-only prefill and decode. |
| `encode MODEL PROMPT` | Return BERT hidden states as JSON rows. |
| `forward MODEL --tokens IDS` | Inspect decoder-only logits and intermediate activations. |

Use `--model PATH` and `--prompt TEXT` for named arguments. Token lists use commas, as in `--tokens '1, 7, 2'`. Place `--` before a positional prompt that starts with `--`.

Execution commands default to one CPU thread and the scalar backend. Use `--backend auto` to select available CPU kernels, `--threads N` to set the thread count, or `--backend avx2` to require AVX2. The `run` and `serve` commands default to `auto`.

## CLI examples

### Generate and tokenize

Download the smaller SmolLM2 GGUF checkpoint for text continuation.

```sh
model=$(nathr pull gguf)
nathr inspect "$model" --summary
nathr tokenize "$model" "Once upon a time"
printf '%s' 'Once upon a time' |
    nathr generate "$model" --stdin --sample --seed 7 --top-k 40 --max-new-tokens 8 --backend auto
```

Use `--greedy` for highest-score selection or `--sample` with `--temperature`, `--top-k`, and `--top-p` to control sampling. Add repeatable `--stop TEXT` or comma-separated `--stop-ids IDS` to end generation early. `--stdin` in `generate` and `infer` reads a single plain-text prompt through EOF and preserves line breaks.

For a checkpoint with an external tokenizer, use `generate MODEL --tokens IDS --json` and decode the returned IDs with that tokenizer.

The same command runs T5 with a source prompt.

```sh
nathr run t5 "translate English to German: The sky is blue." --max-new-tokens 12
```

### Run chat in the terminal

```sh
chat_model=$(nathr pull smollm2)
nathr generate "$chat_model" --messages - --greedy --max-new-tokens 16 --backend auto <<'JSON'
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
embedding_model=$(nathr pull minilm)
nathr inspect "$embedding_model" --summary
nathr infer "$embedding_model" "The sky is blue." --output sentence_embedding --backend auto
```

`infer` returns `output`, `shape`, and flat row-major `values`. Use `--text-pair TEXT` for a BERT text pair, `--token-type-ids IDS` for segment IDs with raw tokens, or `--decoder-tokens IDS` for a T5 decoder prefix. Use `--tokenizer FILE` with a GGUF checkpoint that needs a matching external tokenizer.

### Batch requests

Each input line supplies an ID and one input. Generation lines emit `token` and `done` events. A line with `output` returns a `result` event with a named tensor.

```sh
nathr batch "$model" --backend auto --stats <<'JSONL'
{"id":1,"prompt":"Once upon a time","max_new_tokens":8}
{"id":2,"tokens":[6403,1980,253,655],"max_new_tokens":4,"greedy":false,"top_k":10,"seed":7}
JSONL
```

Text input also emits decoded fragments. Send `{"cancel":2}` to cancel a pending request. Close stdin to let the batch finish. Use `batch --help` for queue, context, and memory limits.

### Benchmark a decoder

```sh
nathr bench "$model" --prompt-length 16 --decode-length 4 --warmups 1 --repetitions 3 --backend auto
```

The JSON report includes prefill and decode throughput, inter-token timings, and KV memory.

## OpenAI-compatible API

Start the local server with the default `smollm2` model.

```sh
nathr serve
```

Set your client's base URL to `http://127.0.0.1:8000/v1` and its model to `smollm2`. The server runs locally and accepts requests without credentials. If the client requires an API key, use a placeholder such as `local`.

From another terminal, list models and send a chat request.

```sh
curl -fsS http://127.0.0.1:8000/v1/models
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"smollm2","messages":[{"role":"user","content":"What is the capital of France?"}],"temperature":0,"max_tokens":16}'
```

With the OpenAI Python SDK, use the same endpoint.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")
reply = client.chat.completions.create(
    model="smollm2",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    temperature=0,
    max_tokens=16,
)
print(reply.choices[0].message.content)
```

| Endpoint | Use |
| --- | --- |
| `GET /health` | Worker status and request counts |
| `GET /model` | Checkpoint capabilities, output shapes, and request limits |
| `GET /v1/models`, `GET /v1/models/{id}` | List and retrieve model IDs for OpenAI-compatible clients |
| `POST /infer` | A batch-style request without an ID, for generation or a named tensor |
| `POST /v1/completions` | Generate from one prompt or token array |
| `POST /v1/chat/completions` | Generate from text messages |
| `POST /v1/responses` | Generate from text or message history in the current source build |
| `POST /v1/embeddings` | Embed text or token arrays with a checkpoint that exposes `sentence_embedding` |

Set `stream=true` for incremental output. Responses requests are stateless and text-only. Send conversation history in `input` for each turn on a chat checkpoint. For embeddings, start `nathr serve minilm` and use the model ID `minilm`. Embedding requests accept up to 16 inputs. Each server loads one checkpoint. See `nathr serve --help` for ports and limits.

## Build from source

### Prerequisites

Build on x86-64 Linux with CMake 3.25+, Ninja, GCC 12+ or Clang 16+, and ICU 70+ development libraries (`libicu-dev` on Debian or Ubuntu). Source setup uses Git, Bash, Python 3.10+, `curl` or `wget`, and `flock`.

```sh
git clone https://github.com/MohamedElashri/nathr.git
cd nathr
./setup.sh instruct
export PATH="$PWD/build/release:$PATH"
nathr run --models-dir models/cache
```

To build with an existing checkpoint, run `cmake --preset release`, `cmake --build --preset release`, and `ctest --preset release` from the source tree.

To install into a local prefix, run `cmake --install build/release --prefix "$PWD/build/install"` and use `./build/install/bin/nathr`. CMake consumers can use `find_package(nathr CONFIG REQUIRED)` and link `nathr::nathr`.
### Check model compatibility

From the source tree, run the checker after downloading the named checkpoints. It checks file hashes, token IDs, tensor values, and generation where supported.

```sh
nathr pull gguf --models-dir models/cache
nathr pull smollm2 --models-dir models/cache
python3 tests/compatibility.py gguf instruct
python3 tests/compatibility.py gguf --measure --threads 2 --repetitions 3
```

Use `all` to check every setup checkpoint.
