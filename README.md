# Nathr

Nathr runs small text Transformers locally. Chat, continue a prompt, and inspect model outputs from the terminal. The same runtime is available through an OpenAI-compatible local API and as a C++ library.

## Quick start

The prebuilt release targets Ubuntu 24.04 on x86-64. Have Bash, Python 3.10+, `ca-certificates`, `libicu74`, `libstdc++6`, `coreutils`, `tar`, and `flock` from `util-linux` available. Use `curl` or `wget` to fetch the installer. Use `curl` for model downloads if you pin the older `v0.6.0` release.

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

Run `nathr run` without a prompt to start an interactive session.

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
| `serve [MODEL]` | Serve one or more models through the local OpenAI-compatible API. |
| `bench MODEL` or `bench --serve` | Measure a decoder or benchmark a running server. |
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

Run Qwen3-0.6B with its official Q8_0 GGUF checkpoint (about 640 MB). The run command downloads it on first use.

```sh
nathr run qwen3 "What is the capital of France?"
nathr serve qwen3
```

Other pinned chat models include TinyLlama-1.1B and Qwen3-1.7B.

```sh
nathr run tinyllama "What is the capital of France?"
nathr run qwen3-1.7b "What is the capital of France?"
```

For a local checkpoint using a supported message format, pass `--chat-format chatml`, `qwen3`, or `zephyr` to select it explicitly. The default `auto` recognizes supported checkpoint templates. Use `--chat-format none` for plain text completion.

Qwen3 chat uses plain text with thinking mode disabled. Pass a JSON message array for explicit conversation history.

```sh
chat_model=$(nathr pull qwen3)
nathr generate "$chat_model" --messages - --greedy --max-new-tokens 16 --backend auto <<'JSON'
[
  {"role":"user","content":"What is the capital of France?"},
  {"role":"assistant","content":"Paris."},
  {"role":"user","content":"What country is it in?"}
]
JSON
```

Use `--messages FILE` to read the same JSON array from disk. Messages can begin with a system message, then alternate user and assistant turns and end with a user message. The checkpoint must have a supported chat template.

Add `--logprobs` to include sampled-token log probabilities in the JSON output, or `--top-logprobs N` to include up to 20 alternatives. Scores come from the raw model distribution before sampling filters.

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

### Apply a LoRA adapter

Use the checkpoint the adapter was trained on and a local PEFT directory containing `adapter_config.json` and `adapter_model.safetensors`. Nathr supports LoRA on Llama-style models, including SmolLM2 GGUF, and GPT-2.

```sh
nathr run smollm2 "Explain why the sky looks blue." --lora ./adapter
nathr serve smollm2 --lora tuned=./adapter
```

Select `"model":"tuned"` in API requests. The base model and adapter appear in `GET /v1/models`. Repeat `--lora NAME=DIR` to register more adapters for the primary checkpoint. Each request selects one adapter and shares the loaded base weights. Use `--lora DIR` with `generate`, `infer`, or `forward`, or register a name with `batch --lora NAME=DIR` and pass that name in the JSONL `lora` field.

### Benchmark a decoder

```sh
nathr bench "$model" --prompt-length 16 --decode-length 4 --warmups 1 --repetitions 3 --backend auto
```

The JSON report includes prefill and decode throughput, inter-token timings, and KV memory.

### Benchmark a running server

Start `nathr serve` in another terminal, then run a concurrent streaming workload.

```sh
nathr bench --serve --workload mixed --concurrency 4 --requests 16
nathr bench --serve --workload shared-prefix --prompt-length 512 --warmups 1 --seed 43
```

The workloads send synthetic token IDs. The report includes time to first token, approximate inter-token percentiles, queue time, throughput, and the worker's peak KV bytes since startup. Mixed workloads also report latency separately for short and long requests. Start fresh servers with the same seed to compare settings. Use `--no-prefix-cache` for a cache-disabled trial or `--max-queue-bypasses 2` to test admission under a tight KV budget.

Try `--speculative-tokens 4` with `nathr batch`, `nathr run`, or `nathr serve` to enable prompt lookup for greedy decoder requests without a JSON schema. Compare with speculation disabled using repeated-text prompts. Batch statistics and native `/infer` results report draft and acceptance counts.

See the [CPU measurements](docs/inference-engine.md#cpu-measurements) for latency, throughput, and peak RAM on four checkpoints.

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

Chat Completions accepts `logprobs=True` and `top_logprobs=0` through `20`. Scores are supported in streamed and non-streamed replies.

| Endpoint | Use |
| --- | --- |
| `GET /health` | Worker status, request counts, and KV memory |
| `GET /metrics` | Per-model Prometheus metrics for scheduler load, KV usage, cache hits, and latency |
| `GET /model` | Checkpoint capabilities, output shapes, and request limits |
| `GET /v1/models`, `GET /v1/models/{id}` | List and retrieve model IDs for OpenAI-compatible clients |
| `POST /infer` | A batch-style request without an ID, for generation or a named tensor |
| `POST /v1/completions` | Generate from one prompt or token array |
| `POST /v1/chat/completions` | Generate from text messages |
| `POST /v1/responses` | Generate from text or message history |
| `POST /v1/embeddings` | Embed text or token arrays with a checkpoint that exposes `sentence_embedding` |
| `POST /v1/rerank` | Rank documents with a text-pair checkpoint that returns one logit |

Set `stream=true` for incremental output. Responses requests are stateless and text-only. Send conversation history in `input` for each turn on a chat checkpoint.

For a Qwen3 logits row from `/infer`, start `nathr serve qwen3 --max-result-values 151936 --max-buffer-bytes 4194304`.

```sh
nathr serve smollm2 --also minilm
```

Use `smollm2` for generation and `minilm` for embedding up to 16 inputs per request. Repeat `--also` to load more checkpoints. The first model handles requests without a model ID. Each model has its own worker and KV budget. Use `/model?model=minilm` or `/health?model=minilm` to inspect one worker.

Set `--memory-budget-mib` to share a memory allowance across model workers.

```sh
nathr serve smollm2 --also minilm --memory-budget-mib 512 --kv-budget-mib 128
```

The primary model loads at startup. Additional models load on first inference or capability inspection at `/model`. Model discovery and health list configured models without loading them. Idle workers are evicted in least-recently-used order when another model needs room. Active requests keep their worker, so a request that cannot fit can retry after they drain.

The allowance accounts for checkpoint files, conservative native workspace estimates, KV capacity, and a 256 MiB adapter allowance on the primary worker when LoRA is configured. It excludes process overhead such as tokenizer data and HTTP buffers. `/health` reports the total allowance, reserved bytes, and each worker's residency and memory breakdown.

Scrape `http://127.0.0.1:8000/metrics` for all models or add `?model=smollm2` for one model. Queue and KV gauges reflect the last scheduler boundary. Latency histograms use seconds and measure worker activity. The server keeps a spare connection slot for scrapes when inference slots are full. Model counters restart when a worker is evicted and loaded again.

Encoder workers group similar token lengths into bounded batches. To rerank, add `--also tinybert`
and send a query and up to 16 documents.

```sh
curl -fsS http://127.0.0.1:8000/v1/rerank \
  -H 'Content-Type: application/json' \
  -d '{"model":"tinybert","query":"Capital of France?","documents":["Paris","Berlin"],"top_n":1}'
```

Results include original document indices and sigmoid relevance scores in descending order.

Add `--lora-root ./adapters` to enable loading and unloading without restarting. `POST /v1/load_lora_adapter` accepts `{"lora_name":"tuned","lora_path":"my-adapter"}`, where the path resolves beneath that directory. `POST /v1/unload_lora_adapter` accepts `{"lora_name":"tuned"}` once the adapter's active and queued requests finish. The primary worker holds up to 16 adapters and 256 MiB of decoded adapter weights. Keep adapter files available and unchanged so workers can reload them after eviction.

### Structured outputs and function calls

Use `response_format` with Chat Completions or `text.format` with Responses. Both accept `json_object` and `json_schema` on decoder-only checkpoints with a byte-level tokenizer. The inference worker masks tokens before sampling and stops when the object is complete.

The schema subset supports object properties and required fields, array items, scalar types, and scalar `enum` and `const`. The root must be an object. Objects generate only declared properties, with up to 32 properties per object and eight nesting levels. Numbers use bounded plain decimal forms. A reply that exhausts its token budget is marked with `finish_reason="length"` or a Responses status of `incomplete`.

On the supported SmolLM2 ChatML profile, Chat Completions also supports one named function call per turn. Supply `tools` and a named `tool_choice`. Use the returned `tool_calls` entry after a `tool_calls` finish, run the function in your application, then send the assistant message and a `tool` message with the matching `tool_call_id`. The function's parameters follow the same schema subset.

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
nathr pull qwen3 --models-dir models/cache
python3 tests/compatibility.py qwen3
python3 tests/compatibility.py gguf --measure --threads 2 --repetitions 3
```

Use `all` to check every setup checkpoint.
