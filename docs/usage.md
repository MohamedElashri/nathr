# Nathr usage guide

Start with the [README](../README.md#quick-start) to install Nathr and run a prompt.

## Installation options

Use wget to install, or add `--no-model` to defer the default model download until first use.

```sh
wget -qO- https://raw.githubusercontent.com/MohamedElashri/nathr/main/install.sh | bash
curl -fsSL https://raw.githubusercontent.com/MohamedElashri/nathr/main/install.sh | bash -s -- --no-model
```

For an offline install, download `install.sh`, `nathr-linux-x86_64.tar.gz`, and its `.sha256` file from the same [release](https://github.com/MohamedElashri/nathr/releases). Keep the files together and run `bash install.sh --archive nathr-linux-x86_64.tar.gz`. Use a cached checkpoint or pass a local checkpoint path.

## CLI input and output

`run` and `serve` select available CPU kernels by default. The lower-level commands default to scalar execution. Use `--backend auto` to select available kernels or `--backend avx2` to require AVX2 support.

`generate` and `infer` read a plain-text prompt through EOF with `--stdin`. Pass `--tokens 1,7,2` to supply raw token IDs. Use `--tokenizer FILE` to supply a matching tokenizer sidecar to a supported GGUF checkpoint. For named tensors, `inspect MODEL --summary` lists the outputs. `infer` returns the output name, shape, and flat row-major values. BERT text pairs use `--text-pair TEXT`, and raw BERT inputs can include `--token-type-ids IDS`. T5 accepts a decoder prefix with `--decoder-tokens IDS`.

Use `--sample` with `--temperature`, `--top-k`, `--top-p`, or `--seed` to control sampling. Repeat `--stop TEXT` or pass comma-separated `--stop-ids IDS` to end generation early. Add `--logprobs` to include chosen-token scores or `--top-logprobs N` for up to 20 alternatives. Scores use the model distribution before sampling filters.

### Chat history

Pass a message array with `--messages FILE` or pipe it into `--messages -`. An optional system message comes first. Alternate user and assistant messages and end with a user message. Qwen3 uses its non-thinking chat format.

```sh
model=$(nathr pull qwen3)
nathr generate "$model" --messages - --max-new-tokens 16 --backend auto <<'JSON'
[
  {"role":"user","content":"What is the capital of France?"},
  {"role":"assistant","content":"Paris."},
  {"role":"user","content":"What country is it in?"}
]
JSON
```

### Batch requests

`batch` reads one JSON request per line. A generation request emits `token` and `done` events. Supply `output` to request a named tensor. Send `{"cancel":2}` to cancel a pending request with ID 2. Close stdin to let pending requests finish.

```sh
model=$(nathr pull gguf)
nathr batch "$model" --backend auto <<'JSONL'
{"id":1,"prompt":"Once upon a time","max_new_tokens":8}
{"id":2,"tokens":[6403,1980,253,655],"max_new_tokens":4}
JSONL
```

## Use the OpenAI Python SDK

Start `nathr serve smollm2` in another terminal. Install the `openai` package in your Python environment with `python -m pip install openai`, then run this example.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")
reply = client.chat.completions.create(
    model="smollm2",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    max_tokens=16,
)
print(reply.choices[0].message.content)
```

## Serve multiple models

Stop the quick-start server and start this one. It serves chat with SmolLM2, embeddings with MiniLM, and reranking with TinyBERT. Repeat `--also` to add more models. Requests without a model ID go to the primary model.

```sh
nathr serve smollm2 --also minilm --also tinybert
```

```sh
curl -fsS http://127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"minilm","input":["The sky is blue.","The grass is green."]}'

curl -fsS http://127.0.0.1:8000/v1/rerank \
  -H 'Content-Type: application/json' \
  -d '{"model":"tinybert","query":"Capital of France?","documents":["Paris","Berlin"],"top_n":1}'
```

Both endpoints take up to 16 inputs or documents per request. Reranking returns original indices and relevance scores in descending order.

Use `--memory-budget-mib` for on-demand loading and eviction of idle workers. For example, `nathr serve smollm2 --also minilm --memory-budget-mib 512 --kv-budget-mib 128` starts with SmolLM2 and loads MiniLM when requested. The shared budget accounts for checkpoint files, estimated native workspace, KV capacity, and a 256 MiB adapter allowance when LoRA is configured. Process overhead such as tokenizers and HTTP buffers sits outside that estimate.

### HTTP endpoints

| Endpoint | Use |
| --- | --- |
| GET `/v1/models`, `/v1/models/{id}` | List all configured models or look up one ID. |
| GET `/model`, `/health`, `/metrics` | Inspect capabilities, worker health and memory, or Prometheus metrics. Add `?model=ID` to select one worker. |
| POST `/v1/chat/completions`, `/v1/completions` | Generate from messages or continue a prompt. |
| POST `/v1/responses` | Generate from text or messages. Send history in `input` on each turn. |
| POST `/v1/embeddings`, `/v1/rerank` | Embed inputs or rank documents for a query. |
| POST `/infer` | Use native generation fields or request a named tensor. |

Set `stream` to `true` for streamed generation. Chat Completions accepts `logprobs` set to `true` and `top_logprobs` from 0 to 20. For a full Qwen3 logits row from `/infer`, start the server with `--max-result-values 151936 --max-buffer-bytes 4194304`. Prometheus counters reset when a worker is evicted and loaded again.

### Structured outputs and function calls

Use `response_format` on Chat Completions or `text.format` on Responses. Both accept `json_object` and `json_schema` on supported decoder-only models with a byte-level tokenizer. The schema root must be an object. Nathr supports properties, required fields, array items, scalar types, and scalar `enum` and `const`. Limits include 32 properties per object and eight nesting levels. A reply that runs out of tokens finishes with `length` in Chat Completions or `incomplete` in Responses.

The supported SmolLM2 chat profile accepts one named function call per turn. Send `tools` and a named `tool_choice`. After a `tool_calls` finish, run the function in your application and send the assistant message back with a `tool` message containing the matching `tool_call_id`.

## LoRA adapters

Use the base checkpoint the adapter was trained on and a PEFT directory containing `adapter_config.json` and `adapter_model.safetensors`.

```sh
nathr run smollm2 "Explain why the sky looks blue." --lora ./adapter
nathr serve smollm2 --lora tuned=./adapter
```

Select `tuned` as the model in API requests. Repeat `--lora NAME=DIR` to register more adapters for the primary model. To manage adapters at runtime, start the server with `--lora-root ./adapters`. For an adapter in `./adapters/my-adapter`, POST `{"lora_name":"tuned","lora_path":"my-adapter"}` to `/v1/load_lora_adapter`. POST `{"lora_name":"tuned"}` to `/v1/unload_lora_adapter` after its requests finish. Keep the files available for worker reloads. The primary worker holds up to 16 adapters and 256 MiB of decoded adapter weights.

## Benchmark inference

```sh
model=$(nathr pull gguf)
nathr bench "$model" --prompt-length 16 --decode-length 4 --warmups 1 --repetitions 3 --backend auto
```

With `nathr serve` running, use `nathr bench --serve --workload mixed --concurrency 4 --requests 16` to measure streaming requests. Use the same `--seed` and restart the server when comparing settings such as `--no-prefix-cache`. `--speculative-tokens 4` enables prompt lookup for unconstrained greedy decoder requests. See [CPU measurements](inference-engine.md#cpu-measurements) for results and commands.

T5 and FLAN-T5 take plain text prompts on `/v1/completions`. Repeated requests with the same source tokens can reuse the source encoding. Pass `--no-prefix-cache` to disable reuse.

## Build and verify

With the [build prerequisites](../README.md#prerequisites) installed, build and test from the repository root.

```sh
cmake --preset release
cmake --build --preset release
ctest --preset release
```

Run `./build/release/nathr run /path/to/model.gguf` with an existing checkpoint. For a local installation, run `cmake --install build/release --prefix "$PWD/build/install"` and use `./build/install/bin/nathr`. CMake consumers can use `find_package(nathr CONFIG REQUIRED)` and link `nathr::nathr`.

### Check model compatibility

Download the checkpoints you want to verify, then run the checker from the repository root.

```sh
./build/release/nathr pull gguf --models-dir models/cache
./build/release/nathr pull qwen3 --models-dir models/cache
python3 tests/compatibility.py gguf qwen3
```

Use `all` after downloading every pinned checkpoint. Add `--measure --threads 2 --repetitions 3` to run the optional mixed-length GGUF benchmark.
