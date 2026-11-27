# Nathr

Nathr is a small CPU inference engine for text Transformers. It runs chat, text generation, embeddings, and reranking locally through a terminal CLI, an OpenAI-compatible HTTP API, or a C++ library.

## Quick start

The prebuilt release targets Ubuntu 24.04 on x86-64. You need Bash, Python 3.10+, `ca-certificates`, `libicu74`, `libstdc++6`, `coreutils`, `tar`, `flock` from `util-linux`, and `curl` or `wget`.

Run these commands as your regular user. The installer verifies the release and downloads the default chat model, SmolLM2-135M-Instruct (about 270 MB).

```sh
curl -fsSL https://raw.githubusercontent.com/MohamedElashri/nathr/main/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"
nathr run
```

Type a prompt and press Enter. Nathr keeps the conversation until you type `/reset`. Type `/exit` or press Ctrl+D to quit. Add the `export` line to your shell startup file to keep `nathr` on PATH in new terminals.

Rerun the installer to update. See the [installation options](docs/usage.md#installation-options) for wget, deferred downloads, and offline installation.

## Run a model

```sh
nathr models
nathr pull qwen3
nathr run qwen3 "What is the capital of France?"
```

`nathr models` lists checkpoint names and cache status. With `run` or `serve`, a named model downloads on first use. `nathr pull` downloads it ahead of time and prints its local path. Other chat models include `smollm2`, `tinyllama`, `qwen2.5`, and `qwen3-1.7b`. The model defaults to `smollm2`. Omit the prompt to start an interactive session.

The cache lives at `${XDG_CACHE_HOME:-$HOME/.cache}/nathr`. Set `NATHR_MODELS` or pass `--models-dir DIR` to use a different directory.

For a local checkpoint, pass the GGUF file or the directory containing `config.json` and `model.safetensors`. For text input, the Safetensors directory also needs the checkpoint’s `tokenizer.json`.

```sh
nathr inspect ./model.gguf --summary
nathr run ./model.gguf "Once upon a time"
```

Use `--threads N` to choose the CPU thread count and `--max-new-tokens N` to limit the reply. Nathr recognizes supported chat templates automatically. Use `--chat-format chatml`, `qwen3`, or `zephyr` to choose a matching format explicitly, or `--chat-format none` for text completion.

## OpenAI-compatible API

Start the server and leave it running in its terminal.

```sh
nathr serve smollm2
```

Send a chat request from a second terminal.

```sh
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"smollm2","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":16}'
```

Read the reply in `choices[0].message.content`. For an OpenAI-compatible client, set the base URL to `http://127.0.0.1:8000/v1` and the model to `smollm2`. The server listens on localhost and accepts requests without credentials. If your client requires a key, use `local`. GET `/v1/models` lists the available model IDs. Press Ctrl+C in the server terminal to stop it.

See the [serving guide](docs/usage.md#serve-multiple-models) for embeddings, reranking and memory limits, or the [Python SDK example](docs/usage.md#use-the-openai-python-sdk).

## CLI reference

Run `nathr --help` for the command list and `nathr COMMAND --help` for that command's options.

| Command | Use |
| --- | --- |
| `run [MODEL] [PROMPT]` | Chat, complete text, or return a named tensor in the terminal. |
| `models`, `pull MODEL` | List pinned checkpoints or download one. |
| `inspect MODEL --summary` | See checkpoint capabilities and output names. |
| `generate MODEL PROMPT` | Generate text with explicit sampling and stopping options. |
| `infer MODEL PROMPT --output NAME` | Return a named tensor as JSON. |
| `tokenize MODEL PROMPT` | Encode text or decode token IDs passed with `--tokens`. |
| `batch MODEL` | Process JSONL requests with one loaded checkpoint. |
| `serve [MODEL]` | Start the local API. |
| `bench MODEL`, `bench --serve` | Measure native inference or HTTP serving. |
| `encode MODEL PROMPT`, `forward MODEL --tokens IDS` | Inspect BERT hidden states or decoder logits and activations. |

### CLI examples

Use a quantized checkpoint to inspect token IDs and generate a short continuation. `--json` includes the generated token IDs and finish reason.

```sh
model=$(nathr pull gguf)
nathr tokenize "$model" "Once upon a time"
nathr generate "$model" "Once upon a time" --greedy --max-new-tokens 8 --backend auto --json
```

Send a plain-text instruction to FLAN-T5 Small.

```sh
nathr run flan-t5 "What is the capital of Italy?"
```

Request a sentence embedding from MiniLM.

```sh
model=$(nathr pull minilm)
nathr infer "$model" "The sky is blue." --output sentence_embedding --backend auto
```

See the [usage guide](docs/usage.md) for batch input, LoRA, chat history, log probabilities, and server options.

## Build from source

### Prerequisites

Use x86-64 Linux with Git, Bash, Python 3.10+, CMake 3.25+, Ninja, GCC 12+ or Clang 16+, and ICU 70+ development headers (`libicu-dev` on Debian and Ubuntu). The setup script also needs `flock` and `curl` or `wget`.

```sh
git clone https://github.com/MohamedElashri/nathr.git
cd nathr
./setup.sh smollm2
./build/release/nathr run --models-dir models/cache
```

`setup.sh` downloads the selected checkpoint, builds Nathr, and runs the tests. See [Build and verify](docs/usage.md#build-and-verify) for using an existing checkpoint or installing the C++ library.

The [inference engine guide](docs/inference-engine.md) explains the implementation, supported checkpoint formats, and [CPU measurements](docs/inference-engine.md#cpu-measurements).
