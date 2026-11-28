# Nathr

**Run small language models on your CPU.**

Chat in the terminal, run a one-off prompt, or connect an OpenAI-compatible client to your local Nathr server. Nathr also runs models for embeddings and reranking.

[Latest release](https://github.com/MohamedElashri/nathr/releases/latest) · [User guide](docs/usage.md) · [How the engine works](docs/inference-engine.md)

## Quick start

The prebuilt release targets Ubuntu 24.04 on x86-64. It requires Bash, Python 3.10+, `ca-certificates`, `libicu74`, `libstdc++6`, `coreutils`, `tar`, `flock` from `util-linux`, and `curl` or `wget`.

Run these commands as your regular user. The installer verifies the release and downloads its default chat model.

```sh
curl -fsSL https://raw.githubusercontent.com/MohamedElashri/nathr/main/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"
nathr run
```

Type a message and press Enter. Use `/reset` to clear the conversation, or `/exit` or Ctrl+D to quit. Add the `export` line to your shell startup file to keep `nathr` on PATH.

Use `nathr update` to upgrade or `nathr update --check` to check for a release. If your version predates the update command, run the installer again. See the [installation options](docs/usage.md#installation-options).

## Run a model

```sh
nathr models
nathr run qwen3 "Explain what a checksum is in one sentence." --stats
nathr run minilm "The sky is blue."
```

Nathr downloads named models on first use. Run `nathr pull MODEL` to download one ahead of time, or pass a local checkpoint path to `nathr run`. See the [model and cache guide](docs/usage.md#models-and-cache) for available quantizations.

`--stats` shows latency and tokens per second. The MiniLM example returns a sentence embedding.

## OpenAI-compatible API

Start the server in one terminal.

```sh
nathr serve qwen3
```

Send a prompt from another terminal.

```sh
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":16}'
```

Read the reply in `choices[0].message.content`. In an OpenAI-compatible client, set the base URL to `http://127.0.0.1:8000/v1` and the model to `qwen3`. If the client requires an API key, enter `local`. See the [Python SDK example](docs/usage.md#use-the-openai-python-sdk).

Press Ctrl+C in the server terminal to stop it.

## Documentation

- [CLI reference and examples](docs/usage.md#cli-reference)
- [Serving multiple models](docs/usage.md#serve-multiple-models)
- [Inference statistics and tuning](docs/usage.md#inference-statistics)
- [Build from source and use the C++ library](docs/usage.md#build-and-verify)

## License

Nathr is licensed under the GNU General Public License, version 3 or later (`GPL-3.0-or-later`). See [LICENSE](LICENSE).

Bundled third-party components retain their license notices. Model checkpoints use the licenses set by their publishers.
