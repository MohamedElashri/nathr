# Nathr

Nathr is a C++20 inference engine for the base SmolLM2-135M language model on x86-64 Linux. It generates text continuations on the CPU using local Safetensors or single-file GGUF v3 checkpoints. The runtime works offline.

GGUF weights stay packed in memory. Nathr supports scalar and AVX2 execution, an FP32 or FP16 key-value cache, and contexts up to 8,192 tokens. The CLI can inspect checkpoints, tokenize text, and measure prefill and decode performance. [Supported checkpoints and SHA-256 hashes](models/manifest.json)

Applications can embed Nathr as a library. A loaded model shares its weights across sessions, and each sequence has its own cache and random state. [C++ API](include/nathr)

## Quick start

Build with CMake 3.25+, Ninja, and GCC 12+ or Clang 16+.

```sh
cmake --preset release
cmake --build --preset release
build/release/nathr generate "$MODEL" "Once upon a time"
```

Set `MODEL` to a GGUF file or a Safetensors directory containing `config.json`, `model.safetensors`, and `tokenizer.json`. Run `build/release/nathr --help` for options.

For C++ integration, install with `cmake --install build/release --prefix "$PWD/build/install"`. Set `CMAKE_PREFIX_PATH` to the install directory, call `find_package(nathr CONFIG REQUIRED)`, and link `nathr::nathr`.
