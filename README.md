# Nathr

Nathr runs base SmolLM2-135M on x86-64 Linux. The CLI and C++20 library load local Safetensors or single-file GGUF v3 weights and run inference offline.

## Build

Use CMake 3.25 or newer, Ninja, and GCC 12 or Clang 16 or newer.

```sh
cmake --preset release
cmake --build --preset release
ctest --preset release
```

## Run

Set `MODEL` to a local GGUF file or a directory containing `config.json`, `model.safetensors`, and `tokenizer.json`. [Supported checkpoints and SHA-256 hashes](models/manifest.json)

```sh
build/release/nathr generate --model "$MODEL" --prompt "Once upon a time" \
  --max-new-tokens 32 --greedy --backend auto --threads 4
```

Use `--kv-cache f16` to halve KV storage. The maximum context is 8,192 tokens. `--backend auto` selects AVX2 when available and otherwise uses scalar kernels.

Run `build/release/nathr --help` for the `inspect`, `tokenize`, `forward`, `generate`, and `bench` commands.

## C++ library

```sh
cmake --install build/release --prefix "$PWD/build/install"
```

Set `CMAKE_PREFIX_PATH` to the install prefix when configuring your application.

```cmake
find_package(nathr CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE nathr::nathr)
```

Share a `nathr::Model` across sequences and create one `nathr::Session` per sequence. Logits are borrowed until the next session mutation. [Public headers](include/nathr)
