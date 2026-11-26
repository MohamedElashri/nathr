# Nathr

Nathr is a C++20 inference engine for base SmolLM2-135M on x86-64 Linux. It loads local Hugging Face Safetensors and GGUF v3 from the frozen SmolLM2 inventory. Weights retain their serialized encoding. Inference uses FP32 activations, FP64 reductions and a configurable FP32 or FP16 KV cache. The binary uses standard system libraries. [Implementation plan and qualification status](plan.md)

## Build

Use CMake 3.25+, Ninja, and GCC 12+ or Clang 16+. Configuration, compilation, and the model-free tests work offline.

```sh
cmake --preset release
cmake --build --preset release --parallel
ctest --preset release
```

The in-tree CMake target is `nathr::nathr`. Use `cmake --install build/release --prefix "$PWD/build/install"` to stage the CLI, library, headers, CMake package and dependency notices. An out-of-tree consumer can point `CMAKE_PREFIX_PATH` at that prefix, call `find_package(nathr CONFIG REQUIRED)`, and link `nathr::nathr`. The install smoke test relocates the prefix before building and running a consumer. Developer presets `dev`, `asan-ubsan` and `tsan` use separate build directories.

## Run the pinned base model

Download weights explicitly with the Python 3.12 standard-library fetcher. The full oracle environment is only needed for differential verification.

```sh
python3.12 reference/models.py fetch --repo HuggingFaceTB/SmolLM2-135M --output models/cache
MODEL="$PWD/models/cache/HuggingFaceTB/SmolLM2-135M/93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
BIN="$PWD/build/release/nathr"
"$BIN" inspect --model "$MODEL"
"$BIN" tokenize --model "$MODEL" --prompt 'Once upon a time'
"$BIN" forward --model "$MODEL" --tokens 1,2,3 --trace traces.safetensors
"$BIN" generate --model "$MODEL" --prompt 'Once upon a time' --greedy --max-new-tokens 32
"$BIN" bench --model "$MODEL" --context 512 --prompt-length 128 --decode-length 32 --json
```

Use `--tokens 1,2,3` in place of `--prompt` for generation from IDs. With `--tokens --json`, generation only needs model weights and configuration. Text commands use the pinned ByteLevel tokenizer pipeline. Encoding inserts no boundary token by default. Pass `--bos` explicitly, including for empty text. Stochastic generation uses temperature 1, no top-k or nucleus restriction, and seed 0 by default. Use `--seed`, `--temperature`, `--top-k`, and `--top-p` to change these settings. Sampling and execution are separate.

`forward` is uncached and capped at 128 tokens by default. Pass `--cached` for longer sequences. The default response includes the final greedy token ID. `--json` on `forward` includes all final-position logits. `--trace` produces bounded F32 Safetensors with tensors in `[position,feature]` order, including flattened head axes for Q, K and V. Use `--logits-f32` to stream one little-endian FP32 vocabulary row per input token to stdout, without a header. The streaming option is exclusive with `--trace` and `--json`. `bench` uses the Phase 0 deterministic teacher-forced workload. `--backend scalar` is the default. `auto` selects AVX2 when both the CPU and operating system support it. Use `--backend avx2` to require that path. F16 GEMV falls back to scalar when F16C is unavailable. `--threads N` selects one to 64 CPU threads per model, including the caller, and defaults to one.

Cached `forward`, `generate` and `bench` accept `--prefill-chunk 1..32` and `--kv-cache f32|f16`. The default chunk is 16 tokens, and a chunk of one selects repeated GEMV. FP32 is the default cache. FP16 rounds K and V on write and occupies 180 MiB for the full 8,192-token SmolLM2 cache. Both formats use the same FP64 attention reductions and FP32 activation boundaries.

## Frozen GGUF coverage

Pass an individual GGUF file as `--model`. Inspection is structural and reports the tensor table, storage encodings, metadata, and available linear kernels even for files with inspection-only encodings. Execution supports the 21 tensor encodings found in the frozen published files and controlled BF16. Legacy, K and I blocks use their original packed layout. AVX2 accelerates the floating, Q4_0 and Q8_0 GEMV kernels when available. The other packed types select observable scalar kernels. The [manifest](models/manifest.json) records the source revisions and encoding inventory. [Packed weight layouts](docs/quantization.md)

```sh
GGUF=/path/to/SmolLM2.Q4_0.gguf
"$BIN" inspect --model "$GGUF" --backend auto
"$BIN" generate --model "$GGUF" --prompt 'Once upon a time' --backend auto --greedy
```

The GGUF payload is mapped read-only and stays packed throughout execution. Keep the file contents intact while a model or session is alive. The two audited tokenizer fingerprints use their embedded vocabulary and merges with the canonical SmolLM2 pipeline. An alternate fingerprint requires `--tokenizer /path/to/tokenizer.json` pointing at the pinned HF tokenizer. Nathr verifies the sidecar hash, vocabulary order, and merge table. Integer-ID execution works without a sidecar.

## Library behavior

A model shares immutable weights and one bounded CPU worker pool. Create one session per sequence and pass `SessionOptions` to select chunk size and KV storage. Sessions own their KV cache, workspace, history and random generator. Independent sessions can run concurrently on the same model. Large projections use the shared pool one at a time. Small projections and callbacks execute on the calling thread. Session operations and callbacks must not reenter or mutate the same session concurrently.

`prefill` accepts a nonempty prompt on an empty session. `append` accepts more tokens on a ready session. Both validate the complete input before mutation. A failure during execution invalidates the session until `reset`. The last logits view is borrowed until the next mutating operation. `LogitsMode::none` skips the output projection, and `all_tokens` delivers each row to a synchronous callback. Reset restores the original seed. A generated EOS or final budget token is emitted without being appended. The JSON response reports that pending ID for continuation. A request may therefore emit one more token than it consumes.

The tokenizer rejects invalid UTF-8 input. Raw token decoding returns bytes. Display decoding buffers partial code points and emits replacement characters for invalid sequences or an incomplete final flush. The pinned BPE drops initial byte symbols absent from its vocabulary. [API, memory limits and numerical contracts](docs/architecture.md)

## Reference verification

Build the Release preset and install the [reference lock](reference/requirements.lock) using the Phase 0 commands in the [plan](plan.md). For the full native trace, floating storage and tokenizer comparisons, run

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python reference/phase1_verify.py \
  --binary build/release/nathr --model "$MODEL" --output fixtures/local/phase1 --floating-variants
```

`build/release/nathr_tests --long` exercises a tiny model through position 8191 in both cache precisions. `build/release/nathr_tests --real-allocations "$MODEL"` verifies warmed allocation counts and full-size FP32 and FP16 KV caches. The binary embeds no weights. The JSON parser and generated Unicode tables carry their licenses beside the source.

The Phase 2 verifier checks all published headers and compares the controlled execution slice with the independent same-weight oracle. Use the [Phase 0 conversion commands](plan.md) to prepare controlled GGUF files and the separate-output fixture.

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python reference/phase2_verify.py \
  --binary build/release/nathr --model "$MODEL" --published models/cache \
  --controlled models/cache/controlled --untied models/cache/controlled/untied.f32.gguf \
  --output fixtures/local/phase2
```

Run the matched scalar and AVX2 benchmark on a quiet physical core. The harness records the samples and enforces the repeatability limits from the implementation plan.

```sh
python reference/phase2_bench.py --binary build/release/nathr --probes build/release/nathr_tests \
  --controlled models/cache/controlled --cpu 12 --output benchmarks/phase2-local.json
```

The [local benchmark results](benchmarks/phase2.json) exceeded the 5% repeatability limit. Phase 2 performance qualification remains open.

The full coverage verifier compares every frozen file with an independent PyTorch model using FP64 intermediates and explicit FP32 activation boundaries. It also records drift from the original eager FP32 oracle and runs the pinned llama.cpp probe for an additional greedy cross-check. Descriptive corpus quality uses eager FP32 models and reports teacher-forced NLL, perplexity, KL divergence and top-five overlap. The verifier checks streamed logits on the three added controlled conversions.

The local run qualified all 50 published artifacts and all eight controlled conversions. Scalar and AVX2 traces matched the rounded oracle on all 483 tensors per artifact, in cached and uncached runs. [Coverage report](models/phase3-coverage.json)

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python reference/phase3_verify.py --binary build/release/nathr --native "$MODEL" \
  --published models/cache --controlled "$CONTROLLED" --extended "$EXTENDED" \
  --llama-probe "$LLAMA_PROBE" --work fixtures/local/phase3 --output fixtures/local/phase3.json
```

The three `$EXTENDED` controls come from the pinned F32 conversion. Use `llama-quantize INPUT_F32 OUTPUT RECIPE 3` with `Q5_1`, `Q4_K_M` and `IQ4_NL`, naming each output `SmolLM2.RECIPE.gguf`. The [implementation plan](plan.md) specifies the pinned tools and fixture commands.

The Phase 4 report covers threaded parity across all 58 artifacts, FP16 cache accuracy and memory, and matched prefill and decode runs with the Phase 3 executable and pinned llama.cpp. The local timing batches exceeded the 5% repeatability limit. [Phase 4 measurements](benchmarks/phase4.json)

The release validation commands cover the full pinned model set, differential tokenization, cache checks, sanitizer builds and bounded parser and codec fuzzing. The sustained benchmark reports TTFT through the first logits after session creation, token-to-token latency, and mapped and owned memory. [Validation and benchmark reproduction](docs/validation.md)
