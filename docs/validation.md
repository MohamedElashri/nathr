# Validation and measurements

Run build and model-free tests with the Release preset. CTest includes the CLI and a relocated CMake consumer. The GCC and Clang Debug and Release matrix, separate ASan/UBSan and TSan builds, and developer fuzz smoke jobs run from the C++ workflow. The pinned-model workflow is manual and consumes a pre-populated cache on a self-hosted Linux x64 runner labelled `nathr-model-cache`. Hosted job completion is tracked separately from local reproduction. [Validation workflow](../.github/workflows/validation.yml)

## Model checks

Use the hashed Python 3.12 lock and the pinned sources recorded in Phase 0. The runner verifies the entire frozen published inventory, controlled and native weights, independent block fixtures, scalar and eligible ISA traces, FP16 cache rounding and a second deterministic tokenizer differential on the native tokenizer and both GGUF fingerprints. It also exercises the synthetic 8,192-token boundary and checks real FP16 and FP32 allocation sizes. [Implementation ledger and source pins](../plan.md)

```sh
python reference/phase5_verify.py --binary build/release/nathr --tests build/release/nathr_tests \
  --native "$HF_MODEL_DIR" --published "$PUBLISHED" --controlled "$CONTROLLED" --extended "$EXTENDED" \
  --untied "$UNTIED" --llama-probe "$LLAMA_PROBE" --llama-source "$LLAMA_SOURCE" \
  --ggml-bridge "$GGML_BRIDGE" --work fixtures/local/validation
```

Prepare the manual runner with a local wheel cache at `uv-cache/`, native `hf/`, published `models/`, `controlled/`, `extended/`, and the untied fixture. Set `NATHR_VALIDATION_CACHE` to that parent directory. Place the pinned llama.cpp archive extraction at `llama-source/` with its compiled `llama-probe` and `ggml-blocks.so` beside it. For release-tier timing, retain the verified `phase3-baseline` executable there and provide `NATHR_VALIDATION_CORES`, `NATHR_VALIDATION_SMT_CORES`, and `NATHR_VALIDATION_ONE_CORE` as explicit Linux CPU lists for the controlled runner.

The release tier also fills a real Q4_0 session to 8,192 tokens with FP16 KV. To reproduce it locally, run the same model with an 8,191-token prefill followed by one append. This prefill took about 79 minutes on the local Phase 5 machine.

```sh
build/release/nathr bench --model "$CONTROLLED/SmolLM2.Q4_0.gguf" \
  --backend auto --threads 4 --kv-cache f16 --prefill-chunk 32 --context 8192 \
  --prompt-length 8191 --decode-length 1 --warmups 0 --repetitions 1 --json
```

## Bounded defensive fuzzing

The developer fuzz build uses Clang libFuzzer, AddressSanitizer and UndefinedBehaviorSanitizer. The input corpora come from the checked-in tiny containers, tokenizer examples, and packed blocks. The optional fuzz targets are separate from the normal CLI and only exercise local libraries and fixtures.

```sh
cmake -S . -B build/fuzz -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_COMPILER=clang++ -DBUILD_TESTING=OFF -DNATHR_BUILD_FUZZERS=ON
cmake --build build/fuzz --parallel --target \
  nathr_fuzz_safetensors nathr_fuzz_gguf nathr_fuzz_tokenizer nathr_fuzz_quant
python3 reference/phase5_fuzz.py --build build/fuzz --work fixtures/local/fuzz --seed 58053 --seconds 45
```

Each run uses a fresh work directory for the logs, seed hashes, updated corpora and any findings. The release ledger records the actual duration, executed inputs and any local sanitizer options.

## Performance

`bench` reports model-load time and session-creation time separately. TTFT is the latency from the start of prefill to the first computed logits and excludes both setup steps, tokenization and sampling. Inter-token samples are measured between successive logits during teacher-forced decode. The sustained decode benchmark uses a fixed 128-token prompt and 128 decode tokens, with a 512-token FP32 cache. Peak RSS is process-wide. Encoded, mapped and owned weight bytes and session memory have separate fields.

```sh
python3 reference/phase5_bench.py --binary build/release/nathr --native "$HF_MODEL_DIR" \
  --controlled "$CONTROLLED" --cpus 8,10,12,14 --output fixtures/local/release-timing.json
```

The benchmark hashes the models before loading them, so its model-load measurements may use warm filesystem pages. The matched performance gates use the Phase 4 workload and unchanged limits. A supported, quiet host and stable paired batches are required for performance qualification. Shared-host results are descriptive. The standalone correctness tests have no timing gates. Quantization quality uses the fixed 112-target corpus and is recorded separately from timing and same-weight parity. [Matched benchmark methodology](../plan.md#phase-4-performance-engineering-v04)

The Phase 5 evidence includes the raw sustained samples, machine and binary metadata, fixed-corpus quality, and the failed matched Q8_0 repeatability retry. [Phase 5 results](../benchmarks/phase5.json)
