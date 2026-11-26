# Implementation Plan: Nathr

Private implementation ledger for Nathr, a standalone C++ LLM inference engine. This file is the canonical plan and progress record. Keep it current as implementation proceeds. Do not create separate phase plans, progress reports or implementation ledgers.

Target: v1.0, SmolLM2-135M on x86-64 Linux.
Status: Phases 0, 1 and 3 are complete. The Phase 2 GGUF execution slice is validated, with performance qualification deferred at the user’s direction. Phase 4 implementation and correctness validation are complete, with local performance qualification blocked on repeatability. Phase 5 hardening, packaging and local correctness validation are complete. Release qualification remains blocked on controlled timing and hosted verification.

Project identifiers: repository and executable `nathr`, C++ namespace `nathr`, library `libnathr`, CMake target `nathr::nathr`.

## Goal

Build a small, independently implemented, CPU-first LLM inference engine in C++20 that loads real pretrained SmolLM2-135M weights, tokenizes text, executes the transformer, maintains a KV cache and generates tokens. Support the original floating-point checkpoint and all distinct weight encodings required by a pinned inventory of published base-model GGUF quantizations. The runtime must not link to an existing ML, tensor, tokenizer or inference framework.

Nathr is both a useful standalone executable/library and a research platform for studying numerical correctness, memory traffic, quantization, CPU kernels and inference performance. A portable scalar implementation is the permanent internal reference within the CPU backend. PyTorch is an independent external numerical oracle, not a runtime dependency. The initial hardware target is x86-64 CPU only; architecture and state ownership must permit future non-CPU backends without rewriting transformer logic.

Success means that a fresh checkout can build offline (after obtaining the compiler/toolchain), run synthetic tests, and, with explicitly downloaded local model artifacts, perform validated generation and reproducible benchmarking. A user of the C++ binary does not need Python, an NVIDIA GPU, internet access or ML libraries.

## Scope

In:

- C++20 library and CLI, CMake/CTest, Linux x86-64, GCC and Clang.
- Minimal tensor descriptors, immutable weight storage, aligned buffers, session workspaces, checked arithmetic and errors.
- Local Hugging Face Safetensors checkpoint loading and GGUF v3 loading; reject unsupported variants explicitly.
- SmolLM2-135M base model, Llama-family decoder architecture, tied embeddings, GQA, RMSNorm, RoPE, SwiGLU and causal attention.
- Exact model-specific byte-level BPE tokenization and decoding, including special tokens and Unicode pre-tokenization.
- FP32 reference activations and cache with FP64 intermediates within sensitive operations. F32, F16 and BF16 weights and the conventional, K-quant and I-quant encodings required by the frozen compatibility manifest.
- Independent block decoding and bounded-memory quantized computation; mixed tensor types in one checkpoint.
- Teacher-forced forward execution, prefill, decode, contiguous FP32 and FP16 KV cache, sampling, deterministic greedy inference, token streaming and context limits.
- One CPU backend with a permanent scalar execution policy, x86 SIMD dispatch and persistent CPU thread pool; distinct GEMV/decode and GEMM/prefill execution paths.
- Python-only reference environment, numerical fixtures, independent GGUF reference decoding, perplexity and performance measurement.
- Model inspection, diagnostics, documented C++ API, packaging and repeatable validation.

Out:

- Training, autograd, fine-tuning, LoRA, general-purpose tensor framework or graph compiler.
- GPU/ARM kernels, distributed inference, HTTP serving, continuous batching, paged attention, prefix caching, speculative decoding and multimodality.
- Reduced-precision persistent activations and INT8 KV cache. Transient INT8 activation quantization for selected integer dot-product kernels is an explicit late-phase optimization, with separate numerical and quality gates; FP32 activations remain the reference. FP16 KV storage is in scope.
- Quantizer/calibration implementation, GPTQ/AWQ and other serialization ecosystems, arbitrary Llama variants, arbitrary tokenizers and arbitrary GGUF architectures.
- Chat templates and Instruct checkpoint compatibility as a v1.0 requirement. The architecture should accommodate the official Instruct checkpoint later without a separate model implementation.
- Downloading weights inside the executable, telemetry, network services, package manager integration and guarantees about performance on untested hardware.

## Constraints

- Preserve existing public behavior unless intentionally changed. Inspect repository instructions, status, build conventions and existing user work before editing.
- Keep changes small and reviewable. Use conventional commits when requested; do not commit, push or overwrite unrelated work without authorization.
- Use standard C++20 and compiler intrinsics where appropriate. Standard threading, filesystem and streams are sufficient for the baseline. Permit a small POSIX wrapper for read-only mmap on Linux, with buffered loading as an alternative.
- Permit one vendored, version-pinned general-purpose JSON parser (`nlohmann/json`) for model metadata and tokenizer files. Record its license and version. Do not write a general JSON parser. No FetchContent or network downloads during ordinary configure/build/test.
- No ggml, llama.cpp, BLAS, Eigen, ONNX Runtime, PyTorch, tokenizers or similar libraries in the runtime. Python packages, reference binaries and developer tooling are isolated under reference/test workflows. `Threads::Threads` and normal toolchain/system libraries are permitted. Verify actual link dependencies.
- All weight math, block dequantization, transformer execution, scheduling within a session, tokenization and sampling are implemented in this repository. Specifications and independent reference implementations may be consulted; do not silently copy third-party code or tables without checking provenance, license and notices. Encoding-defined lookup tables may be derived or vendored with explicit provenance if independently validated.
- Correctness before optimization. Never replace a failing comparison with a looser tolerance simply to make CI pass. Keep scalar reference code readable and independent of SIMD implementation.
- Do not hardcode SmolLM2 dimensions inside generic kernels. Validate model metadata against the supported architecture contract. Reject incompatible configurations and unknown tensor encodings rather than guessing.
- No undefined behavior from unaligned loads, aliasing, integer overflow, invalid shift, invalid object lifetime or signed SIMD saturation. Do not use raw reinterpret casts of untrusted file data into native structs.
- No global `-march=native`, `-ffast-math` or implicit machine-dependent instruction requirements in distributable binaries. Build specialized kernels separately; scalar is always available. FMA contraction policy must be explicit in reference builds.
- Keep weights immutable and shared across sessions. Cache, scratch, RNG and positions belong to individual sessions. Sessions are not concurrently mutable; independent sessions may execute against a shared model under documented backend thread-safety rules.
- Runtime must not silently expand a full quantized checkpoint to FP32. A debugging/reference conversion is allowed outside the production path. Report resident/mapped weight bytes separately from cache and scratch.
- No per-token heap allocation in the compute hot path after session creation and documented warmup (tracing, CLI output, caller-owned strings and optional debug mode excluded). Bound all context, workspace and file-derived allocations.
- Do not claim compatibility, numerical parity or speedup without recorded evidence. A skipped ISA path or unavailable reference tool is not a pass.

## Current State

- The starting workspace contained only `plan.md`. `git status --short --branch` returned “not a git repository”, and the read-only `.git`, `.agents` and `.codex` mounts were empty. No governing AGENTS.md or existing build targets were exposed.
- Phase 0 added the pinned [model manifest](models/manifest.json), isolated reference tools and lock, and small fixtures. Phase 1 adds the C++20 `nathr::nathr` library, CLI, scalar CPU runtime, and offline tests.
- Verified on AMD Ryzen 7 7735HS, Zorin OS 18.1, Linux 7.0.0-43-generic, little-endian x86-64. `lscpu -p=CPU,CORE,SOCKET` showed eight distinct cores, two logical CPUs per core, and affinity 0–15. L1d/L1i are 32 KiB per core, L2 is 512 KiB per core and shared L3 is 16 MiB. `free -h` showed 59 GiB RAM, and `df -h . /tmp` showed 250 GiB free at the start of the audit.
- GCC 13.3.0 and Homebrew Clang 23.1.6 both passed a disposable `cmake -S . -B <compiler> -G Ninja -DCMAKE_CXX_COMPILER=<compiler>`, build and CTest probe with C++20, `std::span`, `std::jthread`, `-march=x86-64` and `-ffp-contract=off`. CMake 3.28.3 and Ninja 1.11.1 were installed, along with clang-format and clang-tidy. Direct CPUID and XGETBV measured AVX2, FMA, F16C and OSXSAVE with XCR0=0x207. AVX-512F was absent. `perf stat -e task-clock true` was denied. GCC ASan/UBSan ran the probe with `ASAN_OPTIONS=detect_leaks=0`. The local TSan startup limitation is recorded below.
- Template provenance is retained from `MohamedElashri/agentic/templates/plan.md` (blob `7de789d139ef41c13e71b564c8b0c03d8bf2ccd2`).

### Model contract and source artifacts

Use `HuggingFaceTB/SmolLM2-135M` as the canonical model, not a similarly named derivative. Phase 0 confirmed the fields below against the pinned model configuration and tensor headers. The width of 64 is derived from H/Nq.

| Field | Verified value |
| --- | --- |
| Architecture | LlamaForCausalLM |
| Layers | 30 |
| Hidden dimension H | 576 |
| FFN dimension I | 1536 |
| Query heads Nq | 9 |
| KV heads Nkv | 3 |
| Head dimension D | 64 |
| Vocabulary V | 49152 |
| Maximum context | 8192 |
| RoPE theta | 100000 |
| RMSNorm epsilon | 1e-5 |
| Activation | SiLU |
| Embeddings | Tied |
| BOS / EOS IDs | 0 / 0 |
| Attention bias / RoPE scaling | false / none |
| HF RoPE interleaving | false |
| Native checkpoint dtype | BF16 for all stored tensors |

Sources (pin immutable revisions; record licenses and SHA-256 digests before full tests):

- https://huggingface.co/HuggingFaceTB/SmolLM2-135M
- https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/main/config.json
- https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/main/tokenizer.json
- https://huggingface.co/QuantFactory/SmolLM2-135M-GGUF
- https://huggingface.co/mradermacher/SmolLM2-135M-GGUF
- https://huggingface.co/mradermacher/SmolLM2-135M-i1-GGUF
- https://github.com/huggingface/safetensors (format specification)
- https://github.com/ggml-org/ggml/blob/master/docs/gguf.md (GGUF specification)
- https://github.com/ggml-org/llama.cpp (reference converter, quant layouts, external benchmark)
- https://github.com/huggingface/transformers (Llama reference and tokenization)

The model revisions, file hashes and tool pins are recorded in Phase 0 and the manifest.

### Quantization compatibility contract

Maintain two separate artifact suites: published artifacts for interoperability and controlled artifacts for experiments. Generate controlled F32/F16 (and BF16 where supported), Q8_0 and Q4_0 artifacts first from the pinned canonical checkpoint using a pinned external llama.cpp converter/quantizer; extend to other feasible recipes as required for diagnostics. Record converter flags, calibration corpus and importance matrix hash if used, output checksums and tokenizer metadata. Do not imply all recipes can be generated for this shape. The controlled suite supplements, never replaces, the published inventory. Neither converter nor quantizer is a Nathr runtime dependency.

"All quantizations" means all distinct recipes/variants published for the canonical base checkpoint in the three named GGUF repositories at the inventory freeze, plus official Safetensors. Do not silently restrict the inventory to convenient encodings. Capture the complete file listing at the pinned revision, including any split artifacts. Identical files may share evidence only if their SHA-256 matches; different quantization files must each pass the release integration gate. New upstream formats published after the freeze are future work unless the manifest is explicitly revised.

The inventory must be machine-readable (`models/manifest.json`) and include source repo/revision, exact filename(s), file size, SHA-256, license/provenance, recipe label, per-tensor dtype histogram (counts, logical parameters and encoded bytes), shape inventory, effective bits per unique stored weight, architecture/tokenizer metadata fingerprint, required encoding IDs, verification status and reason for any exclusion. Large downloaded files and full reference fixtures must not be committed. Download scripts must be explicit, idempotent and hash-verifying; no implicit downloads in tests.

| Family | Anticipated tensor encodings | Required execution policy |
| --- | --- | --- |
| Floating | F32, F16, BF16 | Native storage, FP32 math |
| Legacy | Q8_0, Q4_0, Q4_1, Q5_0, Q5_1 | Block decoding, fused AVX2 for common paths |
| K-quants | Q2_K, Q3_K, Q4_K, Q5_K, Q6_K | Row-aware block decoding, fused hot-path kernels where justified |
| I-quants | IQ1_S/M, IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_NL/XS as encountered | Independent codebook validation; scalar support mandatory for present types |
| Other | Any actual additional tensor IDs in frozen artifacts | Implement and test if required; record exact identity |

Do not confuse model recipes (e.g. Q4_K_M, Q4_K_S, i1-IQ3_M) with actual tensor data types. Quantized blocks have encoding-specific element counts, byte lengths, scales and layouts. In particular, H=576 is not a multiple of 256: a K-quant recipe does not imply that every tensor uses a K-quant encoding. Check divisibility per logical row, not just the total tensor element count. For any optional interleaved layout, enforce the specific format's constraints. Do not pad or silently reinterpret serialized tensors.

If an artifact is corrupt, unavailable, split in an unsupported way or incompatible with the canonical architecture, record the concrete blocker and evidence. A required artifact remains a release blocker until resolved or explicitly removed by the user; do not claim full compatibility by exclusion. If an anticipated encoding has no actual model artifact, document that fact; synthetic tests are optional for out-of-manifest encodings.

### Technical design

#### Repository layout

```text
CMakeLists.txt
CMakePresets.json
README.md
plan.md
include/nathr/        # public C++ API
src/core/                 # checked sizes, dtype, buffer, errors, arena
src/formats/              # safetensors, GGUF, model config, file mapping
src/tokenizer/            # byte-level BPE, pretokenizer, Unicode
src/quantization/         # scalar decoding, storage traits, codebooks
src/backend/             # backend contracts and dispatch
src/backend/cpu/         # shared CPU storage, dispatch and thread pool
src/backend/cpu/scalar/  # permanent independent reference kernels
src/backend/cpu/x86/     # ISA-specific optimized kernels
src/models/llama/        # architecture, weight binding, forward pass
src/runtime/             # model/session, generation, cache, sampling
apps/                    # nathr CLI (inspect, tokenize, forward, generate, bench)
reference/               # isolated Python scripts, locked dependencies
models/manifest.json     # frozen model inventory
third_party/json/        # pinned JSON dependency and license
tests/                  # unit, synthetic integration, numerical, malformed inputs
benchmarks/              # workload definitions, machine-readable results
docs/                   # architecture, numerics, formats, backend, performance
.github/workflows/       # CI
```

Do not create empty placeholder subsystems just to match the tree. Add directories and abstractions when the first implementation requires them. Documentation is user/developer reference material, not a second progress ledger.

#### Ownership and backend boundary

Use one CPU device backend, not separate scalar and SIMD device abstractions. Device backend = memory domain and operation execution; kernel = numerical implementation; execution policy = permitted precision and ISA choices. Scalar kernels are independently implemented and always selectable. Share CPU buffer ownership, cache, session infrastructure and dispatch; never route the scalar oracle through optimized kernels. Keep the device interface small, adding upload/download/synchronization contracts only where concrete use or test tracing requires them. Do not implement speculative GPU infrastructure.

Use a central format registry (serialized type ID, block elements, block bytes, required alignment/row constraints, scalar decode function) and a separate CPU kernel registry (operation, storage encoding/layout, input/accumulator numerical policy, ISA, shape constraints, callable). Resolve dispatch once per operation or cached plan, not per block. Record actual selected kernels. Distinguish serialized encoding from runtime packing, logical layout, activation and accumulator types; integer intermediate accumulation is not the same contract as FP32 accumulation. Reject invalid block subviews; compute row bytes and offsets with checked arithmetic. No general broadcasting or arbitrary strided tensor framework.


- `TensorDesc`: rank/dimensions, logical axis order, physical layout tag, dtype, row stride and checked storage requirements. Quantized tensor storage is block-addressed, not an array of individually addressable scalars.
- `Buffer`: RAII host allocation or read-only mapped storage; `TensorView`: checked descriptor + bounded storage view. File mapping lifetime must exceed all referencing weights.
- `TensorHandle`: backend-owned buffer/view with metadata, offset and lifetime; model code must not assume a host pointer. A CPU implementation can internally use host spans. Backend upload/import of validated mapped weights may be zero-copy when supported.
- `ModelConfig`: all architecture settings, validated head relationships, supported RoPE variant and special IDs. `ModelWeights`: canonical semantic tensor names and layouts; immutable after loading.
- `Model`: owns or shares backend and immutable weights. `Session`: owns model reference, backend-specific cache/workspace, RNG, token history and position. Specify destruction and move semantics; do not expose dangling logits spans.
- Backend interface: minimal allocate/import and explicit transfer for tracing, capabilities, embedding, linear, RMSNorm, RoPE, causal GQA (including cache read/write), SwiGLU, residual add. The model specifies semantic operations; the backend determines physical layouts and kernels. Do not expose x86 intrinsics, POSIX mappings or thread pools through the model API.
- Keep operation dispatch coarse-grained (per layer operation, not per scalar/block). Start synchronous and imperative; no graph executor. Runtime fallback must be explicit and observable. A forced unsupported ISA must fail rather than silently fall back.
- The cache must be backend-owned and accessed through documented operations; the model passes logical positions, sequence lengths, heads and mask semantics. This is the minimal requirement for future device-resident execution.
- Errors: typed exceptions or a coherent explicit result type, with filename/tensor/operation context. Use the contextual public error policy specified in Phase 0. No silent fallback on malformed model input. A failed session operation must leave a documented valid state or invalidate/reset the session.

Proposed public concepts (not fixed ABI): `Model::load`, `Session::create`, `Session::prefill`, `Session::append`, `Session::logits`, `Session::reset`, `generate`, `BackendOptions`. A model can be bound to a single backend instance; transferring a loaded model to another backend is not required. Test-only traces download backend tensors explicitly. Keep generation and sampling separate from model math.

#### Tensor shapes and transformer execution

Use a canonical logical layout: activations `[T,H]`, linear weights `[out,in]`, Q `[T,Nq,D]`, K/V `[T,Nkv,D]`, logits `[T,V]` or `[V]` for the final position only. Each loader translates serialized shape conventions to the same semantic contract. GGUF `ne[0]` is typically the contiguous dimension; do not interpret GGUF shape arrays as HF row-major shapes without mapping.

Expected HF tensor shapes:

| Tensor | Shape |
| --- | --- |
| embed_tokens.weight / tied lm_head | [V,H] |
| input_layernorm.weight, post_attention_layernorm.weight, norm.weight | [H] |
| q_proj.weight | [Nq*D,H] |
| k_proj.weight, v_proj.weight | [Nkv*D,H] |
| o_proj.weight | [H,Nq*D] |
| gate_proj.weight, up_proj.weight | [I,H] |
| down_proj.weight | [H,I] |

Validate actual tensor names, shapes, expected bias policy, layer count, duplicate/missing entries and tied-weight resolution. If both embedding and output tensors exist, validate their semantics rather than assuming they are identical; a GGUF exporter may materialize or quantize them separately. Unrecognized extra metadata may be ignored if safe; unsupported tensors affecting semantics must fail clearly.

For each layer, compute in this order:

```text
u = rms_norm(x, input_norm)
q = Wq @ u; k = Wk @ u; v = Wv @ u
q, k = rope(q, k, absolute_positions)
cache.write(k, v)
a = causal_grouped_query_attention(q, cache, scale=1/sqrt(D))
h = x + Wo @ a
r = rms_norm(h, post_attention_norm)
x = h + Wdown @ (silu(Wgate @ r) * (Wup @ r))
```

Then final RMSNorm and output projection, only for requested logits positions. Provide `LogitsMode::None`, `LastToken`, and `AllTokens` (bounded caller-provided chunks or a sink); do not compute the vocabulary projection for intermediate prompt tokens in normal generation. Implement this in the first scalar path. Use one production transformer execution routine accepting a token span, absolute positions, optional cache and output policy; uncached verification and cached sequential/chunked paths share layer semantics. Select GEMV or GEMM behind `linear()`; the scalar GEMM may use repeated GEMV. Avoid separate copies of the full transformer.

RoPE contract: for head width D, frequency `omega[i] = theta^(-2*i/D)` for `0 <= i < D/2`. In HF split-half layout at position p, rotate pair `(i,i+D/2)` by `p*omega[i]`: `a'=a*cos-b*sin`, `b'=a*sin+b*cos`. In adjacent-pair layout the equivalent coordinate pair is `(2*i,2*i+1)`. Treat Q/K row permutation and activation/cache layout as one coherent mapping; inspect actual converter and pinned artifacts, do not infer rotation convention from a recipe name. Test known basis vectors, position 0, 1, 127 and 8191, and equivalent HF/controlled floating GGUF. Cache post-RoPE K and unrotated V.

 RMSNorm uses `x * rsqrt(mean(x*x)+eps) * weight`; stable softmax subtracts the maximum; GQA maps `query_head / (Nq/Nkv)` to KV head. RoPE rotates Q and K at absolute positions, not V. Confirm HF split-half versus GGUF interleaved/permuted Q/K conventions against actual conversion fixtures. Do not apply both weight permutation and incompatible rotary layout. If canonicalizing quantized Q/K weights would require unpacking/repacking blocks, prefer an explicit loader/backend layout tag or valid row permutation rather than destructive requantization.

Baseline computation uses FP32 inputs, outputs, activations and cache. GEMV, RMSNorm and attention reductions use FP64 intermediates. RoPE and SwiGLU evaluate in FP64 and round to FP32 when writing their output activations. Attention rounds the scaled scores, shifted exponentials and normalized probabilities to FP32 and accumulates the normalization denominator and value-weighted sums in FP64. The independently reconstructed PyTorch model uses these same rounding boundaries for numerical verification. Define all in-place and aliasing guarantees. Reject non-finite logits and invalid sampling inputs with clear errors.

#### File parsing and tokenizer

Safetensors: explicitly reject duplicate JSON keys using parser callbacks/SAX before ordinary object materialization; default JSON parsing alone does not detect duplicates. Enforce a documented header limit, format-specific empty-tensor semantics and exact data-region coverage. Little-endian u64 header length, bounded JSON header, offsets relative to data region, validated contiguous tensor extents, dtype/rank and byte counts. Reject duplicate names/keys where relevant, overflow, overlaps, invalid offsets, truncated data and unsupported sharding. Support the actual official checkpoint layout; optional generic shard support is not required unless the inventory needs it. Tied weights are references, not copied buffers.

GGUF: the frozen inventory requires only little-endian v3 and single-file artifacts. Separate structural parsing/inspection from Llama semantic binding and backend import; inspection must work even if execution support is missing. Validate magic/version/endian, metadata tags and lengths, nested/unsupported arrays, tensor table, alignment, data offsets, block sizes, bounded strings, counts and file extents. Unknown metadata types must fail if they cannot be skipped safely. Decode integers explicitly from little-endian bytes. Support all file shards only if required by frozen inventory; validate shard indices, counts and names. Implement inspection independently of model execution so the manifest can be audited before all encodings run.

Tokenizer: load the exact tokenizer JSON pipeline and token IDs, not a generic approximation. Implement the required normalization/pretokenization, GPT-2-style byte mapping, BPE merge ranks and tie behavior, added/special token matching, decoding, BOS/EOS policy and UTF-8 streaming. Read actual tokenizer configuration to determine which behavior is required. If Unicode category tables are necessary, generate from a pinned Unicode data release with license and generation script; do not rely on locale-dependent `std::regex` or `isalpha`. No extra runtime tokenizer dependency. Explicitly distinguish raw token-byte decoding from terminal UTF-8 display; define invalid-byte handling to match oracle. Base generation has no implicit chat template. Verify the full pipeline (normalizer, added-token flags and matching precedence, ByteLevel regex/use_regex, add_prefix_space, post-processor, decoder and cleanup flags) against pinned files, not a model-family assumption. Test intermediate added-token spans, pretoken segments, byte mapping and merges where observable, and exact final IDs. Differential-fuzz valid UTF-8 including multilingual and combining characters; define invalid UTF-8 input policy separately from raw token-byte output. Implement only the pinned pipeline, not a Unicode regex engine. Keep incomplete UTF-8 token fragments in a streaming buffer and validate flush behavior against the documented policy. Start with simple BPE; optimize merge scheduling only if profiled.

#### Session and KV cache

- Use a fixed-capacity contiguous FP32 cache per layer and KV head for baseline. Logical layout `[layer,kv_head,position,head_dim]` for K and V; physical layout backend-private.
- Add FP16 cache storage in Phase 4. Read into FP32 for attention, apply the baseline attention arithmetic and round on write. FP16 cache at 8192 positions is 180 MiB. Compare against a reference that rounds K/V at identical cache boundaries and separately measure drift versus FP32 cache storage. No INT8 cache in v1.0.
- One sequence per session. Capacity must be within supported context and checked before mutation; reject overflow, do not truncate automatically. A reset clears logical state without needing to zero all allocated bytes; never read uninitialized/stale positions.
- FP32 cache bytes = `2 * L * capacity * Nkv * D * 4`. At 8192 positions this is 377487360 bytes (360 MiB); at 2048 it is 90 MiB. Use configurable capacity rather than automatically allocating maximum for every request.
- Baseline prefill can iterate token-by-token. Optimized prefill processes bounded chunks and uses GEMM; causal mask for chunk position j allows keys through `past_length+j`. Verify arbitrary partitions equal sequential append, within numerical tolerance.
- `prefill/append` consumes supplied IDs and returns next-token logits at the final consumed position. Sampling chooses a token from those logits. If another step is required, append that token exactly once. No duplicate prompt token, off-by-one position, or omitted final prompt logit. Define empty prompt policy and EOS/max-new-tokens behavior explicitly. Session state is Empty, Ready (valid next-token logits if requested), or Invalid (requires reset); stopping generation does not intrinsically invalidate a session. Sampling is separate from append; never advance cache or position just by sampling. Validate capacity and token IDs before mutation. Document failure atomicity (validation failure leaves state unchanged; partial execution failure invalidates session), logits lifetime, reset, consumed-token count and generated-token count. `None` invalidates logits until a subsequent forward requests them. A generated EOS or final budget token may be returned without being appended; document this explicitly, and distinguish emitted versus consumed IDs for continuation.
- For trace and perplexity, expose all-token logits only on request, with bounded chunking. Normal generation must not allocate `[T,V]` logits for long prompts. Avoid full quadratic score buffers from the first scalar implementation: reuse a context-sized score buffer per active query/head (or online softmax). Tiled/online softmax optimization may follow; keep masked positions excluded and stable normalization.
- Greedy uses deterministic tie-break (lowest token ID). Seeded stochastic sampling uses defined RNG and tie ordering, stable temperature, top-k/top-p, valid parameter bounds. Do not promise cross-platform bitwise stochastic parity with PyTorch. Validate distribution and filtering with known logits; greedy and teacher-forced paths are primary numerical tests.

#### x86 computation strategy

- Baseline scalar FP32 GEMV and GEMM. Dense F16/BF16 converted on load or in kernel initially; retain original storage in production. Test half/BF16 conversions, subnormals, infinities, NaNs and round-to-nearest-even where applicable.
- Scalar quant kernels unpack one bounded block at a time and accumulate with FP32 activations. No whole-model expansion. Embedding lookup and output projection must also support quantized weights. Independently test each decoder before executing a model.
- Optimize x86 only after baseline evidence; implement the Q8_0/Q4_0 vertical slice before broad quantization coverage. Separate ISA-specific translation units for AVX2, FMA, F16C; optional SSE2 and AVX-512/VNNI paths if meaningful and testable. SSE2 is part of the normal x86-64 baseline. Detect CPU feature bits AND OS-enabled XMM/YMM/ZMM state (CPUID, OSXSAVE, guarded XGETBV or reliable compiler support). AVX2 does not imply FMA/F16C. Test forced scalar and supported paths; no execution of unsupported opcodes, including during initialization or LTO.
- Persistent thread pool, static row partitions initially; bound worker count, avoid nested pools and excessive barriers. Use measured size-dependent parallelization thresholds, contiguous row partitions and avoid false sharing. Measure physical cores versus SMT. Define behavior for multiple sessions and deterministic per-output reduction order where practical. Handle worker exceptions and pool shutdown correctly.
- Optimize GEMV (decode) separately from blocked GEMM (prefill); use row/tile packing only where beneficial, with explicit memory overhead. Never require a second permanent uncompressed model. Tune alignment, row widths/tails, scale unpacking, activation reuse and vector reductions. Quantized integer dot products that round activations are a different numerical contract; require separate oracle and quality evidence, not silently enabled as exact fused kernels.
- SIMD fallback to scalar per encoding is valid for v1.0 and must be reported by inspection/bench output. All manifest encodings must work, but not all need hand-written SIMD. Prioritize actual profile bottlenecks and common Q8/legacy/K formats.
- Do not implement untestable AVX-512 kernels merely for feature completeness; if no suitable hardware, defer and record. No GPU implementation in this plan.

#### Numerical verification contract

Use a hierarchy: hand-derived and FP64 operation oracles, independent PyTorch FP32 inference, independent GGUF block decoding, scalar C++, optimized C++ and external llama.cpp interoperability. No single reference establishes every contract. Use a pinned CPU-only Python environment with torch, transformers, safetensors, numpy and any reference-only GGUF decoder. Pin Python and package versions (lock file), model revisions, reference converter/llama.cpp commit and fixture schema version. Run `eval()`, inference mode, fixed tokens, explicit eager attention, FP32 weights/activations and controlled thread count. Disable stochastic operations. Never require Python to run the delivered binary.

Use small Safetensors fixtures plus JSON metadata for tensor shape, semantic axes, artifact hash, tokens, positions, expected dtype, seed, reference version and tolerance profile. Generate fixtures outside the C++ implementation. Store tiny synthetic fixtures in repo; cache real model traces outside it. Reference scripts must not call the C++ implementation to generate expected values.

Validation layers:

- FP64 oracle for dot products, RMSNorm, softmax and small attention; use reduction-aware error bounds (e.g. `gamma_n * sum(abs(a_i*b_i))` for an appropriate accumulation model), and test FMA/non-FMA, cancellation, near-zero outputs and finite-value handling. Use exact bit checks for unpacking/conversion where representation is equivalent. Test invariants: RoPE norm/position-zero identity, softmax positivity/sum/shift invariance, causality, GQA mapping, cache partition invariance and valid sampling support. Use distinct valid generated quant blocks versus arbitrary malformed byte fuzzing.

1. Tokenization: exact IDs, special-token policy and exact decoded bytes where applicable. Compare multilingual strings (English, Arabic, Chinese), code, spaces, tabs, newlines, punctuation, combining marks, emoji, invalid byte policy and long repeated merges.
2. Kernels: deterministic hand-derived, randomized and adversarial inputs, odd sizes, tails, unaligned buffers, zero and extreme values. Use explicit dtype conversion/rounding cases and independent packed-block fixtures for all required encodings. Check finite/non-finite classification.
3. Full FP32 model: compare embeddings; each layer's input/post RMSNorm, Q/K/V before/after RoPE, attention scores/probabilities or output, attention projection, MLP gate/up/SwiGLU/down, residual outputs, final norm, logits and K/V cache. Trace one short prompt end-to-end and selected layers/positions for longer sequences. Compare canonical axes, not raw layouts.
4. Quantized models: independently decode the exact GGUF artifact into canonical FP32 weight tensors, reconstruct equivalent PyTorch model with correct tied/output semantics and Q/K permutation (if output is omitted, alias embeddings; if distinct, disable HF weight tying and verify actual final parameter values after model construction), and compare teacher-forced logits against scalar C++. Use independently generated block fixtures (e.g. pinned llama.cpp reference decoder) where Python GGUF support is missing. Never use our decoder as its own oracle. Compare optimized backend against scalar for the same artifact and tokens. Use llama.cpp end-to-end as an additional cross-check, not the sole oracle; its activation quantization and accumulation may differ.
5. Cache equivalence: no-cache full forward vs incremental decoding; chunked vs single-token prefill; reset/reuse; multiple independent sessions; context boundaries including 8192 where practical.
6. Quantization quality: separately compare each quantized artifact against the original BF16 checkpoint executed in FP32 on the same fixed token corpus. Compute teacher-forced NLL/perplexity, logit divergence and top-k agreement. Do not confuse quantization degradation with implementation error. Report corpus identity, token count, BOS/stride/masking and same evaluation context; no repeated scoring of overlap. Quality measurements are descriptive unless a threshold is explicitly justified.

Initial proposed numerical thresholds (calibrate against real independent fixtures in Phase 1 for native weights and Phase 2 for the quantized slice, before optimizing the affected path; log any revisions with rationale):

| Comparison | Initial gate |
| --- | --- |
| IDs, shape, layout, expected exact conversions | Exact |
| FP32 simple elementwise | atol 1e-6, rtol 1e-5 |
| FP32 linear | atol 1e-4, rtol 1e-4; also reduction-aware error analysis |
| Layer hidden states and final logits | relative L2 <= 1e-4; cosine >= 0.99999 |
| Cache equivalence | same relevant kernel/model gates |
| Quantized weights vs independent decoder | exact when arithmetic/rounding equivalent; otherwise encoding-specific justified tolerance |
| Quantized full-model vs same-artifact oracle | initially same model gates; revise only with evidence |
| Greedy next token | exact except explicitly reported near-ties |

Use `abs(actual-expected) <= atol + rtol*abs(expected)` for elementwise checks; report max absolute error, RMSE, relative L2 with denominator floor, cosine, worst index, top-k and non-finite counts. For near-zero vectors use absolute error rather than unstable cosine. A token near-tie exception requires the reference logit margin to be consistent with measured numerical error and must not conceal other failures. Record all exceptions. For long generation compare teacher-forced positions rather than only comparing generated strings, since one token difference changes all subsequent conditioning.

#### Benchmark and observability contract

- Record CPU model, physical/logical cores, ISA, cache sizes, memory configuration when known, OS/kernel, compiler/flags, git revision, model SHA, quantization recipe, backend/kernel coverage, threads, context, prompt/decode lengths and sampling mode.
- Use monotonic clock (`steady_clock`), warmup and repeated runs; report median and dispersion/p95, not best-only. Fix prompt IDs; use teacher-forced decode tokens or identical greedy trace for fair comparisons. Separate cold/warm loading, prompt prefill, decode and sampling/output costs. No terminal printing inside timed kernel loop.
- Report model load latency, TTFT (explicit definition including/excluding load), prefill tokens/s, decode tokens/s, inter-token latency, peak RSS, mapped/resident weights where observable, cache/scratch bytes, and throughput scaling; executable size (stripped/unstripped), runtime link dependencies, and allocation counts. Separate reserved virtual memory, mapped/resident pages, owned bytes and allocator/thread overhead. Distinguish logical bytes from actual measured traffic. `perf` counters are optional and may require host permissions; record availability.
- Workloads: prompt 32/128/512/1024/2048; decode 32/128/256; context 512/2048/8192 as applicable; 1/2/4/8/physical cores; supported backend/precision combinations. Skip invalid prompt+decode > context. Use a representative subset on routine CI and complete matrix on controlled hardware; do not run enormous Cartesian products blindly.
- Kernel microbenchmarks: GEMV/GEMM using actual model dimensions; RMSNorm, RoPE, attention, dequant and fused dot products. Check output so compiler cannot eliminate work. Compare scalar, forced ISA and external llama.cpp using matching artifact, prompt/context/thread settings and documented precision differences. PyTorch CPU is a secondary baseline; note differing implementation and activation types.
- No invented latency or speed target. In Phase 0 establish hardware and external baseline, a fixed workload protocol and proposed memory limits; in Phase 1 measure scalar baseline; before each optimization freeze hardware-specific target and variability threshold. Require measured end-to-end decode improvement and prefill improvement over sequential GEMV on representative workloads, not merely an isolated microkernel speedup. If the gates fail, profile and revise implementation or record a blocking deviation for user decision; do not move the goalposts. Use roofline analysis with measured sustained memory bandwidth and compute throughput; report actual encoded bytes, cache/activation traffic and arithmetic intensity assumptions, not nominal bits or theoretical peak as measurements. Match prompt and teacher-forced decode IDs, active context, cache precision, logits output policy, threads/affinity, warmup, sampling exclusion and thermal conditions across external comparisons; disclose differences in activation quantization.
- Memory budget: derive checked per-session cache and workspace sizes before allocation, cap configurable prompt chunk size and trace output, account for immutable weights, metadata/tokenizer, cache, logits and scratch. Normal generation: no quadratic score allocation, no `[T,V]` logits allocation, no model-sized duplicate/expanded weight buffers and no steady-state compute allocation after warmup. A single FP32 logits vector is 192 KiB; full-context all-position logits would be 1536 MiB. Phase 0 defines testable caps for parser inputs and provisional model/session overhead; Phase 1 records actual allocation baseline, Phase 4 checks optimized budgets. Report any bounded packing overhead explicitly.

## Commands

Phase 0 reference commands and Phase 1 CMake, CLI and independent HF verification commands are implemented. Configuration and CTest are offline. The full GGUF release validation and perplexity tools are planned for later phases.

```bash
# setup (C++ toolchain already installed)
cmake --version
c++ --version
# Optional isolated reference environment; explicit network access for installation.
uv venv --python 3.12 .venv-reference
uv pip sync --python .venv-reference/bin/python --require-hashes \
  --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
  reference/requirements.lock
. .venv-reference/bin/activate

# build and test (baseline, no Python or model weights needed)
cmake --preset dev
cmake --build --preset dev --parallel
ctest --preset dev --output-on-failure
cmake --preset release
cmake --build --preset release --parallel
ctest --preset release --output-on-failure

# sanitizers (separate presets; TSan separate from ASan/UBSan)
cmake --preset asan-ubsan
cmake --build --preset asan-ubsan --parallel
ctest --preset asan-ubsan --output-on-failure
cmake --preset tsan
cmake --build --preset tsan --parallel
ctest --preset tsan --output-on-failure

# lint/format (developer tools only)
cmake --build --preset dev --target format-check
# optional clang-tidy preset/target once configured

# explicitly download or verify pinned weights (reference tool only)
python reference/models.py fetch --manifest models/manifest.json --output models/cache
python reference/models.py verify --manifest models/manifest.json --directory models/cache

# inspect, tokenize and generate; CLI target is nathr
./build/release/nathr inspect --model "$MODEL"
./build/release/nathr tokenize --model "$MODEL" --prompt 'Hello world'
./build/release/nathr forward --model "$MODEL" --tokens 1,2,3 --backend scalar --trace trace.safetensors
./build/release/nathr generate --model "$MODEL" --prompt 'Once upon a time' --backend auto --threads 1 --context 2048 --max-new-tokens 64 --greedy

# numerical fixtures and comparisons (explicit model path)
python reference/generate_fixtures.py --model "$HF_MODEL_DIR" --output fixtures/local
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python reference/phase1_verify.py \
  --binary build/release/nathr --model "$HF_MODEL_DIR" --output fixtures/local/phase1 --floating-variants
./build/release/nathr_tests --long
./build/release/nathr_tests --real-allocations "$HF_MODEL_DIR"

# benchmark (output machine-readable JSON)
taskset -c 0 ./build/release/nathr bench --model "$MODEL" --backend scalar --threads 1 --prompt-length 128 --decode-length 32 --context 512 --repetitions 7 --warmups 2 --json
```

`$MODEL` is a local Safetensors directory or a little-endian v3 GGUF file. `$HF_MODEL_DIR` is the pinned HF checkpoint. The GGUF-specific reference commands are under Phase 2. Document exact argument semantics, trace filenames, environment setup and test labels as implemented. No network access for normal `ctest`. Reference downloads must never run implicitly.

## Acceptance Criteria

- [x] A standalone Linux x86-64 C++20 release binary and reusable library build with GCC and Clang, using only standard/system libraries and the approved JSON parser. Runtime dependency audit passes; no Python or ML framework required.
- [x] Official SmolLM2-135M base Safetensors checkpoint loads directly; F32/F16/BF16 handling and tied weights are verified. All architecture dimensions are metadata-driven.
- [x] Exact tokenizer parity on frozen fixtures and differential fuzz tests; text and token-ID generation work with documented BOS/EOS, UTF-8, sampling and context behavior.
- [x] Scalar CPU model matches independent PyTorch per-layer traces, logits and FP32 KV cache under justified numerical gates; FP16 cache meets its separate rounded-cache reference and quality gates; optimized kernels match scalar under equivalent numerical policy. Cached/uncached and chunked/sequential paths agree.
- [x] Every required frozen GGUF artifact loads and runs; every required encoding has an independent block-decoding oracle and end-to-end same-artifact numerical validation. Mixed tensor formats and Q/K layout conversion are covered. No unapproved exclusions or unresolved artifact blockers.
- [x] Quantized production path preserves packed weight storage; memory accounting verifies no full FP32 expansion. Model weights are shared across sessions, and steady-state compute has no per-token heap allocation under documented conditions.
- [x] Runtime dispatch works on baseline x86-64 and tested ISA extensions without unsupported instructions. Forced unsupported paths fail cleanly; fallbacks are reported. Threaded execution is correct and race-tested.
- [ ] Benchmark suite reports reproducible scalar/optimized/external baselines and quantization quality, with limitations and hardware details. End-to-end decode and prefill improvements meet the predeclared hardware-specific gates; no arbitrary unsupported performance claims.
- [x] Malformed input, resource limits, lifetime, parser/quant-block fuzzing and cache-boundary tests pass. ASan/UBSan and TSan (supported subset) are clean, or environmental limitations explicitly block affected claims. No skipped required validation is marked passed.
- [ ] README and API/architecture/numerical/quantization/performance docs describe usage, exact supported artifacts, backend selection, reproduction commands and limitations. The ledger records completion evidence and deviations. Final diff is reviewed.

## Test Plan

1. **Fast unit tests:** dtype traits, checked arithmetic, alignment/lifetime, binary parser boundaries, half/BF16 conversion, scalar kernels, quant block decoders, tokenizer parity, sampling, CPU feature dispatch and thread-pool behavior. Use custom explicit failure assertions (not C++ `assert` compiled out in release) and CTest. Exercise zero, odd, minimum and maximum valid sizes, invalid shapes, unaligned memory, NaNs, overflow and malformed bytes.
2. **Synthetic transformer integration:** construct tiny deterministic Llama-compatible weights with GQA and tied embeddings, run independent Python fixtures, compare no-cache, cached, chunked and full logits. Include a non-divisible block row rejection, mixed storage types, output-head aliasing, multiple sessions, context reset and EOS tests. Tiny fixtures checked into repo; no Python needed to execute CTest.
3. **Real checkpoint numerical tests:** explicit opt-in label for official artifact and full trace. Use short/medium prompts and context boundary tests. Compare teacher-forced positions, each layer and cache. Save structured metrics and worst-index diagnostics on failure. Validate tokenizer independently before evaluating end-to-end text.
4. **Quantization compatibility:** inventory all required artifact tensor IDs, test independent block fixtures, verify hashes, compare each artifact's logits against independent same-weight PyTorch reconstruction; include scalar and supported optimized paths, plus external llama.cpp checks. Release CI must enumerate the whole frozen manifest, not a sample. Large artifacts are cached outside Git. Keep quality/perplexity distinct from implementation correctness.
5. **Memory and concurrency:** count hot-path allocations, verify mapped lifetime and no weight copies per session; test shared immutable model, separate sessions, thread counts, reset and repeated create/destroy. ASan/UBSan and TSan in separate builds; optional Valgrind; required developer-only bounded libFuzzer campaigns for Safetensors, GGUF, tokenizer and quant block parsing (Clang ASan/UBSan), with minimized regression inputs. Validate memory budget and failure behavior on oversized input.
6. **Performance and regression:** benchmark fixed tokens/threads/models on controlled x86 hardware; record baseline before optimization and compare same artifact after changes. Record peak RSS and cache/workspace accounting. Use CI timing only as informational unless dedicated stable runners exist.
7. **CI and packaging:** Linux GCC/Clang Debug/Release; baseline scalar forced; AVX2/FMA/F16C tests if supported; sanitizer jobs; offline build/test with no ML libraries; verify public headers and install/consumer smoke test. Full model and manifest jobs manual/scheduled/release with explicit artifact cache. Required jobs unavailable are blockers, not implicit successes. Run format/lint and inspect final diff.

### CI tiers and numerical evidence

| Tier | Trigger | Coverage |
| --- | --- | --- |
| Fast | Every commit | Model-free synthetic models, parsers, math invariants, kernels and frozen tokenizer fixtures |
| Extended | Pull requests | Sanitizers, dispatch, concurrency, bounded fuzz smoke tests and integration |
| Model | Manual/scheduled | Official checkpoint, controlled artifacts, representative published artifacts, extended tokenizer differential tests |
| Release | Release candidate | Every frozen artifact, numerical reports, complete required paths and controlled benchmarks |

Use full short-prompt traces and selected layers/positions for long-context tests. Exercise the 8192-token boundary via real-model tests plus inexpensive synthetic boundary cases; bound fixture storage. Version fixture schemas and record oracle versions, artifact hash, canonical axes, operation/layer/position, input tokens, seed, dtype, precision policy and tolerance profile. No fixture generated by Nathr is independent expected output.

### Execution ledger rules

Phases are sequential gates unless a task is explicitly independent. Complete and validate each phase before making the next depend on it. Small code changes and tests should land together. Maintain this file: update checklist evidence, status (`Pending`, `In progress`, `Blocked`, `Complete`), completion notes (commands, results, paths, commit if available) and deviation notes (decision, reason, impact). Do not mark a phase complete when acceptance criteria are unverified. If a blocker cannot be resolved locally, record it here with the exact missing evidence and continue only with independent tasks. Do not create separate planning documents. Never manufacture measured results.

## Phases

### Phase 0: Orientation, feasibility and unresolved-item resolution

Do this phase before committing to production implementation. The purpose is to replace assumptions with reproducible evidence, not merely to repeat this plan. Small disposable Python/external-tool probes and minimal compiler experiments are allowed. Do not create a second inference engine or a separate plan. Record decisions and links to evidence here; keep generated machine-readable artifacts in their appropriate directories. Parallelize independent investigations if useful.

Resolution register

| ID | Status | Evidence and decision |
| --- | --- | --- |
| R01 | Resolved | The workspace held only this plan and empty management mounts. Git 2.43 reported no repository. GCC and Clang both built and passed a disposable C++20 CMake/CTest probe. Host, ISA, profiling and sanitizer findings are under Current State. |
| R02 | Resolved | The pinned canonical checkpoint contains 272 BF16 tensors, with 134,515,008 stored parameters and 269,030,016 encoded weight bytes. The 30,528-byte header has no `lm_head.weight` or bias tensors. Transformers loads the output head as the same parameter as the embedding. The complete audited config and shapes are in [the manifest](models/manifest.json). |
| R03 | Resolved | Verified every published GGUF by streaming the full bytes against the repository's LFS SHA-256. All 50 hashes are distinct. The 14 QuantFactory, 12 mradermacher static and 24 mradermacher importance-matrix files are single-file, little-endian GGUF v3 with 32-byte alignment and 272 stored tensors each. The manifest groups exact tensor shapes and encodings by fingerprint and includes per-file histograms, stored bytes, parameters and effective bits. |
| R04 | Resolved | The manifest enumerates all 21 published type IDs. Pinned gguf-py decodes those types and controlled BF16. Three sampled blocks per type matched the pinned ggml C reference bit for bit, including all nine I-quant types and controlled BF16. Block dimensions and evidence are in the manifest. Rows of width 576 use only types whose block width divides 576. In particular, 256-wide encodings can occur in width-1536 down-projection rows. |
| R05 | Resolved | Generated controlled F32, F16, BF16, Q8_0 and Q4_0 with the pinned external tools and no calibration. The Q4_0 recipe stores the token embedding as Q8_0. All five omit a separate output tensor. Commands and hashes are recorded here and in the manifest. |
| R06 | Resolved | The pinned converter maps Q/K coordinates from split halves to adjacent pairs. The oracle applies the inverse row permutation using 9 Q heads or 3 K heads, with each weight in logical `[out,in]` order. Reconstructed controlled F32 and BF16 weights and logits match native HF exactly. For F16 4,661 state-dict entries change under storage rounding. Reconstructed logits match equivalently rounded HF exactly and differ by at most 9.82e-5 from the native unrounded logits. A synthetic GGUF with a distinct negated output embedding produced exactly negated native logits. |
| R07 | Resolved | All published files share one tokenizer fingerprint. The controlled files share another that explicitly sets `clean_spaces=false`. Both preserve the exact canonical vocabulary and merge ranks. The pinned HF fast tokenizer defines text semantics, and the GGUF sidecar policy is recorded below. The committed tokenizer fixtures cover full pipeline output and partial UTF-8 output. |
| R08 | Resolved | An isolated CPython 3.12 environment installed from the hashed reference lock. One-thread eager FP32 Transformers produced a fixed-ID trace and FP64 examples. The GGUF oracle independently reconstructed the weights and produced finite fixed-ID logits for the verified published and controlled files. Phase 1/2 must still calibrate C++ parity. |
| R09 | Resolved | Public errors, ownership, supported inputs, logits policies, ISA, compiler floor and dependency boundaries are specified below. Production CMake targets and runtime implementation start in Phase 1. |
| R10 | Resolved | The fixed-ID eager PyTorch and CPU llama.cpp baselines are recorded in [the machine-readable benchmark](benchmarks/phase0.json). The protocol and provisional resource ceilings are specified below. Scalar Nathr measurements and optimization targets are later gates. |
| R11 | Resolved | JSON parser, CPU-only reference wheels, converter, quantizer, Unicode data and quant codebook sources are pinned below. The four pinned model cards declare Apache-2.0. Keep these weights out of the tracked workspace and preserve required notices if redistributing model artifacts. Nathr has no production link dependencies to inspect until Phase 1. |
| R12 | Resolved for feasibility | Local disk and RAM accommodated the complete suite and all independent reference runs. Local AVX2/FMA/F16C work under the observed OS state. AVX-512 is unavailable. A hosted runner is not provisioned yet. Phase 1 must provide a working TSan runner and record hosted runner limits before enabling CI jobs that depend on them. |

Deliverables

- [x] Inspect the repository and verify compiler, hardware and tool availability.
- [x] Freeze the native checkpoint and all published GGUF artifacts with byte-verified hashes and actual tensor inventories.
- [x] Convert the controlled slice and verify layouts, storage rounding and tied and untied output behavior.
- [x] Exercise an independent block decoder and the reconstructed PyTorch model across the actual required encodings.
- [x] Audit both GGUF tokenizer fingerprints and the canonical HF stages, and produce representative token and UTF-8 fixtures.
- [x] Install the CPU reference lock, run eager FP32 HF, and generate the short trace and FP64 examples.
- [x] Resolve the production API, ownership, precision and offline dependency decisions.
- [x] Define parser, runtime and benchmark budgets and run both external baselines.
- [x] Record evidence and identify unverified future gates.

#### API and resource decisions

- Define one public `nathr::Error` derived from `std::runtime_error`. Provide a stable error code for invalid arguments, unsupported inputs, malformed inputs, I/O, resource limits and backend failures, and include the file, tensor or operation in its message where applicable. Normal C++ allocation failures can still propagate as `std::bad_alloc`.
- `Model::load` accepts the canonical single-file Safetensors directory with its HF metadata or a local GGUF file. Metadata inspection and integer-ID forward work without tokenizer sidecars. Known GGUF tokenizer fingerprints may use the embedded tables plus the canonical SmolLM2 pipeline. For an unknown fingerprint, require the canonical HF tokenizer sidecars and verify embedded vocabulary and merge equality before using them. A mismatched token-ID table is unsupported for text use. Exact HF text semantics govern both formats.
- `Model` shares read-only weights and its backend. `Session::create` retains model ownership and allocates session-private cache, scratch and RNG. `prefill` operates on Empty, `append` on Ready, and `reset` from any state. Empty spans and context overflow fail before mutation. A compute or sink failure after execution starts marks the session Invalid until reset. A Ready session can have unavailable logits after `LogitsMode::None`. A logits view lasts until the next mutating operation or session destruction. `AllTokens` uses a synchronous bounded sink. Generation returns emitted and consumed counts so continuation can append any final emitted ID still pending. An empty prompt requires an explicit BOS token supplied by the caller.
- Use CMake 3.25, GCC 12 or Clang 16 as the declared minimums. The current installed toolchain versions are the only ones verified in Phase 0. Build x86-64 baseline with the scalar FP32 reference and `-ffp-contract=off`. Resolve format geometry and backend kernel dispatch in separate registries. Use AVX2, FMA and F16C only after checking each required CPU feature and OS state. Keep ISA and precision selection observable. Model runtime and normal CTest stay network-free and Python-free.
- Provisional structural limits for the scoped artifacts are 1 GiB per weight file, 1 MiB Safetensors headers or config JSON, 8 MiB tokenizer JSON or GGUF header plus metadata and tensor table, 512 tensors, rank 4, and 2,048 metadata keys. Cap GGUF strings at 64 KiB, arrays at 65,536 elements and alignment at 4,096 bytes. Validate bounds and checked arithmetic before data-sized allocations. Reject unsupported nested arrays. The audited GGUF headers fit under 2 MiB and the canonical tokenizer fits under 3 MiB.
- Budget 64 MiB of model-owned metadata and tokenizer overhead. Cap prompt chunks at 128 tokens and session scratch including logits at 32 MiB, excluding KV and immutable model bytes. Capacity defaults to 2,048 positions, with 8,192 supported. FP32 KV costs 90 MiB and 360 MiB at those capacities. Bound debug trace output to 64 MiB per run, and stream larger comparisons by chunk. Phase 1 measures owned/mapped bytes, resident bytes and post-warmup allocation counts separately and revises provisional ceilings only with measurements.
- Model-free CI can use a proposed 2 vCPU, 4 GiB RAM and 2 GiB workspace budget. Plan the full model audit on a dedicated 8 GiB RAM and 12 GiB free-disk runner with a pinned cache. Published weights total 5,181,378,112 bytes, plus the native checkpoint, controlled conversions and reference environment. Use the existing CI tiers. Run scalar on every supported x86-64 host, and provide real AVX2/FMA/F16C test coverage for any advertised accelerated paths.

#### Tokenizer contract from the pinned checkpoint

The 49,152-entry BPE vocabulary has 48,900 merges. The pipeline uses no normalizer or postprocessor. First match exact added tokens 0–16 by leftmost-longest match with no single-word restriction, stripping or normalization. Those 17 IDs are marked special. The pretokenizer splits every Unicode numeric character individually, then runs ByteLevel with no prefix space, trimmed offsets and regex enabled. ByteLevel uses GPT-2 contractions, letter/number runs, other non-whitespace runs and the two trailing-whitespace alternatives in the pinned tokenizer source. BPE selects the lowest rank and breaks equal ranks by leftmost position. It has no dropout, unknown token in the BPE model, byte fallback or ignoring of merges. Full text decode uses ByteLevel with cleanup disabled. IDs 0 for BOS, EOS and unknown are configured aliases, and fast encoding leaves boundaries unchanged when special insertion is enabled. The default CPython reference uses the pinned fast tokenizer even though the config class name is `GPT2Tokenizer`.

Use Unicode 17.0.0 General_Category N for the Digits stage. Use Unicode 16.0.0 General_Category L and N plus the 25 White_Space code points for ByteLevel, as observed by comparing the pinned wheel over all Unicode scalar values. Keep these tables versioned independently. Reject invalid UTF-8 text inputs. Raw token decode preserves bytes, and terminal output carries incomplete sequences between tokens and applies UTF-8 lossy replacement on invalid input or an incomplete final flush. The committed examples include three partial prefixes of one emoji which decode to U+FFFD when finalized separately. Use the reference wheel's BPE tie order and special-token matching as the differential oracle. [Tokenizer source v0.22.2](https://github.com/huggingface/tokenizers/tree/v0.22.2/tokenizers/src/pre_tokenizers)

#### Provenance and reproduction

- Canonical model `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, QuantFactory `d948db3614be18259a175aafd7689a70f1cb4e2f`, mradermacher static `bf92313aa80eb55329ae75ccce3743101784c802`, and mradermacher importance matrix `2169229412c9066057acb0b60f068bb791e2b1c3`. The latter includes a hash-verified `imatrix.dat` for provenance. Use the four pinned source listings in [the manifest](models/manifest.json). No model artifacts were excluded.
- llama.cpp `b1b770d90d94cad9e9ea5b993bdbc383a522fb28` matches the installed quantizer and benchmark build 12359. The source tarball SHA-256 is `ae8556670b17c25931cd39eb7c95617650eaf6091bbcff206750583ddde1acb4`. The feasibility C block decoder was rebuilt from this source. Stage the checkpoint files in a directory named `hf` for byte-reproducible conversions. The converter derives display metadata from that basename, and a hash-named Hub snapshot directory introduces a different `general.finetune` field. I-quant tables are specified by its MIT-licensed `ggml/src/ggml-common.h`. The manifest records required type IDs and actual decoder comparisons. Preserve the notice and validate any table incorporated later. [Pinned llama.cpp source](https://github.com/ggml-org/llama.cpp/tree/b1b770d90d94cad9e9ea5b993bdbc383a522fb28)
- Choose nlohmann/json v3.12.0 MIT. The single-header SHA-256 is `aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63` and license SHA-256 is `46a65cffd1ea955132d95a8dd921640714a8d6b537d2e4e482d31145ae95b603`. Vendor both with the first parser in Phase 1. [Upstream v3.12.0](https://github.com/nlohmann/json/tree/v3.12.0)
- UnicodeData.txt SHA-256 is `2e1efc1dcb59c575eedf5ccae60f95229f706ee6d031835247d843c11d96470c` for 17.0.0 and `ff58e5823bd095166564a006e47d111130813dcf8bf234ef79fa51a870edb48f` for 16.0.0. Unicode License v3 SHA-256 is `e7a93b009565cfce55919a381437ac4db883e9da2126fa28b91d12732bc53d96`. Generate tables and carry the license when the C++ tokenizer lands. [Unicode 17 data](https://www.unicode.org/Public/17.0.0/ucd/UnicodeData.txt), [Unicode 16 data](https://www.unicode.org/Public/16.0.0/ucd/UnicodeData.txt), [Unicode license](https://www.unicode.org/license.txt)
- [The reference lock](reference/requirements.lock) was exercised with Ubuntu CPython 3.12.3 and uv 0.9.26. Direct reference dependencies include PyTorch 2.11.0+cpu, Transformers 4.57.6, Safetensors 0.7.0, NumPy 2.4.6, gguf 0.19.0 and SentencePiece 0.2.1 for the converter. Tokenizers 0.22.2 is pinned transitively. Store real traces locally. The eight-token native trace covers 364 FP32 tensors across the model and KV cache. Its Safetensors SHA-256 is `3a11b0820fba10d6b51ab455df84141c5e207a01bc87208d5bbff97f2b4ff941`. Use fixture schema v1, Safetensors with JSON axes/shape/dtype, fixed input IDs, artifact and fixture hashes, and pinned oracle versions. Compare semantic tensor names, shape and axes before metrics. Apply the existing provisional tolerance table and numerical reporting fields. Phase 1 freezes measured gates for native model and cache, and Phase 2 does the same for quantized comparisons.

```bash
# Python 3.12, uv and a local checkout of the pinned llama.cpp commit are prerequisites.
uv venv --python 3.12 .venv-reference
uv pip sync --python .venv-reference/bin/python --require-hashes \
  --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
  reference/requirements.lock
. .venv-reference/bin/activate
python reference/models.py fetch --output models/cache --jobs 6
python reference/models.py verify --directory models/cache
HF_MODEL_DIR="$PWD/models/cache/HuggingFaceTB/SmolLM2-135M/93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
CONTROLLED="$PWD/models/cache/controlled"
mkdir -p "$CONTROLLED" fixtures/local "$PWD/build/reference/hf"
CONVERTER_INPUT="$PWD/build/reference/hf"
for file in config.json generation_config.json special_tokens_map.json tokenizer.json tokenizer_config.json model.safetensors vocab.json merges.txt; do
  ln -s "$HF_MODEL_DIR/$file" "$CONVERTER_INPUT/$file"
done
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
for type in f32 f16 bf16; do
  OMP_NUM_THREADS=4 python "$LLAMA_SOURCE/convert_hf_to_gguf.py" \
    "$CONVERTER_INPUT" --outfile "$CONTROLLED/SmolLM2.$type.gguf" --outtype "$type"
done
llama-quantize "$CONTROLLED/SmolLM2.f32.gguf" "$CONTROLLED/SmolLM2.Q8_0.gguf" Q8_0 4
llama-quantize "$CONTROLLED/SmolLM2.f32.gguf" "$CONTROLLED/SmolLM2.Q4_0.gguf" Q4_0 4
python reference/audit_models.py --directory models/cache --controlled "$CONTROLLED" \
  --output fixtures/local/manifest-audit.json
python reference/generate_fixtures.py --model "$HF_MODEL_DIR" --output fixtures/local
python reference/gguf_oracle.py --model "$HF_MODEL_DIR" --published models/cache \
  --controlled "$CONTROLLED" --output fixtures/local/gguf-oracle.jsonl
python reference/check_untied.py --model "$HF_MODEL_DIR" \
  --controlled-f32 "$CONTROLLED/SmolLM2.f32.gguf" --output-file "$CONTROLLED/untied.f32.gguf"
```

For the block comparison, build the pinned llama.cpp source with `cmake -S "$LLAMA_SOURCE" -B build/reference -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_COMMON=OFF`, then build `ggml-base`. Compile `reference/ggml_blocks.c` with `cc -shared -fPIC`, the source's `ggml/include` and a link and runtime search path for `build/reference/bin` with `-lggml-base`. Pass the resulting shared object to `reference/check_gguf_blocks.py --bridge` with the same model cache and controlled directories. For the tokenizer and timing probes, compile `reference/llama_probe.cpp` with `g++ -std=c++20 -O2`, include `llama.h` and `ggml.h`, and link `-l llama -l ggml-base` from the pinned external installation. `reference/audit_tokenizers.py --probe` accepts that executable and uses vocab-only file loads. `reference/audit_unicode.py` accepts each pinned UnicodeData.txt.

#### Fixed performance protocol

Use token IDs `17 + (i * 7919 + 12345) % (49152 - 17)` for successive positions. The pilot uses 128 prompt plus 32 teacher-forced IDs, capacity 512 where the engine preallocates, one active sequence and FP32 K/V. Request only the final logits of each prefill or decode operation, two workload warmups and seven measured repetitions, with separate prefill and decode timings. Use controlled F32 GGUF for llama.cpp and the equivalent native BF16 expanded to FP32 for PyTorch. Force CPU and disable flash attention in llama.cpp. Torch's DynamicCache grows and copies during decode, and implementation-specific kernels and packing can differ. Model-load timing is a separate warm-filesystem observation. Record affinity, wall-clock samples, median, p95 and median absolute deviation before comparing results. Capture cold-file timings only under reproducible cold-cache conditions, and capture RSS and owned/mapped allocations separately; the pilot has no warm/cold load comparison.

For later gates keep the prompt/decode/context/thread matrix already specified. Pin one thread per physical core first, then measure SMT separately. Use the same affinity and model for comparison and run one workload at a time on an idle host. Treat coefficient of variation above 5% or a median shift above 5% across two repeated batches as unstable and report samples before retesting. Performance targets must exceed measured run-to-run variability and are frozen after the Phase 1 baseline, before Phase 2/4 optimization.

| CPU reference | Threads | Median prefill tokens/s | Median decode tokens/s | Within 5% variation in both timings |
| --- | ---: | ---: | ---: | --- |
| llama.cpp | 1 | 233.1 | 37.0 | Yes |
| llama.cpp | 4 | 525.5 | 57.8 | No |
| PyTorch eager | 1 | 315.2 | 22.0 | No |
| PyTorch eager | 4 | 724.7 | 22.2 | No |

The pilot ran with `taskset -c 0` for one thread and `taskset -c 0,2,4,6` for four threads. Use `llama-probe bench "$CONTROLLED/SmolLM2.f32.gguf" THREADS` and `python reference/benchmark_cpu.py --model "$HF_MODEL_DIR" --threads THREADS`. GNU `time -f '%M'` recorded process peak RSS for each run. A single-core copy of 256 MiB buffers measured 29.1 GB/s median of logical read plus write traffic. Reproduce with `c++ -std=c++20 -O2 reference/bandwidth.cpp -o build/reference/bandwidth` and `taskset -c 0 build/reference/bandwidth`. The copy metric does not measure hardware DRAM traffic. The shared host and observed variance make this pilot descriptive. [Raw samples and run metadata](benchmarks/phase0.json)

Acceptance criteria

- Reproducible native and same-artifact oracles and the complete pinned inventory establish feasibility. Phase 1 measures C++ numerical and scalar performance baselines. Release acceptance still requires every artifact and claimed backend path to pass the corresponding Nathr tests.

Status
- Complete

Completion notes
- Completed the full pinned artifact audit, the native eight-ID trace, 55 GGUF reference forwards, 22 block checks and 13 tokenizer cases per GGUF fingerprint. Production source and build work starts in Phase 1.
- Replayed all five controlled conversions and reproduced their files byte for byte. Replayed the native Safetensors trace and both small fixtures with identical bytes. The offline verifier checked 59 required input files, and the isolated environment passed `uv pip check` with 27 packages.

Deviation notes
- Local LeakSanitizer is unavailable under tracing, and the disposable TSan probe fails at process startup with an unexpected memory mapping. The ASan/UBSan probe ran with leak detection disabled. Phase 1 needs a runnable TSan environment before making any race-free claim. Hardware counters were denied at `perf_event_paranoid=4`.

### Phase 1: Reference inference engine (v0.1)

Checkpoint A - first correct token-ID logits (prioritize over tokenizer, CLI polish and acceleration):
- [x] Establish minimal CMake library/test targets, baseline C++20, explicit CTest assertions, compiler warnings, offline build, dev/release presets and initial GCC/Clang CI. Add only required source files.
- [x] Implement checked arithmetic, bounded byte reader, dtype/layout traits, RAII storage and views. Implement validated Safetensors/config parser and semantic weights, F32/F16/BF16 conversion and tied-weight ownership. Start with buffered loading; mmap only if useful after correctness.
- [x] Implement narrow backend boundary and shared CPU storage with independent scalar operations. No general tensor algebra, graph executor or device abstractions without a demonstrated use.
- [x] Implement FP32 scalar embedding, linear, RMSNorm, RoPE, stable softmax, GQA, SiLU/SwiGLU, residuals and final output using one model execution path. Bounded attention scratch and explicit None/LastToken/AllTokens logits policy from the start. Test each operation with independent FP64/analytical oracle and invariants.
- [x] Execute tiny synthetic GQA and real canonical checkpoint on fixed integer IDs without tokenizer or cache. Export short traces and compare per-op/per-layer/final logits against pinned PyTorch. Freeze justified native numerical gates before optimization; log tolerance revisions and worst-index diagnostics.
- [x] Validate floating storage variants against identical converted weight references; use native half-width weight storage in ordinary execution. Verify model inspection and memory accounting, including mapping lifetime if implemented.

Checkpoint B - generation and baseline:
- [x] Implement pinned tokenizer pipeline, special tokens, UTF-8 byte streaming and frozen fixtures; differential-test stages and random valid Unicode against reference. Define explicit raw-byte and invalid-input policy.
- [x] Implement FP32 contiguous KV cache, session reset/invalid state and position/logit/consumption semantics. Sequential prefill uses shared transformer and avoids intermediate vocabulary projections. Compare cached/uncached and partitioned execution with PyTorch; test capacity and failure paths, including 8192 context.
- [x] Implement separate sampling and generation, deterministic greedy/tie rules and seeded top-k/top-p/temperature, EOS and token accounting. Test terminal and continuation behavior.
- [x] Provide minimal `nathr inspect`, `tokenize`, `forward`, `generate`, `bench` CLI and public library. Validate multiple sessions, allocation bounds, ownership and model-free offline build; add sanitizers and basic malformed-file tests.
- [x] Measure actual scalar prefill/decode, memory and allocation baseline using Phase 0 protocol; set quantitative end-to-end targets and noise thresholds before Phase 2 SIMD. Record measurements and provisional performance decisions in ledger.

Acceptance criteria:
- Checkpoint A yields independent numerical parity before optional infrastructure is added. v0.1 loads canonical native checkpoint, generates from text/IDs, matches reference tokenization and FP32 math/cache gates, uses bounded memory and has reproducible scalar benchmarks. No ML runtime dependencies.

Status:
- Complete for the scalar implementation and reference validation. Shared-host throughput remains informational until the repeatability gate passes.

Completion notes:

- The buffered single-file Safetensors loader validates JSON and tensor layout before allocating the full encoded buffer. F32, F16 and BF16 rows stay in original storage and expand to FP32 values inside scalar operations. Tied embeddings alias storage, and an explicitly materialized tied head must match the stored embedding bytes. A session owns its preallocated FP32 KV cache, workspace and RNG. Uncached forward supports up to 128 positions. Cached prefill uses the same layer operations one token at a time, and validates requests against the full context before mutation.
- Model-free CTest uses checked-in PyTorch logits from a two-layer synthetic GQA model with exact F32/F16/BF16 equivalents. It checks malformed inputs, partial execution failure, cached partitions, reset and ownership, simultaneous sessions, kernel calculations against FP64 and analytical invariants, and sampling and tokenizer semantics. The standalone long test reached absolute position 8191 and verified reset and overflow behavior. The real allocation probe confirmed a 360 MiB cache at the full context and zero C++ allocations in warmed prefill plus four appends.
- [The reference verifier](reference/phase1_verify.py) captured 483 named FP32 tensors per mode on the pinned eight-ID prompt. Cached and uncached traces were bit-identical. F32 models with exactly expanded BF16 weights also produced the same C++ trace bit for bit. F16 was compared with the same independently rounded weights in eager FP32 PyTorch. The worst tensor relative L2 was `3.1503e-5` for BF16 and F32 at `layer.12.down[2,507]`, and `3.3672e-5` for F16 at the same site. Native logits had relative L2 `3.7791e-6` and max absolute error `3.2521e-4` at `[1,880]`. The declared model gates remain relative L2 ≤ 1e-4 and cosine ≥ 0.99999, with exact greedy IDs on this fixture. These gates apply to propagated trace errors. Individual kernel tests use independent reduction-aware or analytical checks. At 64 real tokens, cached and uncached final logits were identical, with relative L2 `6.1815e-7` against PyTorch. Both text and token-ID greedy generation produced the same eight IDs as PyTorch for “Once upon a time”. The real traces and metrics are generated into the explicit local output directory.
- The tokenizer matched the fast HF oracle on 2,146 inputs, including the Phase 0 examples, every ASCII byte, 2,000 seeded Unicode strings and separate emoji-fragment decode checks. The pinned vocabulary lacks 21 initial byte symbols, including six legal ASCII controls. The implementation drops unavailable symbols at the BPE model stage because the checkpoint declares neither an unknown BPE token nor byte fallback. One MiB bounds individual text and decode requests, and runtime text generation never adds a BOS token without `--bos`.
- Local GCC 13.3 and Clang 23.1 Debug/Release builds and the format target were exercised. The release executable links to libc, libm, libgcc_s and libstdc++ plus the platform loader. The model-free workflow is defined for GCC and Clang with separate sanitizer jobs. The synthetic weights, expected logits and Unicode tables reproduced byte for byte. The sandbox exposes no Git metadata or configured hosted runner, so the workflow has local equivalents here.
- The Clang Release binary reproduced all 483 real-checkpoint GCC trace tensors bit for bit.

#### Scalar performance baseline

Both full batches used the pinned BF16 weights, scalar FP32 compute and cache, CPU 0, 128 prompt IDs, 32 teacher-forced decode IDs, capacity 512, two warmups and seven samples. One session reserved 23,592,960 bytes of KV, 224,000 bytes of numeric workspace and 2,048 bytes of token history. The model owned 269,060,552 file bytes, including 269,030,016 encoded weight bytes. Model weights used buffered storage and zero mapped bytes. The verified executable was 587,976 bytes and 493,896 bytes stripped. [Raw samples and accounting](benchmarks/phase1.json)

| Batch | Median prefill tokens/s | Median decode tokens/s | Prefill CV | Decode CV | Peak RSS (KiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.81 | 2.13 | 46.3% | 42.1% | 290,572 |
| 2 | 3.31 | 2.25 | 32.5% | 28.4% | 290,516 |

Prefill and decode medians drifted by 15.1% and 5.5%. Treat both full batches as descriptive. The host was shared and its thermal and frequency behavior was uncontrolled. The loader changed between invocations to validate the header before reading the payload. The scalar compute path was unchanged, and model-load time is reported separately. A 32-prompt, 32-decode control on the final binary also exceeded the within-batch limit with 51.6% prefill CV and 50.0% decode CV. The optional second short batch was stopped once the short workload had failed the gate.

Before Phase 2 SIMD comparisons, repeat the 128/32/512 workload for controlled Q8_0 and Q4_0 against their own packed scalar baselines. Use the same encoding, fixed IDs, physical core, FP32 cache, and last-logits policy in both paths. Keep the Phase 0 repeatability limits of ≤5% coefficient of variation within each batch and ≤5% median drift between two batches. Require ≥10% higher median decode throughput and at most 5% loss of median prefill throughput for the same artifact. When blocked prefill replaces repeated GEMV in Phase 4, require at least 10% higher median prefill throughput on the matched representative workload, with the same repeatability limits. The relative gates exceed the permitted noise and use measurements from an equivalent scalar implementation.

Deviation notes:

- Phase 1 supplied model-free tests and external same-weight comparisons. Work on GGUF loading, SIMD kernels, cache FP16, thread scheduling and broad release CI stays in the later phases.
- The Phase 0 TSan startup problem was overcome in the local test process using `TSAN_OPTIONS=halt_on_error=1 setarch x86_64 -R ctest --preset tsan`. The test covers simultaneous independent sessions sharing a model. LeakSanitizer still fails under local tracing. ASan/UBSan passed with `ASAN_OPTIONS=detect_leaks=0`. Valgrind Memcheck independently reported zero errors and no leaks for the offline suite.
- The full-workload throughput did not meet the Phase 0 repeatability threshold. Rerun a stable matched scalar benchmark on controlled hardware before treating those numbers as an optimization baseline. Hardware counter access remains unavailable under the observed `perf_event_paranoid=4`.

### Phase 2: Quantized vertical slice (v0.2)

Deliverables:
- [x] Implement GGUF structural parser/inspector separately from semantic binder; validate v3 metadata, lengths, alignment, row/block geometry and mixed dtypes; handle required split/format variants. Inspect every artifact even when its encoding lacks execution support.
- [x] Implement floating GGUF loading and proven HF/GGUF layout and tokenizer mappings. Resolve distinct output weights explicitly and independently verify PyTorch reconstruction after initialization.
- [x] Implement centralized quant format registry and Q8_0/Q4_0 block decoders and bounded scalar GEMV/embedding/output paths. Include any additional encodings needed for the controlled vertical-slice artifacts. Validate independent blocks and same-artifact PyTorch inference; freeze quantized parity gates before optimizing.
- [x] Compile separate x86 kernel translation units and baseline dispatch, CPU and OS-state checks, forced scalar and supported ISA controls, fallback diagnostics. Test simulated feature combinations and unsupported ISA rejection. AVX2 does not imply FMA/F16C.
- [x] Implement initial dense and fused Q8_0/Q4_0 GEMV with FP32 activations, using selected ISA only where supported. Validate equivalence to scalar with reduction-aware tolerances. Keep scalar fallback if actual hardware lacks extension; document test limitation rather than claim acceleration.
- [x] Benchmark the full vertical slice, including dequantization, embedding, output projection, decode and storage. Confirm kernel registry, precision policy, weight and session ownership work without duplicating weights or model code. Revise backend boundary only if concrete evidence requires it; document deviations before extending to other formats.

- [ ] Qualify the ≥10% decode gain and ≤5% prefill regression limits with stable matched pairs for both artifacts.

Acceptance criteria:
- Controlled floating GGUF and quantized slice execute end to end with independently verified math, packed storage, tokenizer and dispatch. Tested acceleration meets declared slice gate on supported hardware; unsupported hardware is reported and not silently passed. Architecture is proven with a small quantized workload before broad implementation.

Status:
- GGUF execution and correctness validation are complete. The user approved proceeding to Phase 3 with performance qualification deferred. The Phase 2 timing gate remains open.

Completion notes:

- The structural GGUF inspector matched the manifest across 50 published files and five controlled conversions. It checked physical extents, logical row geometry, tensor types and byte totals independently of execution. All 21 published encoding IDs plus controlled BF16 have structural geometry in the format registry. The frozen files are single-file little-endian v3. Split metadata is exposed by inspection, and split execution is unsupported in the frozen scope. Unknown type IDs are rejected instead of guessing their byte widths.
- A checked read-only mapping owns packed GGUF weight bytes. Loading never expands the full model or allocates a model-sized duplicate. The shared Llama binder reorders Q/K projection outputs into canonical split-half coordinates when writing results, preserving serialization. The backend keeps the Phase 1 cache and RMSNorm paths. A present output tensor is bound independently. The untied fixture with a negated embedding as its distinct output head matched its independently reconstructed PyTorch model.
- Both audited GGUF tokenizer fingerprints passed 2,146 exact cases each, plus four raw-byte fragment checks per fingerprint. The sidecar tests changed the fingerprint, verified acceptance with the canonical sidecar, and checked rejection of an incorrect sidecar hash and an embedded vocabulary mismatch. Text generation uses the same ByteLevel and UTF-8 policies as Phase 1.
- All six execution fixtures passed on both scalar and AVX2, with 483 traced tensors per cached or uncached path. The golden numerical gates remain relative L2 ≤ 1e-4 and cosine ≥ 0.99999 with exact greedy IDs on the short fixture. The largest scalar-vs-PyTorch relative L2 was `6.1659e-5` for Q8_0 and `7.7282e-6` for Q4_0. The largest AVX2-vs-scalar differences were `1.8851e-5` for Q8_0 and `2.8729e-6` for Q4_0. Each backend's cached and uncached traces matched bit for bit. The reduction-aware kernel tests use independently decoded Q4_0 and Q8_0 blocks, with negative scales, subnormals, unaligned inputs and dense tails.
- AVX2 GEMV is built in its own translation unit. Baseline CPUID and OSXSAVE/XCR0 checks precede dispatch. F16C alone governs the extra F16 conversion requirement for the AVX2 path. FMA is detected independently and unused in this phase. Forced unsupported AVX2 yields a typed error, and inspection and benchmarking report the selected kernels. The Q4_0 recipe still exercises a Q8_0 tied embedding and output projection.
- The new scalar binary reproduced the full Phase 1 native BF16 trace byte for byte. The GCC and Clang Debug and Release test matrix passed. ASan/UBSan passed with leak detection disabled and TSan passed with process-local ASLR disabled. Valgrind reported zero errors and leaks for the model-free suite. The tiny GGUF and packed-block fixtures reproduced byte for byte. Hosted CI remains unverified here.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python reference/phase2_verify.py \
  --binary build/release/nathr --model "$HF_MODEL_DIR" --published models/cache \
  --controlled "$CONTROLLED" --untied "$CONTROLLED/untied.f32.gguf" --output fixtures/local/phase2
python reference/phase2_bench.py --binary build/release/nathr --probes build/release/nathr_tests \
  --controlled "$CONTROLLED" --cpu 12 --output benchmarks/phase2-local.json
```

#### Matched performance measurements

The release binary ran the pinned Q8_0 and Q4_0 artifacts on logical CPU 12, physical core 6. Each pair used 128 prompt IDs, 32 teacher-forced decode IDs, context 512, one thread, FP32 KV, last-position logits and the same token sequence. Each batch had two warmups and seven measured samples. Scalar and AVX2 used the same stored weights within each pair. [Raw samples, resource accounting and kernel measurements](benchmarks/phase2.json)

| Artifact | Backend | Prefill tokens/s | Decode tokens/s | Prefill CV | Decode CV | Peak RSS (KiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Q8_0 | scalar | 5.98 | 4.48 | 16.2% | 16.6% | 173,388 |
| Q8_0 | avx2 | 14.96 | 11.12 | 31.6% | 28.3% | 173,404 |
| Q4_0 | scalar | 5.56 | 4.55 | 15.1% | 38.6% | 121,564 |
| Q4_0 | avx2 | 21.25 | 17.23 | 15.2% | 20.4% | 121,560 |

| Artifact | Dequantize 8,192 blocks (ms) | Look up 2,048 embeddings (ms) | Down projection scalar / AVX2 (ms) | Output projection scalar / AVX2 (ms) |
| --- | ---: | ---: | ---: | ---: |
| Q8_0 | 0.103 | 0.687 | 1.832 / 0.210 | 60.135 / 15.485 |
| Q4_0 | 0.074 | 0.842 | 1.181 / 0.181 | 47.511 / 7.982 |

Component figures are sample medians. The allocation probes recorded zero C++ heap allocations in warmed prefill and append on both backends and both artifact formats. The Q4_0 down projection used Q4_0 and its tied output used Q8_0. The files stayed mapped in their original packed representation.

Deviation notes:

- The inherited Phase 1 repeatability issue persisted on physical core 7 (logical CPU 14). The prior binary's 32/32/512 retest had 45.4% prefill CV and 44.0% decode CV. CPU time was close to wall time. The host continued to use the observed `powersave` governor, and this evidence does not isolate the cause of the variance. [Phase 1 timing evidence](benchmarks/phase1.json)

- Every measured end-to-end batch exceeded 5% variation in both prefill and decode. The harness stopped after each first matched pair once the within-batch requirement failed. The throughput values are descriptive. Performance qualification still requires two stable pairs for each artifact on a controlled host under the unchanged improvement and repeatability gates.

### Phase 3: Complete frozen quantization coverage (v0.3)

Deliverables:
- [x] Implement remaining actual encodings (legacy, K-quants, I-quants and other required IDs) in priority order determined by encoded bytes, parameter distribution, actual model execution and profiling, not filename alone.
- [x] For each encoding: define exact block/row contract, provenance and independently generated fixtures; validate scale/minimum reconstruction, nibble/sign/codebook ordering, boundaries, half scales, and valid generated adversarial blocks. Separate malformed-byte robustness tests. Add scalar packed linear, embedding and output support as applicable.
- [x] Extend controlled artifact suite where feasible and validate every published manifest artifact against its independent dequantized PyTorch oracle, with exact tokenizer policy, canonical Q/K layout and separate output weights when present. External llama.cpp is an additional interoperability check, not the sole oracle.
- [x] Publish machine-readable coverage report: artifact hash, recipe, actual encoding counts/parameters/bytes, verified reference source, numerical metrics, memory, tokenizer status and selected/fallback kernels. Block on every unresolved required artifact; do not remove without user approval.
- [x] Evaluate quality on pinned fixed token corpus separately from correctness: teacher-forced NLL/perplexity, divergence and top-k; distinguish controlled versus published variants and mask overlapping tokens. Stream logits by chunk.

Acceptance criteria:
- Every required frozen published artifact and controlled test artifact has a verified execution path without full-model dequantization. Every required actual encoding has independent block and end-to-end validation; scalar fallback remains acceptable. All numerical and compatibility blockers are resolved or phase remains blocked.

Status:
- Complete. All 50 published files and eight controlled conversions qualified.

Completion notes:

- The frozen inventory put IQ4_NL first among the remaining formats by encoded footprint (1,127,208,960 bytes across the published files), followed by Q5_1 (465,813,504 bytes) and Q5_0 (433,382,400 bytes). The 32-element formats cover most width-576 matrices. The registered 256-element formats bind only on compatible row widths, including width-1536 down projections. Phase 3 adds all 17 remaining scalar decoders, so execution covers all 21 published encoding IDs plus controlled BF16.
- All 136 new synthetic packed blocks were decoded bit for bit by Nathr, pinned gguf-py, and the pinned GGML C bridge. Model-free tests exercise unaligned storage, two-row GEMV and embeddings, finite half scales including subnormals and signed zero, scale/minimum planes, signed payloads, codebook indexes and reduction-aware FP64 bounds. The previous Q4_0 and Q8_0 cases remain in the same offline suite. Malformed-file cases keep structural validation separate from valid generated blocks.
- FP64 intermediates are now the common scalar and AVX2 policy for sensitive operations. GEMV rounds once per output row. Attention accumulates its scores, normalization denominator and weighted values in FP64, with explicit FP32 staging before softmax and at the activation and cache boundaries. This replaces the earlier Q5_1-specific reduction. A cancellation regression covers scalar and SIMD with a small term that FP32 summation discarded.
- The codebook generator verified all 4,608 grids, 128 parity sign masks and 16 IQ4_NL levels against the independent gguf-py representation. It extracts table data only from the SHA-verified MIT header at the pinned llama.cpp revision. The runtime retains the notice alongside the generated header. [Block and row contracts](docs/quantization.md)
- The controlled suite adds Q5_1, Q4_K_M and IQ4_NL from the original pinned F32 GGUF using the pinned quantizer with three threads and no importance matrix. Each output reproduced byte for byte. The original published inventory and its checksums are retained.
- The GCC and Clang Debug and Release builds passed the offline suite. GCC ASan/UBSan and separate TSan passed with the existing process-local ASLR workaround. LeakSanitizer was disabled locally. Valgrind Memcheck reported zero errors and no outstanding allocations. The original Phase 1 and Phase 2 PyTorch verifiers also passed on the revised binary, including native BF16, exact expansion to F32, the F16 variant, generation and the separate-output GGUF. Phase 1's largest short-trace relative L2 was `1.6893e-5`. The six Phase 2 execution fixtures reached at most `1.8666e-5`. A warmed run on i1-Q4_K_M allocated zero heap objects across 128 prefill tokens and four appends and verified the 360 MiB full-length FP32 KV cache.
- All 58 GGUFs passed hash, structure, tokenizer and execution checks. Each of the four scalar/AVX2 and cached/uncached paths had zero measured tensor error across all 483 tensors per model against the independent PyTorch reference with FP64 intermediates and FP32 activation boundaries. All greedy IDs also matched the original eager FP32 oracle and the published manifest. The two audited tokenizer fingerprints each passed 2,146 cases and four UTF-8 fragment checks. Peak RSS was 98,476 to 567,940 KiB. All 58 used mapped file weights with zero owned file bytes. The pinned llama.cpp probe agreed on 355 of 464 greedy positions. [Coverage report](models/phase3-coverage.json)
- The fixed corpus has 112 target tokens across three documents. Descriptive quantization quality still uses eager FP32 PyTorch with independently decoded weights, 32-token windows, stride 24 and a mask over eight overlapping targets where applicable. The native perplexity of `40.3773` and all artifact quality metrics reproduced exactly. The three new controlled models also matched the rounded reference on each scored row in the streamed-logit check. [Corpus fixture](fixtures/phase3/corpus.json)

```bash
# The source, C bridge, reference environment and F32 conversion are from Phase 0.
python reference/generate_iq_codebooks.py "$LLAMA_SOURCE/ggml/src/ggml-common.h" \
  /tmp/nathr-iq-codebooks.hpp --verify-oracle
cmp /tmp/nathr-iq-codebooks.hpp src/quantization/iq_codebooks.hpp
python reference/generate_phase3_fixtures.py --bridge "$GGML_BRIDGE" \
  --tokenizer "$HF_MODEL_DIR/tokenizer.json" --output fixtures/local/phase3-fixtures
diff -qr fixtures/phase3 fixtures/local/phase3-fixtures

EXTENDED="$PWD/models/cache/phase3-controlled"
mkdir -p "$EXTENDED" fixtures/local/phase3
for type in Q5_1 Q4_K_M IQ4_NL; do
  llama-quantize "$CONTROLLED/SmolLM2.f32.gguf" "$EXTENDED/SmolLM2.$type.gguf" "$type" 3
done
# Rebuild the Phase 0 llama.cpp probe from reference/llama_probe.cpp with its forward mode.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python reference/phase3_verify.py --binary build/release/nathr --native "$HF_MODEL_DIR" \
  --published models/cache --controlled "$CONTROLLED" --extended "$EXTENDED" \
  --llama-probe "$LLAMA_PROBE" --work fixtures/local/phase3 --output fixtures/local/phase3.json
```

Deviation notes:
- The user explicitly authorized deferring performance qualification and proceeding with Phase 3. The Phase 2 thresholds and raw measurements remain unchanged. Phase 3 covers correctness, compatibility and descriptive quality.
- Under the earlier FP32 accumulation policy, i1-Q4_K_M differed from the eager oracle by `3.1175e-4` in AVX2 logits and i1-Q4_K_S differed by `6.2035e-4` at the final layer output. For i1-Q4_K_M, the first three rows of an eager FP32 three-token run alone differed from the eight-token run by `4.0851e-4` relative to the eight-token logits norm. For the full eight tokens, eager FP32 logits differ from the rounded oracle by `5.2466e-4`. These comparisons showed that eager FP32 batching and summation were sensitive enough to exceed the gate on identical weights.
- Phase 3 now applies the same FP64-intermediate policy to every runtime artifact and verifies it against the independently evaluated PyTorch operations in `reference/rounded_oracle.py`. Activations and KV storage remain FP32. Relative L2, cosine, exact greedy, cache and ISA gates retain their original limits. The report preserves the eager FP32 discrepancy for every artifact, including three above the layer or logits limit. `reference/phase3_verify.py` accepts repeated `--artifact FILENAME` arguments for targeted checks and exits nonzero for any required gate failure.

### Phase 4: Performance engineering (v0.4)

Deliverables:
- [x] Profile full model and hot kernels using controlled workloads and roofline estimates; record memory bandwidth/compute measurements where available. Confirm per-workload optimization targets before changing kernels.
- [x] Implement persistent standard-library CPU thread pool with size-dependent thresholds, bounded workers, static row partitioning and explicit concurrency/shutdown/error behavior. Test worker lifecycle, race safety and multi-session policy; measure physical cores versus SMT.
- [x] Extend dense/quantized fused GEMV for hot encodings and eligible x86 features; preserve numerical policy and reported fallback. Avoid optimizing every format by default. Optional AVX-512/VNNI only with actual validation hardware and demonstrated benefit.
- [x] Implement bounded chunked prefill and narrow blocked GEMM for actual model shapes; dense and selected quantized paths reuse weights across prompt tokens. Test past-length masks, arbitrary chunks and odd tile sizes against reference. No generic BLAS library.
- [x] Optimize attention cache access/online softmax only if beneficial, keeping bounded memory and numerical invariants. Include long-context tests.
- [x] Implement FP16 KV cache with explicit rounding and baseline attention arithmetic. Compare against the rounded-cache PyTorch reference and separately measure quality relative to FP32 cache storage. Verify exact memory allocation accounting and 180 MiB full-context storage.
- [x] Implement and evaluate selected integer dot-product kernels for Q8_0 and/or Q4_0, using transient quantized activations. Validate against independent equivalently quantized reference; check signedness, saturation, overflow, scale corrections and quality separately. Retain only beneficial, validated kernels; expose as explicit precision policy until evidence justifies default. Record negative results rather than forcing approximation.
- [ ] Demonstrate end-to-end decode and prefill improvements over scalar/repeated GEMV on predefined workloads, compare to external engines with matching tokens/context/cache/threads/logit policy. Investigate largest gaps. Verify no model-sized duplicate, no quadratic score/output allocations, no hot-path heap allocation, and bounded scratch budgets. Document performance and any regressions with dispatch rules.

Acceptance criteria:
- Required optimized prefill/decode gates meet measured improvement beyond benchmark variability, with no correctness regression. All frozen artifacts retain explicit supported paths; FP16 cache is independently validated; integer experiments have documented evidence and are enabled only if justified. Memory and concurrency requirements pass. Unmet performance gates are recorded as blockers or explicit user-approved deviations.

Status:
- Implementation and correctness validation complete. Performance qualification is blocked by measured variability. The earlier Phase 2 timing deferral is tracked separately from its completed correctness work.
- The Phase 5 retry repeated the exact Q8_0 workload on the recorded Phase 3 and Phase 4 binaries. First-batch prefill/decode CV was 34.47%/39.61% on the one-core AVX2 baseline, 23.19%/28.06% for four-core repeated GEMV and 31.21%/44.71% for four-core blocked prefill. Each pair exceeded the 5% limit. Q8_0 alone established the mandatory gate remained blocked on this host.

Completion notes:
- The Phase 3 release executable and component probe were saved before optimization. A one-core `32/8/512` pilot on CPU 12 with one warmup and three measured runs gave 24.68/19.83 prefill/decode tokens/s for F32, 25.67/19.92 for Q8_0 and 23.77/19.04 for Q4_0. Cached weights exceed the shared 16 MiB L3. A 256 MiB-buffer copy gave 20.21 GB/s median logical traffic. Hardware counters remained unavailable with `perf_event_paranoid=4`. These exploratory results select dense, Q8_0 and Q4_0 kernels for tuning. The matched `128/32/512` acceptance workloads retain the 5% repeatability and drift limits and 10% prefill and decode improvement targets for the representative comparisons.
- A simple decode model uses two floating-point operations per stored weight, one logical read of the encoded weights per token and excludes KV reads, temporary activations and decoding work. For the controlled F32, Q8_0 and Q4_0 files, this gives roughly 0.50, 1.88 and 2.99 FLOP per encoded byte. The pilot's equivalent logical weight rates are 10.67, 2.85 and 1.71 GB/s respectively. Comparison with the copy rate motivates reuse for dense prefill and reducing per-block compute for quantized decode. These rates are estimates from throughput and artifact geometry.
- One thread pool belongs to each model. The calling thread shares contiguous row ranges with up to 63 workers. Work below 16 output rows or 32,768 estimated products per participant stays on the caller. Independent sessions own scratch and cache, and large projections serialize at the shared pool boundary. The pool drains each submission before propagating a worker's exception, reuses workers after a failed job and joins them at model destruction. CTest exercises simultaneous submissions, failure recovery and thread shutdown. The TSan run passes on the local process-ASLR workaround. The Q8_0 pilot gave 135.24 prefill and 52.78 decode tokens/s on four physical cores, and 83.83 and 36.86 on four logical CPUs across two physical cores.
- Cached prefill and multi-token append use chunks of up to 32 tokens, with 16 by default and one for repeated GEMV. The blocked GEMM packs at most 16 input tokens per tile and decodes weight blocks once per row per tile. Scalar iterations and the selected AVX2 kernels keep FP64 reductions and FP32 row outputs. For masked attention, Q/K/V for a whole chunk can be written before attention because each query reads only through its absolute position. CTest checks odd chunks, split append, row/tile tails and both KV formats at the 8,192-token boundary. All 58 frozen artifacts passed the four-thread Phase 3 numerical matrix with zero measured error on all 483 tensors per path. Original native and separate-output reference tests remain in the regression run.
- FP16 stores the two banks as binary16 with round-to-nearest-even and reads them as FP32 for the established attention math. The real allocation probe checked 188,743,680 bytes at 8,192 tokens, half the FP32 size. The warmed four-thread FP16 run recorded zero C++ allocations across 128 prefill tokens and four appends, including worker allocations. The independent PyTorch cache applies the same storage boundary for native, Q8_0 and Q4_0. All 483 tensors per tested path and all 112 streamed corpus logits per model matched exactly. FP16 versus FP32 cache top-1 agreement was 100% on that corpus and mean KL divergence was at most `4.93e-7`.
- Both CPU execution policies share the same attention arithmetic. The value accumulation tile grew from eight to 64 elements to reduce repeated cache-row visits, with a bounded score vector and no full attention matrix. The narrow-tile comparison checked exact outputs at lengths 128, 2,048 and 8,192.
- The Q8_0 int8 experiment quantizes activations in signed 32-element blocks, sign-extends before multiplying, uses unsaturated int32 sums and applies input and weight scales in FP64. The AVX2 and scalar code sums matched at signed extrema and for every row of the real projections. An independently quantized PyTorch model matched all 483 tensors for both tested chunk sizes. On the 112-target corpus, int8 versus FP32 activation top-1 agreement was 91.96% and mean KL was `0.009248`. The experimental whole-model path reduced prefill throughput in both batches and exceeded the timing variability limit. FP32 activations remain the production policy. [Experimental source patch](reference/q8-int8-experiment.patch)
- The 64-value attention tile took 0.184, 2.852 and 17.448 ms at the respective tested lengths, compared with 0.361, 6.198 and 30.285 ms for the eight-value tile. The one-core exact GEMV component samples were mixed. Q8_0 output projection took 15.07 ms in the post-change probe versus 9.45 ms in the pre-change probe, and Q4_0 took 12.02 ms versus 11.86 ms. End-to-end timings and individual samples are retained separately.
- Release and Debug CTest passed with GCC and Clang. ASan/UBSan and TSan passed with the existing local options, including disabled LeakSanitizer and the process-local ASLR workaround. Valgrind reported zero errors and no outstanding allocations. The release executable links only to the ordinary C and C++ system libraries. The representative four-thread FP32 `128/32/512` runs retained packed mappings with zero owned file bytes and accounted for 702,464 workspace bytes, 23,592,960 cache bytes and 2,048 history bytes per session.

#### Matched timing evidence

The baseline uses Phase 3 AVX2 on CPU 12. Repeated GEMV and blocked prefill use the Phase 4 binary on four physical cores with chunk sizes one and 16. Each batch uses the same artifact and deterministic IDs, FP32 KV, last-position logits, two warmups and seven measurements. The prefill timer ends at the first logits and excludes model load and tokenization. The pinned llama.cpp probe uses the same workload and four physical cores. Its arithmetic and kernel choices differ, particularly for quantized models. The table shows first-batch medians in tokens/s.

| Artifact | Phase 3 prefill / decode | Repeated GEMV prefill / decode | Blocked prefill / decode | llama.cpp prefill / decode |
| --- | ---: | ---: | ---: | ---: |
| F32 | 12.73 / 12.46 | 23.30 / 19.00 | 59.07 / 18.95 | 138.09 / 17.29 |
| Q8_0 | 8.10 / 5.84 | 13.34 / 13.55 | 42.30 / 16.53 | 31.64 / 4.78 |
| Q4_0 | 8.97 / 5.67 | 14.94 / 11.29 | 24.70 / 9.47 | 142.57 / 52.42 |

First-batch CV ranged from 11.56% to 48.70% across the three artifacts and Nathr paths. The F32 full repeat also failed the drift gate. Q8_0's SMT run measured 23.60 prefill and 9.89 decode tokens/s across four logical CPUs on two cores. The exact numeric equality between chunked and sequential caches makes the observed decode differences between the two Phase 4 paths a useful indicator of timing instability. The Q4_0 gap to llama.cpp is substantial in the local samples. Different activation arithmetic and kernel implementations are plausible contributors, and the unavailable hardware counters and unstable timings limit attribution. [Raw timings, correctness and memory results](benchmarks/phase4.json)

```bash
# Use the Phase 3 release executable captured before Phase 4 as PHASE3_BIN.
mkdir -p fixtures/local
python3 reference/phase4_bench.py --baseline "$PHASE3_BIN" --binary build/release/nathr \
  --controlled "$CONTROLLED" --llama-probe "$LLAMA_PROBE" --one-core 12 \
  --physical 8,10,12,14 --smt 12,13,14,15 --output fixtures/local/phase4-timing.json
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python reference/phase4_verify.py --binary build/release/nathr --native "$HF_MODEL_DIR" \
  --controlled "$CONTROLLED" --output fixtures/local/phase4-cache.json
cmake --build --preset release --target nathr_phase4_probes
taskset -c 12 build/release/nathr_phase4_probes attention
taskset -c 12 build/release/nathr_phase4_probes integer "$CONTROLLED/SmolLM2.Q8_0.gguf"
```

Use `--threads 4 --workers 3` with the Phase 3 coverage verifier for the full threaded regression. To reproduce the rejected integer experiment, apply `reference/q8-int8-experiment.patch` with `patch -p1` in a disposable copy of the source, build Release and use the experimental binary with `phase4_verify.py --integer`. In that build `bench --activation q8-int8` selects the prototype. Its matched comparison uses `--activation f32` and otherwise the same `128/32/512` command, four physical cores, two warmups and seven repetitions.

Deviation notes:
- The fixed 5% repeatability and drift limits and the 10% throughput targets remain in force. All three first batches exceeded the variability limit. Q4_0's runner stopped after the failed batch. The original Q8_0 repeat was interrupted after the first batch failed. Its subprocess saved an additional baseline sample after cancellation, which may have overlapped follow-up diagnostics. The report excludes that repeat from qualification and keeps the follow-up timing descriptive. No performance gate is marked passed.
- AVX-512 and VNNI remain outside the tested paths. FP32 activations and FP64 intermediates remain the exact reference policy. FP16 KV storage is opt-in, and the rejected Q8_0 activation approximation is reproducible as a separate experiment.

### Phase 5: Hardening, validation, packaging and release (v1.0)

Deliverables:
- [x] Run complete frozen manifest, native and controlled artifact tests, tokenizer parity/differential fuzz, scalar and all tested ISA paths, FP32/FP16 cache equivalence, long-context boundary and failure-state tests. Verify reference package/artifact hashes and fixtures.
- [x] Run GCC/Clang Debug/Release, ASan/UBSan, separate TSan, offline build and dependency audit. Run bounded coverage-guided fuzz campaigns for Safetensors, GGUF, tokenizer and quant blocks; preserve minimized regression cases. Record duration, seeds and unavailable tests.
- [ ] Run release benchmark matrix on controlled hardware, including sustained decoding, prefill, TTFT, peak RSS, mapped/owned allocations, executable size and quality results. Publish machine-readable reproducibility metadata and comparison limitations. No timing gates on shared CI.
- [x] Finalize `nathr::nathr` consumer/installation smoke test, public headers, CLI help/error handling, documentation and notices. Document precise artifact compatibility, optional sidecar requirements, kernel fallback, numerical policy and memory/CPU limits.
- [ ] Review final diff, remove unused abstractions and temporary artifacts without touching user work. Verify all top-level acceptance criteria against recorded evidence; do not label skipped mandatory checks as passed. Release/tag only with authorization.

Acceptance criteria:
- All top-level acceptance criteria have recorded evidence. Nathr is independently implemented, usable, numerically validated, resource-bounded and reproducible on the stated x86 target. Required unresolved gaps block v1.0 unless user explicitly approves a scoped deviation. No unrelated changes remain.

Status:
- Blocked for v1.0. Packaging, hardening, the frozen artifact replay and the real full-context boundary check passed locally. Sustained characterization is recorded. Controlled matched timing, hosted verification and Git-based final diff review remain open.

Completion notes:
- The offline GCC 13.3 and Clang 23.1 Debug and Release matrix passed the model-free CTest suite, including CLI and relocated static package tests. A separate GCC shared-library Release build passed the same tests. The installed CLI, CMake target, public headers and bundled notices are documented in the [README](README.md) and [architecture guide](docs/architecture.md). The CLI's direct dynamic dependencies are `libstdc++.so.6`, `libm.so.6`, `libgcc_s.so.1` and `libc.so.6`. QEMU's Nehalem profile produced the same tiny-model output under automatic and forced scalar dispatch and rejected forced AVX2.
- GCC ASan/UBSan and separate TSan passed two model-free tests each. TSan used the process-local ASLR workaround. Valgrind Memcheck reported zero errors and freed all 39,755 allocations. Four Clang ASan/UBSan libFuzzer campaigns used seed 58053 for 45 requested seconds each. Safetensors, GGUF, tokenizer and packed quantization executed 14,414, 23,446, 9,556 and 1,816,492 inputs respectively, with no findings. Initial corpora came from the checked-in tiny fixtures, tokenizer examples and packed-block fixtures. Clang static analysis, ShellCheck for literal workflow shell steps and the formatting target passed. [Validation commands](docs/validation.md)
- A clean Python 3.12 environment installed all 27 pinned reference packages offline with `uv pip sync --offline --require-hashes`. CI defines a model-free fuzz smoke job and a manual cached-model workflow with a release tier that requires stable matched timing. [C++ checks](.github/workflows/cpp.yml) and [pinned model validation](.github/workflows/validation.yml)
- The replay verified all 59 required native and published manifest files by SHA-256. The native BF16 checkpoint, F32/F16 conversions, eight-token generation, and the Phase 2 sidecar and separate-output cases passed. Across all 58 frozen GGUF artifacts and 22 encodings, four cached/uncached scalar and AVX2 paths matched all 483 short-trace tensors exactly against the rounded PyTorch oracle. The native short trace reached a maximum relative L2 of `1.6893e-5`. The native tokenizer and both GGUF fingerprints each passed 10,402 differential cases with seed 59031 and four partial UTF-8 prefix checks. The independent block fixtures, corpus and codebooks replayed byte for byte.
- Native, Q8_0 and Q4_0 FP16 cache paths matched the rounded PyTorch reference on all 483 trace tensors per path and 112 streamed corpus logits per model. FP16 versus FP32 cache storage preserved all corpus top-1 decisions and reached a maximum mean KL divergence of `4.93e-7`. Both precisions passed the synthetic 8,192-token boundary and rejected the next append. The warmed four-thread Q8_0 allocation probe recorded zero C++ allocations and checked both real full-context cache sizes.
- The real four-thread Q4_0 check completed an 8,191-token prefill and one append with FP16 KV. It used 188,743,680 cache bytes, 1,138,688 workspace bytes and 283,156 KiB peak RSS. The shared-host prefill took about 79 minutes. This one-shot check verifies the boundary without a timing gate.
- The local four-core, 128/128/512 sustained run recorded one warmup and three samples per model, including 384 inter-token latencies per model. Median prefill/decode throughput was 18.29/12.49 tokens/s for native, 28.63/10.87 for Q8_0, and 28.60/9.36 for Q4_0. Median TTFT was 7.00, 4.47 and 4.48 seconds respectively. Peak RSS was 291,164, 173,588 and 121,600 KiB. Both quantized files stayed mapped with zero owned file bytes. The GCC Release CLI was 814,880 bytes, or 698,704 bytes stripped. The fixed 112-target corpus yielded perplexity 40.38, 41.33 and 50.71 respectively. [Phase 5 measurements and validation evidence](benchmarks/phase5.json)
- Reviewed the changed workspace files and confirmed the archived Q8_0 integer experiment still applies with a dry run. The integer dispatch remains disabled in production.

Deviation notes:
- LeakSanitizer could not start under sandbox tracing. Local sanitizer and fuzz runs disabled leak detection. Valgrind independently covered the model-free suite.
- The shared-host sustained samples overlapped the Q4_0 full-context check on disjoint physical cores of the same CPU package. These samples characterize the local run. The 5% matched repeatability and the speedup gates remain open.
- The sandbox exposes empty Git metadata and no configured controlled model runner. Hosted validation and Git-based final diff review remain unverified. The installed package version is 0.4.0.
