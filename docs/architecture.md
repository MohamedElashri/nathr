# Runtime architecture

The `nathr::nathr` target exports C++20 headers and the system thread dependency. The installed executable and library use the same inference code. The standard build requires a Linux x86-64 toolchain. Python and the reference packages run only the external validation tools.

## Ownership and concurrency

`nathr::Model::load` creates a handle to immutable weight storage and a CPU pool. Copies of a model share both. The GGUF path maps the original file read-only. Keep a mapped file intact until the model and all sessions using it are destroyed. Safetensors validates its header and extents before reading the encoded payload into one owned buffer. An omitted output head shares the embedding table. A present GGUF output head is bound independently.

`nathr::Session::create` retains a model handle and owns the KV cache, scratch, history and sampling RNG. Use one session per sequence. Independent sessions can run concurrently on a shared model. The pool executes large projection jobs one at a time. Work ranges are disjoint, the caller participates, and callbacks execute on the submitting thread. A session itself requires external synchronization for concurrent access.

Prefill takes an empty session. Append takes a ready session and accepts a nonempty token span. Argument errors leave the session intact. An execution or callback failure makes the session invalid until reset. Reset discards the history and restores the starting random seed. Logit spans borrow session storage, and each callback span is valid only for the callback. All-token logits are emitted in absolute position order. Trace events describe intermediate values within each chunk and layer.

## Precision and dispatch

The packed block decoder writes at most one block of temporary FP32 weights. The core math accumulates GEMV, normalization and attention in FP64. It writes FP32 activations at operation boundaries. `--backend auto` selects AVX2 when the processor and operating system expose it. Explicit `avx2` requires support. F16 GEMV requires F16C for its AVX2 version. Other packed formats use the scalar row decoder and batch implementation, with the same rounding contract. An integer activation experiment is kept in the development references. [Packed block layouts and provenance](quantization.md)

Prefill and larger appends reuse weight blocks across a tile of up to 16 input tokens. The cache-aware attention mask reads up to each query's absolute position. The backend accumulates weighted values in fixed-size tiles and reuses one context-length score vector. There is one vocabulary-sized logits row. FP16 cache rounds post-RoPE K and unrotated V on write and converts them back to FP32 on read. [Frozen artifact manifest](../models/manifest.json)

## Runtime limits

| Resource | Limit |
| --- | --- |
| CPU pool | 1–64 participants including the calling thread |
| Uncached forward | 128 tokens |
| Session context | Model limit, at most 8,192 tokens |
| Default context | `min(2048, model context)` |
| Session prefill chunk | 1–32 tokens, default 16 |
| Per-session KV allocation | At most 512 MiB |
| Numeric scratch | At most 32 MiB |
| Weight file | At most 1 GiB |
| GGUF header and Safetensors JSON header | At most 8 MiB and 1 MiB respectively |
| Tokenizer JSON and input | At most 8 MiB and 1 MiB respectively |
| CLI trace tensor data | At most 63 MiB |

For the pinned model the KV cache at 8,192 tokens uses 360 MiB with FP32 storage or 180 MiB with FP16 storage. Model-owned, mapped-file, cache, workspace and history byte counts are distinct from process peak RSS. Parallel workers have system-managed stacks in addition to the measured session buffers.

## Artifact compatibility

The qualified set consists of the native base checkpoint, the pinned single-file little-endian GGUF v3 artifacts, and the controlled conversions in the validation reports. Source revisions, hashes, tensor types and tokenizer fingerprints are in the manifest. Known GGUF vocabularies use embedded tables with the canonical ByteLevel pipeline. A different GGUF fingerprint requires the exact pinned `tokenizer.json` as a sidecar. The sidecar verifier checks both the hash and compatibility with the embedded token and merge tables. Token-ID inference needs no tokenizer.

Dependency notices for the JSON parser, Unicode data and GGML codebook data are installed under the package documentation directory. Model files and their source notices are distributed by their respective repositories.
