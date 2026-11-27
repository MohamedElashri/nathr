# Build a small LLM inference engine from scratch

This guide builds a small CPU inference engine for pretrained decoder-only Transformers. Start with fixed token IDs and a scalar forward pass. Then add a tokenizer, a key-value cache, and a generation loop. Keeping these pieces separate lets you swap checkpoints, test each operation, and add CPU optimizations without changing the model equations.

Nathr is the worked implementation. SmolLM2-135M supplies concrete dimensions for the Llama walkthrough. GPT-2 shares its decoder runtime, and BERT and T5 show how the same tensor readers and CPU kernels work in an encoder and a sequence-to-sequence model. [Checkpoint support and limits](#checkpoint-support-and-limits)

```mermaid
flowchart LR
    Checkpoint --> Reader -->|Configuration and tensor views| Decoder
    Prompt --> Tokenizer -->|Token IDs| Decoder
    Decoder -->|Logits| Sampler
    Sampler -->|Next input ID| Decoder
    Sampler -->|Emitted ID| TokenBytes["Token-to-byte lookup"] --> Text["UTF-8 stream"]
```

## 1. Separate the checkpoint, model, and runtime

An inference engine evaluates the model with fixed weights. Given tokens `t[0]` through `t[p]`, the output at position `p` is a vector of next-token scores called *logits*. Index `i` in that vector corresponds to vocabulary token `i`. One forward pass can compute a logit vector at every position. For generation, the vector after the last input token is the one used to select the next token.

| Component | Responsibility |
| --- | --- |
| Checkpoint reader | Locate named tensors, their encodings, and metadata in a file or directory. |
| Model architecture | Interpret configuration, bind weight names and shapes, and define the forward pass. |
| Tokenizer | Map text to IDs and generated IDs back to bytes using the checkpoint's vocabulary and rules. |
| Session and generation | Track positions and mutable state, request logits, and sample the next ID. |
| CPU kernels | Evaluate operations on the tensors and encodings passed by the model. |

Choose one architecture and an accessible checkpoint first. List the operations the checkpoint requires, such as normalization, positional encoding, attention, and the feed-forward activation. Read the architecture name, dimensions, and options from its configuration, then validate them against your implementation before binding weights. Use a default for absent fields only when the architecture's format defines one. For example, Nathr's Llama loader uses the query-head count when the KV-head count is absent. It checks the head geometry, token ID range, and expected tensor shapes. [Nathr Llama configuration](../src/models/llama/config.cpp) · [Weight binding](../src/models/llama/model.cpp)

The worked decoder uses `L` layers, hidden width `H`, feed-forward width `F`, `Hq` query heads, and `Hkv` key/value heads. Each head has `D` coordinates. For the Llama geometry implemented here, `H = Hq * D`. Let `V` be the vocabulary size and `T` the context declared by the checkpoint. Keep session capacity and memory budgets separate from `T` so a model that supports a long context can run in a smaller session. [Nathr session limits](../src/runtime/session.cpp)

Use the logical weight shape `[output_width, input_width]`. A linear projection has one dot product per output row. The embedding table `[V, H]` maps a token ID to its initial activation. The output projection maps the final activation back to `V` logits. The following shapes describe the worked decoder.

| Projection | Weight shape |
| --- | --- |
| Query `Wq` | `[Hq * D, H]` |
| Key `Wk`, value `Wv` | `[Hkv * D, H]` each |
| Attention output `Wo` | `[H, Hq * D]` |
| Feed-forward gate `Wgate`, up `Wup` | `[F, H]` each |
| Feed-forward down `Wdown` | `[H, F]` |

For SmolLM2-135M, `L = 30`, `H = 576`, `F = 1536`, `Hq = 9`, `Hkv = 3`, and `D = 64`. That gives 576 query coordinates and 192 each for keys and values. `V = 49152` and `T = 8192`. These numbers describe the example checkpoint. Qwen3-0.6B has a residual width of 1,024 and 16 query heads of width 128, so its attention output projection maps 2,048 coordinates back to 1,024. It applies learned RMSNorm gains to each query and key head before RoPE. [SmolLM2-135M configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/config.json) · [Qwen3-0.6B GGUF](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/blob/23749fefcc72300e3a2ad315e1317431b06b590a/Qwen3-0.6B-Q8_0.gguf)

## 2. Read the checkpoint into tensor views

Represent a tensor with its element encoding, dimensions, and a read-only view of its bytes. Keep the backing storage alive as long as the model uses those views. Parse sizes with checked arithmetic, check tensor bounds before forming views, and reject unsupported encodings before execution. A format reader can expose tensors and metadata through this interface without knowing the decoder's layer equations. Nathr uses the same tensor view for two formats. [Nathr tensor view](../src/formats/tensor.hpp)

**Safetensors.** Read the eight-byte little-endian header length, parse the JSON header, and use each tensor's `data_offsets` to locate its half-open byte range relative to the start of the payload. Use checked arithmetic to compute the expected byte count from the shape and encoding. Nathr checks ranges for overlap, gaps, and bounds before it exposes any views, then loads the file into an owned buffer. FP32, FP16, and BF16 weights retain their stored encodings. The surrounding directory supplies `config.json` and `tokenizer.json` for text generation. [Safetensors format](https://github.com/safetensors/safetensors#format) · [Nathr loader](../src/formats/safetensors.cpp)

**GGUF.** Check the file signature and version, read typed metadata and tensor descriptors, and locate the aligned tensor data section. Tensor offsets are relative to that section. GGUF puts the fastest-varying dimension first, so a matrix's serialized dimensions read `[input_width, output_width]`. For packed encodings, validate each row against the block width and count bytes by blocks. Nathr uses a read-only mapping and retains metadata for the architecture declared by the file. The architecture loader interprets those fields. [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) · [Nathr GGUF reader](../src/formats/gguf.cpp)

Bind tensor names and storage conventions in the architecture code. In Nathr's example, a Hugging Face Llama checkpoint calls its first query matrix `model.layers.0.self_attn.q_proj.weight`. The corresponding GGUF name is `blk.0.attn_q.weight`. For Llama, Nathr also maps GGUF query and key projection outputs to the split-half order its rotary kernel expects. Qwen3 GGUF stores the projections in split-half order. The original weight bytes stay in place. If the model ties its embeddings, the output projection reuses the embedding view. [Nathr weight binding](../src/models/llama/model.cpp) · [Rotary row mapping](../src/formats/tensor.hpp)

## 3. Implement a scalar forward pass

Begin with a small sequence of known token IDs and dense floating-point weights. A scalar FP32 projection can be written as follows. The caller supplies validated contiguous weights and an output buffer sized to the number of rows. FP64 accumulation followed by FP32 storage gives a straightforward reference. Nathr also decodes lower-precision weights as the kernel reads them. [Nathr scalar kernels](../src/backend/cpu/scalar.cpp)

```cpp
#include <cstddef>
#include <span>

void project_f32(std::span<const float> weights,
                 std::span<const float> input,
                 std::span<float> output) {
    for (std::size_t row = 0; row < output.size(); ++row) {
        const auto w = weights.subspan(row * input.size(), input.size());
        double sum = 0.0;
        for (std::size_t col = 0; col < input.size(); ++col) {
            sum += static_cast<double>(w[col]) * input[col];
        }
        output[row] = static_cast<float>(sum);
    }
}
```

With weight rows `[1, 2, 3]` and `[-1, 0, 4]` and input `[2, 1, -1]`, the output is `[1, -6]`. This case catches an accidental transpose or row-stride error before either gets into the Transformer. The rest of this section works through Nathr's Llama-style block. For a different architecture, use the layer order and operations expected by that checkpoint.

### Normalize and project

Each layer starts with a vector `x` of width `H` for every token. Root mean square normalization (RMSNorm) scales it using the root mean square of its coordinates and a learned gain vector `g`.

```text
scale = 1 / sqrt(sum(x[i] * x[i] for i in 0..H-1) / H + epsilon)
norm[i] = x[i] * scale * g[i]
```

Apply `Wq`, `Wk`, and `Wv` to the normalized vector. Split their outputs into heads of width `D`. Each head is a contiguous span, so head `h` begins at coordinate `h * D`.

### Rotate queries and keys

Rotary position embeddings (RoPE) rotate pairs of coordinates within each query and key head. For the split-half layout, pair coordinate `j` with `j + D/2`. Their angle is proportional to the zero-based absolute token position `p`. [RoFormer paper](https://arxiv.org/abs/2104.09864) · [Nathr rotary kernel](../src/backend/cpu/scalar.cpp)

```text
angle = p * theta^(-2*j/D)               for j in 0..D/2-1
(a, b) = (head[j], head[j + D/2])
head[j]       = a*cos(angle) - b*sin(angle)
head[j + D/2] = a*sin(angle) + b*cos(angle)
```

Precompute the inverse frequencies from `theta` when loading the model. Position zero is a useful test because the rotation is the identity there. Use the token's position in the full sequence during both prefill and decode. Values pass through unchanged.

### Compute causal grouped-query attention

For each query head at position `p`, compute a dot product with its corresponding key head at every position from `0` through `p`, inclusive. Grouped-query attention shares a key/value head across several query heads. With nine query heads and three key/value heads, query heads `0–2` use key/value head `0`, heads `3–5` use `1`, and heads `6–8` use `2`. [Llama 2 architecture](https://arxiv.org/abs/2307.09288) · [Nathr attention kernel](../src/backend/cpu/scalar.cpp)

```text
kv_head = query_head / (Hq / Hkv)        # integer division
score[i] = dot(query[p, query_head], key[i, kv_head]) / sqrt(D)
prob = softmax(score[0..p])
head_output = sum(prob[i] * value[i, kv_head] for i in 0..p)
```

Implement softmax by subtracting the largest score before exponentiation, then dividing each exponential by their sum. Scores `[0, ln(3)]` yield weights `[0.25, 0.75]`. If the value rows are `[4, 0]` and `[0, 8]`, the weighted output is `[1, 6]`. At position zero the output for each query head is simply its corresponding value head at position zero. Reuse one score buffer sized to the longest prefix the call or session can process, one query head at a time.

### Add the residuals and produce logits

Concatenate the attention heads, apply `Wo` to get a vector of width `H`, then add it to `x`. Normalize that sum with the layer's second RMSNorm gain. The feed-forward block computes two projections of width `F`, applies `SiLU(g) = g / (1 + exp(-g))` to the gate projection, and multiplies it element by element by the up projection. Apply `Wdown` and add a second residual. After the last layer, apply a final RMSNorm and multiply by the output matrix. SmolLM2-135M uses the embedding table for this final projection. [Layer execution and tracing](../src/models/llama/model.cpp)

The full uncached computation for a short token sequence looks like this in pseudocode. Each projection or normalization on two-dimensional activations applies to each token independently.

```text
x = embedding[token_ids]
for layer in layers:
    u = rms_norm(x, layer.input_gain)
    q = rope(project(layer.Wq, u), positions)
    k = rope(project(layer.Wk, u), positions)
    v = project(layer.Wv, u)
    a = causal_attention(q, k, v)
    x = x + project(layer.Wo, a)

    u = rms_norm(x, layer.ffn_gain)
    gate = silu(project(layer.Wgate, u))
    up = project(layer.Wup, u)
    x = x + project(layer.Wdown, gate * up)

return project(output_weights, rms_norm(x[last_position], final_gain))
```

This path can generate text by evaluating the whole prefix after each new token. It also gives you a reference for checking the cached path with the same input.

## 4. Turn text into token IDs

The model was trained with a specific tokenizer. Give the engine an entry point that accepts token IDs so you can check the decoder before implementing text input. For end-to-end generation, use the checkpoint's actual vocabulary, token IDs, and tokenization rules. A tokenizer with the same vocabulary size can still assign different IDs or split the same text differently.

Byte-level byte pair encoding (BPE) is one worked example. The serialized vocabulary uses a reversible byte alphabet. For example, it can represent ASCII space as `Ġ` (U+0120). Invert that mapping on load so each ID has a byte sequence. The merge list gives the priority of each pair and the vocabulary gives the ID of the merged token.

1. Recognize configured added tokens and copy their IDs directly. Split the remaining text into pieces using the checkpoint's pretokenizer, which defines where whitespace, digits, letters, and punctuation can be grouped.
2. For each piece, start with the vocabulary IDs for its individual UTF-8 bytes. Handle missing byte tokens according to the tokenizer's rules. Work on each piece independently to keep merges within pretokenizer boundaries.
3. Find the adjacent pair with the lowest merge rank. Resolve equal ranks by taking the leftmost pair. Replace it with its merged ID and repeat until no rule applies.
4. Emit the surviving IDs in order.

For a toy piece `abc`, suppose the rules say `b + c → bc` at rank `0` and `a + b → ab` at rank `1`. The first merge produces `[a, bc]`. The `a + b` rule can no longer apply because `b` is part of another token. A simple implementation rescans all pairs after each merge. Nathr links the surviving nodes, queues candidates by rank, and updates only the neighbors of a merge. Queued candidates carry node versions so stale pairs can be skipped. [Byte-level BPE implementation](../src/tokenizer/tokenizer.cpp)

Nathr loads the vocabulary and merge ranks from `tokenizer.json` or GGUF metadata. It supports ByteLevel, individual-digit splitting followed by ByteLevel, and the Qwen byte-level pretokenizer. For GGUF, control token types come from metadata alongside the token texts. If the caller supplies a matching `tokenizer.json`, called a sidecar, Nathr checks the complete vocabulary-to-ID mapping and ordered merges, and checks token types and pretokenizer behavior where it can compare them. A caller can also use an external tokenizer and pass its IDs to the decoder. [Nathr tokenizer loading](../src/tokenizer/tokenizer.cpp)

Check the IDs directly on prompts with leading spaces, newlines, digits, non-ASCII text, and special tokens. Check boundary IDs such as beginning-of-sequence (BOS) and end-of-sequence (EOS) separately from the rule that decides when to insert or stop on those IDs.

## 5. Keep past keys and values in a session

Causality means that appending token `p + 1` leaves the activations for positions `0..p` unchanged. For an attention-based decoder, save each layer's keys and values as you process tokens. The next query uses those stored vectors and the new token's Q, K, and V. When the model applies RoPE to Q and K, store the rotated keys so you can use them directly on later steps.

Share the immutable weights across sessions. Give each session its own cache, temporary buffers, token history, and random number generator. Nathr maps groups of 16 consecutive positions to independently allocated pages. Within each page, keys and values occupy separate banks indexed by `[layer][kv_head][position_in_page][coordinate]`. Attention reads the visible rows one page at a time. [Nathr cache layout](../src/backend/cpu/scalar.hpp) · [PagedAttention paper](https://arxiv.org/abs/2309.06180)

```text
KV payload bytes = 2 * L * Hkv * allocated_positions * D * bytes_per_element
```

The cache starts empty. Each call allocates any pages needed for its input before running the model. The last page stops at the configured capacity. When a SmolLM2-135M session fills all 8,192 slots, its cache payload is 360 MiB in FP32 or 180 MiB in FP16. Weights and temporary buffers are separate. Nathr defaults to a session capacity of up to 2,048 tokens. FP16 rounds cache entries on write and converts them back to FP32 on read. [Nathr cache storage and attention](../src/backend/cpu/scalar.cpp)

**Prefill** runs the input prompt through the layers and fills the cache. Nathr evaluates groups of up to 16 tokens by default. A group can project and store all its keys and values before computing attention because a query at `p` reads only through `p`. Compute the vocabulary projection for the last prompt token when generating. Earlier prompt tokens still need to pass through all layers so their K and V can serve later tokens.

**Decode** processes each newly appended token at the current history length. Its queries attend to earlier cached entries and the new entry. The cache avoids recomputing earlier tokens' projections and layers. The new query still visits the entire prefix, so the attention work in each decode step grows with the occupied context.

Validate token IDs and remaining capacity before mutating the session. Nathr exposes logits after a successful call that requests them. A failure during execution can leave a partially written cache, so the session remains invalid until reset. Reset clears the logical history and restores the original random seed while retaining the buffers. [Session API](../include/nathr/nathr.hpp) · [Session state transitions](../src/runtime/session.cpp)

## 6. Sample tokens and stream text

Greedy decoding takes the maximum logit, with the first ID winning a tie. It is a convenient way to check the end-to-end model before introducing random sampling. For sampling, sort candidates by descending logit, keep the top `k` if specified, then compute weights with `exp((logit - max_logit) / temperature)`. Top-p retains the shortest prefix whose cumulative weight reaches `p` times the total weight of those candidates. Draw a uniform number over the retained weight mass and select the first token whose cumulative weight exceeds it. Nathr uses the session's seeded RNG for the draw. [Sampler implementation](../src/runtime/generation.cpp)

```text
session.prefill(prompt_ids)
for step in 0..max_new_tokens-1:
    id = sample(session.logits())
    emit(id)
    if id == eos or step + 1 == max_new_tokens:
        break
    session.append([id])
```

Use the checkpoint's EOS ID and any additional stop IDs in the stopping rule. Nathr reads supported sampling defaults from the checkpoint configuration, with `generation_config.json` taking precedence. The logits at the end of prefill already predict the first new token. Appending a sampled ID computes logits for the following token. With `P` prompt tokens and a positive budget `N`, this loop needs room to consume at most `P + N - 1` tokens. Nathr reports the final emitted ID as `pending` so the caller can append it when resuming the session. [Nathr generation API](../include/nathr/generation.hpp) · [Request defaults](../apps/requests.cpp)

Look up each emitted ID's byte sequence and pass it to an incremental UTF-8 decoder. A token can end with byte `C3`, and the next can start with `A9`. Together they encode `é`. To stop on text, hold back any suffix that could start a stop string. Drop a complete match and release an unmatched suffix when generation ends. For chat, render the checkpoint template before tokenization so the model sees its expected role markers. [Text streaming](../src/runtime/generation.cpp) · [Chat rendering](../apps/requests.cpp)

### Schedule multiple requests

The scheduler gives active requests one decode token per step, then divides the remaining token budget among prompt chunks. The causal decoder packs scheduled rows into shared projection calls. Each row carries its absolute position and its request’s KV table, so it attends only to the visible rows of its own sequence. T5 encodes one whole source as a scheduling unit. [Batch scheduler](../src/runtime/batch.cpp) · [Decoder input rows](../src/models/decoder.hpp)

Admission reserves enough KV capacity for a request’s token budget. FIFO is the default. A configured bypass limit lets fitting requests pass a memory-blocked queue head, then holds the queue until that request fits. The engine allocates pages on demand from a budgeted pool. [Batch scheduler](../src/runtime/batch.cpp) · [KV page pool](../src/backend/cpu/scalar.hpp)

The decoder scheduler indexes full prompt pages by their exact token IDs and the parent page's identity. Later requests can share those pages and skip their prefill rows. A fully cached prompt replays its final token with read-only KV access to obtain the first logits. Shared pages are immutable, with unreferenced leaves reclaimed under memory pressure. [Prefix cache](../src/runtime/batch.cpp) · [Serving benchmark](../README.md#benchmark-a-running-server)

The terminal and HTTP frontends send JSONL requests to one persistent worker per model. The terminal keeps chat history in the current session. The API routes by model ID and adapts generation and embeddings to the batch request format. It bounds client count, request bodies, and response buffering, and cancels a request between scheduler steps if a buffer fills or a client disconnects. [Frontends](../apps/serve.py) · [HTTP endpoints](../README.md#openai-compatible-api)

## 7. Execute packed weights and make the CPU path faster

Add packed formats behind the row-dot interface. As a first implementation, unpack one block into a small local buffer, convert its values to FP32, and accumulate their products with the corresponding activation coordinates. Keep the full matrix packed in its backing storage. For a row of `N` weights, its byte stride is `N / elements_per_block * bytes_per_block`. Validate the divisibility when loading the tensor. [Tensor encodings](../src/formats/tensor.hpp) · [Block decoders](../src/quantization/decoders.cpp)

| Encoding | Block of 32 weights | Bytes per block |
| --- | --- | ---: |
| `Q8_0` | FP16 scale `d`, followed by 32 signed 8-bit integers | 34 |
| `Q4_0` | FP16 scale `d`, followed by 16 bytes containing two 4-bit values each | 18 |

`Q8_0` reconstructs a weight as `d * q`. In `Q4_0`, the low nibbles form the first 16 weights and the high nibbles form the second 16. A nibble `n` represents `d * (n - 8)`. With `d = 0.25` and the first packed byte `0xB2`, weights `0` and `16` are `-1.5` and `0.75`. A known byte pattern makes a useful test for sign conversion, nibble order, and scale decoding.

With one input token, a projection is a matrix-vector product. With several, a tile lets the kernel reuse each weight row across multiple input tokens. Nathr packs activation coordinates for up to 16 tokens next to each other within each column and reuses the packed tile across projections with the same input. It distributes sufficiently large ranges of output rows to a persistent worker pool. Each worker writes its own rows. [Projection and tiling](../src/backend/cpu/kernels.cpp) · [Worker pool](../src/backend/cpu/thread_pool.cpp)

Select kernels when binding the weights. On x86, check CPU feature bits and whether the OS saves the vector register state before entering AVX2 code. Nathr has AVX2 projection kernels for common encodings and a scalar path for other supported encodings or systems without usable AVX2. Its AVX2 FP16 projection also requires F16C for conversion. Compare optimized rows against the scalar version on ordinary lengths and tail sizes. [CPU dispatch](../src/backend/cpu/kernels.cpp) · [AVX2 kernels](../src/backend/cpu/x86/avx2.cpp)

## 8. Verify each interface before comparing generated text

Work outward from the first place results can diverge. Use an independent implementation or a small hand-computed case as a reference. A matching greedy token only checks the location of the maximum logit. Compare the entire vector to catch errors elsewhere in the output.

| Check | What to compare |
| --- | --- |
| Loader and kernels | Expected shapes and row values, including quantized blocks built from known bytes |
| Transformer | Embeddings, normalization, Q and K before and after RoPE when used, attention output, residuals, and logits |
| Cache | Cached and uncached logits for the same prefix, then logits after appending one fixed token at a time |
| Tokenizer | Token IDs and decoded bytes for the same prompt and special-token settings |
| Generation | Greedy IDs, EOS handling, context boundaries, and UTF-8 across token boundaries |

### Run the checks with Nathr

The checked-in FP32 fixture exercises both forward paths without downloading model weights. Run these commands from the repository root. [Build requirements](../README.md#prerequisites)

```sh
cmake --preset release
cmake --build --preset release
ctest --preset release

bin=build/release/nathr
fixture=tests/fixtures/model/f32
"$bin" forward "$fixture" --tokens 1,7,2 --json
"$bin" forward "$fixture" --tokens 1,7,2 --cached --prefill-chunk 1 --json
```

Both Llama fixture calls return 23 logits and a greedy ID of `2`. Compare their `logits` arrays, then rerun the two forward calls with `fixture=tests/fixtures/gpt2` to exercise the GPT-2 path. The runtime tests compare full forward, chunked prefill, and repeated append. The GPT-2 fixture also checks logits against a Transformers reference. [Nathr runtime tests](../tests/runtime.cpp)

`forward --trace FILE` writes intermediate activations for a short sequence. `forward --logits-f32` streams one little-endian FP32 vocabulary row per token for comparisons against an independent implementation using the same checkpoint. Use exact comparisons when the operations and their numeric behavior match. When precision or reduction order differs, use a tolerance that accounts for the expected rounding error. [CLI reference](../README.md#cli-reference)

Once the numeric path agrees, try the CLI examples with full SmolLM2 and GPT-2 checkpoints. Reuse the prompt IDs returned by `tokenize` for reproducible forward comparisons. The `bench` command uses synthetic IDs to measure prefill and decode without tokenization or sampling. Keep the checkpoint, backend, thread count, context, cache precision, and input lengths fixed when comparing implementations. [CLI examples](../README.md#cli-examples)

The compatibility checker verifies SHA-256 hashes, token IDs, sampled tensor values, and seeded generation across standalone and batched execution where supported. Transformers 4.57.6 supplies the dense tensor and two-turn chat reference values. SmolLM2 GGUF tensor values and seeded samples come from Nathr. Qwen3 GGUF token IDs, sampled logits, and chat prompt IDs come from llama.cpp 1.1.0. The optional mixed-length GGUF benchmark reports short-request time to first token, p95 inter-token latency, output throughput, and KV memory for one and three active requests. [Run the compatibility checker](../README.md#check-model-compatibility) · [Reference values](../tests/checkpoints.json)

## 9. Bring up a second checkpoint

Choose a checkpoint that differs in vocabulary size, head geometry, or context. For the same architecture, bind the tensors from its metadata and repeat the fixed-ID comparison. If it uses a different tokenizer, get reference IDs externally and check the decoder before enabling text input. Nathr can exercise compatible checkpoints this way with `forward --tokens` and `generate --tokens IDS --json`. Use IDs from the new checkpoint's tokenizer for those tests. [Nathr CLI](../apps/nathr.cpp)

For an architecture with different layer equations, add the configuration and weight binding rules alongside its forward pass. Define any additional state the session must carry. Reuse the tensor readers, sampling loop, and kernels whose interfaces still fit. Validate the new forward pass on a tiny fixture, then compare its uncached and cached outputs before testing a long continuation.

Nathr's GPT-2 decoder follows this pattern. It binds learned position embeddings and affine LayerNorm weights, transposes the stored projections once when it loads the checkpoint, and splits a combined projection into queries, keys, and values. The session requests workspace from the decoder and supplies the same KV cache to both architectures. [Decoder interface](../src/models/decoder.hpp) · [GPT-2 implementation](../src/models/gpt2/model.cpp) · [Run a named checkpoint](../README.md#run-a-model)

### Add an encoder and cross-attention

BERT learns a position embedding for each input position. Each layer projects the entire sequence to Q, K, and V, then lets every query see every input token. The last layer returns one hidden-state row per input token. To verify bidirectionality, change the final token and compare the first token's new hidden state against a reference. [BERT encoder](../src/models/bert/model.cpp) · [BERT reference checks](../tests/runtime.cpp)

T5's encoder also sees the full input. Its self-attention adds a learned bias based on the bucketed distance between each query and key. During decoding, each layer reads two sets of keys and values. The causal self-attention cache grows with each decoder token. The cross-attention cache is projected once per decoder layer from the final encoder output. The decoder begins with the configured start token and uses the same sampling loop as the decoder-only models. [T5 model](../src/models/t5/model.cpp) · [T5 paper](https://jmlr.org/papers/volume21/20-074/20-074.pdf)

### Expose checkpoint outputs

Use a tensor name and shape as the common inference contract. A checkpoint describes which outputs exist and how many input tokens they take. A BERT checkpoint can expose all hidden states or bind weights for its pooler and a classification head. Sentence Transformers checkpoints describe extra pooling and normalization in their module configuration. T5 logits are indexed by decoder position and can be compared using an explicit decoder prefix. Generation wraps logits with a sampling loop and request state. [Model inference API](../include/nathr/nathr.hpp) · [Output examples](../README.md#request-a-named-output)

## Checkpoint support and limits

Nathr reads dimensions, attention head counts, context length, and boundary token IDs from the checkpoint. It loads single-file weights up to 1 GiB and supports vocabularies up to 196,608 tokens. Session capacity also depends on the KV cache budget. [Model limits](../src/core/limits.hpp) · [Session allocation](../src/runtime/session.cpp)

| Architecture | Supported configuration |
| --- | --- |
| Llama-style | GGUF v3 or Safetensors with RMSNorm, SwiGLU, unscaled full-width RoPE, and bias-free projections |
| Qwen3 | Dense GGUF v3 with per-head query and key RMSNorm and unscaled, full-width RoPE. The pinned checkpoint is Qwen3-0.6B Q8_0 |
| GPT-2 | Safetensors with learned positions, affine LayerNorm, and the GPT-2 GELU variant |
| BERT | Safetensors with absolute position embeddings and GELU. Uses unpadded inputs and optional segment IDs. Outputs depend on the checkpoint's heads and pooling modules |
| T5 | Safetensors for original ReLU T5. The decoder uses FP32 caches. The source and decoder are each limited to 512 positions or the checkpoint context, whichever is smaller |

The built-in tokenizers cover byte-level BPE, BERT WordPiece, and T5 SentencePiece Unigram. WordPiece can encode text pairs and segment IDs. Compatible GGUF files can embed the tokenizer. Raw token IDs let callers use an external tokenizer. Chat rendering supports SmolLM2 ChatML and the pinned Qwen3 template in plain-text, non-thinking mode. [Tokenizer loading](../src/tokenizer/tokenizer.cpp) · [WordPiece and Unigram](../src/tokenizer/subword.cpp) · [Chat rendering](../apps/requests.cpp)
