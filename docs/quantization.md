# GGUF packed weight formats

Nathr stores GGUF tensors in their serialized form. In the tensor table, the first GGUF dimension is contiguous. Every row must contain a whole number of blocks. A row of width `W` occupies `W / elements_per_block * bytes_per_block` bytes, independent of the other dimensions. In SmolLM2, width 576 can use 32-element blocks. Width 1536 can also use 256-element blocks.

Fields below are in increasing byte-offset order. Integer planes and F16 fields are little-endian. A paired-nibble stream emits all low-nibble values before the high-nibble values for that group. For Q/K weights Nathr writes each projected value into its canonical split-half coordinate, with the serialized bytes left in adjacent-pair order. The CPU backend decodes one block at a time for scalar GEMV and embedding lookups. Scalar and AVX2 GEMV multiply decoded FP32 values and accumulate products in FP64, then round each row output to FP32. Packed weights, activations and cache storage retain their existing layouts.

## Legacy blocks

| Encoding | ID | Elements | Bytes | Fields and reconstruction |
| --- | ---: | ---: | ---: | --- |
| Q4_0 | 2 | 32 | 18 | F16 scale `d`, then 16 nibble pairs. Use `d * (q - 8)`. |
| Q4_1 | 3 | 32 | 20 | F16 scale `d` and minimum `m`, then 16 nibble pairs. Use `d * q + m`. |
| Q5_0 | 6 | 32 | 22 | F16 scale `d`, 32-bit high plane, then 16 nibble pairs. Combine the low nibble with high bit `i` for element `i`. Use `d * (q - 16)`. |
| Q5_1 | 7 | 32 | 24 | F16 scale and minimum, then the Q5 high plane and low nibbles. Use `d * q + m`. |
| Q8_0 | 8 | 32 | 34 | F16 scale followed by 32 signed bytes. Use `d * q`. |

## K blocks

Each K block has 256 elements. Q2_K, Q3_K and Q6_K apply one scale per 16 elements. Q4_K and Q5_K apply one scale and minimum per 32 elements.

| Encoding | ID | Bytes | Fields and reconstruction |
| --- | ---: | ---: | --- |
| Q2_K | 10 | 84 | 16 packed scale/minimum nibbles, 64 two-bit code bytes, F16 base scale and base minimum. Within each 128-element half, the four planes are 32 elements wide. Scale is the low nibble and minimum is the high nibble. Use `d * scale * q - dmin * minimum`. |
| Q3_K | 11 | 110 | 32 high-plane bytes, 64 low two-bit code bytes, 12 packed scale bytes, F16 base scale. The scale low nibbles occupy bytes 0–7 of the scale field and the high two bits occupy bytes 8–11. Use `scale - 32`. The high plane subtracts four from a low code when its bit is zero. |
| Q4_K | 12 | 144 | F16 base scale and base minimum, 12 packed scale bytes, 128 nibble bytes. The first four scale/minimum values use the low six bits of bytes 0–3 and 4–7 respectively. Later values use nibbles from bytes 8–11 plus the upper two bits of the corresponding earlier bytes. Subtract the scaled minimum. |
| Q5_K | 13 | 176 | Q4_K fields with a 32-byte high plane inserted before the low nibbles. High bit `g` in a byte belongs to group `g` at the same lane. Subtract the scaled minimum. |
| Q6_K | 14 | 210 | 128 low-nibble bytes, 64 high-two-bit bytes, 16 signed scale bytes, F16 base scale. Within each 128-element half, low plane width is 64 and high plane width is 32. Combine planes, subtract 32, then multiply by the signed group scale and base scale. |

## I blocks

The grid codecs use fixed lookup tables. Packed indices select magnitudes in lane order. A sign mask applies one bit per lane. For seven-bit sign indices, the eighth bit gives even parity across the eight signs. IQ1_S and IQ1_M use a ternary grid with an additive delta of +0.125 or −0.125.

| Encoding | ID | Elements | Bytes | Fields and reconstruction |
| --- | ---: | ---: | ---: | --- |
| IQ1_S | 19 | 256 | 50 | F16 base scale, 32 low index bytes and eight 16-bit group words. Each word carries four three-bit index extensions, a three-bit scale field and a delta sign. Group scale is `d * (1 + 2 * scale_code)`. |
| IQ1_M | 29 | 256 | 56 | 32 low index bytes, 16 bytes of high-index/delta nibbles and four 16-bit scale words. The upper nibble of each scale word supplies a quarter of the block F16 scale. The lower 12 bits supply four three-bit scales for 16-element groups. Each index extension uses the lower three nibble bits and the fourth bit is the delta sign. |
| IQ2_XXS | 16 | 256 | 66 | F16 base scale, then eight groups of eight bytes. Each group has four grid indices and a 32-bit word with four seven-bit parity sign indices and a four-bit scale code. Scale is `d * (0.5 + scale_code) / 4`. A grid emits eight values. |
| IQ2_XS | 17 | 256 | 74 | F16 base scale, 32 16-bit words with nine index bits and seven parity sign bits, then eight packed scale bytes. Each nibble scales 16 values using the same IQ2 scale formula. |
| IQ2_S | 22 | 256 | 82 | F16 base scale, 32 low index bytes, 32 full sign masks, eight two-bit index-extension bytes and eight packed scale bytes. Scale is per 16 values and uses the IQ2 formula. |
| IQ3_XXS | 18 | 256 | 98 | F16 base scale, 64 four-lane grid indices and eight 32-bit scale/sign words. Adjacent grids share an eight-lane parity sign mask. Scale is per 32 values and is `d * (0.5 + scale_code) / 2`. |
| IQ3_S | 21 | 256 | 110 | F16 base scale, 64 low index bytes, eight high-index bytes, 32 eight-lane sign masks and four scale bytes. A scale nibble serves 32 values and uses `d * (1 + 2 * scale_code)`. |
| IQ4_NL | 20 | 32 | 18 | F16 scale and 16 nibble pairs. The nibble selects one of 16 signed nonlinear levels, then the level is scaled. |
| IQ4_XS | 23 | 256 | 136 | F16 base scale, a 16-bit high scale plane, four low scale-nibble bytes and 128 nibble bytes. Each signed six-bit scale code is biased by 32 and applies to one IQ4_NL group. |

The lookup table generator checks the pinned GGML source hash before extracting only encoding constants. The tables and parity signs are compared exhaustively with the independently packed gguf-py representation. The runtime has no GGML linkage. [Pinned GGML data](https://github.com/ggml-org/llama.cpp/blob/b1b770d90d94cad9e9ea5b993bdbc383a522fb28/ggml/src/ggml-common.h)

The packed fixtures cover finite half scales including subnormals, zero and negative zero, maximum values, full index planes and nontrivial signs. Both pinned external decoders agree bit for bit on those fixtures. The CPU suite checks unpacking, row strides, embeddings and reduction-aware dot products. Structural malformed-file tests exercise input bounds separately. [Lookup-table license](../src/quantization/LICENSE.ggml)
