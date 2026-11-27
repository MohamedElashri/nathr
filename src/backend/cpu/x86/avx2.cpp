#include "backend/cpu/x86/avx2.hpp"
#include "backend/cpu/batch.hpp"
#include "backend/cpu/scalar.hpp"
#include "quantization/decoders.hpp"

#include <immintrin.h>

namespace nathr::detail::x86 {
namespace {

double horizontal_sum(__m256d vec) {
    const auto pairs = _mm_add_pd(_mm256_castpd256_pd128(vec), _mm256_extractf128_pd(vec, 1));
    return _mm_cvtsd_f64(_mm_add_sd(pairs, _mm_unpackhi_pd(pairs, pairs)));
}

__m256d accumulate4(__m256d total, __m128 weights, const float* activations) {
    return _mm256_add_pd(total,
                         _mm256_mul_pd(_mm256_cvtps_pd(weights), _mm256_cvtps_pd(_mm_loadu_ps(activations))));
}

__m256d accumulate8(__m256d total, __m256 weights, const float* activations) {
    total = accumulate4(total, _mm256_castps256_ps128(weights), activations);
    return accumulate4(total, _mm256_extractf128_ps(weights, 1), activations + 4);
}

__m256d signed_products(__m128i packed, const float* input) {
    const auto integers = _mm256_cvtepi32_pd(_mm_cvtepi8_epi32(packed));
    return _mm256_mul_pd(integers, _mm256_cvtps_pd(_mm_loadu_ps(input)));
}

__m256d block_products(__m128i codes, const float* input) {
    const auto first =
        _mm256_add_pd(signed_products(codes, input), signed_products(_mm_srli_si128(codes, 4), input + 4));
    const auto second = _mm256_add_pd(signed_products(_mm_srli_si128(codes, 8), input + 8),
                                      signed_products(_mm_srli_si128(codes, 12), input + 12));
    return _mm256_add_pd(first, second);
}

template <DType type> float quantized_dot(const std::byte* weights, const float* input, std::size_t count) {
    constexpr auto stride = encoding(type)->bytes;
    auto total0 = _mm256_setzero_pd();
    auto total1 = _mm256_setzero_pd();
    for (std::size_t start = 0; start < count; start += 32, weights += stride) {
        const auto scale = _mm256_set1_pd(half_to_float(read_unaligned<std::uint16_t>(weights)));
        auto a = read_unaligned<__m128i>(weights + 2);
        __m128i b;
        if constexpr (type == DType::q8_0) {
            b = read_unaligned<__m128i>(weights + 18);
        } else {
            const auto mask = _mm_set1_epi8(15);
            const auto zero_point = _mm_set1_epi8(8);
            b = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(a, 4), mask), zero_point);
            a = _mm_sub_epi8(_mm_and_si128(a, mask), zero_point);
        }
        // The block scale is shared by all 32 codes. Apply it after the
        // FP64 partial reductions and keep two independent row accumulators.
        total0 = _mm256_add_pd(total0, _mm256_mul_pd(block_products(a, input + start), scale));
        total1 = _mm256_add_pd(total1, _mm256_mul_pd(block_products(b, input + start + 16), scale));
    }
    return static_cast<float>(horizontal_sum(_mm256_add_pd(total0, total1)));
}

__m128i extract_codes(const std::byte* data, unsigned shift, int mask) {
    const auto codes = read_unaligned<__m128i>(data);
    return _mm_and_si128(_mm_srl_epi16(codes, _mm_cvtsi32_si128(shift)), _mm_set1_epi8(mask));
}

__m256 code_weights(__m128i codes, __m256 scale) {
    return _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(codes)), scale);
}

template <DType type, class Emit> void unpack_k(const std::byte* block, Emit emit) {
    constexpr bool affine = type == DType::q2_k || type == DType::q4_k || type == DType::q5_k;
    constexpr unsigned scale_offset = type == DType::q2_k   ? 80
                                      : type == DType::q3_k ? 108
                                      : type == DType::q6_k ? 208
                                                            : 0;
    const auto base = half_to_float(read_unaligned<std::uint16_t>(block + scale_offset));
    float base_min = 0.f;
    if constexpr (affine)
        base_min = half_to_float(read_unaligned<std::uint16_t>(block + scale_offset + 2));

    for (unsigned tile = 0; tile < 16; ++tile) {
        __m128i codes;
        float scale, minimum = 0.f;
        if constexpr (type == DType::q4_k || type == DType::q5_k) {
            constexpr auto code_offset = type == DType::q5_k ? 48 : 16;
            const auto group = tile / 2;
            const auto offset = (tile % 2) * 16;
            const auto params = k_scale(block + 4, group);
            scale = base * params.scale;
            minimum = base_min * params.minimum;
            codes = extract_codes(block + code_offset + (group / 2) * 32 + offset, 4 * (group % 2), 15);
            if constexpr (type == DType::q5_k) {
                const auto high = extract_codes(block + 16 + offset, group, 1);
                codes = _mm_or_si128(codes, _mm_slli_epi16(high, 4));
            }
        } else if constexpr (type == DType::q6_k) {
            const auto subblock = tile % 8;
            const auto low =
                extract_codes(block + (tile / 8) * 64 + (subblock % 4) * 16, 4 * (subblock / 4), 15);
            const auto high =
                extract_codes(block + 128 + (tile / 8) * 32 + (tile % 2) * 16, 2 * (subblock / 2), 3);
            codes = _mm_sub_epi8(_mm_or_si128(low, _mm_slli_epi16(high, 4)), _mm_set1_epi8(32));
            scale = base * read_unaligned<std::int8_t>(block + 192 + tile);
        } else {
            constexpr auto code_offset = type == DType::q2_k ? 16 : 32;
            const auto offset = (tile % 2) * 16;
            codes = extract_codes(block + code_offset + (tile / 8) * 32 + offset, 2 * ((tile / 2) % 4), 3);
            if constexpr (type == DType::q3_k) {
                const auto nonnegative = extract_codes(block + offset, tile / 2, 1);
                codes = _mm_sub_epi8(codes, _mm_slli_epi16(_mm_xor_si128(nonnegative, _mm_set1_epi8(1)), 2));
                scale = base * q3_k_scale(block + 96, tile);
            } else {
                const auto packed = std::to_integer<unsigned>(block[tile]);
                scale = base * (packed & 15);
                minimum = base_min * (packed >> 4);
            }
        }
        const auto d = _mm256_set1_ps(scale);
        auto low = code_weights(codes, d);
        auto high = code_weights(_mm_srli_si128(codes, 8), d);
        if constexpr (affine) {
            const auto m = _mm256_set1_ps(minimum);
            low = _mm256_sub_ps(low, m);
            high = _mm256_sub_ps(high, m);
        }
        emit(low, high);
    }
}

template <DType type> void decode_k(const std::byte* block, float* decoded) {
    unpack_k<type>(block, [&](auto first, auto second) {
        _mm256_storeu_ps(decoded, first);
        _mm256_storeu_ps(decoded + 8, second);
        decoded += 16;
    });
}

template <DType type> float k_dot(const std::byte* weights, const float* input, std::size_t count) {
    constexpr auto format = *encoding(type);
    auto first = _mm256_setzero_pd();
    auto second = _mm256_setzero_pd();
    // Use the scalar decoder's FP32 weights and keep FP64 dot accumulators.
    for (std::size_t offset = 0; offset < count; offset += format.elements, weights += format.bytes) {
        unpack_k<type>(weights, [&](auto low, auto high) {
            first = accumulate8(first, low, input);
            second = accumulate8(second, high, input + 8);
            input += 16;
        });
    }
    return static_cast<float>(horizontal_sum(_mm256_add_pd(first, second)));
}

} // namespace

KernelChoice k_kernel(DType type) {
    switch (type) {
    case DType::q2_k:
        return {k_dot<DType::q2_k>, "avx2-q2_k", decode_k<DType::q2_k>,
                batch<DType::q2_k, decode_k<DType::q2_k>>};
    case DType::q3_k:
        return {k_dot<DType::q3_k>, "avx2-q3_k", decode_k<DType::q3_k>,
                batch<DType::q3_k, decode_k<DType::q3_k>>};
    case DType::q4_k:
        return {k_dot<DType::q4_k>, "avx2-q4_k", decode_k<DType::q4_k>,
                batch<DType::q4_k, decode_k<DType::q4_k>>};
    case DType::q5_k:
        return {k_dot<DType::q5_k>, "avx2-q5_k", decode_k<DType::q5_k>,
                batch<DType::q5_k, decode_k<DType::q5_k>>};
    case DType::q6_k:
        return {k_dot<DType::q6_k>, "avx2-q6_k", decode_k<DType::q6_k>,
                batch<DType::q6_k, decode_k<DType::q6_k>>};
    default:
        return {};
    }
}

float f32_dot(const std::byte* weights, const float* input, std::size_t count) {
    auto total = _mm256_setzero_pd();
    auto second = _mm256_setzero_pd();
    std::size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        total = accumulate8(total, read_unaligned<__m256>(weights + i * 4), input + i);
        second = accumulate8(second, read_unaligned<__m256>(weights + (i + 8) * 4), input + i + 8);
    }
    total = _mm256_add_pd(total, second);
    for (; i + 8 <= count; i += 8)
        total = accumulate8(total, read_unaligned<__m256>(weights + i * 4), input + i);
    double sum = horizontal_sum(total);
    for (; i < count; ++i)
        sum += static_cast<double>(read_unaligned<float>(weights + i * 4)) * input[i];
    return static_cast<float>(sum);
}

__attribute__((target("avx2,f16c"))) float f16_dot(const std::byte* weights, const float* input,
                                                   std::size_t count) {
    auto total = _mm256_setzero_pd();
    std::size_t i = 0;
    for (; i + 8 <= count; i += 8)
        total = accumulate8(total, _mm256_cvtph_ps(read_unaligned<__m128i>(weights + i * 2)), input + i);
    double sum = horizontal_sum(total);
    for (; i < count; ++i)
        sum += static_cast<double>(half_to_float(read_unaligned<std::uint16_t>(weights + i * 2))) * input[i];
    return static_cast<float>(sum);
}

float bf16_dot(const std::byte* weights, const float* input, std::size_t count) {
    auto total = _mm256_setzero_pd();
    std::size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        const auto ints = _mm256_cvtepu16_epi32(read_unaligned<__m128i>(weights + i * 2));
        total = accumulate8(total, _mm256_castsi256_ps(_mm256_slli_epi32(ints, 16)), input + i);
    }
    double sum = horizontal_sum(total);
    for (; i < count; ++i)
        sum += static_cast<double>(bf16_to_float(read_unaligned<std::uint16_t>(weights + i * 2))) * input[i];
    return static_cast<float>(sum);
}

float q8_0_dot(const std::byte* weights, const float* input, std::size_t count) {
    return quantized_dot<DType::q8_0>(weights, input, count);
}

float q4_0_dot(const std::byte* weights, const float* input, std::size_t count) {
    return quantized_dot<DType::q4_0>(weights, input, count);
}

void f32_batch(const std::byte* w, const float* x, std::size_t width, std::size_t lanes, float* out) {
    batch<DType::f32>(w, x, width, lanes, out);
}
void f16_batch(const std::byte* w, const float* x, std::size_t width, std::size_t lanes, float* out) {
    batch<DType::f16>(w, x, width, lanes, out);
}
void bf16_batch(const std::byte* w, const float* x, std::size_t width, std::size_t lanes, float* out) {
    batch<DType::bf16>(w, x, width, lanes, out);
}
void q8_0_batch(const std::byte* w, const float* x, std::size_t width, std::size_t lanes, float* out) {
    batch<DType::q8_0, decode_q8_0>(w, x, width, lanes, out);
}
void q4_0_batch(const std::byte* w, const float* x, std::size_t width, std::size_t lanes, float* out) {
    batch<DType::q4_0, decode_q4_0>(w, x, width, lanes, out);
}

} // namespace nathr::detail::x86
