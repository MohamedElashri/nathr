#include "backend/cpu/x86/avx2.hpp"
#include "backend/cpu/batch.hpp"
#include "backend/cpu/scalar.hpp"

#include <cstring>
#include <immintrin.h>

namespace nathr::detail::x86 {
namespace {

template <class Value> Value read(const std::byte* p) {
    Value value;
    std::memcpy(&value, p, sizeof(value));
    return value;
}

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
        const auto scale = _mm256_set1_pd(half_to_float(read<std::uint16_t>(weights)));
        auto a = read<__m128i>(weights + 2);
        __m128i b;
        if constexpr (type == DType::q8_0) {
            b = read<__m128i>(weights + 18);
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

} // namespace

float f32_dot(const std::byte* weights, const float* input, std::size_t count) {
    auto total = _mm256_setzero_pd();
    auto second = _mm256_setzero_pd();
    std::size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        total = accumulate8(total, read<__m256>(weights + i * 4), input + i);
        second = accumulate8(second, read<__m256>(weights + (i + 8) * 4), input + i + 8);
    }
    total = _mm256_add_pd(total, second);
    for (; i + 8 <= count; i += 8)
        total = accumulate8(total, read<__m256>(weights + i * 4), input + i);
    double sum = horizontal_sum(total);
    for (; i < count; ++i)
        sum += static_cast<double>(read<float>(weights + i * 4)) * input[i];
    return static_cast<float>(sum);
}

__attribute__((target("avx2,f16c"))) float f16_dot(const std::byte* weights, const float* input,
                                                   std::size_t count) {
    auto total = _mm256_setzero_pd();
    std::size_t i = 0;
    for (; i + 8 <= count; i += 8)
        total = accumulate8(total, _mm256_cvtph_ps(read<__m128i>(weights + i * 2)), input + i);
    double sum = horizontal_sum(total);
    for (; i < count; ++i)
        sum += static_cast<double>(half_to_float(read<std::uint16_t>(weights + i * 2))) * input[i];
    return static_cast<float>(sum);
}

float bf16_dot(const std::byte* weights, const float* input, std::size_t count) {
    auto total = _mm256_setzero_pd();
    std::size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        const auto ints = _mm256_cvtepu16_epi32(read<__m128i>(weights + i * 2));
        total = accumulate8(total, _mm256_castsi256_ps(_mm256_slli_epi32(ints, 16)), input + i);
    }
    double sum = horizontal_sum(total);
    for (; i < count; ++i)
        sum += static_cast<double>(bf16_to_float(read<std::uint16_t>(weights + i * 2))) * input[i];
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
    batch<DType::q8_0, unpack_block<DType::q8_0>>(w, x, width, lanes, out);
}
void q4_0_batch(const std::byte* w, const float* x, std::size_t width, std::size_t lanes, float* out) {
    batch<DType::q4_0, unpack_block<DType::q4_0>>(w, x, width, lanes, out);
}

} // namespace nathr::detail::x86
