#pragma once

#include "backend/cpu/kernels.hpp"
#include "core/input.hpp"

#include <variant>

namespace nathr::detail {

float half_to_float(std::uint16_t bits) noexcept;
std::uint16_t float_to_half(float value) noexcept;
float bf16_to_float(std::uint16_t bits) noexcept;

struct Cache {
    Cache(const ModelConfig& config, std::size_t capacity, KvPrecision precision = KvPrecision::f32);
    template <class T>
    [[nodiscard]] std::span<T> row(bool value, std::size_t layer, std::size_t head, std::size_t position) {
        const auto index =
            (value ? bank_elements : 0) + ((layer * kv_heads + head) * capacity + position) * width;
        return std::span(std::get<std::vector<T>>(storage)).subspan(index, width);
    }
    [[nodiscard]] std::size_t bytes() const noexcept;
    const KvPrecision precision;
    std::size_t capacity, kv_heads, width, bank_elements;
    std::variant<std::vector<float>, std::vector<std::uint16_t>> storage;
};

struct Scalar {
    static void embedding(TensorView table, Token token, std::span<float> output);
    static void rms_norm(TensorView weight, std::span<const float> input, std::span<float> output,
                         float epsilon);
    static void rope(std::span<float> query, std::span<float> key, std::span<const float> frequency,
                     std::size_t position);
    static void store(Cache& cache, std::size_t layer, std::size_t position, std::span<const float> key,
                      std::span<const float> value);
    static void attention(const ModelConfig& config, std::size_t layer, std::size_t position, Cache* cache,
                          std::span<const float> keys, std::span<const float> values,
                          std::span<const float> query, std::span<float> output, std::span<float> scores);
    static void softmax(std::span<float> scores);
    static void swiglu(std::span<float> gate, std::span<const float> up);
    static void add(std::span<float> output, std::span<const float> residual);
};

} // namespace nathr::detail
