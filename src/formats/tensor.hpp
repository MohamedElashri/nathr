#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace nathr::detail {

enum class DType : std::uint32_t {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    iq2_xxs = 16,
    iq2_xs = 17,
    iq3_xxs = 18,
    iq1_s = 19,
    iq4_nl = 20,
    iq3_s = 21,
    iq2_s = 22,
    iq4_xs = 23,
    iq1_m = 29,
    bf16 = 30,
    id_limit,
};

struct Encoding {
    DType type;
    std::string_view name;
    std::size_t elements;
    std::size_t bytes;
};

// GGUF encoding names and serialized block sizes.
inline constexpr Encoding encodings[] = {
    {DType::f32, "F32", 1, 4},
    {DType::f16, "F16", 1, 2},
    {DType::q4_0, "Q4_0", 32, 18},
    {DType::q4_1, "Q4_1", 32, 20},
    {DType::q5_0, "Q5_0", 32, 22},
    {DType::q5_1, "Q5_1", 32, 24},
    {DType::q8_0, "Q8_0", 32, 34},
    {DType::q2_k, "Q2_K", 256, 84},
    {DType::q3_k, "Q3_K", 256, 110},
    {DType::q4_k, "Q4_K", 256, 144},
    {DType::q5_k, "Q5_K", 256, 176},
    {DType::q6_k, "Q6_K", 256, 210},
    {DType::iq2_xxs, "IQ2_XXS", 256, 66},
    {DType::iq2_xs, "IQ2_XS", 256, 74},
    {DType::iq3_xxs, "IQ3_XXS", 256, 98},
    {DType::iq1_s, "IQ1_S", 256, 50},
    {DType::iq4_nl, "IQ4_NL", 32, 18},
    {DType::iq3_s, "IQ3_S", 256, 110},
    {DType::iq2_s, "IQ2_S", 256, 82},
    {DType::iq4_xs, "IQ4_XS", 256, 136},
    {DType::iq1_m, "IQ1_M", 256, 56},
    {DType::bf16, "BF16", 1, 2},
};

constexpr const Encoding* encoding(DType type) {
    for (const auto& item : encodings) {
        if (item.type == type)
            return &item;
    }
    return nullptr;
}

struct TensorDesc {
    DType dtype{};
    std::array<std::size_t, 4> shape{};
    std::size_t rank{};
    std::size_t elements{};

    [[nodiscard]] std::span<const std::size_t> dimensions() const noexcept {
        return std::span(shape).first(rank);
    }
};

struct TensorView {
    TensorDesc desc;
    std::span<const std::byte> bytes;
    // Nonzero means serialized rows have adjacent RoPE components.
    std::size_t rotary_head_width{};
};

inline std::size_t canonical_row(TensorView tensor, std::size_t stored) {
    if (!tensor.rotary_head_width)
        return stored;
    const auto within = stored % tensor.rotary_head_width;
    return stored - within + within / 2 + (within % 2) * (tensor.rotary_head_width / 2);
}

} // namespace nathr::detail
