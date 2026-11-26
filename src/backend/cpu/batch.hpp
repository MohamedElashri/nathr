#pragma once

#include "backend/cpu/scalar.hpp"

#include <bit>
#include <cstring>

namespace nathr::detail {

// Give each translation unit its own baseline or AVX2 instantiations.
namespace {

template <class Value> Value read_unaligned(const std::byte* bytes) {
    Value value;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
}

template <DType dtype> float value_at(const std::byte* bytes, std::size_t index) {
    static_assert(std::endian::native == std::endian::little);
    if constexpr (dtype == DType::f32) {
        return read_unaligned<float>(bytes + index * sizeof(float));
    } else {
        const auto bits = read_unaligned<std::uint16_t>(bytes + index * sizeof(std::uint16_t));
        if constexpr (dtype == DType::bf16)
            return bf16_to_float(bits);
        else
            return half_to_float(bits);
    }
}

template <std::size_t lanes, DType dtype, BlockDecode unpack>
void batch_row(const std::byte* bytes, const float* input, std::size_t width, float* output) {
    constexpr auto format = encoding(dtype);
    std::array<double, lanes> sums{};
    const auto add = [&](double weight, const float* activations) {
        for (std::size_t lane = 0; lane < lanes; ++lane)
            sums[lane] += weight * activations[lane];
    };
    if constexpr (format->elements == 1) {
        for (std::size_t column = 0; column < width; ++column)
            add(value_at<dtype>(bytes, column), input + column * lanes);
    } else {
        std::array<float, format->elements> block;
        for (std::size_t column = 0; column < width; column += block.size(), bytes += format->bytes) {
            unpack(bytes, block.data());
            for (std::size_t j = 0; j < block.size(); ++j)
                add(block[j], input + (column + j) * lanes);
        }
    }
    for (std::size_t lane = 0; lane < lanes; ++lane)
        output[lane] = static_cast<float>(sums[lane]);
}

template <DType dtype, BlockDecode unpack = nullptr>
void batch(const std::byte* bytes, const float* input, std::size_t width, std::size_t lanes, float* output) {
    if (lanes == 4)
        batch_row<4, dtype, unpack>(bytes, input, width, output);
    else if (lanes == 8)
        batch_row<8, dtype, unpack>(bytes, input, width, output);
    else
        batch_row<batch_tile, dtype, unpack>(bytes, input, width, output);
}

} // namespace
} // namespace nathr::detail
