#pragma once

#include "formats/tensor.hpp"
#include <nathr/nathr.hpp>

namespace nathr::detail {

class ThreadPool;
inline constexpr std::size_t batch_tile = 16;

using RowDot = float (*)(const std::byte*, const float*, std::size_t);
using BlockDecode = void (*)(const std::byte*, float*);
using RowBatch = void (*)(const std::byte*, const float*, std::size_t, std::size_t, float*);
struct KernelChoice {
    RowDot dot{};
    std::string_view name;
    BlockDecode unpack{};
    RowBatch batch{};
};

struct FeatureInputs {
    bool avx{}, osxsave{}, avx2{}, fma{}, f16c{};
    std::uint64_t xcr0{};
};

constexpr CpuFeatures usable_features(FeatureInputs input) {
    const bool avx_state = input.avx && input.osxsave && (input.xcr0 & 6) == 6;
    return {avx_state && input.avx2, avx_state && input.fma, avx_state && input.f16c};
}

KernelChoice scalar_kernel(DType type);
KernelChoice select_kernel(DType type, ExecutionPolicy policy, CpuFeatures features);
void project(const KernelChoice& kernel, const ThreadPool& pool, TensorView weight,
             std::span<const float> input, std::span<float> output, std::size_t tokens,
             std::span<float> packed);

} // namespace nathr::detail
