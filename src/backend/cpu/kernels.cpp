#include "backend/cpu/kernels.hpp"
#include "backend/cpu/thread_pool.hpp"
#include "backend/cpu/x86/avx2.hpp"

#include <algorithm>
#include <cpuid.h>

namespace nathr {

CpuFeatures cpu_features() noexcept {
    unsigned eax, ebx, ecx, edx;
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx))
        return {};
    detail::FeatureInputs input{};
    input.avx = ecx & bit_AVX;
    input.osxsave = ecx & bit_OSXSAVE;
    input.fma = ecx & bit_FMA;
    input.f16c = ecx & bit_F16C;
    if (input.avx && input.osxsave) {
        unsigned lo, hi;
        asm volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        input.xcr0 = std::uint64_t{lo} | (std::uint64_t{hi} << 32);
    }
    input.avx2 = __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) && (ebx & bit_AVX2);
    return detail::usable_features(input);
}

namespace detail {

KernelChoice select_kernel(DType type, ExecutionPolicy policy, CpuFeatures features) {
    if (policy == ExecutionPolicy::avx2 && !features.avx2)
        throw Error(ErrorCode::unsupported, "AVX2 was requested but CPU or OS AVX2 state is unavailable");
    if (policy != ExecutionPolicy::scalar && features.avx2) {
        switch (type) {
        case DType::f32:
            return {x86::f32_dot, "avx2-f32", nullptr, x86::f32_batch};
        case DType::f16:
            if (features.f16c)
                return {x86::f16_dot, "avx2-f16c", nullptr, x86::f16_batch};
            break;
        case DType::bf16:
            return {x86::bf16_dot, "avx2-bf16", nullptr, x86::bf16_batch};
        case DType::q8_0:
            return {x86::q8_0_dot, "avx2-q8_0", nullptr, x86::q8_0_batch};
        case DType::q4_0:
            return {x86::q4_0_dot, "avx2-q4_0", nullptr, x86::q4_0_batch};
        default:
            break;
        }
    }
    return scalar_kernel(type);
}

namespace {
void project_rows(RowDot kernel, TensorView weight, std::span<const float> input, std::span<float> output,
                  std::size_t begin, std::size_t end) {
    const auto& format = *encoding(weight.desc.dtype);
    const auto stride = input.size() / format.elements * format.bytes;
    for (std::size_t row = begin; row < end; ++row) {
        output[canonical_row(weight, row)] =
            kernel(weight.bytes.data() + row * stride, input.data(), input.size());
    }
}
} // namespace

void linear_with(RowDot kernel, TensorView weight, std::span<const float> input, std::span<float> output) {
    project_rows(kernel, weight, input, output, 0, output.size());
}

void project(const KernelChoice& kernel, const ThreadPool& pool, TensorView weight,
             std::span<const float> input, std::span<float> output, std::size_t tokens,
             std::span<float> packed) {
    const auto width = input.size() / tokens;
    const auto rows = output.size() / tokens;
    const auto& format = *encoding(weight.desc.dtype);
    const auto stride = width / format.elements * format.bytes;
    for (std::size_t start = 0; start < tokens; start += batch_tile) {
        const auto count = std::min(batch_tile, tokens - start);
        if (count == 1) {
            pool.rows(rows, width, [&](std::size_t begin, std::size_t end) {
                project_rows(kernel.dot, weight, input.subspan(start * width, width),
                             output.subspan(start * rows, rows), begin, end);
            });
            continue;
        }
        // A contiguous set of activations for each column lets a row of weights
        // serve several tokens without changing any individual dot's reduction order.
        const auto lanes = count <= 4 ? 4 : count <= 8 ? 8 : batch_tile;
        for (std::size_t column = 0; column < width; ++column) {
            for (std::size_t i = 0; i < count; ++i)
                packed[column * lanes + i] = input[(start + i) * width + column];
            std::fill_n(packed.data() + column * lanes + count, lanes - count, 0.f);
        }
        pool.rows(rows, width * count, [&](std::size_t begin, std::size_t end) {
            for (std::size_t r = begin; r < end; ++r) {
                std::array<float, batch_tile> projected;
                kernel.batch(weight.bytes.data() + r * stride, packed.data(), width, lanes, projected.data());
                const auto row = canonical_row(weight, r);
                for (std::size_t i = 0; i < count; ++i)
                    output[(start + i) * rows + row] = projected[i];
            }
        });
    }
}

} // namespace detail
} // namespace nathr
