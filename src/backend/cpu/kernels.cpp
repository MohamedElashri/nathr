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

void project(const ThreadPool& pool, std::span<const KernelChoice> kernels,
             std::span<const Projection> projections, std::span<const float> input, std::size_t tokens,
             std::span<float> packed) {
    const auto width = input.size() / tokens;
    std::size_t total_rows = 0;
    for (const auto& projection : projections)
        total_rows += projection.output.size() / tokens;
    for (std::size_t start = 0; start < tokens; start += batch_tile) {
        const auto count = std::min(batch_tile, tokens - start);
        const auto lanes = count <= 2 ? 2 : count <= 4 ? 4 : count <= 8 ? 8 : batch_tile;
        if (count > 1) {
            // Pack each activation tile once for every weight matrix that uses it.
            for (std::size_t column = 0; column < width; ++column) {
                for (std::size_t i = 0; i < count; ++i)
                    packed[column * lanes + i] = input[(start + i) * width + column];
                std::fill_n(packed.data() + column * lanes + count, lanes - count, 0.f);
            }
        }
        pool.rows(total_rows, width * count, [&](std::size_t begin, std::size_t end) {
            for (const auto& [weight, output] : projections) {
                const auto rows = output.size() / tokens;
                if (begin < rows) {
                    const auto& format = *encoding(weight.desc.dtype);
                    const auto stride = width / format.elements * format.bytes;
                    const auto& kernel = kernels[static_cast<std::size_t>(weight.desc.dtype)];
                    const auto last = std::min(end, rows);
                    if (count == 1) {
                        const auto* activations = input.data() + start * width;
                        auto result = output.subspan(start * rows, rows);
                        for (std::size_t r = begin; r < last; ++r) {
                            result[canonical_row(weight, r)] =
                                kernel.dot(weight.bytes.data() + r * stride, activations, width);
                        }
                    } else {
                        for (std::size_t r = begin; r < last; ++r) {
                            std::array<float, batch_tile> projected;
                            kernel.batch(weight.bytes.data() + r * stride, packed.data(), width, lanes,
                                         projected.data());
                            const auto row = canonical_row(weight, r);
                            for (std::size_t i = 0; i < count; ++i)
                                output[(start + i) * rows + row] = projected[i];
                        }
                    }
                }
                if (end <= rows)
                    break;
                begin -= std::min(begin, rows);
                end -= rows;
            }
        });
    }
}

} // namespace detail
} // namespace nathr
