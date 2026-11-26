#include "backend/cpu/scalar.hpp"
#include "backend/cpu/batch.hpp"
#include "quantization/decoders.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <functional>
#include <type_traits>

namespace nathr::detail {

float half_to_float(std::uint16_t bits) noexcept {
    const auto sign = (std::uint32_t{bits} & 0x8000) << 16;
    const auto exponent = (bits >> 10) & 31;
    const auto fraction = std::uint32_t{bits} & 1023;
    if (!exponent) {
        const float magnitude = std::ldexp(static_cast<float>(fraction), -24);
        return sign ? -magnitude : magnitude;
    }
    const auto expanded = static_cast<std::uint32_t>(exponent == 31 ? 255 : exponent + 112);
    return std::bit_cast<float>(sign | (expanded << 23) | (fraction << 13));
}

std::uint16_t float_to_half(float value) noexcept {
    const auto full = std::bit_cast<std::uint32_t>(value);
    const auto sign = (full >> 16) & 0x8000;
    const auto magnitude = full & 0x7fffffff;
    if (magnitude > 0x7f800000)
        return static_cast<std::uint16_t>(sign | 0x7e00 | ((magnitude >> 13) & 0x1ff));
    if (magnitude >= 0x47800000)
        return static_cast<std::uint16_t>(sign | 0x7c00);
    if (magnitude >= 0x38800000) {
        const auto rounded = magnitude + 0xfff + ((magnitude >> 13) & 1);
        return static_cast<std::uint16_t>(sign | ((rounded - 0x38000000) >> 13));
    }
    if (magnitude < 0x33000000)
        return static_cast<std::uint16_t>(sign);
    const auto shift = 126 - (magnitude >> 23);
    const auto significand = (magnitude & 0x7fffff) | 0x800000;
    const auto truncated = significand >> shift;
    const auto rounded = significand + ((1u << (shift - 1)) - 1) + (truncated & 1);
    return static_cast<std::uint16_t>(sign | (rounded >> shift));
}

float bf16_to_float(std::uint16_t bits) noexcept {
    return std::bit_cast<float>(std::uint32_t{bits} << 16);
}

template <DType type> void unpack_block(const std::byte* input, float* output) {
    static_assert(type == DType::q8_0 || type == DType::q4_0);
    std::uint16_t bits;
    std::memcpy(&bits, input, sizeof(bits));
    const auto scale = half_to_float(bits);
    const auto* bytes = input + 2;
    if constexpr (type == DType::q8_0) {
        for (std::size_t i = 0; i < 32; ++i)
            output[i] = scale * std::bit_cast<std::int8_t>(std::to_integer<std::uint8_t>(bytes[i]));
    } else {
        for (std::size_t i = 0; i < 16; ++i) {
            const int pair = std::to_integer<unsigned>(bytes[i]);
            output[i] = scale * ((pair & 15) - 8);
            output[i + 16] = scale * ((pair >> 4) - 8);
        }
    }
}

template void unpack_block<DType::q8_0>(const std::byte*, float*);
template void unpack_block<DType::q4_0>(const std::byte*, float*);

namespace {

template <DType dtype, BlockDecode unpack = nullptr>
float row_dot(const std::byte* bytes, const float* input, std::size_t count) {
    double total = 0;
    if constexpr (encoding(dtype)->elements > 1) {
        constexpr auto format = encoding(dtype);
        std::array<float, format->elements> block;
        for (std::size_t i = 0; i < count; i += block.size(), bytes += format->bytes) {
            unpack(bytes, block.data());
            for (std::size_t j = 0; j < block.size(); ++j)
                total += static_cast<double>(block[j]) * input[i + j];
        }
    } else {
        for (std::size_t i = 0; i < count; ++i)
            total += static_cast<double>(value_at<dtype>(bytes, i)) * input[i];
    }
    return static_cast<float>(total);
}

template <DType type, BlockDecode unpack> KernelChoice packed(std::string_view name) {
    return {row_dot<type, unpack>, name, unpack, batch<type, unpack>};
}

void dispatch(DType dtype, const auto& operation) {
    switch (dtype) {
    case DType::f32:
        return operation.template operator()<DType::f32>();
    case DType::f16:
        return operation.template operator()<DType::f16>();
    case DType::bf16:
        return operation.template operator()<DType::bf16>();
    default:
        throw Error(ErrorCode::unsupported, "expected a floating-point tensor encoding");
    }
}

double dot(std::span<const float> left, std::span<const float> right) {
    double result = 0;
    for (std::size_t i = 0; i < left.size(); ++i)
        result += static_cast<double>(left[i]) * right[i];
    return result;
}

} // namespace

KernelChoice scalar_kernel(DType type) {
    switch (type) {
    case DType::f32:
        return {row_dot<DType::f32>, "scalar-f32", nullptr, batch<DType::f32>};
    case DType::f16:
        return {row_dot<DType::f16>, "scalar-f16", nullptr, batch<DType::f16>};
    case DType::bf16:
        return {row_dot<DType::bf16>, "scalar-bf16", nullptr, batch<DType::bf16>};
    case DType::q4_0:
        return packed<DType::q4_0, unpack_block<DType::q4_0>>("scalar-q4_0");
    case DType::q8_0:
        return packed<DType::q8_0, unpack_block<DType::q8_0>>("scalar-q8_0");
    case DType::iq4_nl:
        return packed<DType::iq4_nl, decode_iq4_nl>("scalar-iq4_nl");
    case DType::q5_1:
        return packed<DType::q5_1, decode_q5_1>("scalar-q5_1");
    case DType::q5_0:
        return packed<DType::q5_0, decode_q5_0>("scalar-q5_0");
    case DType::q4_1:
        return packed<DType::q4_1, decode_q4_1>("scalar-q4_1");
    case DType::q5_k:
        return packed<DType::q5_k, decode_q5_k>("scalar-q5_k");
    case DType::q6_k:
        return packed<DType::q6_k, decode_q6_k>("scalar-q6_k");
    case DType::q4_k:
        return packed<DType::q4_k, decode_q4_k>("scalar-q4_k");
    case DType::q3_k:
        return packed<DType::q3_k, decode_q3_k>("scalar-q3_k");
    case DType::iq3_s:
        return packed<DType::iq3_s, decode_iq3_s>("scalar-iq3_s");
    case DType::iq4_xs:
        return packed<DType::iq4_xs, decode_iq4_xs>("scalar-iq4_xs");
    case DType::iq2_xs:
        return packed<DType::iq2_xs, decode_iq2_xs>("scalar-iq2_xs");
    case DType::q2_k:
        return packed<DType::q2_k, decode_q2_k>("scalar-q2_k");
    case DType::iq3_xxs:
        return packed<DType::iq3_xxs, decode_iq3_xxs>("scalar-iq3_xxs");
    case DType::iq2_s:
        return packed<DType::iq2_s, decode_iq2_s>("scalar-iq2_s");
    case DType::iq2_xxs:
        return packed<DType::iq2_xxs, decode_iq2_xxs>("scalar-iq2_xxs");
    case DType::iq1_m:
        return packed<DType::iq1_m, decode_iq1_m>("scalar-iq1_m");
    case DType::iq1_s:
        return packed<DType::iq1_s, decode_iq1_s>("scalar-iq1_s");
    default:
        return {};
    }
}

Cache::Cache(const ModelConfig& config, std::size_t limit, KvPrecision format)
    : precision(format), capacity(limit), kv_heads(config.kv_heads), width(config.head_dim),
      bank_elements(product({config.layers, kv_heads, capacity, width}, "KV cache")) {
    const auto count = product({2, bank_elements}, "KV cache");
    if (precision == KvPrecision::f16)
        storage.emplace<std::vector<std::uint16_t>>(count);
    else
        storage.emplace<std::vector<float>>(count);
}

std::size_t Cache::bytes() const noexcept {
    return std::visit([](const auto& bank) { return bank.capacity() * sizeof(bank.front()); }, storage);
}

void Scalar::embedding(TensorView table, Token token, std::span<float> output) {
    const auto& format = *encoding(table.desc.dtype);
    if (const auto unpack = scalar_kernel(table.desc.dtype).unpack) {
        const auto blocks = output.size() / format.elements;
        const auto* start = table.bytes.data() + token * blocks * format.bytes;
        for (std::size_t i = 0; i < blocks; ++i)
            unpack(start + i * format.bytes, output.data() + i * format.elements);
        return;
    }
    dispatch(table.desc.dtype, [&]<DType dtype> {
        const auto start = token * output.size();
        for (std::size_t i = 0; i < output.size(); ++i)
            output[i] = value_at<dtype>(table.bytes.data(), start + i);
    });
}

void Scalar::linear(TensorView weight, std::span<const float> input, std::span<float> output) {
    const auto kernel = scalar_kernel(weight.desc.dtype);
    if (!kernel.dot)
        throw Error(ErrorCode::unsupported, "no scalar kernel for tensor encoding");
    linear_with(kernel.dot, weight, input, output);
}

void Scalar::rms_norm(TensorView weight, std::span<const float> input, std::span<float> output,
                      float epsilon) {
    const double scale = 1.0 / std::sqrt(dot(input, input) / static_cast<double>(input.size()) + epsilon);
    dispatch(weight.desc.dtype, [&]<DType dtype> {
        for (std::size_t i = 0; i < input.size(); ++i)
            output[i] = static_cast<float>((input[i] * scale) * value_at<dtype>(weight.bytes.data(), i));
    });
}

void Scalar::rope(std::span<float> query, std::span<float> key, std::span<const float> frequency,
                  std::size_t position) {
    if (position == 0)
        return;
    const auto half = frequency.size();
    for (std::size_t i = 0; i < half; ++i) {
        const double angle = static_cast<double>(position) * frequency[i];
        const double cosine = std::cos(angle);
        const double sine = std::sin(angle);
        for (auto values : {query, key}) {
            for (std::size_t j = i; j < values.size(); j += 2 * half) {
                const auto a = values[j];
                const auto b = values[j + half];
                values[j] = static_cast<float>(a * cosine - b * sine);
                values[j + half] = static_cast<float>(b * cosine + a * sine);
            }
        }
    }
}

void Scalar::store(Cache& cache, std::size_t layer, std::size_t position, std::span<const float> key,
                   std::span<const float> value) {
    const auto write = [&](bool is_value, std::span<const float> source) {
        for (std::size_t kv = 0; kv < cache.kv_heads; ++kv) {
            const auto values = source.subspan(kv * cache.width, cache.width);
            if (cache.precision == KvPrecision::f32) {
                std::ranges::copy(values, cache.row<float>(is_value, layer, kv, position).begin());
            } else {
                auto dest = cache.row<std::uint16_t>(is_value, layer, kv, position);
                for (std::size_t i = 0; i < values.size(); ++i) {
                    const auto half = float_to_half(values[i]);
                    if ((half & 0x7c00) == 0x7c00)
                        throw Error(ErrorCode::backend, "KV value exceeds finite FP16 range");
                    dest[i] = half;
                }
            }
        }
    };
    write(false, key);
    write(true, value);
}

void Scalar::softmax(std::span<float> scores) {
    const double maximum = *std::ranges::max_element(scores);
    double sum_exp = 0;
    for (float& score : scores) {
        score = static_cast<float>(std::exp(score - maximum));
        sum_exp += score;
    }
    for (float& score : scores)
        score = static_cast<float>(score / sum_exp);
}

namespace {
struct DecodedHalves {
    std::array<float, 1u << 16> values;

    DecodedHalves() noexcept {
        for (std::size_t i = 0; i < values.size(); ++i)
            values[i] = half_to_float(static_cast<std::uint16_t>(i));
    }
};

const auto& decoded_half_values() {
    // Shared by sessions. Initialize only when the first FP16 cache is read.
    static const DecodedHalves table;
    return table.values;
}

template <class T, class Decode>
void attend(const ModelConfig& config, std::size_t layer, std::size_t position, Cache* cache,
            std::span<const float> keys, std::span<const float> values, std::span<const float> query,
            std::span<float> output, std::span<float> scores, Decode decode) {
    const auto width = config.head_dim;
    const auto kv_width = config.kv_heads * width;
    const auto group = config.query_heads / config.kv_heads;
    const auto scale = 1.0 / std::sqrt(static_cast<double>(width));
    scores = scores.first(position + 1);
    for (std::size_t head = 0; head < config.query_heads; ++head) {
        const auto kv = head / group;
        const auto q = query.subspan(head * width, width);
        auto out = output.subspan(head * width, width);
        const auto get = [&](bool value, std::size_t i) -> std::span<const T> {
            if constexpr (std::is_same_v<T, float>) {
                if (!cache)
                    return (value ? values : keys).subspan(i * kv_width + kv * width, width);
            }
            return cache->row<T>(value, layer, kv, i);
        };
        for (std::size_t i = 0; i <= position; ++i) {
            const auto k = get(false, i);
            double sum = 0;
            for (std::size_t d = 0; d < width; ++d)
                sum += static_cast<double>(q[d]) * decode(k[d]);
            scores[i] = static_cast<float>(sum * scale);
        }
        Scalar::softmax(scores);
        for (std::size_t offset = 0; offset < width; offset += 64) {
            std::array<double, 64> sums{};
            const auto lanes = std::min(sums.size(), width - offset);
            for (std::size_t i = 0; i <= position; ++i) {
                const auto v = get(true, i);
                for (std::size_t d = 0; d < lanes; ++d)
                    sums[d] += static_cast<double>(scores[i]) * decode(v[offset + d]);
            }
            for (std::size_t d = 0; d < lanes; ++d)
                out[offset + d] = static_cast<float>(sums[d]);
        }
    }
}
} // namespace

void Scalar::attention(const ModelConfig& config, std::size_t layer, std::size_t position, Cache* cache,
                       std::span<const float> keys, std::span<const float> values,
                       std::span<const float> query, std::span<float> output, std::span<float> scores) {
    if (cache && cache->precision == KvPrecision::f16) {
        const auto& decoded = decoded_half_values();
        attend<std::uint16_t>(config, layer, position, cache, keys, values, query, output, scores,
                              [&decoded](std::uint16_t bits) { return decoded[bits]; });
    } else {
        attend<float>(config, layer, position, cache, keys, values, query, output, scores, std::identity{});
    }
}

void Scalar::swiglu(std::span<float> gate, std::span<const float> up) {
    for (std::size_t i = 0; i < gate.size(); ++i) {
        const double g = gate[i];
        gate[i] = static_cast<float>((g / (1.0 + std::exp(-g))) * up[i]);
    }
}

void Scalar::add(std::span<float> output, std::span<const float> residual) {
    for (std::size_t i = 0; i < output.size(); ++i)
        output[i] += residual[i];
}

} // namespace nathr::detail
