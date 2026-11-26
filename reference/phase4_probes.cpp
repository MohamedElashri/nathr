#include "backend/cpu/scalar.hpp"
#include "formats/gguf.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <numeric>

namespace {
using namespace nathr;
using namespace nathr::detail;

template <class Operation> Json measure(const Operation& op, std::span<const float> output) {
    std::array<double, 7> samples{};
    double checksum = 0;
    for (int repetition = -2; repetition < static_cast<int>(samples.size()); ++repetition) {
        const auto start = std::chrono::steady_clock::now();
        op();
        const auto end = std::chrono::steady_clock::now();
        if (repetition >= 0)
            samples[repetition] = std::chrono::duration<double, std::milli>(end - start).count();
        checksum += std::accumulate(output.begin(), output.end(), 0.0);
    }
    auto sorted = samples;
    std::ranges::sort(sorted);
    return {{"samples_ms", samples}, {"median_ms", sorted[3]}, {"checksum", checksum}};
}

struct Activation {
    float scale;
    std::array<std::int8_t, 32> codes;
};

void quantize(std::span<const float> x, std::span<Activation> blocks) {
    for (auto& block : blocks) {
        float largest = 0;
        for (float value : x.first(32))
            largest = std::max(largest, std::abs(value));
        block.scale = largest / 127.f;
        block.codes.fill(0);
        if (block.scale != 0) {
            for (std::size_t i = 0; i < 32; ++i)
                block.codes[i] = static_cast<std::int8_t>(
                    std::clamp(std::lrint(static_cast<double>(x[i]) / block.scale), -127L, 127L));
        }
        x = x.subspan(32);
    }
}

__attribute__((target("avx2"))) __m256i signed_pairs(const std::byte* a, const std::int8_t* b) {
    __m128i lhs, rhs;
    std::memcpy(&lhs, a, sizeof(lhs));
    std::memcpy(&rhs, b, sizeof(rhs));
    return _mm256_madd_epi16(_mm256_cvtepi8_epi16(lhs), _mm256_cvtepi8_epi16(rhs));
}

__attribute__((target("avx2"))) int signed_dot(const std::byte* weight, const std::int8_t* activation) {
    const auto sums =
        _mm256_add_epi32(signed_pairs(weight, activation), signed_pairs(weight + 16, activation + 16));
    const auto halves = _mm_add_epi32(_mm256_castsi256_si128(sums), _mm256_extracti128_si256(sums, 1));
    const auto lower = _mm_add_epi32(halves, _mm_srli_si128(halves, 8));
    return _mm_cvtsi128_si32(_mm_add_epi32(lower, _mm_srli_si128(lower, 4)));
}

float integer_row(const std::byte* weight, std::span<const Activation> blocks, bool vectorized) {
    double sum = 0;
    for (const auto& block : blocks) {
        std::uint16_t bits;
        std::memcpy(&bits, weight, sizeof(bits));
        int dot = 0;
        if (vectorized)
            dot = signed_dot(weight + 2, block.codes.data());
        else {
            for (std::size_t j = 0; j < 32; ++j)
                dot += std::bit_cast<std::int8_t>(weight[2 + j]) * block.codes[j];
        }
        sum += static_cast<double>(half_to_float(bits)) * block.scale * dot;
        weight += 34;
    }
    return static_cast<float>(sum);
}

Json integer_probe(TensorView weight, std::size_t rows, std::size_t columns) {
    if (weight.desc.dtype != DType::q8_0)
        throw std::runtime_error("integer probe requires Q8_0 weights");
    const auto stride = columns / 32 * 34;
    std::vector<float> x(columns), exact(rows), approximate(rows);
    std::vector<Activation> blocks(columns / 32);
    for (std::size_t i = 0; i < x.size(); ++i)
        x[i] = static_cast<float>(static_cast<int>((i * 7919) % 1071) - 535) / 535;
    const auto base = select_kernel(DType::q8_0, ExecutionPolicy::avx2, cpu_features());
    const auto exact_samples = measure([&] { linear_with(base.dot, weight, x, exact); }, exact);
    const auto integer_samples = measure(
        [&] {
            quantize(x, blocks);
            for (std::size_t row = 0; row < rows; ++row)
                approximate[row] = integer_row(weight.bytes.data() + row * stride, blocks, true);
        },
        approximate);
    double error = 0, norm = 0;
    for (std::size_t row = 0; row < rows; ++row) {
        if (approximate[row] != integer_row(weight.bytes.data() + row * stride, blocks, false))
            throw std::runtime_error("vector and scalar signed integer dot disagree");
        const auto delta = static_cast<double>(approximate[row]) - exact[row];
        error += delta * delta;
        norm += static_cast<double>(exact[row]) * exact[row];
    }
    return {{"rows", rows},
            {"columns", columns},
            {"fp32_activation", exact_samples},
            {"int8_activation", integer_samples},
            {"relative_l2", std::sqrt(error / norm)}};
}

void narrow_attention(const ModelConfig& c, Cache& cache, std::span<const float> q, std::span<float> output,
                      std::span<float> scores) {
    const auto width = cache.width;
    const auto scale = 1.0 / std::sqrt(static_cast<double>(width));
    for (std::size_t head = 0; head < c.query_heads; ++head) {
        const auto kv = head / (c.query_heads / c.kv_heads);
        for (std::size_t position = 0; position < scores.size(); ++position) {
            auto key = cache.row<float>(false, 0, kv, position);
            double dot = 0;
            for (std::size_t d = 0; d < width; ++d)
                dot += static_cast<double>(q[head * width + d]) * key[d];
            scores[position] = static_cast<float>(dot * scale);
        }
        Scalar::softmax(scores);
        for (std::size_t offset = 0; offset < width; offset += 8) {
            std::array<double, 8> sums{};
            for (std::size_t position = 0; position < scores.size(); ++position) {
                auto value = cache.row<float>(true, 0, kv, position);
                for (std::size_t j = 0; j < sums.size(); ++j)
                    sums[j] += static_cast<double>(scores[position]) * value[offset + j];
            }
            for (std::size_t j = 0; j < sums.size(); ++j)
                output[head * width + offset + j] = static_cast<float>(sums[j]);
        }
    }
}

Json attention_probe() {
    ModelConfig config{};
    config.layers = 1;
    config.query_heads = 9;
    config.kv_heads = 3;
    config.head_dim = 64;
    std::array<float, 9 * 64> q{}, reference{}, actual{};
    for (std::size_t i = 0; i < q.size(); ++i)
        q[i] = static_cast<float>(i % 31) / 31;
    Json report;
    for (std::size_t context : {128u, 2048u, 8192u}) {
        Cache cache(config, context);
        std::vector<float> scores(context);
        std::array<float, 3 * 64> key{}, value{};
        for (std::size_t position = 0; position < context; ++position) {
            for (std::size_t d = 0; d < value.size(); ++d) {
                key[d] = static_cast<float>((position + d * 19) % 101) / 101;
                value[d] = static_cast<float>((position * 11 + d) % 167) / 167;
            }
            Scalar::store(cache, 0, position, key, value);
        }
        const auto before =
            measure([&] { narrow_attention(config, cache, q, reference, scores); }, reference);
        const auto after = measure(
            [&] { Scalar::attention(config, 0, context - 1, &cache, {}, {}, q, actual, scores); }, actual);
        if (actual != reference)
            throw std::runtime_error("attention cache tile differs from baseline");
        report[std::to_string(context)] = {{"tile8", before}, {"tile64", after}};
    }
    return report;
}
} // namespace

int main(int argc, char** argv) try {
    if (argc == 2 && std::string_view(argv[1]) == "attention") {
        std::cout << attention_probe().dump(2) << '\n';
        return 0;
    }
    if (argc != 3 || std::string_view(argv[1]) != "integer" || !cpu_features().avx2)
        throw std::runtime_error("usage: nathr_phase4_probes attention | integer Q8_0.gguf (requires AVX2)");
    std::array<std::byte, 32> extrema;
    std::array<std::int8_t, 32> activations;
    extrema.fill(std::byte{0x80});
    for (const auto sign : {-127, 127}) {
        activations.fill(static_cast<std::int8_t>(sign));
        if (signed_dot(extrema.data(), activations.data()) != -128 * sign * 32)
            throw std::runtime_error("signed-extrema integer dot");
    }
    Gguf file(argv[2]);
    const auto c = file.config();
    const auto down = file.take("blk.0.ffn_down.weight", {c.hidden, c.intermediate});
    const auto output =
        file.take(c.tied_embeddings ? "token_embd.weight" : "output.weight", {c.vocabulary, c.hidden});
    const Json report{{"down", integer_probe(down, c.hidden, c.intermediate)},
                      {"output", integer_probe(output, c.vocabulary, c.hidden)}};
    std::cout << report.dump(2) << '\n';
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
