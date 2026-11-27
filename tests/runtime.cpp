#include <nathr/batch.hpp>
#include <nathr/generation.hpp>
#include <nathr/tokenizer.hpp>

#include "allocation_counter.hpp"
#include "backend/cpu/scalar.hpp"
#include "backend/cpu/thread_pool.hpp"
#include "core/sha256.hpp"
#include "formats/gguf.hpp"
#include "formats/safetensors.hpp"
#include "models/llama/config.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <new>
#include <sys/types.h>
#include <thread>
#include <unistd.h>

namespace {
using namespace nathr;
using namespace nathr::detail;
const std::filesystem::path fixtures = NATHR_FIXTURES;
const auto fixture = fixtures / "model";
constexpr std::array<Token, 8> ids{1, 7, 2, 18, 0, 3, 21, 13};

void expect(bool truth, std::string_view description) {
    if (!truth)
        throw std::runtime_error(std::string(description));
}

void rejects(const auto& operation, ErrorCode code) {
    try {
        operation();
    } catch (const Error& error) {
        if (error.code() != code)
            throw;
        return;
    }
    throw std::runtime_error("expected a typed error");
}

void close(float actual, double expected, double epsilon, std::string_view description) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > epsilon) {
        throw std::runtime_error(std::string(description) + ": actual " + std::to_string(actual) +
                                 ", expected " + std::to_string(expected));
    }
}

template <std::size_t Size> TensorView floats(const std::array<float, Size>& values) {
    return {{DType::f32, {}, 1, values.size()}, std::as_bytes(std::span(values))};
}

void project(const KernelChoice& kernel, const ThreadPool& pool, TensorView weight,
             std::span<const float> input, std::span<float> output, std::size_t tokens,
             std::span<float> packed) {
    std::array<KernelChoice, static_cast<std::size_t>(DType::id_limit)> kernels{};
    kernels[static_cast<std::size_t>(weight.desc.dtype)] = kernel;
    const std::array targets{Projection{weight, output}};
    detail::project(pool, kernels, targets, input, tokens, packed);
}

void kernels() {
    close(half_to_float(0x3c00), 1.0, 0, "F16 one");
    close(half_to_float(1), std::ldexp(1.0, -24), 0, "F16 subnormal");
    close(half_to_float(0x7bff), 65504, 0, "F16 maximum finite");
    expect(std::signbit(half_to_float(0x8000)), "F16 negative zero");
    expect(std::isinf(half_to_float(0x7c00)) && std::isnan(half_to_float(0x7e00)), "F16 specials");
    close(bf16_to_float(0xc040), -3.0, 0, "BF16 minus three");
    expect(std::bit_cast<std::uint32_t>(bf16_to_float(1)) == 0x10000u, "BF16 subnormal");
    expect(std::isinf(bf16_to_float(0x7f80)) && std::isnan(bf16_to_float(0x7fc0)), "BF16 specials");

    const auto ref = read_json(fixtures / "math.json", mib);
    const ThreadPool serial(1);
    const auto scalar = scalar_kernel(DType::f32);
    const std::array x{1.0f, -2.0f, 3.0f};
    const std::array y{-3.0f, 2.0f, 4.0f};
    std::array<float, 1> dot{};
    project(scalar, serial, floats(y), x, dot, 1, {});
    close(dot[0], ref.at("dot").get<double>(), 0, "FP64 reference linear");

    std::array<std::byte, sizeof(y) + 1> bytes{};
    std::ranges::copy(std::as_bytes(std::span(y)), bytes.begin() + 1);
    project(scalar, serial, {floats(y).desc, std::span(bytes).subspan(1)}, x, dot, 1, {});
    close(dot[0], ref.at("dot").get<double>(), 0, "unaligned linear");

    std::array<float, 513> cancellation{}, weights{};
    double sum = 0, magnitude = 0;
    for (std::size_t i = 0; i < cancellation.size(); ++i) {
        weights[i] = (i % 2 ? 1.0f : -1.0f) * (1.0f + static_cast<float>(i % 7) / 65.0f);
        cancellation[i] = 1.0f + static_cast<float>(i % 13) / 53.0f;
        const double term = static_cast<double>(weights[i]) * static_cast<double>(cancellation[i]);
        sum += term;
        magnitude += std::abs(term);
    }
    project(scalar, serial, floats(weights), cancellation, dot, 1, {});
    const double n_u = (weights.size() + 1) * std::numeric_limits<float>::epsilon() / 2;
    close(dot[0], sum, n_u / (1 - n_u) * magnitude, "FP64 linear cancellation bound");

    std::array<float, 17> cancelling_weights{}, ones;
    cancelling_weights[0] = 0x1p25f;
    cancelling_weights[8] = 1.0f;
    cancelling_weights[16] = -0x1p25f;
    ones.fill(1.0f);
    const auto cancelling_view = floats(cancelling_weights);
    for (const auto policy : {ExecutionPolicy::scalar, ExecutionPolicy::automatic}) {
        const auto kernel = select_kernel(DType::f32, policy, cpu_features());
        project(kernel, serial, cancelling_view, ones, dot, 1, {});
        close(dot[0], 1.0, 0.0, "preserve small term through scalar and SIMD cancellation");
    }

    std::array<float, 3> norm{};
    Scalar::rms_norm(floats(y), x, norm, 1e-5f);
    for (std::size_t i = 0; i < norm.size(); ++i)
        close(norm[i], ref["rmsnorm"][i].get<double>(), 1e-6, "FP64 RMSNorm");
    auto s = x;
    Scalar::softmax(s);
    for (std::size_t i = 0; i < s.size(); ++i)
        close(s[i], ref["softmax"][i].get<double>(), 1e-6, "FP64 softmax");
    auto shifted = std::array{1001.0f, 998.0f, 1003.0f};
    Scalar::softmax(shifted);
    expect(s == shifted, "softmax shift invariance");
    std::array keys{0.0f, 1.0f, 2.0f, 2.0f, 1.0f, -3.0f};
    std::array values{1.0f, 2.0f, 0.0f, 4.0f, -2.0f, 0.0f};
    std::array query{1.0f, -2.0f, 3.0f, 1.0f, -2.0f, 3.0f};
    std::array<float, 6> attention{};
    std::array<float, 2> scores{};
    auto config = ModelConfig{};
    config.head_dim = 3;
    config.kv_heads = 1;
    config.query_heads = 2;
    Scalar::attention(config, 0, 1, nullptr, keys, values, query, attention, scores);
    for (std::size_t i = 0; i < 2; ++i) {
        close(attention[i], ref["attention"]["output"][i].get<double>(), 1e-6, "FP64 GQA");
        expect(attention[i] == attention[i + 3], "grouped query mapping");
    }
    Scalar::attention(config, 0, 0, nullptr, keys, values, query, attention, scores);
    expect(attention == std::array{1.0f, 2.0f, 0.0f, 1.0f, 2.0f, 0.0f}, "future positions are masked");

    // Compare FP16 reads with FP32 cache entries rounded through the same half bits.
    config.layers = 1;
    config.head_dim = 66;
    Cache rounded(config, 3), packed(config, 3, KvPrecision::f16);
    rounded.ensure(3);
    packed.ensure(3);
    constexpr std::uint16_t half_bits[] = {0x8000, 0x0001, 0x83ff, 0x0400, 0xb2a1, 0x3c00, 0x4355};
    for (const bool value : {false, true}) {
        for (std::size_t pos = 0; pos < 3; ++pos) {
            auto full = rounded.row<float>(value, 0, 0, pos);
            auto half = packed.row<std::uint16_t>(value, 0, 0, pos);
            for (std::size_t d = 0; d < half.size(); ++d) {
                half[d] = half_bits[(d + 3 * pos + value) % std::size(half_bits)];
                full[d] = half_to_float(half[d]);
            }
        }
    }
    std::array<float, 132> packed_query{}, rounded_out{}, packed_out{};
    std::array<float, 3> rounded_scores{}, packed_scores{};
    for (std::size_t d = 0; d < packed_query.size(); ++d)
        packed_query[d] = static_cast<float>(static_cast<int>(d % 19) - 9) / 16.f;
    Scalar::attention(config, 0, 2, &rounded, {}, {}, packed_query, rounded_out, rounded_scores);
    Scalar::attention(config, 0, 2, &packed, {}, {}, packed_query, packed_out, packed_scores);
    expect(packed_out == rounded_out && packed_scores == rounded_scores,
           "FP16 cached GQA matches exact reads");

    const std::array frequency{1.0f, 0.125f};
    for (std::size_t pos : {0, 1, 127, 8191}) {
        std::array q{1.0f, 0.0f, 0.0f, 1.0f};
        auto k = q;
        Scalar::rope(q, k, frequency, pos);
        close(q[0], std::cos(static_cast<double>(pos)), 1e-7, "RoPE split half real");
        close(q[2], std::sin(static_cast<double>(pos)), 1e-7, "RoPE split half imaginary");
        close(q[1], -std::sin(static_cast<double>(pos) / 8), 1e-7, "RoPE second coordinate");
        expect(q == k, "same RoPE convention applied to Q and K");
    }
    std::array gate{-3.0f, 0.0f, 2.0f};
    Scalar::swiglu(gate, x);
    close(gate[0], -3.0 / (1 + std::exp(3.0)), 1e-7, "SiLU negative");
    close(gate[1], 0, 0, "SiLU zero");
    close(gate[2], 6.0 / (1 + std::exp(-2.0)), 1e-6, "SiLU positive");
    Scalar::add(gate, x);
    close(gate[1], -2, 0, "residual addition");
}

class Temporary {
  public:
    Temporary() {
        auto pattern = std::filesystem::temp_directory_path() / "nathr-test-XXXXXX";
        std::string path = pattern.string();
        auto* result = mkdtemp(path.data());
        if (!result)
            throw std::runtime_error("mkdtemp failed");
        directory = result;
    }
    ~Temporary() {
        std::error_code error;
        std::filesystem::remove_all(directory, error);
    }
    std::filesystem::path directory;
};

void parser() {
    const auto parse = [](std::string_view content) {
        return parse_json(std::as_bytes(std::span(content)), "test.json");
    };
    expect(parse(R"({"a":{"x":1},"b":{"x":2}})").is_object(), "nested unique JSON keys");
    rejects([&] { (void)parse(R"({"a":{"x":1,"x":2}})"); }, ErrorCode::malformed);
    rejects([&] { (void)parse(R"({"a":-1,})"); }, ErrorCode::malformed);
    const std::string_view masked_tail("{}\0{}", 5);
    rejects([&] { (void)parse(masked_tail); }, ErrorCode::malformed);
    rejects([&] { (void)product({std::numeric_limits<std::size_t>::max(), 2}, "test"); },
            ErrorCode::resource_limit);

    Temporary temp;
    const auto path = temp.directory / "model.safetensors";
    const auto write = [&](std::string_view header, std::string_view payload) {
        const auto length = static_cast<std::uint64_t>(header.size());
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(&length), sizeof(length));
        file.write(header.data(), static_cast<std::streamsize>(header.size()));
        file.write(payload.data(), static_cast<std::streamsize>(payload.size()));
    };
    write(
        R"({"a":{"shape":[],"dtype":"F32","data_offsets":[0,4]},"b":{"shape":[0],"dtype":"BF16","data_offsets":[4,4]}})",
        "abcd");
    SafeTensors valid(path);
    expect(valid.take("a", {}).bytes.size() == 4 && valid.take("b", {0}).bytes.empty(),
           "scalar and empty tensors");
    for (
        auto header :
        {R"({"a":{"shape":[2],"dtype":"F16","data_offsets":[0,4]},"b":{"shape":[2],"dtype":"F16","data_offsets":[0,4]}})",
         R"({"a":{"shape":[1],"dtype":"F16","data_offsets":[2,4]}})",
         R"({"a":{"shape":[1],"dtype":"F16","data_offsets":[0,2]}})",
         R"({"a":{"shape":[1],"dtype":"F16","data_offsets":[0,2],"shape":[2]}})",
         R"({"a":{"shape":[1],"dtype":"F16","data_offsets":[-1,1]}})",
         R"({"a":{"shape":[3],"dtype":"F16","data_offsets":[0,6]}})"}) {
        write(header, "abcd");
        rejects([&] { SafeTensors malformed(path); }, ErrorCode::malformed);
    }
    write(R"({"a":{"shape":[1],"dtype":"I32","data_offsets":[0,4]}})", "abcd");
    rejects([&] { SafeTensors unsupported(path); }, ErrorCode::unsupported);
    write(std::string(mib + 1, ' '), "");
    rejects([&] { SafeTensors oversized(path); }, ErrorCode::resource_limit);
}

void compare(std::span<const float> actual, std::span<const float> expected, double tolerance = 1e-6) {
    expect(actual.size() == expected.size(), "output shape");
    for (std::size_t i = 0; i < actual.size(); ++i)
        close(actual[i], expected[i], tolerance, "fixture output");
}

std::size_t allocation_test(const Model& model, std::span<const Token> prompt, Token next,
                            SessionOptions options = {}) {
    auto session = Session::create(model, std::min<std::size_t>(512, model.config().context), 0, options);
    session.prefill(prompt);
    for (unsigned i = 0; i < 4; ++i)
        session.append(std::span(&next, 1));
    session.reset();
    start_allocation_count();
    try {
        session.prefill(prompt);
        for (unsigned i = 0; i < 4; ++i)
            session.append(std::span(&next, 1));
    } catch (...) {
        (void)stop_allocation_count();
        throw;
    }
    return stop_allocation_count();
}

void paged_cache() {
    const auto model = Model::load(fixture / "f32");
    const auto& config = model.config();
    constexpr auto page = Cache::page_tokens;
    constexpr auto capacity = 2 * page + 3;
    std::array<Token, capacity> tokens{};
    for (std::size_t i = 0; i < tokens.size(); ++i)
        tokens[i] = static_cast<Token>((7 * i + 3) % config.vocabulary);
    const auto prompt = std::span(tokens);
    std::vector<std::vector<float>> reference;
    model.forward(prompt, LogitsMode::all_tokens, {}, [&](std::size_t pos, std::span<const float> values) {
        expect(pos == reference.size(), "uncached output order");
        reference.emplace_back(values.begin(), values.end());
    });
    for (auto precision : {KvPrecision::f32, KvPrecision::f16}) {
        const auto bytes_per_token = 2 * config.layers * config.kv_heads * config.head_dim *
                                     (precision == KvPrecision::f16 ? sizeof(std::uint16_t) : sizeof(float));
        auto session = Session::create(model, capacity, 0, {precision, 7});
        expect(session.memory().cache_bytes == 0, "new session has no KV pages");
        const auto tolerance = precision == KvPrecision::f32 ? 0.0 : 8e-4;
        const LogitsSink cmp = [&](std::size_t pos, std::span<const float> values) {
            compare(values, reference.at(pos), tolerance);
        };
        session.prefill(prompt.first(page), LogitsMode::all_tokens, cmp);
        expect(session.memory().cache_bytes == page * bytes_per_token, "first KV page");
        const auto invalid = static_cast<Token>(config.vocabulary);
        rejects([&] { session.append(std::span(&invalid, 1)); }, ErrorCode::invalid_argument);
        expect(session.memory().cache_bytes == page * bytes_per_token, "invalid input allocates no page");
        session.append(prompt.subspan(page, 1), LogitsMode::all_tokens, cmp);
        expect(session.memory().cache_bytes == 2 * page * bytes_per_token, "append opens second KV page");
        session.append(prompt.subspan(page + 1), LogitsMode::all_tokens, cmp);
        expect(session.memory().cache_bytes == capacity * bytes_per_token, "final KV page fits capacity");
        session.reset();
        expect(session.memory().cache_bytes == capacity * bytes_per_token, "reset retains allocated pages");
        session.prefill(prompt.first(2), LogitsMode::all_tokens, cmp);
        compare(session.logits(), reference.at(1), tolerance);
        expect(allocation_test(model, prompt.first(page - 1), tokens[page - 1], {precision, 7}) == 0,
               "warmed append across a page boundary allocates no heap");
    }
}

void batch_scheduler() {
    const auto model = Model::load(fixture / "f32");
    BatchOptions options;
    options.context = 64;
    options.max_active = 3;
    options.max_queued = 3;
    options.step_tokens = 6;
    options.session.prefill_chunk = 4;
    const SamplingOptions sampling{false, 0.8, 7, 0.95};
    const BatchRequest long_prompt{1, std::vector<Token>(37, 4), 5, 37, sampling};
    const BatchRequest short_prompt{2, {1, 7}, 4, 11, sampling};
    const BatchRequest short_later{4, {3, 9, 6}, 5, 91, sampling};
    const auto standalone = [&](const BatchRequest& request) {
        auto session = Session::create(model, options.context, request.seed, options.session);
        session.prefill(request.tokens);
        Sampler sampler(request.sampling, model.config().vocabulary);
        std::vector<Token> result;
        (void)generate(session, sampler, model.config().eos, request.max_new_tokens,
                       [&](Token token) { result.push_back(token); });
        return result;
    };
    std::array<std::vector<Token>, 5> observed;
    const auto record = [&](const std::vector<BatchEvent>& events) {
        for (const auto& event : events) {
            expect(event.kind != BatchEventKind::error, event.message);
            if (event.kind == BatchEventKind::token) {
                expect(event.token.has_value() && event.elapsed_ms >= 0, "batch token event");
                const auto next = observed.at(event.id).size() + 1;
                expect(event.emitted == next && event.inter_token_ms.has_value() == (next > 1),
                       "batch output timing and sequence");
                observed.at(event.id).push_back(*event.token);
            }
        }
    };

    BatchScheduler scheduler(model, options);
    scheduler.submit(long_prompt);
    scheduler.submit(short_prompt);
    rejects([&] { scheduler.submit(short_prompt); }, ErrorCode::invalid_argument);
    record(scheduler.step());
    expect(observed[1].empty() && observed[2].size() == 1, "short prompt progresses before long prefill");
    scheduler.submit({3, std::vector<Token>(22, 8), 4, 0, {}});
    record(scheduler.step());
    const auto cancelled = scheduler.cancel(3);
    expect(cancelled && cancelled->kind == BatchEventKind::cancelled && cancelled->emitted == 0,
           "cancel request after partial prefill");
    expect(!scheduler.cancel(3), "cancel drops request ID");
    scheduler.submit(short_later);
    for (unsigned i = 0; i < 64 && !scheduler.empty(); ++i)
        record(scheduler.step());
    expect(scheduler.empty(), "mixed batch drains");
    for (const auto* request : {&long_prompt, &short_prompt, &short_later})
        expect(observed[request->id] == standalone(*request), "batch sampling matches standalone");
    expect(observed[3].empty(), "cancelled request emits no later token");
    expect(scheduler.peak_kv_bytes() > 0, "batch tracks peak KV payload");

    options.max_queued = 1;
    options.kv_budget_bytes = Cache::required_bytes(model.config(), 5, KvPrecision::f32);
    BatchScheduler bounded(model, options);
    bounded.submit({1, {1, 7, 2}, 3, 0, {}});
    rejects([&] { bounded.submit({2, {1}, 1, 0, {}}); }, ErrorCode::resource_limit);
    (void)bounded.step();
    bounded.submit({2, {1, 7, 2}, 3, 0, {}});
    expect(std::ranges::none_of(bounded.step(), [](const BatchEvent& event) { return event.id == 2; }),
           "KV budget holds second request in queue");
    expect(bounded.peak_kv_bytes() == options.kv_budget_bytes, "KV payload stays within reservation");
    expect(bounded.cancel(2).has_value(), "cancel waiting request");
    rejects([&] { bounded.submit({3, {1, 7, 2}, 4, 0, {}}); }, ErrorCode::resource_limit);
    if (!bounded.empty())
        expect(bounded.cancel(1).has_value(), "cancel active request");
    bounded.submit({3, {1, 7, 2}, 3, 0, {}});
    (void)bounded.step();
    expect(bounded.kv_page_reuses() > 0 && bounded.resident_kv_bytes() <= options.kv_budget_bytes,
           "request pages return to the shared pool");

    options.max_queued = 3;
    options.max_queued_tokens = 3;
    BatchScheduler token_limited(model, options);
    token_limited.submit({1, {1, 2}, 1, 0, {}});
    rejects([&] { token_limited.submit({2, {3, 4}, 1, 0, {}}); }, ErrorCode::resource_limit);
    expect(!token_limited.contains(2) && token_limited.cancel(1).has_value(),
           "queued token rejection leaves the queue intact");
    token_limited.submit({2, {3, 4, 5}, 1, 0, {}});
    (void)token_limited.step();
    token_limited.submit({3, {6, 7, 8}, 1, 0, {}});

    BatchOptions enough;
    enough.context = 32;
    BatchScheduler stopped(model, enough);
    stopped.submit({1, {4, 4, 4, 4}, 4, 0, {}, {22, 20}});
    const auto events = stopped.step();
    expect(events.size() == 2 && events[0].token == 20 && events[1].kind == BatchEventKind::stop &&
               stopped.empty(),
           "additional stop IDs finish a batch request at the sampled match");
}

void decoder_prefix_cache() {
    const auto run = [](BatchScheduler& engine, const BatchRequest& input, std::size_t expected_hit) {
        engine.submit(input);
        std::vector<Token> tokens;
        while (!engine.empty()) {
            for (const auto& event : engine.step()) {
                expect(event.kind != BatchEventKind::error, event.message);
                expect(event.cached_prompt_tokens == expected_hit, "exact computed prefix length");
                if (event.token)
                    tokens.push_back(*event.token);
            }
        }
        expect(!tokens.empty(), "cached decoder generates tokens");
        return tokens;
    };
    const auto model = Model::load(fixture / "f32");
    const auto gpt2 = Model::load(fixtures / "gpt2");
    for (auto precision : {KvPrecision::f32, KvPrecision::f16}) {
        BatchOptions options;
        options.context = 64;
        options.step_tokens = 8;
        options.session = {precision, 7};
        options.kv_budget_bytes = Cache::required_bytes(model.config(), 37, precision);
        BatchScheduler scheduler(model, options);
        BatchRequest request{1, std::vector<Token>(33, 4), 5, 19, {false, 0.8, 7, 0.95}};
        const auto cold = run(scheduler, request, 0);
        expect(run(scheduler, request, 32) == cold, "warm cache preserves seeded output");

        options.prefix_cache = false;
        BatchScheduler uncached(model, options);
        request.tokens.back() = 7;
        expect(run(scheduler, request, 32) == run(uncached, request, 0), "diverging suffix preserves output");
        request.tokens.pop_back();
        expect(run(scheduler, request, 32) == run(uncached, request, 0),
               "page-aligned prompts obtain logits");

        request.tokens.front() = 8;
        expect(run(scheduler, request, 0) == run(uncached, request, 0),
               "later pages depend on earlier tokens");
        request.tokens.front() = 4;
        expect(run(scheduler, request, 0) == run(uncached, request, 0),
               "idle pages are evicted under pressure");
        expect(scheduler.peak_kv_bytes() <= options.kv_budget_bytes &&
                   scheduler.resident_kv_bytes() <= options.kv_budget_bytes,
               "shared and idle pages stay within the KV budget");

        options.context = gpt2.config().context;
        options.kv_budget_bytes = Cache::required_bytes(gpt2.config(), options.context, precision);
        options.prefix_cache = true;
        BatchScheduler learned_positions(gpt2, options);
        request.tokens.resize(Cache::page_tokens);
        const auto gpt2_cold = run(learned_positions, request, 0);
        expect(run(learned_positions, request, Cache::page_tokens) == gpt2_cold,
               "GPT-2 replays a page-aligned prompt without changing generated tokens");
    }
}

void queue_bypass() {
    const auto model = Model::load(fixture / "f32");
    for (const auto count : {0u, 1u}) {
        BatchOptions options;
        options.context = 32;
        options.max_active = 2;
        options.step_tokens = 4;
        options.session.prefill_chunk = 3;
        options.kv_budget_bytes = Cache::required_bytes(model.config(), 24, KvPrecision::f32);
        options.max_queue_bypasses = count;
        BatchScheduler scheduler(model, options);
        scheduler.submit({1, std::vector<Token>(20, 4), 1, 0, {}});
        expect(scheduler.step().empty(), "long request is prefilling");
        scheduler.submit({2, std::vector<Token>(12, 7), 1, 0, {}});
        scheduler.submit({3, {4}, 1, 0, {}});
        scheduler.submit({4, {4}, 1, 0, {}});
        std::vector<RequestId> finished;
        while (!scheduler.empty()) {
            for (const auto& event : scheduler.step()) {
                expect(event.kind != BatchEventKind::error, event.message);
                if (!event.token)
                    finished.push_back(event.id);
            }
        }
        const auto position = [&](RequestId id) { return std::ranges::find(finished, id); };
        expect(finished.size() == 4 && (position(3) < position(1)) == (count == 1) &&
                   position(4) > position(1) && scheduler.queue_bypasses() == count,
               "only the configured number of short requests pass a blocked head");
    }
}

void threaded_cache_and_chunks() {
    for (unsigned bits = 0; bits <= 0xffff; ++bits) {
        if ((bits & 0x7c00) != 0x7c00)
            expect(float_to_half(half_to_float(bits)) == bits, "finite half storage round trip");
    }
    expect(float_to_half(1.f + 0x1p-11f) == 0x3c00 && float_to_half(1.f + 0x1.8p-10f) == 0x3c02,
           "halfway rounds to even");
    expect(float_to_half(0x1p-25f) == 0 && float_to_half(-0x1p-25f) == 0x8000 &&
               float_to_half(65520.f) == 0x7c00,
           "half underflow, sign and overflow rounding");

    rejects([] { ThreadPool invalid(0); }, ErrorCode::invalid_argument);
    ThreadPool pool(4);
    constexpr std::size_t rows = 259, columns = 521, tokens = 19;
    std::vector<float> weights(rows * columns), input(tokens * columns);
    for (std::size_t i = 0; i < weights.size(); ++i)
        weights[i] = static_cast<float>(static_cast<int>(i % 31) - 15) / 16;
    for (std::size_t i = 0; i < input.size(); ++i)
        input[i] = static_cast<float>(static_cast<int>(i % 19) - 9) / 16;
    const TensorView matrix{{DType::f32, {rows, columns}, 2, weights.size()},
                            std::as_bytes(std::span(weights))};
    const auto scalar = scalar_kernel(DType::f32);
    const ThreadPool serial(1);
    std::vector<float> expected(tokens * rows);
    for (std::size_t i = 0; i < tokens; ++i)
        project(scalar, serial, matrix, std::span(input).subspan(i * columns, columns),
                std::span(expected).subspan(i * rows, rows), 1, {});
    const auto selected = select_kernel(DType::f32, ExecutionPolicy::automatic, cpu_features());
    constexpr auto extra_rows = rows / 2;
    const TensorView prefix{{DType::f32, {extra_rows, columns}, 2, extra_rows * columns},
                            matrix.bytes.first(extra_rows * columns * sizeof(float))};
    const auto submit = [&](std::span<float> output, std::span<float> extra) {
        std::array<float, columns * batch_tile> packed{};
        const auto count = output.size() / rows;
        const std::array targets{Projection{matrix, output}, Projection{prefix, extra}};
        detail::project(pool, std::span(&selected, 1), targets, std::span(input).first(count * columns),
                        count, packed);
    };
    std::vector<float> first(expected.size()), second(expected.size());
    std::vector<float> first_extra(tokens * extra_rows), second_extra(tokens * extra_rows);
    bool failed = false;
    try {
        pool.rows(rows, columns, [](std::size_t begin, std::size_t) {
            if (begin > 0)
                throw std::runtime_error("worker test");
        });
    } catch (const std::runtime_error&) {
        failed = true;
    }
    expect(failed, "CPU worker exception reaches the submitting caller");
    {
        std::jthread a([&] { submit(first, first_extra); });
        std::jthread b([&] { submit(second, second_extra); });
    }
    compare(first, expected, 0);
    compare(second, expected, 0);
    for (std::size_t i = 0; i < tokens; ++i) {
        const auto reference = std::span(expected).subspan(i * rows, extra_rows);
        compare(std::span(first_extra).subspan(i * extra_rows, extra_rows), reference, 0);
        compare(std::span(second_extra).subspan(i * extra_rows, extra_rows), reference, 0);
    }
    std::array<float, 2 * rows> pair{};
    std::array<float, 2 * extra_rows> pair_extra{};
    submit(pair, pair_extra);
    compare(pair, std::span(expected).first(pair.size()), 0);
    compare(pair_extra, std::span(first_extra).first(pair_extra.size()), 0);

    auto model = Model::load(fixture / "f32", {ExecutionPolicy::automatic, 3});
    auto run = [&](KvPrecision precision, std::size_t chunk) {
        SessionOptions options{precision, chunk};
        auto session = Session::create(model, 19, 0, options);
        std::vector<float> logits;
        const auto sink = [&](std::size_t position, std::span<const float> values) {
            expect(position * model.config().vocabulary == logits.size(), "absolute logit positions");
            logits.insert(logits.end(), values.begin(), values.end());
        };
        session.prefill(ids, LogitsMode::all_tokens, sink);
        session.append(ids, LogitsMode::all_tokens, sink);
        session.append(std::span(ids).first(3), LogitsMode::all_tokens, sink);
        const auto& c = model.config();
        expect(session.memory().cache_bytes ==
                   2 * c.layers * 19 * c.kv_heads * c.head_dim * (precision == KvPrecision::f16 ? 2 : 4),
               "KV precision bytes");
        return logits;
    };
    for (auto precision : {KvPrecision::f16, KvPrecision::f32}) {
        const auto sequential = run(precision, 1);
        for (auto chunk : {3u, 7u, 16u, 32u})
            compare(run(precision, chunk), sequential, 0);
    }
    expect(allocation_test(model, ids, 3, {KvPrecision::f16, 3}) == 0,
           "warmed FP16 chunking and append allocate no heap");
    rejects([&] { (void)Session::create(model, 19, 0, {KvPrecision::f32, 33}); },
            ErrorCode::invalid_argument);
}

void model_and_sessions() {
    SafeTensors oracle(fixture / "expected.safetensors");
    const auto data = oracle.take("logits", {ids.size(), 23});
    std::vector<float> expected(data.desc.elements);
    std::memcpy(expected.data(), data.bytes.data(), data.bytes.size());
    for (auto storage : {"f32", "f16", "bf16"}) {
        auto model = Model::load(fixture / storage);
        const auto memory = model.memory();
        expect(memory.tensors == 20 && model.config().tied_embeddings, "tied tensor inventory");
        expect(memory.encoded_weight_bytes ==
                   memory.stored_parameters * (std::string_view(storage) == "f32" ? 4 : 2),
               "native storage accounting");

        std::size_t projected = 0;
        model.forward(ids, LogitsMode::all_tokens, {}, [&](std::size_t pos, std::span<const float> logits) {
            compare(logits, std::span(expected).subspan(pos * 23, 23));
            ++projected;
        });
        expect(projected == ids.size(), "all positions delivered");
        std::vector<float> last(23);
        model.forward(ids, LogitsMode::last_token, last);
        compare(last, std::span(expected).last(23));

        auto session = Session::create(model, ids.size());
        std::size_t logits_calls = 0;
        const TraceSink counter = [&](const TraceEvent& event) {
            if (event.operation == "logits")
                ++logits_calls;
        };
        session.prefill(std::span(ids).first(3), LogitsMode::none, {}, counter);
        rejects([&] { (void)session.logits(); }, ErrorCode::invalid_argument);
        expect(logits_calls == 0, "None avoids output projection");
        const std::array<Token, 2> bad{1, 23};
        rejects([&] { session.append(bad); }, ErrorCode::invalid_argument);
        expect(session.history().size() == 3 && session.state() == SessionState::ready,
               "invalid input is atomic");
        session.append(std::span(ids).subspan(3, 2), LogitsMode::none, {}, counter);
        session.append(std::span(ids).last(3), LogitsMode::last_token, {}, counter);
        expect(logits_calls == 1, "LastToken projects only the final token");
        compare(session.logits(), last, 0);
        rejects([&] { session.append(std::span(ids).first(1)); }, ErrorCode::invalid_argument);
        expect(session.memory().cache_bytes == 2 * 2 * 2 * 2 * ids.size() * sizeof(float),
               "cache memory accounting");
        session.reset();
        const LogitsSink broken = [](std::size_t, std::span<const float>) { throw std::logic_error("sink"); };
        bool caught = false;
        try {
            session.prefill(ids, LogitsMode::all_tokens, broken);
        } catch (const std::logic_error&) {
            caught = true;
        }
        expect(caught && session.state() == SessionState::invalid, "partial failure invalidates session");
        rejects([&] { session.append(std::span(ids).first(1)); }, ErrorCode::invalid_argument);
        session.reset();
        session.prefill(std::span(ids).first(2));
        session.append(session.history());
        expect(session.history().size() == 4 && session.history()[2] == ids[0], "append from history view");
    }

    auto shared = Model::load(fixture / "f32");
    auto a = Session::create(shared);
    auto b = Session::create(shared);
    {
        std::jthread first([&] { a.prefill(ids); });
        std::jthread second([&] { b.prefill(ids); });
    }
    compare(a.logits(), b.logits(), 0);
    expect(allocation_test(shared, ids, 3) == 0, "warmed scalar prefill and append allocate no heap");

    const auto detached = [] {
        auto local = Model::load(fixture / "bf16");
        return Session::create(local, ids.size(), 153);
    };
    auto survivor = detached();
    survivor.prefill(ids);
    compare(survivor.logits(), std::span(expected).last(23));

    Temporary temp;
    const auto config_path = temp.directory / "config.json";
    auto config = read_json(fixture / "f32/config.json", mib);
    config["max_position_embeddings"] = 16384;
    config["bos_token_id"] = 7;
    config["eos_token_id"] = 13;
    std::ofstream(config_path) << config;
    std::filesystem::copy_file(fixture / "f32/model.safetensors", temp.directory / "model.safetensors");
    const auto variant = Model::load(temp.directory);
    expect(variant.config().context == 16384 && variant.config().bos == 7 && variant.config().eos == 13,
           "checkpoint sets context and boundary tokens");
    auto short_session = Session::create(variant, ids.size());
    short_session.prefill(ids);
    compare(short_session.logits(), std::span(expected).last(23));
}

void gpt2_and_sessions() {
    const auto path = fixtures / "gpt2";
    const auto model = Model::load(path);
    expect(model.architecture() == "gpt2" && model.config().context == 24 && model.config().bos == 17 &&
               model.config().eos == 22,
           "GPT-2 checkpoint configuration");
    const auto expected = read_json(path / "logits.json", mib).get<std::vector<std::vector<float>>>();
    expect(expected.size() == ids.size(), "GPT-2 reference length");
    std::vector<std::vector<float>> actual;
    model.forward(ids, LogitsMode::all_tokens, {}, [&](std::size_t pos, std::span<const float> logits) {
        expect(pos == actual.size(), "GPT-2 forward positions");
        compare(logits, expected.at(pos), 3e-6);
        actual.emplace_back(logits.begin(), logits.end());
    });
    expect(actual.size() == ids.size(), "all GPT-2 forward positions delivered");
    for (auto precision : {KvPrecision::f32, KvPrecision::f16}) {
        auto session = Session::create(model, 24, 0, {precision, 2});
        const LogitsSink sink = [&](std::size_t pos, std::span<const float> logits) {
            compare(logits, actual.at(pos), precision == KvPrecision::f32 ? 0 : 8e-4);
        };
        session.prefill(std::span(ids).first(5), LogitsMode::all_tokens, sink);
        session.append(std::span(ids).subspan(5), LogitsMode::all_tokens, sink);
        expect(std::ranges::equal(session.history(), ids), "GPT-2 cache tracks appended positions");
        compare(session.logits(), actual.back(), precision == KvPrecision::f32 ? 0 : 8e-4);
    }
    expect(allocation_test(model, ids, 4) == 0, "warmed GPT-2 prefill and append allocate no heap");
}

void encoder_models() {
    const auto bert_path = fixtures / "bert";
    const auto bert = Model::load(bert_path);
    const auto hidden = read_json(bert_path / "hidden.json", mib);
    const auto check_hidden = [&](const std::vector<Token>& tokens, const Json& reference) {
        const auto actual = bert.encode(tokens);
        expect(actual.size() == tokens.size() * bert.config().hidden, "BERT output shape");
        for (std::size_t i = 0; i < tokens.size(); ++i)
            compare(std::span(actual).subspan(i * bert.config().hidden, bert.config().hidden),
                    reference.at(i).get<std::vector<float>>(), 3e-6);
        return actual;
    };
    const auto base = check_hidden(hidden.at("ids").get<std::vector<Token>>(), hidden.at("hidden_states"));
    const auto altered =
        check_hidden(hidden.at("altered_ids").get<std::vector<Token>>(), hidden.at("altered_hidden_states"));
    expect(std::abs(base.front() - altered.front()) > 1e-3f, "BERT first token uses the final token");
    rejects([&] { (void)bert.encode({}); }, ErrorCode::invalid_argument);
    rejects([&] { (void)Session::create(bert); }, ErrorCode::unsupported);

    const auto t5_path = fixtures / "t5";
    const auto t5 = Model::load(t5_path, {ExecutionPolicy::automatic, 2});
    const auto ref = read_json(t5_path / "logits.json", mib);
    const auto source = ref.at("source").get<std::vector<Token>>();
    const auto target = ref.at("target").get<std::vector<Token>>();
    const auto logits = ref.at("logits").get<std::vector<std::vector<float>>>();
    expect(t5.config().bos == target.front(), "T5 decoder start");
    auto session = T5Session::create(t5, source, target.size());
    for (std::size_t i = 0; i < target.size(); ++i) {
        expect(session.length() == i + 1, "T5 cached decoder length");
        compare(session.logits(), logits.at(i), 2e-5);
        if (i + 1 < target.size())
            session.append(std::span(target).subspan(i + 1, 1));
    }
    rejects([&] { session.append(std::span(target).first(1)); }, ErrorCode::invalid_argument);
    compare(session.logits(), logits.back(), 2e-5);
    rejects([&] { (void)T5Session::create(bert, source); }, ErrorCode::unsupported);
    rejects([&] { (void)t5.encode(source); }, ErrorCode::unsupported);
    rejects([&] { t5.forward(source, LogitsMode::none); }, ErrorCode::unsupported);

    auto generation = T5Session::create(t5, source, 5);
    Sampler greedy({}, t5.config().vocabulary);
    std::vector<Token> emitted;
    const auto result =
        generate(generation, greedy, t5.config().eos, 5, [&](Token token) { emitted.push_back(token); });
    expect(emitted == ref.at("greedy").get<std::vector<Token>>() && result.emitted == emitted.size(),
           "T5 greedy output matches reference");
    const auto evaluated = t5.infer({source, {}, target}, "logits");
    expect(evaluated.shape == std::vector<std::size_t>{target.size(), t5.config().vocabulary},
           "T5 named output shape");
    for (std::size_t i = 0; i < target.size(); ++i)
        compare(std::span(evaluated.values).subspan(i * t5.config().vocabulary, t5.config().vocabulary),
                logits.at(i), 2e-5);
    BatchScheduler scheduler(t5);
    scheduler.submit({1, source, 5, 0, {}});
    scheduler.submit({2, source, 5, 0, {}});
    std::array<std::vector<Token>, 2> scheduled;
    for (unsigned i = 0; i < 64 && !scheduler.empty(); ++i) {
        for (const auto& event : scheduler.step()) {
            expect(event.kind != BatchEventKind::error, event.message);
            if (event.token)
                scheduled.at(event.id - 1).push_back(*event.token);
        }
    }
    expect(scheduler.empty() && scheduled[0] == emitted && scheduled[1] == emitted,
           "T5 batch matches standalone");
}

void checkpoint_outputs() {
    const auto path = fixtures / "bert";
    const auto tokenizer = Tokenizer::load_model(path);
    const auto ref = read_json(path / "outputs.json", mib);
    expect(tokenizer.encode(ref.at("single_text").get<std::string>()) ==
               ref.at("single_ids").get<std::vector<Token>>(),
           "WordPiece normalization and special IDs");
    const auto pair =
        tokenizer.encode_pair(ref.at("first").get<std::string>(), ref.at("second").get<std::string>());
    expect(pair.ids == ref.at("ids").get<std::vector<Token>>() &&
               pair.token_type_ids == ref.at("types").get<std::vector<Token>>(),
           "WordPiece pair IDs and token types");
    const auto model = Model::load(path);
    expect(!model.supports_generation() && model.outputs().size() == 4, "checkpoint output capabilities");
    const auto short_ids = tokenizer.encode("The sky.");
    const auto different = tokenizer.encode_pair("The sky.", "A tree.");
    const std::array inputs{InferenceInput{pair.ids, pair.token_type_ids}, InferenceInput{short_ids},
                            InferenceInput{different.ids, different.token_type_ids}};
    for (const auto& output : model.outputs()) {
        const auto batch = model.infer_batch(inputs, output.name);
        expect(batch.size() == inputs.size(), "encoder batch result count");
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            const auto standalone = model.infer(inputs[i], output.name);
            expect(batch[i].shape == standalone.shape, "encoder batch shape and order");
            compare(batch[i].values, standalone.values, 3e-6);
        }
    }
    for (const auto name : {"pooler_output", "sentence_embedding", "logits"}) {
        const auto result = model.infer({pair.ids, pair.token_type_ids}, name);
        const auto reference = ref.at(name).get<std::vector<float>>();
        expect(result.shape == std::vector<std::size_t>{reference.size()}, "named tensor shape");
        compare(result.values, reference, 3e-6);
    }
    rejects([&] { (void)model.infer({pair.ids}, "absent"); }, ErrorCode::unsupported);
    const std::vector<Token> bad_types(pair.ids.size(), 2);
    rejects([&] { (void)model.infer({pair.ids, bad_types}, "logits"); }, ErrorCode::invalid_argument);
    const auto long_input = std::vector<Token>(13, 1);
    rejects([&] { (void)model.infer({long_input}, "sentence_embedding"); }, ErrorCode::invalid_argument);

    const auto t5_path = fixtures / "t5";
    const auto unigram = Tokenizer::load_model(t5_path);
    const auto t5 = read_json(t5_path / "tokenizer-reference.json", mib);
    const auto ids = unigram.encode(t5.at("text").get<std::string>());
    expect(ids == t5.at("ids").get<std::vector<Token>>() &&
               unigram.decode(ids, true) == t5.at("decoded").get<std::string>(),
           "Unigram scores, unknown IDs, whitespace, and template");
}

void tokenizer_and_sampling() {
    const auto tokenizer = Tokenizer::load(fixture / "tokenizer.json");
    expect(tokenizer.encode("abc<x>>ab") == std::vector<Token>({256, 258, 4}),
           "BPE merges and longest special");
    expect(tokenizer.encode("\x04").empty(), "missing byte is skipped when byte fallback is disabled");
    expect(tokenizer.decode(std::array<Token, 3>{256, 258, 4}, true) == "abcab", "skip special decode");
    expect(tokenizer.pretokenize("  a 1٢") == std::vector<std::string>({"Ġ", "Ġa", "Ġ", "1", "Ù¢"}),
           "ordered regex stages");
    rejects([&] { (void)tokenizer.encode("\xc0\xaf"); }, ErrorCode::invalid_argument);
    rejects([&] { (void)tokenizer.encode("\xed\xa0\x80"); }, ErrorCode::invalid_argument);
    Utf8Stream stream;
    expect(stream.push("\xf0\x9f").empty() && stream.push("\x99").empty() && stream.push("\x82") == "🙂",
           "cross-token UTF-8 buffering");
    expect(stream.push("\xe2\x82").empty() && stream.finish() == "�" && stream.finish().empty(),
           "partial terminal UTF-8 sequence");
    expect(stream.push("\xe2\x82x\xff") == "�x�", "lossy UTF-8 invalid byte boundaries");
    GenerationText stop(tokenizer, {"bcab", "bca"});
    expect(stop.push(256) == "a" && stop.push(4).empty() && stop.stopped() && stop.finish().empty(),
           "matching stop suppresses the matched suffix and the rest of its final token");
    GenerationText unicode_stop(tokenizer, {"éab"});
    expect(unicode_stop.push(195).empty() && unicode_stop.push(169).empty() && unicode_stop.push(4).empty() &&
               unicode_stop.stopped(),
           "UTF-8 stop spans token boundaries");
    GenerationText incomplete_stop(tokenizer, {"abcd"});
    expect(incomplete_stop.push(256).empty() && incomplete_stop.finish() == "abc",
           "unmatched stop prefix is flushed at the output limit");

    std::mt19937_64 rng(1);
    std::array logits{3.0f, 3.0f, -1.0f, 1.0f};
    Sampler greedy({}, 4);
    expect(greedy.select(logits, rng) == 0, "lowest greedy tie ID");
    Sampler top_k({false, 1, 1, 1}, 4);
    Sampler top_p({false, 1, 0, 0.1}, 4);
    for (int i = 0; i < 12; ++i) {
        expect(top_k.select(logits, rng) == 0 && top_p.select(logits, rng) == 0, "top-k/top-p retain best");
    }
    Sampler stochastic({false, 0.8, 2, 1}, 4);
    auto copy = rng;
    for (int i = 0; i < 30; ++i) {
        const auto id = stochastic.select(logits, rng);
        expect(id < 2 && id == stochastic.select(logits, copy), "seeded filtered stochastic sampling");
    }
    logits[0] = std::numeric_limits<float>::infinity();
    rejects([&] { (void)greedy.select(logits, rng); }, ErrorCode::invalid_argument);
    rejects([&] { (void)stochastic.select(logits, rng); }, ErrorCode::invalid_argument);
    rejects([] { Sampler invalid({false, 0, 0, 1}, 4); }, ErrorCode::invalid_argument);

    const auto model = Model::load(fixture / "f32");
    auto session = Session::create(model, 32);
    session.prefill(ids);
    Sampler sampler({}, model.config().vocabulary);
    const Token first = session.sample(sampler);
    expect(session.history().size() == ids.size(), "sampling leaves cache position unchanged");
    std::vector<Token> emitted;
    const auto sink = [&](Token id) { emitted.push_back(id); };
    const auto eos = generate(session, sampler, first, 4, sink);
    expect(eos.stopped_at_eos && eos.pending == first && eos.emitted == 1 && eos.consumed == 0,
           "EOS stays pending");
    const auto budget = generate(session, sampler, (first + 1) % model.config().vocabulary, 1, sink);
    expect(!budget.stopped_at_eos && budget.pending == first && budget.emitted == 1 && budget.consumed == 0,
           "last budget token stays pending");
    session.append(std::span(&first, 1));
    expect(session.history().size() == ids.size() + 1 && session.history().back() == first,
           "continuation appends once");
}

void gguf_tokenizer_metadata() {
    Temporary temp;
    const auto model = temp.directory / "tokenizer.gguf";
    constexpr std::array<std::string_view, 7> tokens{"1", "🙂", "2", "12", "x", "y", "xy"};
    constexpr std::array control_types{1, 3, 1, 1, 1, 1, 1};
    const auto write_model = [&](std::string_view pre, std::span<const int> types) {
        std::vector<char> bytes;
        const auto number = [&](std::uint64_t value, unsigned width) {
            for (unsigned i = 0; i < width; ++i)
                bytes.push_back(static_cast<char>((value >> (i * 8)) & 0xff));
        };
        const auto string = [&](std::string_view value) {
            number(value.size(), 8);
            bytes.insert(bytes.end(), value.begin(), value.end());
        };
        const auto entry = [&](std::string_view key, unsigned type) {
            string(key);
            number(type, 4);
        };
        number(0x46554747, 4);
        number(3, 4);
        number(0, 8);
        number(types.empty() ? 6 : 7, 8);
        entry("example.block_count", 4);
        number(2, 4);
        entry("general.architecture", 8);
        string("example");
        entry("tokenizer.ggml.model", 8);
        string("gpt2");
        entry("tokenizer.ggml.pre", 8);
        string(pre);
        entry("tokenizer.ggml.tokens", 9);
        number(8, 4);
        number(tokens.size(), 8);
        for (auto token : tokens)
            string(token);
        entry("tokenizer.ggml.merges", 9);
        number(8, 4);
        number(2, 8);
        string("1 2");
        string("x y");
        if (!types.empty()) {
            entry("tokenizer.ggml.token_type", 9);
            number(5, 4);
            number(types.size(), 8);
            for (int type : types)
                number(type, 4);
        }
        bytes.resize((bytes.size() + 31) / 32 * 32);
        std::ofstream file(model, std::ios::binary);
        file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    };

    const auto sidecar = temp.directory / "tokenizer.json";
    auto json = read_json(fixture / "tokenizer.json", mib);
    const auto sequence = json.at("pre_tokenizer");
    json["pre_tokenizer"] = sequence.at("pretokenizers").at(1);
    json["pre_tokenizer"].erase("use_regex");
    json["decoder"].erase("use_regex");
    json["post_processor"] = {{"type", "ByteLevel"}, {"add_prefix_space", true}, {"trim_offsets", false}};
    auto& bpe = json.at("model");
    bpe.erase("type");
    bpe["continuing_subword_prefix"] = "";
    bpe["end_of_word_suffix"] = "";
    bpe["vocab"] = Json::object();
    for (std::size_t id = 0; id < tokens.size(); ++id)
        bpe["vocab"][tokens[id]] = id;
    bpe["merges"] = Json::array({{"1", "2"}, {"x", "y"}});
    json["added_tokens"] = Json::array({{{"id", 1},
                                         {"content", "🙂"},
                                         {"special", true},
                                         {"normalized", true},
                                         {"single_word", false},
                                         {"lstrip", false},
                                         {"rstrip", false}}});
    const auto write_sidecar = [&] { std::ofstream(sidecar) << json; };
    write_sidecar();
    expect(Tokenizer::load(sidecar).encode("12🙂xy") == std::vector<Token>{3, 1, 6},
           "legacy ByteLevel tokenizer serialization");

    write_model("gpt-2", control_types);
    expect(Gguf(model).architecture().at("example.block_count") == 2,
           "GGUF retains metadata for the declared architecture regardless of key order");
    rejects([&] { (void)Model::load(model); }, ErrorCode::unsupported);
    const auto embedded = Tokenizer::load_model(model);
    const std::vector<Token> expected{3, 1, 6};
    expect(embedded.vocabulary() == tokens.size() && embedded.encode("12🙂xy") == expected &&
               embedded.decode(expected) == "12🙂xy" && embedded.decode(expected, true) == "12xy",
           "GGUF uses the declared pretokenizer, vocabulary, and control token type");
    expect(Tokenizer::load_model(model, sidecar).encode("12🙂xy") == expected,
           "GGUF matches sidecars using pair arrays or strings");
    std::swap(bpe["merges"][0], bpe["merges"][1]);
    write_sidecar();
    rejects([&] { (void)Tokenizer::load_model(model, sidecar); }, ErrorCode::malformed);
    std::swap(bpe["merges"][0], bpe["merges"][1]);
    write_sidecar();

    write_model("smollm", control_types);
    expect(Tokenizer::load_model(model).encode("12🙂xy") == std::vector<Token>{0, 2, 1, 6},
           "digit isolation comes from tokenizer metadata");
    rejects([&] { (void)Tokenizer::load_model(model, sidecar); }, ErrorCode::malformed);
    write_model("gpt-2", std::span(control_types).first(2));
    rejects([&] { (void)Tokenizer::load_model(model); }, ErrorCode::malformed);
    write_model("unrecognized", {});
    rejects([&] { (void)Tokenizer::load_model(model); }, ErrorCode::unsupported);
    expect(Tokenizer::load_model(model, sidecar).encode("12🙂xy") == expected,
           "matching sidecar supplies missing tokenizer behavior");
}

void gguf_layout() {
    const auto golden = fixtures / "tiny.gguf";
    const Gguf metadata(golden);
    expect(metadata.alignment() == 32 && metadata.records().size() == 20 &&
               load_llama_config(metadata).tied_embeddings,
           "GGUF inventory");
    expect(metadata.memory().mapped_file_bytes == std::filesystem::file_size(golden),
           "GGUF mapping accounting");
    Gguf structural(golden);
    const auto table = structural.take("token_embd.weight", {23, 8});
    expect(table.bytes.size() == 23 * 8 * sizeof(float), "GGUF row-major logical shapes");
    rejects([&] { (void)structural.take("token_embd.weight", {23, 8}); }, ErrorCode::malformed);

    Temporary temp;
    const auto local = temp.directory / "only-model.gguf";
    std::filesystem::copy_file(golden, local);
    auto survivor = [&] {
        const auto loaded = Model::load(local);
        expect(loaded.kernel(0) == "scalar-f32", "GGUF default scalar execution");
        return Session::create(loaded, 512);
    }();
    std::filesystem::remove(local);
    survivor.prefill(ids);
    auto hf = Session::create(Model::load(fixture / "f32"), 512);
    hf.prefill(ids);
    compare(survivor.logits(), hf.logits(), 0);

    const auto valid = read_file(golden, mib);
    const auto malformed = [&](auto edit, ErrorCode code) {
        auto data = valid;
        edit(data);
        std::ofstream output(local, std::ios::binary);
        output.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
        output.close();
        rejects([&] { Gguf invalid(local); }, code);
    };
    malformed([](auto& data) { data[0] = std::byte{0}; }, ErrorCode::malformed);
    malformed([](auto& data) { data[4] = std::byte{2}; }, ErrorCode::unsupported);
    malformed([](auto& data) { std::reverse(data.begin() + 4, data.begin() + 8); }, ErrorCode::unsupported);
    malformed([](auto& data) { data[9] = std::byte{4}; }, ErrorCode::resource_limit);
    malformed([](auto& data) { data[24] = std::byte{255}; }, ErrorCode::malformed);
    malformed(
        [](auto& data) {
            data[25] = std::byte{255};
            data[26] = std::byte{2};
        },
        ErrorCode::malformed);
    const auto location = [&](std::string_view name) {
        const std::string_view header(reinterpret_cast<const char*>(valid.data()), metadata.data_offset());
        const auto offset = header.find(name);
        expect(offset != header.npos, "locate GGUF fixture record");
        return offset + name.size();
    };
    const auto q = location("blk.0.attn_q.weight");
    const auto k = location("blk.0.attn_k.weight");
    malformed([&](auto& data) { data[q + 4 + 16] = std::byte{2}; }, ErrorCode::malformed);
    malformed([&](auto& data) { data[q + 4 + 16] = std::byte{100}; }, ErrorCode::unsupported);
    malformed([&](auto& data) { data[q + 4 + 16 + 4] |= std::byte{1}; }, ErrorCode::malformed);
    malformed(
        [&](auto& data) {
            std::copy_n(data.begin() + static_cast<std::ptrdiff_t>(k + 24), 8,
                        data.begin() + static_cast<std::ptrdiff_t>(q + 24));
        },
        ErrorCode::malformed);
    const auto arch = location("general.architecture");
    malformed(
        [&](auto& data) {
            data[arch] = std::byte{9};
            data[arch + 4] = std::byte{9};
        },
        ErrorCode::malformed);
    malformed([&](auto& data) { data[arch] = std::byte{99}; }, ErrorCode::malformed);
    malformed(
        [&](auto& data) {
            const auto& last = metadata.records().at("token_embd.weight");
            const auto high = metadata.data_offset() + last.offset + last.view.bytes.size();
            data.resize(high - 1);
        },
        ErrorCode::malformed);
}

void packed_kernels() {
    expect(sha256("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "empty SHA-256");
    expect(sha256("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "SHA-256");
    expect(!usable_features({true, false, true, true, true, 6}).avx2, "OSXSAVE gate");
    expect(!usable_features({true, true, true, true, true, 2}).avx2, "YMM state gate");
    expect(!usable_features({false, true, true, true, true, 6}).avx2, "CPUID AVX gate");
    const auto avx2_no_extras = usable_features({true, true, true, false, false, 6});
    expect(avx2_no_extras.avx2 && !avx2_no_extras.fma && !avx2_no_extras.f16c, "independent feature bits");
    expect(select_kernel(DType::f16, ExecutionPolicy::avx2, avx2_no_extras).name == "scalar-f16",
           "F16C fallback");
    expect(select_kernel(DType::f32, ExecutionPolicy::automatic, {}).name == "scalar-f32",
           "automatic scalar fallback");
    rejects([] { (void)select_kernel(DType::q4_0, ExecutionPolicy::avx2, {}); }, ErrorCode::unsupported);
    expect(!select_kernel(static_cast<DType>(31), ExecutionPolicy::automatic, avx2_no_extras).dot,
           "unknown encoding");
    const auto features = cpu_features();
    const auto golden = read_json(fixtures / "quant/basic.json", mib);
    for (const auto type : {DType::q4_0, DType::q8_0}) {
        const auto& description = *encoding(type);
        const auto scalar = scalar_kernel(type);
        const auto& entry = golden.at(description.name);
        std::vector<std::byte> packed(1); // exercise unaligned payload addresses
        std::vector<float> decoded;
        for (std::size_t index = 0; index < entry.at("blocks").size(); ++index) {
            const auto& hex = entry["blocks"][index].get_ref<const std::string&>();
            expect(hex.size() == 2 * description.bytes, "serialized fixture block size");
            const auto offset = packed.size();
            for (std::size_t i = 0; i < hex.size(); i += 2)
                packed.push_back(static_cast<std::byte>(std::stoul(hex.substr(i, 2), nullptr, 16)));
            std::array<float, 32> block{};
            scalar.unpack(packed.data() + offset, block.data());
            for (std::size_t i = 0; i < block.size(); ++i) {
                expect(std::bit_cast<std::uint32_t>(block[i]) ==
                           entry["fp32_bits"][index][i].get<std::uint32_t>(),
                       "independent GGUF block decoder");
                decoded.push_back(block[i]);
            }
        }
        const auto storage = std::span(packed).subspan(1);
        std::vector<float> input(decoded.size() + 1);
        double exact = 0, magnitude = 0;
        for (std::size_t i = 0; i < decoded.size(); ++i) {
            input[i + 1] = static_cast<float>(static_cast<int>(i % 17) - 8) / 9;
            const double term = static_cast<double>(input[i + 1]) * decoded[i];
            exact += term;
            magnitude += std::abs(term);
        }
        const auto gamma = (decoded.size() + 1) * std::numeric_limits<float>::epsilon() / 2;
        const auto limit = gamma / (1 - gamma) * magnitude;
        close(scalar.dot(storage.data(), input.data() + 1, decoded.size()), exact, limit,
              "scalar packed dot");
        if (features.avx2)
            close(select_kernel(type, ExecutionPolicy::avx2, features)
                      .dot(storage.data(), input.data() + 1, decoded.size()),
                  exact, limit, "AVX2 packed dot");
        TensorView embedding{{type, {1, decoded.size()}, 2, decoded.size()}, storage};
        std::vector<float> row(decoded.size());
        Scalar::embedding(embedding, 0, row);
        expect(row == decoded, "packed embedding row");
    }
    if (features.avx2) {
        std::array<float, 13> weights{}, input{};
        for (std::size_t i = 0; i < weights.size(); ++i) {
            weights[i] = static_cast<float>(i);
            input[i] = static_cast<float>(i % 3);
        }
        std::array<std::byte, sizeof(weights) + 1> packed{};
        std::ranges::copy(std::as_bytes(std::span(weights)), packed.begin() + 1);
        const auto reference = scalar_kernel(DType::f32).dot(packed.data() + 1, input.data(), input.size());
        close(select_kernel(DType::f32, ExecutionPolicy::avx2, features)
                  .dot(packed.data() + 1, input.data(), input.size()),
              reference, 0, "AVX2 dense tail and unaligned loads");
        constexpr std::array<std::uint16_t, 4> half{0xbc00, 0x4000, 0x4200, 0x3c00};
        constexpr std::array<std::uint16_t, 4> brain{0xbf80, 0x4000, 0x4040, 0x3f80};
        for (const auto type : {DType::f16, DType::bf16}) {
            if (type == DType::f16 && !features.f16c)
                continue;
            const auto& encodings = type == DType::f16 ? half : brain;
            for (std::size_t i = 0; i < weights.size(); ++i) {
                const auto value = encodings[i % encodings.size()];
                std::memcpy(packed.data() + 1 + i * sizeof(value), &value, sizeof(value));
            }
            const auto expected = scalar_kernel(type).dot(packed.data() + 1, input.data(), input.size());
            close(select_kernel(type, ExecutionPolicy::avx2, features)
                      .dot(packed.data() + 1, input.data(), input.size()),
                  expected, 0, "AVX2 narrow dense tail");
        }
    }
}

void packed_encodings() {
    const auto atlas = read_json(fixtures / "quant/packed.json", mib);
    const auto features = cpu_features();
    const ThreadPool serial(1);
    for (const auto& format : encodings) {
        const auto scalar = scalar_kernel(format.type);
        expect(scalar.dot, format.name);
        if (format.elements == 1 || format.type == DType::q4_0 || format.type == DType::q8_0)
            continue;
        const auto& entry = atlas.at(format.name);
        const auto& blocks = entry.at("blocks");
        expect(blocks.size() % 2 == 0 && scalar.unpack, "packed fixture row layout");
        expect(select_kernel(format.type, ExecutionPolicy::automatic, features).name == scalar.name,
               "observable packed scalar fallback");
        std::vector<std::byte> storage(1);
        std::vector<float> weights, block(format.elements);
        for (std::size_t i = 0; i < blocks.size(); ++i) {
            const auto& raw = blocks[i].get_ref<const std::string&>();
            const auto& expected = entry["fp32_le"][i].get_ref<const std::string&>();
            expect(raw.size() == 2 * format.bytes && expected.size() == 8 * format.elements,
                   "reference block dimensions");
            const auto start = storage.size();
            for (std::size_t j = 0; j < raw.size(); j += 2)
                storage.push_back(static_cast<std::byte>(std::stoul(raw.substr(j, 2), nullptr, 16)));
            scalar.unpack(storage.data() + start, block.data());
            for (std::size_t j = 0; j < block.size(); ++j) {
                std::uint32_t bits = 0;
                for (unsigned b = 0; b < 4; ++b) {
                    const auto value = std::stoul(expected.substr(8 * j + 2 * b, 2), nullptr, 16);
                    bits |= static_cast<std::uint32_t>(value) << (8 * b);
                }
                expect(std::bit_cast<std::uint32_t>(block[j]) == bits,
                       std::string(format.name) + ": FP32 block differs from independent oracles");
                weights.push_back(std::bit_cast<float>(bits));
            }
        }
        const auto width = weights.size() / 2;
        const TensorView view{{format.type, {2, width}, 2, weights.size()}, std::span(storage).subspan(1)};
        std::vector<float> decoded(width), activations(width);
        std::array<float, 2> projected{};
        for (std::size_t i = 0; i < width; ++i)
            activations[i] = static_cast<float>(static_cast<int>(i % 23) - 11) / 12;
        project(scalar, serial, view, activations, projected, 1, {});
        const auto gamma = width * std::numeric_limits<double>::epsilon() / 2;
        for (std::size_t row = 0; row < 2; ++row) {
            const auto expected = std::span(weights).subspan(row * width, width);
            Scalar::embedding(view, row, decoded);
            expect(std::ranges::equal(expected, decoded), "packed embedding across rows and blocks");
            double dot = 0, magnitude = 0;
            for (std::size_t i = 0; i < width; ++i) {
                const double term = static_cast<double>(expected[i]) * activations[i];
                dot += term;
                magnitude += std::abs(term);
            }
            const auto rounding = std::abs(dot) * std::numeric_limits<float>::epsilon() / 2 +
                                  std::numeric_limits<float>::denorm_min() / 2.0;
            close(projected[row], dot, gamma / (1 - gamma) * magnitude + rounding, format.name);
        }
    }
}

void context_boundary(KvPrecision precision) {
    const auto model = Model::load(fixture / "f32");
    auto session = Session::create(model, 8192, 0, {precision, 16});
    const std::vector<Token> prompt(8191, 1);
    session.prefill(prompt, LogitsMode::none);
    session.append(std::span(ids).first(1));
    expect(session.history().size() == 8192 && session.logits().size() == model.config().vocabulary,
           "position 8191");
    rejects([&] { session.append(std::span(ids).first(1)); }, ErrorCode::invalid_argument);
    session.reset();
    session.prefill(ids);
    std::vector<float> baseline(model.config().vocabulary);
    auto sequential = Session::create(model, 8192, 0, {precision, 1});
    sequential.prefill(ids);
    std::ranges::copy(sequential.logits(), baseline.begin());
    compare(session.logits(), baseline, 0);
}

} // namespace

int main(int argc, char** argv) try {
    if (argc == 2 && std::string_view(argv[1]) == "--long") {
        context_boundary(KvPrecision::f32);
        context_boundary(KvPrecision::f16);
    } else if (argc == 1) {
        kernels();
        parser();
        model_and_sessions();
        paged_cache();
        batch_scheduler();
        decoder_prefix_cache();
        queue_bypass();
        gpt2_and_sessions();
        encoder_models();
        checkpoint_outputs();
        threaded_cache_and_chunks();
        tokenizer_and_sampling();
        gguf_tokenizer_metadata();
        gguf_layout();
        packed_kernels();
        packed_encodings();
    } else {
        return 2;
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
