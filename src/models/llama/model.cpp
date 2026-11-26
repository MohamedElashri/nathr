#include "models/llama/model.hpp"

#include <algorithm>
#include <cmath>
#include <system_error>
#include <type_traits>

namespace nathr::detail {

Scratch::Scratch(const ModelConfig& c, std::size_t positions, std::size_t block) {
    const auto hidden_rows = product({block, c.hidden}, "hidden workspace");
    const auto kv_rows = product({block, c.kv_heads, c.head_dim}, "KV workspace");
    const auto intermediate_rows = product({block, c.intermediate}, "feed-forward workspace");
    const auto packing =
        block == 1 ? 0 : product({batch_tile, std::max(c.hidden, c.intermediate)}, "matrix tile workspace");
    const auto widths =
        std::array{hidden_rows, hidden_rows,       kv_rows,           kv_rows, hidden_rows, hidden_rows,
                   hidden_rows, intermediate_rows, intermediate_rows, packing, positions,   c.vocabulary};
    std::size_t count = 0;
    for (const auto width : widths)
        count = sum(count, width, "workspace");
    if (count > scratch_limit / sizeof(float))
        throw Error(ErrorCode::resource_limit, "workspace exceeds 32 MiB");
    storage.resize(count);
    auto remaining = std::span(storage);
    const auto buffers = std::array{&hidden,     &query, &keys, &values, &norm,   &attention,
                                    &projection, &gate,  &up,   &packed, &scores, &logits};
    static_assert(buffers.size() == widths.size());
    for (std::size_t i = 0; i < widths.size(); ++i) {
        *buffers[i] = remaining.first(widths[i]);
        remaining = remaining.subspan(widths[i]);
    }
}

std::size_t Scratch::bytes() const noexcept {
    return storage.capacity() * sizeof(float);
}

void validate_tokens(std::span<const Token> tokens, const ModelConfig& config, std::size_t limit) {
    if (tokens.empty())
        throw Error(ErrorCode::invalid_argument, "token span must be nonempty");
    if (tokens.size() > limit)
        throw Error(ErrorCode::invalid_argument, "token span exceeds capacity");
    if (std::ranges::any_of(tokens, [&](Token token) { return token >= config.vocabulary; })) {
        throw Error(ErrorCode::invalid_argument, "token ID exceeds model vocabulary");
    }
}

void validate_mode(LogitsMode mode, const LogitsSink& sink) {
    if ((mode != LogitsMode::all_tokens && mode != LogitsMode::last_token && mode != LogitsMode::none) ||
        ((mode == LogitsMode::all_tokens) != static_cast<bool>(sink))) {
        throw Error(ErrorCode::invalid_argument,
                    "all-token logits mode requires a sink, other modes take no sink");
    }
}

} // namespace nathr::detail

namespace nathr {
using namespace detail;

namespace {
std::filesystem::path weights_path(const std::filesystem::path& directory) {
    std::error_code error;
    if (std::filesystem::exists(directory / "model.safetensors.index.json", error)) {
        throw Error(ErrorCode::unsupported, directory.string() + ": sharded Safetensors");
    }
    if (error)
        throw Error(ErrorCode::io, directory.string() + ": " + error.message());
    return directory / "model.safetensors";
}

std::span<float> row(std::span<float> values, std::size_t width, std::size_t index) {
    return values.subspan(width * index, width);
}
} // namespace

Model::Impl::WeightFile Model::Impl::open(const std::filesystem::path& path, bool hf_directory) {
    if (hf_directory)
        return WeightFile(std::in_place_type<SafeTensors>, weights_path(path));
    return WeightFile(std::in_place_type<Gguf>, path);
}

Model::Impl::Impl(const std::filesystem::path& path, bool hf_directory, BackendOptions options)
    : config(hf_directory ? read_config(path / "config.json") : ModelConfig{}),
      tensors(open(path, hf_directory)), thread_count(options.threads), pool(thread_count) {
    if (!hf_directory)
        config = std::get<Gguf>(tensors).config();
    std::visit([&](auto& file) { bind(file, path); }, tensors);
    const auto features = cpu_features();
    const bool use_avx2 = options.policy != ExecutionPolicy::scalar && features.avx2;
    active_backend = use_avx2 ? "avx2" : "scalar";
    const auto kernel_for = [&](TensorView tensor) {
        const auto type = tensor.desc.dtype;
        auto& kernel = kernels[static_cast<std::uint32_t>(type)];
        if (!kernel.dot)
            kernel = select_kernel(type, options.policy, features);
    };
    kernel_for(embedding);
    kernel_for(output);
    kernel_for(norm);
    for (const auto& l : layers) {
        for (auto tensor : {l.input_norm, l.q, l.k, l.v, l.o, l.post_attention_norm, l.gate, l.up, l.down})
            kernel_for(tensor);
    }
    frequency.resize(config.head_dim / 2);
    for (std::size_t i = 0; i < frequency.size(); ++i) {
        const double power = static_cast<double>(i) / static_cast<double>(frequency.size());
        frequency[i] = static_cast<float>(1.0 / std::pow(static_cast<double>(config.rope_theta), power));
    }
}

template <class File> void Model::Impl::bind(File& file, const std::filesystem::path& path) {
    constexpr bool gguf = std::is_same_v<File, Gguf>;
    const auto h = config.hidden;
    const auto kv = config.kv_heads * config.head_dim;
    const auto ff = config.intermediate;
    const auto take = [&](std::string_view name, std::initializer_list<std::size_t> shape) {
        auto view = file.take(name, shape);
        const auto format = encoding(view.desc.dtype);
        if (!executable_encoding(view.desc.dtype) || (shape.size() == 1 && format->elements != 1)) {
            throw Error(ErrorCode::unsupported, path.string() + ": no execution kernel for " +
                                                    std::string(name) + " (" + std::string(format->name) +
                                                    ")");
        }
        return view;
    };
    embedding = take(gguf ? "token_embd.weight" : "model.embed_tokens.weight", {config.vocabulary, h});
    norm = take(gguf ? "output_norm.weight" : "model.norm.weight", {h});
    output = embedding;
    if (!config.tied_embeddings) {
        output = take(gguf ? "output.weight" : "lm_head.weight", {config.vocabulary, h});
    } else if constexpr (!gguf) {
        if (file.contains("lm_head.weight")) {
            const auto stored = take("lm_head.weight", {config.vocabulary, h});
            if (stored.desc.dtype != embedding.desc.dtype ||
                !std::ranges::equal(stored.bytes, embedding.bytes))
                throw Error(ErrorCode::malformed, path.string() + ": tied lm_head differs from embedding");
        }
    }
    layers.reserve(config.layers);
    for (std::size_t i = 0; i < config.layers; ++i) {
        const auto prefix = (gguf ? "blk." : "model.layers.") + std::to_string(i) + ".";
        auto layer =
            LayerWeights{take(prefix + (gguf ? "attn_norm.weight" : "input_layernorm.weight"), {h}),
                         take(prefix + (gguf ? "attn_q.weight" : "self_attn.q_proj.weight"), {h, h}),
                         take(prefix + (gguf ? "attn_k.weight" : "self_attn.k_proj.weight"), {kv, h}),
                         take(prefix + (gguf ? "attn_v.weight" : "self_attn.v_proj.weight"), {kv, h}),
                         take(prefix + (gguf ? "attn_output.weight" : "self_attn.o_proj.weight"), {h, h}),
                         take(prefix + (gguf ? "ffn_norm.weight" : "post_attention_layernorm.weight"), {h}),
                         take(prefix + (gguf ? "ffn_gate.weight" : "mlp.gate_proj.weight"), {ff, h}),
                         take(prefix + (gguf ? "ffn_up.weight" : "mlp.up_proj.weight"), {ff, h}),
                         take(prefix + (gguf ? "ffn_down.weight" : "mlp.down_proj.weight"), {h, ff})};
        if constexpr (gguf)
            layer.q.rotary_head_width = layer.k.rotary_head_width = config.head_dim;
        layers.push_back(layer);
    }
    if (!file.bound_all())
        throw Error(ErrorCode::unsupported,
                    path.string() + ": unexpected tensor " + std::string(file.first_unbound()));
    if constexpr (gguf)
        file.discard_metadata();
}

void Model::Impl::execute(std::span<const Token> tokens, std::size_t start, Cache* cache, Scratch& w,
                          LogitsMode mode, const LogitsSink& sink, const TraceSink& trace) const {
    const auto h = config.hidden;
    const auto kv = config.kv_heads * config.head_dim;
    const auto ff = config.intermediate;
    const auto count = tokens.size();
    const auto hidden = w.hidden.first(count * h);
    const auto queries = w.query.first(count * h);
    const auto keys = w.keys.first(count * kv);
    const auto values = w.values.first(count * kv);
    const auto norms = w.norm.first(count * h);
    const auto attention = w.attention.first(count * h);
    const auto projection = w.projection.first(count * h);
    const auto gate = w.gate.first(count * ff);
    const auto up = w.up.first(count * ff);
    const auto linear = [&](TensorView weight, std::span<const float> input, std::span<float> result,
                            std::size_t rows) {
        project(kernels[static_cast<std::uint32_t>(weight.desc.dtype)], pool, weight, input, result, rows,
                w.packed);
    };
    const auto record = [&](std::string_view name, std::size_t layer, std::size_t index,
                            std::span<const float> data) {
        if (trace)
            trace({name, layer, start + index, data});
    };
    for (std::size_t i = 0; i < count; ++i) {
        auto activation = row(hidden, h, i);
        Scalar::embedding(embedding, tokens[i], activation);
        record("embedding", TraceEvent::model_level, i, activation);
    }

    for (std::size_t layer = 0; layer < layers.size(); ++layer) {
        const auto& weights = layers[layer];
        for (std::size_t i = 0; i < count; ++i) {
            auto activation = row(hidden, h, i);
            auto norm_row = row(norms, h, i);
            record("layer_input", layer, i, activation);
            Scalar::rms_norm(weights.input_norm, activation, norm_row, config.rms_epsilon);
            record("input_norm", layer, i, norm_row);
        }
        linear(weights.q, norms, queries, count);
        linear(weights.k, norms, keys, count);
        linear(weights.v, norms, values, count);
        for (std::size_t i = 0; i < count; ++i) {
            auto q = row(queries, h, i);
            auto k = row(keys, kv, i);
            auto v = row(values, kv, i);
            record("q_before_rope", layer, i, q);
            record("k_before_rope", layer, i, k);
            record("value", layer, i, v);
            Scalar::rope(q, k, frequency, start + i);
            record("q_after_rope", layer, i, q);
            record("key_after_rope", layer, i, k);
            if (cache)
                Scalar::store(*cache, layer, start + i, k, v);
        }
        for (std::size_t i = 0; i < count; ++i) {
            auto result = row(attention, h, i);
            Scalar::attention(config, layer, start + i, cache, keys, values, row(queries, h, i), result,
                              w.scores);
            record("attention", layer, i, result);
        }
        linear(weights.o, attention, projection, count);
        for (std::size_t i = 0; i < count; ++i) {
            auto activation = row(hidden, h, i);
            auto result = row(projection, h, i);
            auto norm_row = row(norms, h, i);
            record("attention_projection", layer, i, result);
            Scalar::add(activation, result);
            record("attention_residual", layer, i, activation);
            Scalar::rms_norm(weights.post_attention_norm, activation, norm_row, config.rms_epsilon);
            record("post_attention_norm", layer, i, norm_row);
        }
        linear(weights.gate, norms, gate, count);
        linear(weights.up, norms, up, count);
        for (std::size_t i = 0; i < count; ++i) {
            auto g = row(gate, ff, i);
            auto u = row(up, ff, i);
            record("gate", layer, i, g);
            record("up", layer, i, u);
            Scalar::swiglu(g, u);
            record("swiglu", layer, i, g);
        }
        linear(weights.down, gate, projection, count);
        for (std::size_t i = 0; i < count; ++i) {
            auto result = row(projection, h, i);
            auto activation = row(hidden, h, i);
            record("down", layer, i, result);
            Scalar::add(activation, result);
            record("layer_output", layer, i, activation);
        }
    }
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        if (mode == LogitsMode::none || (mode == LogitsMode::last_token && i + 1 != tokens.size()))
            continue;
        const auto final_norm = w.norm.first(h);
        Scalar::rms_norm(norm, row(hidden, h, i), final_norm, config.rms_epsilon);
        record("final_norm", TraceEvent::model_level, i, final_norm);
        linear(output, final_norm, w.logits, 1);
        if (!std::ranges::all_of(w.logits, [](float val) { return std::isfinite(val); })) {
            throw Error(ErrorCode::backend, "output projection produced non-finite logits");
        }
        record("logits", TraceEvent::model_level, i, w.logits);
        if (mode == LogitsMode::all_tokens)
            sink(start + i, w.logits);
    }
}

Model Model::load(const std::filesystem::path& path, BackendOptions options) {
    if (!options.threads || options.threads > max_cpu_threads)
        throw Error(ErrorCode::invalid_argument, "CPU thread count must be between 1 and 64");
    if (options.policy != ExecutionPolicy::scalar && options.policy != ExecutionPolicy::automatic &&
        options.policy != ExecutionPolicy::avx2)
        throw Error(ErrorCode::unsupported, "unknown execution policy");
    if (options.policy == ExecutionPolicy::avx2 && !cpu_features().avx2)
        throw Error(ErrorCode::unsupported, "AVX2 was requested but CPU or OS AVX2 state is unavailable");
    std::error_code error;
    const bool hf_directory = std::filesystem::is_directory(path, error);
    if (error)
        throw Error(ErrorCode::io, path.string() + ": " + error.message());
    return Model(std::make_shared<Impl>(path, hf_directory, options));
}

const ModelConfig& Model::config() const noexcept {
    return impl_->config;
}

ModelMemory Model::memory() const noexcept {
    return std::visit([](const auto& file) { return file.memory(); }, impl_->tensors);
}

std::string_view Model::backend() const noexcept {
    return impl_->active_backend;
}

unsigned Model::threads() const noexcept {
    return impl_->thread_count;
}

std::string_view Model::kernel(std::uint32_t id) const noexcept {
    return id < impl_->kernels.size() ? impl_->kernels[id].name : std::string_view{};
}

void Model::forward(std::span<const Token> tokens, LogitsMode mode, std::span<float> last_logits,
                    const LogitsSink& sink, const TraceSink& trace) const {
    validate_tokens(tokens, config(), std::min(config().context, prompt_chunk));
    validate_mode(mode, sink);
    if ((mode == LogitsMode::last_token && last_logits.size() != config().vocabulary) ||
        (mode != LogitsMode::last_token && !last_logits.empty())) {
        throw Error(ErrorCode::invalid_argument, "last-token mode requires a vocabulary-sized output span");
    }
    Scratch scratch(config(), tokens.size(), tokens.size());
    impl_->execute(tokens, 0, nullptr, scratch, mode, sink, trace);
    if (mode == LogitsMode::last_token)
        std::ranges::copy(scratch.logits, last_logits.begin());
}

} // namespace nathr
