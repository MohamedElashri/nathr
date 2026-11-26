#include "models/bert/model.hpp"

#include <algorithm>
#include <cmath>

namespace nathr::detail {

ModelConfig bert_config(const Json& json, const std::filesystem::path& path) {
    try {
        if (json.value("position_embedding_type", std::string("absolute")) != "absolute" ||
            json.value("is_decoder", false) || json.value("add_cross_attention", false) ||
            json.value("hidden_act", std::string("gelu")) != "gelu" ||
            !json.value("pruned_heads", Json::object()).empty()) {
            throw Error(ErrorCode::unsupported, path.string() + ": unsupported BERT configuration");
        }
        const auto field = [&](std::string_view name) { return natural(json.at(name), name); };
        ModelConfig c{};
        c.layers = field("num_hidden_layers");
        c.hidden = field("hidden_size");
        c.intermediate = field("intermediate_size");
        c.query_heads = c.kv_heads = field("num_attention_heads");
        c.vocabulary = field("vocab_size");
        c.context = field("max_position_embeddings");
        c.rms_epsilon = json.value("layer_norm_eps", 1e-12f);
        if (!c.layers || c.layers > 24 || !c.hidden || !c.intermediate || !c.query_heads ||
            c.hidden % c.query_heads || !c.context || !c.vocabulary || c.vocabulary > max_vocabulary ||
            !(c.rms_epsilon > 0) || !std::isfinite(c.rms_epsilon)) {
            throw Error(ErrorCode::unsupported, path.string() + ": unsupported BERT dimensions or epsilon");
        }
        c.head_dim = c.hidden / c.query_heads;
        return c;
    } catch (const Json::exception& error) {
        throw Error(ErrorCode::malformed, path.string() + ": " + error.what());
    }
}

Bert::Bert(SafeTensors& file, const std::filesystem::path& path, const ModelConfig& config, const Json& json,
           CpuBackend& backend)
    : config_(config), backend_(backend) {
    const auto h = config.hidden, f = config.intermediate;
    const std::string prefix = file.contains("bert.embeddings.word_embeddings.weight") ? "bert." : "";
    const auto take = [&](std::string_view name, std::initializer_list<std::size_t> shape) {
        return backend.bind(file.take(name, shape), path, name);
    };
    const auto linear = [&](const std::string& name, std::size_t in, std::size_t out) {
        return Affine{take(name + ".weight", {out, in}), take(name + ".bias", {out})};
    };
    const auto norm = [&](const std::string& name) {
        if (file.contains(name + ".gamma"))
            return Affine{take(name + ".gamma", {h}), take(name + ".beta", {h})};
        return Affine{take(name + ".weight", {h}), take(name + ".bias", {h})};
    };
    const auto embeddings = prefix + "embeddings.";
    tokens_ = take(embeddings + "word_embeddings.weight", {config.vocabulary, h});
    positions_ = take(embeddings + "position_embeddings.weight", {config.context, h});
    const auto segments = natural(json.value("type_vocab_size", Json(2)), "BERT type_vocab_size");
    if (!segments)
        throw Error(ErrorCode::unsupported, path.string() + ": BERT requires segment embeddings");
    segments_ = take(embeddings + "token_type_embeddings.weight", {segments, h});
    embedding_norm_ = norm(embeddings + "LayerNorm");
    layers_.reserve(config.layers);
    for (std::size_t i = 0; i < config.layers; ++i) {
        const auto layer = prefix + "encoder.layer." + std::to_string(i) + ".";
        const auto self = layer + "attention.self.";
        const auto attention = layer + "attention.output.";
        layers_.push_back({linear(self + "query", h, h), linear(self + "key", h, h),
                           linear(self + "value", h, h), linear(attention + "dense", h, h),
                           norm(attention + "LayerNorm"), linear(layer + "intermediate.dense", h, f),
                           linear(layer + "output.dense", f, h), norm(layer + "output.LayerNorm")});
    }
    // Checkpoints can include a pooler or task head. Encoding uses only the underlying encoder.
}

namespace {
struct BertWorkspace final : Workspace {
    BertWorkspace(const ModelConfig& c, std::size_t count) {
        const auto rows = product({count, c.hidden}, "BERT activation workspace");
        allocate({{&query, rows},
                  {&key, rows},
                  {&value, rows},
                  {&attended, rows},
                  {&projected, rows},
                  {&ff, product({count, c.intermediate}, "BERT feed-forward workspace")},
                  {&packed, product({batch_tile, std::max(c.hidden, c.intermediate)}, "BERT matrix tile")},
                  {&scores, count}});
    }
    std::span<float> query, key, value, attended, projected, ff;
};
} // namespace

std::vector<float> Bert::encode(std::span<const Token> tokens) const {
    const auto h = config_.hidden, count = tokens.size();
    BertWorkspace w(config_, count);
    std::vector<float> hidden(w.query.size());
    const auto affine = [&](const Affine& weights, std::span<const float> input, std::span<float> output) {
        backend_.project(weights.weight, input, output, count, w.packed);
        const auto width = output.size() / count;
        for (std::size_t i = 0; i < count; ++i)
            Scalar::add_bias(row(output, width, i), weights.bias);
    };
    const auto normalize = [&](const Affine& norm, std::span<float> input) {
        Scalar::layer_norm(norm.weight, norm.bias, input, input, config_.rms_epsilon);
    };
    const auto segment = w.attended.first(h);
    Scalar::embedding(segments_, 0, segment);
    for (std::size_t i = 0; i < count; ++i) {
        auto activation = row(hidden, h, i);
        auto position = w.projected.first(h);
        Scalar::embedding(tokens_, tokens[i], activation);
        Scalar::embedding(positions_, i, position);
        Scalar::add(activation, segment);
        Scalar::add(activation, position);
        normalize(embedding_norm_, activation);
    }
    const auto scale = 1.0 / std::sqrt(static_cast<double>(config_.head_dim));
    for (const auto& layer : layers_) {
        affine(layer.query, hidden, w.query);
        affine(layer.key, hidden, w.key);
        affine(layer.value, hidden, w.value);
        for (std::size_t i = 0; i < count; ++i)
            Scalar::attention_range(config_, 0, nullptr, w.key, w.value, row(w.query, h, i),
                                    row(w.attended, h, i), w.scores, scale);
        affine(layer.attention, w.attended, w.projected);
        Scalar::add(hidden, w.projected);
        for (std::size_t i = 0; i < count; ++i)
            normalize(layer.attention_norm, row(hidden, h, i));
        affine(layer.ff_in, hidden, w.ff);
        for (float& value : w.ff)
            value = static_cast<float>(0.5 * value * (1.0 + std::erf(value / std::sqrt(2.0))));
        affine(layer.ff_out, w.ff, w.projected);
        Scalar::add(hidden, w.projected);
        for (std::size_t i = 0; i < count; ++i)
            normalize(layer.ff_norm, row(hidden, h, i));
    }
    if (!std::ranges::all_of(hidden, [](float value) { return std::isfinite(value); }))
        throw Error(ErrorCode::backend, "BERT encoder produced non-finite output");
    return hidden;
}

} // namespace nathr::detail
