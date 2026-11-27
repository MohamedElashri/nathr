#include "models/bert/model.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <numeric>

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
    const auto check_index_buffer = [&](std::string_view name, bool positions) {
        if (!file.contains(name))
            return;
        const auto tensor = file.take(name, {1, config.context});
        if (tensor.desc.dtype != DType::i64)
            throw Error(ErrorCode::unsupported, path.string() + ": expected I64 in " + std::string(name));
        for (std::size_t i = 0; i < config.context; ++i) {
            if (little_integer(tensor.bytes.subspan(i * 8, 8)) != (positions ? i : 0))
                throw Error(ErrorCode::unsupported,
                            path.string() + ": unsupported index buffer " + std::string(name));
        }
    };
    check_index_buffer(embeddings + "position_ids", true);
    check_index_buffer(embeddings + "token_type_ids", false);
    segments_count_ = natural(json.value("type_vocab_size", Json(2)), "BERT type_vocab_size");
    if (!segments_count_)
        throw Error(ErrorCode::unsupported, path.string() + ": BERT requires segment embeddings");
    segments_ = take(embeddings + "token_type_embeddings.weight", {segments_count_, h});
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
    const auto pooler = prefix + "pooler.dense";
    if (file.contains(pooler + ".weight"))
        pooler_ = linear(pooler, h, h);
    if (json.value("architectures", Json::array()) == Json::array({"BertForSequenceClassification"})) {
        const auto labels = json.value("id2label", Json::object());
        label_count_ =
            natural(json.value("num_labels", Json(labels.empty() ? 2 : labels.size())), "BERT num_labels");
        if (!pooler_ || !label_count_ || label_count_ > max_vocabulary)
            throw Error(ErrorCode::unsupported, path.string() + ": invalid BERT classifier");
        classifier_ = linear("classifier", h, label_count_);
    }
    if (std::filesystem::exists(path / "modules.json")) {
        try {
            const auto modules = read_json(path / "modules.json", mib);
            constexpr std::string_view root = "sentence_transformers.models.";
            if (!modules.is_array() || modules.size() < 2 || modules.size() > 3 ||
                modules[0].at("type") != std::string(root) + "Transformer" || modules[0].at("path") != "" ||
                modules[1].at("type") != std::string(root) + "Pooling" ||
                (modules.size() == 3 && modules[2].at("type") != std::string(root) + "Normalize")) {
                throw Error(ErrorCode::unsupported, path.string() + ": unsupported sentence model modules");
            }
            const auto directory = std::filesystem::path(modules[1].at("path").get<std::string>());
            if (directory.empty() || directory.has_parent_path() || directory.is_absolute() ||
                directory == "." || directory == "..")
                throw Error(ErrorCode::malformed, path.string() + ": invalid pooling module path");
            const auto pooling = read_json(path / directory / "config.json", mib);
            const bool mean = pooling.value("pooling_mode_mean_tokens", false);
            const bool cls = pooling.value("pooling_mode_cls_token", false);
            const bool max = pooling.value("pooling_mode_max_tokens", false);
            if (static_cast<int>(mean) + cls + max != 1 ||
                natural(pooling.at("word_embedding_dimension"), "pooling width") != h ||
                pooling.value("pooling_mode_mean_sqrt_len_tokens", false) ||
                pooling.value("pooling_mode_weightedmean_tokens", false) ||
                pooling.value("pooling_mode_lasttoken", false))
                throw Error(ErrorCode::unsupported, path.string() + ": unsupported sentence pooling");
            pooling_ = mean ? Pooling::mean : cls ? Pooling::cls : Pooling::maximum;
            normalize_embedding_ = modules.size() == 3;
            embedding_context_ = config.context;
            if (std::filesystem::exists(path / "sentence_bert_config.json")) {
                const auto sentence = read_json(path / "sentence_bert_config.json", mib);
                const auto limit = natural(sentence.value("max_seq_length", Json(config.context)),
                                           "sentence max_seq_length");
                if (!limit || sentence.value("do_lower_case", false))
                    throw Error(ErrorCode::unsupported,
                                path.string() + ": unsupported sentence input options");
                embedding_context_ = std::min(config.context, limit);
            }
        } catch (const Json::exception& error) {
            throw Error(ErrorCode::malformed, path.string() + ": " + error.what());
        }
    }
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

    static std::size_t row_limit(const ModelConfig& c) {
        constexpr auto capacity = max_workspace_bytes / sizeof(float);
        const auto packed = batch_tile * std::max(c.hidden, c.intermediate);
        return packed >= capacity ? 0 : (capacity - packed) / (5 * c.hidden + c.intermediate + 1);
    }
};
} // namespace

std::vector<float> Bert::encode(std::span<const Token> tokens, std::span<const Token> token_types) const {
    const InferenceInput input{tokens, token_types};
    return encode(std::span(&input, 1));
}

std::vector<float> Bert::encode(std::span<const InferenceInput> inputs) const {
    const auto h = config_.hidden;
    std::size_t count = 0;
    for (const auto& [tokens, types, _] : inputs) {
        if ((!types.empty() && types.size() != tokens.size()) ||
            std::ranges::any_of(types, [&](Token id) { return id >= segments_count_; }))
            throw Error(ErrorCode::invalid_argument, "invalid token_type_ids");
        count += tokens.size();
    }
    BertWorkspace w(config_, count);
    std::vector<float> hidden(w.query.size());
    const auto add_bias = [&](std::span<float> output, TensorView bias) {
        const auto width = output.size() / count;
        for (std::size_t i = 0; i < count; ++i)
            Scalar::add_bias(row(output, width, i), bias);
    };
    const auto affine = [&](const Affine& weights, std::span<const float> input, std::span<float> output) {
        backend_.project(weights.weight, input, output, count, w.packed);
        add_bias(output, weights.bias);
    };
    const auto normalize = [&](const Affine& norm, std::span<float> input) {
        Scalar::layer_norm(norm.weight, norm.bias, input, input, config_.rms_epsilon);
    };
    const auto segment = w.attended.first(h);
    std::size_t start = 0;
    for (const auto& [tokens, types, _] : inputs) {
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            auto activation = row(hidden, h, start + i);
            auto position = w.projected.first(h);
            Scalar::embedding(tokens_, tokens[i], activation);
            Scalar::embedding(positions_, i, position);
            Scalar::embedding(segments_, types.empty() ? 0 : types[i], segment);
            Scalar::add(activation, segment);
            Scalar::add(activation, position);
            normalize(embedding_norm_, activation);
        }
        start += tokens.size();
    }
    const auto scale = 1.0 / std::sqrt(static_cast<double>(config_.head_dim));
    for (const auto& layer : layers_) {
        backend_.project(
            {{layer.query.weight, w.query}, {layer.key.weight, w.key}, {layer.value.weight, w.value}}, hidden,
            count, w.packed);
        add_bias(w.query, layer.query.bias);
        add_bias(w.key, layer.key.bias);
        add_bias(w.value, layer.value.bias);
        start = 0;
        for (const auto& input : inputs) {
            const auto length = input.tokens.size();
            const auto keys = w.key.subspan(start * h, length * h);
            const auto values = w.value.subspan(start * h, length * h);
            const auto scores = w.scores.first(length);
            for (std::size_t i = start; i < start + length; ++i)
                Scalar::attention_range(config_, 0, nullptr, keys, values, row(w.query, h, i),
                                        row(w.attended, h, i), scores, scale);
            start += length;
        }
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

std::vector<OutputInfo> Bert::outputs() const {
    std::vector<OutputInfo> result{{"hidden_states", config_.hidden, config_.context, true}};
    if (pooler_)
        result.push_back({"pooler_output", config_.hidden, config_.context, false});
    if (pooling_ != Pooling::none)
        result.push_back({"sentence_embedding", config_.hidden, embedding_context_, false});
    if (classifier_)
        result.push_back({"logits", label_count_, config_.context, false});
    return result;
}

TensorResult Bert::infer(InferenceInput input, const OutputInfo& output) const {
    auto hidden = encode(input.tokens, input.token_type_ids);
    if (output.name == "hidden_states")
        return {{input.tokens.size(), config_.hidden}, std::move(hidden)};
    return project_output(hidden, input.tokens.size(), output);
}

TensorResult Bert::project_output(std::span<const float> hidden, std::size_t count,
                                  const OutputInfo& output) const {
    const auto width = config_.hidden;
    std::vector<float> result(output.width);
    if (output.name == "sentence_embedding") {
        for (std::size_t d = 0; d < width; ++d) {
            if (pooling_ == Pooling::cls) {
                result[d] = hidden[d];
                continue;
            }
            double sum = 0;
            float max = hidden[d];
            for (std::size_t i = 0; i < count; ++i) {
                const auto value = hidden[i * width + d];
                sum += value;
                max = std::max(max, value);
            }
            result[d] = pooling_ == Pooling::mean ? static_cast<float>(sum / count) : max;
        }
        if (normalize_embedding_) {
            const auto norm = std::max(
                1e-12, std::sqrt(std::inner_product(result.begin(), result.end(), result.begin(), 0.0)));
            for (auto& value : result)
                value = static_cast<float>(value / norm);
        }
    } else {
        std::vector<float> pooled(width);
        backend_.project(pooler_->weight, hidden.first(width), pooled, 1, {});
        Scalar::add_bias(pooled, pooler_->bias);
        for (auto& value : pooled)
            value = std::tanh(value);
        if (output.name == "pooler_output")
            result = std::move(pooled);
        else {
            backend_.project(classifier_->weight, pooled, result, 1, {});
            Scalar::add_bias(result, classifier_->bias);
        }
    }
    if (!std::ranges::all_of(result, [](float value) { return std::isfinite(value); }))
        throw Error(ErrorCode::backend, "BERT output contains non-finite values");
    return {{output.width}, std::move(result)};
}

std::vector<TensorResult> Bert::infer_batch(std::span<const InferenceInput> inputs,
                                            const OutputInfo& output) const {
    std::vector<std::size_t> order(inputs.size());
    std::iota(order.begin(), order.end(), 0);
    std::ranges::sort(order, {}, [&](std::size_t index) { return inputs[index].tokens.size(); });
    std::vector<TensorResult> results(inputs.size());
    const auto limit = std::min<std::size_t>(1024, BertWorkspace::row_limit(config_));
    for (std::size_t start = 0; start < order.size();) {
        std::array<InferenceInput, 16> group;
        const auto bucket = std::bit_width(inputs[order[start]].tokens.size());
        std::size_t count = 0, tokens = 0;
        do {
            group[count] = inputs[order[start + count]];
            tokens += group[count++].tokens.size();
            if (count == group.size() || start + count == order.size() || tokens >= limit)
                break;
            const auto next = inputs[order[start + count]].tokens.size();
            if (std::bit_width(next) != bucket || next > limit - tokens)
                break;
        } while (true);
        if (count == 1) {
            results[order[start++]] = infer(group.front(), output);
            continue;
        }
        const auto hidden = encode(std::span(group).first(count));
        std::size_t offset = 0;
        for (std::size_t i = 0; i < count; ++i) {
            const auto length = group[i].tokens.size();
            const auto slice = std::span(hidden).subspan(offset, length * config_.hidden);
            offset += slice.size();
            results[order[start + i]] =
                output.sequence ? TensorResult{{length, config_.hidden}, {slice.begin(), slice.end()}}
                                : project_output(slice, length, output);
        }
        start += count;
    }
    return results;
}

} // namespace nathr::detail
