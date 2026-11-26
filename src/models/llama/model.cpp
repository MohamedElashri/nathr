#include "models/llama/model.hpp"

#include "formats/gguf.hpp"
#include "formats/safetensors.hpp"

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace nathr::detail {
namespace {

struct LlamaWorkspace final : Workspace {
    LlamaWorkspace(const ModelConfig& c, std::size_t positions, std::size_t block);
    std::span<float> hidden, query, keys, values, norm, attention, projection, gate, up;
};

LlamaWorkspace::LlamaWorkspace(const ModelConfig& c, std::size_t positions, std::size_t block) {
    const auto hidden_rows = product({block, c.hidden}, "hidden workspace");
    const auto kv_rows = product({block, c.kv_heads, c.head_dim}, "KV workspace");
    const auto intermediate_rows = product({block, c.intermediate}, "feed-forward workspace");
    const auto packing =
        block == 1 ? 0 : product({batch_tile, std::max(c.hidden, c.intermediate)}, "matrix tile workspace");
    allocate({{&hidden, hidden_rows},
              {&query, hidden_rows},
              {&keys, kv_rows},
              {&values, kv_rows},
              {&norm, hidden_rows},
              {&attention, hidden_rows},
              {&projection, hidden_rows},
              {&gate, intermediate_rows},
              {&up, intermediate_rows},
              {&packed, packing},
              {&scores, positions},
              {&logits, c.vocabulary}});
}

struct LayerWeights {
    TensorView input_norm, q, k, v, o, post_attention_norm, gate, up, down;
};

class Llama final : public Decoder {
  public:
    template <class File>
    Llama(File& file, const std::filesystem::path& path, const ModelConfig& config, CpuBackend& backend);
    std::unique_ptr<Workspace> workspace(std::size_t positions, std::size_t block) const override {
        return std::make_unique<LlamaWorkspace>(config, positions, block);
    }
    void execute(Workspace& scratch, const Evaluation& evaluation) const override;

  private:
    const ModelConfig& config;
    const CpuBackend& backend;
    TensorView embedding, output, norm;
    std::vector<LayerWeights> layers;
    std::vector<float> frequency;
};

template <class File>
Llama::Llama(File& file, const std::filesystem::path& path, const ModelConfig& c, CpuBackend& cpu)
    : config(c), backend(cpu) {
    constexpr bool gguf = std::is_same_v<File, Gguf>;
    const auto h = config.hidden;
    const auto kv = config.kv_heads * config.head_dim;
    const auto ff = config.intermediate;
    const auto take = [&](std::string_view name, std::initializer_list<std::size_t> shape) {
        return cpu.bind(file.take(name, shape), path, name);
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
    if (const auto name = file.first_unbound(); !name.empty())
        throw Error(ErrorCode::unsupported, path.string() + ": unexpected tensor " + std::string(name));
    if constexpr (gguf)
        file.discard_metadata();
    frequency.resize(config.head_dim / 2);
    for (std::size_t i = 0; i < frequency.size(); ++i) {
        const double power = static_cast<double>(i) / static_cast<double>(frequency.size());
        frequency[i] = static_cast<float>(1.0 / std::pow(static_cast<double>(config.rope_theta), power));
    }
}

void Llama::execute(Workspace& scratch, const Evaluation& evaluation) const {
    auto& w = static_cast<LlamaWorkspace&>(scratch);
    const auto tokens = evaluation.tokens;
    const auto start = evaluation.start;
    auto* const cache = evaluation.cache;
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
                            std::size_t rows) { backend.project(weight, input, result, rows, w.packed); };
    for (std::size_t i = 0; i < count; ++i) {
        auto activation = row(hidden, h, i);
        Scalar::embedding(embedding, tokens[i], activation);
        evaluation.record("embedding", TraceEvent::model_level, i, activation);
    }

    for (std::size_t layer = 0; layer < layers.size(); ++layer) {
        const auto& weights = layers[layer];
        for (std::size_t i = 0; i < count; ++i) {
            auto activation = row(hidden, h, i);
            auto norm_row = row(norms, h, i);
            evaluation.record("layer_input", layer, i, activation);
            Scalar::rms_norm(weights.input_norm, activation, norm_row, config.rms_epsilon);
            evaluation.record("input_norm", layer, i, norm_row);
        }
        linear(weights.q, norms, queries, count);
        linear(weights.k, norms, keys, count);
        linear(weights.v, norms, values, count);
        for (std::size_t i = 0; i < count; ++i) {
            auto q = row(queries, h, i);
            auto k = row(keys, kv, i);
            auto v = row(values, kv, i);
            evaluation.record("q_before_rope", layer, i, q);
            evaluation.record("k_before_rope", layer, i, k);
            evaluation.record("value", layer, i, v);
            Scalar::rope(q, k, frequency, start + i);
            evaluation.record("q_after_rope", layer, i, q);
            evaluation.record("key_after_rope", layer, i, k);
            if (cache)
                Scalar::store(*cache, layer, start + i, k, v);
        }
        for (std::size_t i = 0; i < count; ++i) {
            auto result = row(attention, h, i);
            Scalar::attention(config, layer, start + i, cache, keys, values, row(queries, h, i), result,
                              w.scores);
            evaluation.record("attention", layer, i, result);
        }
        linear(weights.o, attention, projection, count);
        for (std::size_t i = 0; i < count; ++i) {
            auto activation = row(hidden, h, i);
            auto result = row(projection, h, i);
            auto norm_row = row(norms, h, i);
            evaluation.record("attention_projection", layer, i, result);
            Scalar::add(activation, result);
            evaluation.record("attention_residual", layer, i, activation);
            Scalar::rms_norm(weights.post_attention_norm, activation, norm_row, config.rms_epsilon);
            evaluation.record("post_attention_norm", layer, i, norm_row);
        }
        linear(weights.gate, norms, gate, count);
        linear(weights.up, norms, up, count);
        for (std::size_t i = 0; i < count; ++i) {
            auto g = row(gate, ff, i);
            auto u = row(up, ff, i);
            evaluation.record("gate", layer, i, g);
            evaluation.record("up", layer, i, u);
            Scalar::swiglu(g, u);
            evaluation.record("swiglu", layer, i, g);
        }
        linear(weights.down, gate, projection, count);
        for (std::size_t i = 0; i < count; ++i) {
            auto result = row(projection, h, i);
            auto activation = row(hidden, h, i);
            evaluation.record("down", layer, i, result);
            Scalar::add(activation, result);
            evaluation.record("layer_output", layer, i, activation);
        }
    }
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        if (!evaluation.wants_logits(i))
            continue;
        const auto final_norm = w.norm.first(h);
        Scalar::rms_norm(norm, row(hidden, h, i), final_norm, config.rms_epsilon);
        evaluation.record("final_norm", TraceEvent::model_level, i, final_norm);
        linear(output, final_norm, w.logits, 1);
        evaluation.publish_logits(i, w.logits);
    }
}

} // namespace

std::unique_ptr<Decoder> load_llama(SafeTensors& file, const std::filesystem::path& path,
                                    const ModelConfig& config, CpuBackend& backend) {
    return std::make_unique<Llama>(file, path, config, backend);
}

std::unique_ptr<Decoder> load_llama(Gguf& file, const std::filesystem::path& path, const ModelConfig& config,
                                    CpuBackend& backend) {
    return std::make_unique<Llama>(file, path, config, backend);
}

} // namespace nathr::detail
