#include "models/llama/model.hpp"

#include "formats/gguf.hpp"
#include "formats/safetensors.hpp"

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace nathr::detail {
namespace {

struct RotaryWorkspace final : Workspace {
    RotaryWorkspace(const ModelConfig& c, std::size_t positions, std::size_t block);
    std::span<float> hidden, query, keys, values, norm, attention, projection, gate, up;
};

RotaryWorkspace::RotaryWorkspace(const ModelConfig& c, std::size_t positions, std::size_t block) {
    const auto hidden_rows = product({block, c.hidden}, "hidden workspace");
    const auto query_width = product({c.query_heads, c.head_dim}, "query width");
    const auto query_rows = product({block, query_width}, "query workspace");
    const auto kv_rows = product({block, c.kv_heads, c.head_dim}, "KV workspace");
    const auto intermediate_rows = product({block, c.intermediate}, "feed-forward workspace");
    const auto packing = block == 1 ? 0
                                    : product({batch_tile, std::max({c.hidden, c.intermediate, query_width})},
                                              "matrix tile workspace");
    allocate({{&hidden, hidden_rows},
              {&query, query_rows},
              {&keys, kv_rows},
              {&values, kv_rows},
              {&norm, hidden_rows},
              {&attention, query_rows},
              {&projection, hidden_rows},
              {&gate, intermediate_rows},
              {&up, intermediate_rows},
              {&packed, packing},
              {&scores, positions},
              {&logits, c.vocabulary}});
}

struct LayerWeights {
    TensorView input_norm, q, k, v, o, post_attention_norm, gate, up, down;
    TensorView query_norm{}, key_norm{};
    TensorView query_bias{}, key_bias{}, value_bias{};
};

class RotaryDecoder final : public Decoder {
  public:
    template <class File>
    RotaryDecoder(File& file, const std::filesystem::path& path, const ModelConfig& config,
                  CpuBackend& backend, std::string_view architecture = "llama");
    std::unique_ptr<Workspace> workspace(std::size_t positions, std::size_t block) const override {
        return std::make_unique<RotaryWorkspace>(config, positions, block);
    }
    void execute_batch(Workspace& scratch, const EvaluationBatch& batch) const override;
    [[nodiscard]] std::vector<LoraTarget> lora_targets() const override;

  private:
    const ModelConfig& config;
    const CpuBackend& backend;
    TensorView embedding, output, norm;
    std::vector<LayerWeights> layers;
    std::vector<float> frequency;
};

template <class File>
RotaryDecoder::RotaryDecoder(File& file, const std::filesystem::path& path, const ModelConfig& c,
                             CpuBackend& cpu, std::string_view architecture)
    : config(c), backend(cpu) {
    constexpr bool gguf = std::is_same_v<File, Gguf>;
    const auto h = config.hidden;
    const auto query_width = product({config.query_heads, config.head_dim}, "query width");
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
        auto layer = LayerWeights{
            take(prefix + (gguf ? "attn_norm.weight" : "input_layernorm.weight"), {h}),
            take(prefix + (gguf ? "attn_q.weight" : "self_attn.q_proj.weight"), {query_width, h}),
            take(prefix + (gguf ? "attn_k.weight" : "self_attn.k_proj.weight"), {kv, h}),
            take(prefix + (gguf ? "attn_v.weight" : "self_attn.v_proj.weight"), {kv, h}),
            take(prefix + (gguf ? "attn_output.weight" : "self_attn.o_proj.weight"), {h, query_width}),
            take(prefix + (gguf ? "ffn_norm.weight" : "post_attention_layernorm.weight"), {h}),
            take(prefix + (gguf ? "ffn_gate.weight" : "mlp.gate_proj.weight"), {ff, h}),
            take(prefix + (gguf ? "ffn_up.weight" : "mlp.up_proj.weight"), {ff, h}),
            take(prefix + (gguf ? "ffn_down.weight" : "mlp.down_proj.weight"), {h, ff})};
        if (architecture == "qwen3") {
            layer.query_norm = take(prefix + "attn_q_norm.weight", {config.head_dim});
            layer.key_norm = take(prefix + "attn_k_norm.weight", {config.head_dim});
        } else if (architecture == "qwen2") {
            const auto bias = [&](std::string_view suffix, std::size_t width) {
                const auto weight = take(prefix + std::string(suffix), {width});
                if (encoding(weight.desc.dtype)->elements != 1)
                    throw Error(ErrorCode::unsupported,
                                path.string() + ": Qwen2 attention biases must be dense");
                return weight;
            };
            layer.query_bias = bias("attn_q.bias", query_width);
            layer.key_bias = bias("attn_k.bias", kv);
            layer.value_bias = bias("attn_v.bias", kv);
        } else if constexpr (gguf) {
            layer.q.rotary_head_width = layer.k.rotary_head_width = config.head_dim;
        }
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

std::vector<LoraTarget> RotaryDecoder::lora_targets() const {
    std::vector<LoraTarget> targets{{"model.embed_tokens", embedding, true}, {"lm_head", output}};
    targets.reserve(2 + 7 * layers.size());
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const auto prefix = "model.layers." + std::to_string(i) + ".";
        const auto& w = layers[i];
        targets.insert(targets.end(), {{prefix + "self_attn.q_proj", w.q},
                                       {prefix + "self_attn.k_proj", w.k},
                                       {prefix + "self_attn.v_proj", w.v},
                                       {prefix + "self_attn.o_proj", w.o},
                                       {prefix + "mlp.gate_proj", w.gate},
                                       {prefix + "mlp.up_proj", w.up},
                                       {prefix + "mlp.down_proj", w.down}});
    }
    return targets;
}

void RotaryDecoder::execute_batch(Workspace& scratch, const EvaluationBatch& batch) const {
    auto& w = static_cast<RotaryWorkspace&>(scratch);
    const auto h = config.hidden;
    const auto query_width = product({config.query_heads, config.head_dim}, "query width");
    const auto kv = config.kv_heads * config.head_dim;
    const auto ff = config.intermediate;
    const auto count = batch.size();
    const auto hidden = w.hidden.first(count * h);
    const auto queries = w.query.first(count * query_width);
    const auto keys = w.keys.first(count * kv);
    const auto values = w.values.first(count * kv);
    const auto norms = w.norm.first(count * h);
    const auto attention = w.attention.first(count * query_width);
    const auto projection = w.projection.first(count * h);
    const auto gate = w.gate.first(count * ff);
    const auto up = w.up.first(count * ff);
    const auto* adapters = batch.adapters();
    const auto linear = [&](TensorView weight, std::span<const float> input, std::span<float> result) {
        backend.project(weight, input, result, count, w.packed, adapters);
    };
    batch.for_each([&](const Evaluation& e, std::size_t i, std::size_t index, std::size_t) {
        auto activation = row(hidden, h, index);
        Scalar::embedding(embedding, e.tokens[i], activation);
        if (e.lora)
            e.lora->embed(embedding, e.tokens[i], activation);
        e.record("embedding", TraceEvent::model_level, i, activation);
    });

    for (std::size_t layer = 0; layer < layers.size(); ++layer) {
        const auto& weights = layers[layer];
        batch.for_each([&](const Evaluation& e, std::size_t i, std::size_t index, std::size_t) {
            auto activation = row(hidden, h, index);
            auto norm_row = row(norms, h, index);
            e.record("layer_input", layer, i, activation);
            Scalar::rms_norm(weights.input_norm, activation, norm_row, config.rms_epsilon);
            e.record("input_norm", layer, i, norm_row);
        });
        backend.project({{weights.q, queries}, {weights.k, keys}, {weights.v, values}}, norms, count,
                        w.packed, adapters);
        batch.for_each([&](const Evaluation& e, std::size_t i, std::size_t index, std::size_t) {
            auto q = row(queries, query_width, index);
            auto k = row(keys, kv, index);
            auto v = row(values, kv, index);
            if (!weights.query_norm.bytes.empty()) {
                const auto normalize_heads = [&](TensorView weight, std::span<float> heads) {
                    for (std::size_t offset = 0; offset < heads.size(); offset += config.head_dim) {
                        auto head = heads.subspan(offset, config.head_dim);
                        Scalar::rms_norm(weight, head, head, config.rms_epsilon);
                    }
                };
                normalize_heads(weights.query_norm, q);
                normalize_heads(weights.key_norm, k);
            }
            if (!weights.query_bias.bytes.empty()) {
                Scalar::add_bias(q, weights.query_bias);
                Scalar::add_bias(k, weights.key_bias);
                Scalar::add_bias(v, weights.value_bias);
            }
            e.record("q_before_rope", layer, i, q);
            e.record("k_before_rope", layer, i, k);
            e.record("value", layer, i, v);
            Scalar::rope(q, k, frequency, e.start + i);
            e.record("q_after_rope", layer, i, q);
            e.record("key_after_rope", layer, i, k);
            if (e.cache && !e.replay_cached)
                Scalar::store(*e.cache, layer, e.start + i, k, v);
        });
        batch.for_each([&](const Evaluation& e, std::size_t i, std::size_t index, std::size_t offset) {
            auto result = row(attention, query_width, index);
            Scalar::attention(config, layer, e.start + i, e.cache,
                              keys.subspan(offset * kv, e.tokens.size() * kv),
                              values.subspan(offset * kv, e.tokens.size() * kv),
                              row(queries, query_width, index), result, w.scores);
            e.record("attention", layer, i, result);
        });
        linear(weights.o, attention, projection);
        batch.for_each([&](const Evaluation& e, std::size_t i, std::size_t index, std::size_t) {
            auto activation = row(hidden, h, index);
            auto result = row(projection, h, index);
            auto norm_row = row(norms, h, index);
            e.record("attention_projection", layer, i, result);
            Scalar::add(activation, result);
            e.record("attention_residual", layer, i, activation);
            Scalar::rms_norm(weights.post_attention_norm, activation, norm_row, config.rms_epsilon);
            e.record("post_attention_norm", layer, i, norm_row);
        });
        backend.project({{weights.gate, gate}, {weights.up, up}}, norms, count, w.packed, adapters);
        batch.for_each([&](const Evaluation& e, std::size_t i, std::size_t index, std::size_t) {
            auto g = row(gate, ff, index);
            auto u = row(up, ff, index);
            e.record("gate", layer, i, g);
            e.record("up", layer, i, u);
            Scalar::swiglu(g, u);
            e.record("swiglu", layer, i, g);
        });
        linear(weights.down, gate, projection);
        batch.for_each([&](const Evaluation& e, std::size_t i, std::size_t index, std::size_t) {
            auto result = row(projection, h, index);
            auto activation = row(hidden, h, index);
            e.record("down", layer, i, result);
            Scalar::add(activation, result);
            e.record("layer_output", layer, i, activation);
        });
    }
    project_logits(w, batch, backend, output, hidden, norms.first(h), [&](auto input, auto result) {
        Scalar::rms_norm(norm, input, result, config.rms_epsilon);
    });
}

} // namespace

std::unique_ptr<Decoder> load_llama(SafeTensors& file, const std::filesystem::path& path,
                                    const ModelConfig& config, CpuBackend& backend) {
    return std::make_unique<RotaryDecoder>(file, path, config, backend);
}

std::unique_ptr<Decoder> load_gguf_decoder(Gguf& file, const std::filesystem::path& path,
                                           const ModelConfig& config, CpuBackend& backend,
                                           std::string_view architecture) {
    return std::make_unique<RotaryDecoder>(file, path, config, backend, architecture);
}

} // namespace nathr::detail
