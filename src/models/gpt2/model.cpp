#include "models/gpt2/model.hpp"

#include "formats/safetensors.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace nathr::detail {

ModelConfig gpt2_config(const Json& json, const std::filesystem::path& config_path) {
    const auto path = config_path.string();
    try {
        if ((json.contains("architectures") &&
             json.at("architectures") != Json::array({"GPT2LMHeadModel"})) ||
            json.value("activation_function", std::string("gelu_new")) != "gelu_new" ||
            !json.value("scale_attn_weights", true) || json.value("scale_attn_by_inverse_layer_idx", false) ||
            json.value("reorder_and_upcast_attn", false) || json.value("add_cross_attention", false)) {
            throw Error(ErrorCode::unsupported, path + ": unsupported GPT-2 configuration");
        }
        const auto field = [&](std::string_view name) {
            return natural(json.at(name), path + ": " + std::string(name));
        };
        ModelConfig c{};
        c.layers = field("n_layer");
        c.hidden = field("n_embd");
        c.query_heads = c.kv_heads = field("n_head");
        c.vocabulary = field("vocab_size");
        c.context = field("n_positions");
        c.intermediate = json.contains("n_inner") && !json.at("n_inner").is_null()
                             ? field("n_inner")
                             : product({4, c.hidden}, "GPT-2 feed-forward width");
        if (!c.layers || c.layers > 39 || !c.hidden || !c.intermediate || !c.query_heads ||
            c.hidden % c.query_heads || !c.context || !c.vocabulary || c.vocabulary > max_vocabulary ||
            (json.contains("n_ctx") && field("n_ctx") != c.context)) {
            throw Error(ErrorCode::unsupported, path + ": unsupported GPT-2 dimensions");
        }
        c.head_dim = c.hidden / c.query_heads;
        c.rms_epsilon = json.value("layer_norm_epsilon", 1e-5f);
        c.tied_embeddings = json.value("tie_word_embeddings", true);
        if (!(c.rms_epsilon > 0) || !std::isfinite(c.rms_epsilon))
            throw Error(ErrorCode::malformed, path + ": invalid LayerNorm epsilon");
        const auto bos = field("bos_token_id");
        if (bos >= c.vocabulary)
            throw Error(ErrorCode::malformed, path + ": special token ID exceeds vocabulary");
        c.bos = static_cast<Token>(bos);
        c.eos = primary_eos_id(json.at("eos_token_id"), c.vocabulary, path + ": eos_token_id");
        return c;
    } catch (const Json::exception& error) {
        throw Error(ErrorCode::malformed, path + ": " + error.what());
    }
}

namespace {

struct Gpt2Workspace final : Workspace {
    Gpt2Workspace(const ModelConfig& c, std::size_t positions, std::size_t block) {
        const auto hidden_rows = product({block, c.hidden}, "hidden workspace");
        const auto qkv_rows = product({3, hidden_rows}, "QKV workspace");
        const auto ff_rows = product({block, c.intermediate}, "feed-forward workspace");
        const auto packing =
            block == 1 ? 0
                       : product({batch_tile, std::max(c.hidden, c.intermediate)}, "matrix tile workspace");
        allocate({{&hidden, hidden_rows},
                  {&norm, hidden_rows},
                  {&qkv, qkv_rows},
                  {&keys, hidden_rows},
                  {&values, hidden_rows},
                  {&attention, hidden_rows},
                  {&projection, hidden_rows},
                  {&ff, ff_rows},
                  {&packed, packing},
                  {&scores, positions},
                  {&logits, c.vocabulary}});
    }
    std::span<float> hidden, norm, qkv, keys, values, attention, projection, ff;
};

struct Affine {
    TensorView weight, bias;
};

struct LayerWeights {
    Affine input_norm, qkv, attention_output, post_attention_norm, ff_in, ff_out;
};

void validate_mask(TensorView mask, std::size_t context, const std::filesystem::path& path,
                   std::string_view name) {
    if (mask.desc.dtype != DType::f32)
        throw Error(ErrorCode::unsupported,
                    path.string() + ": expected an FP32 causal mask for " + std::string(name));
    // Some GPT-2 checkpoints serialize this deterministic buffer. Attention builds the same mask from
    // positions.
    for (std::size_t i = 0; i < context; ++i) {
        for (std::size_t j = 0; j < context; ++j) {
            float value;
            std::memcpy(&value, mask.bytes.data() + (i * context + j) * sizeof(value), sizeof(value));
            if (value != (j <= i ? 1.f : 0.f))
                throw Error(ErrorCode::malformed,
                            path.string() + ": invalid causal mask " + std::string(name));
        }
    }
}

class Gpt2 final : public Decoder {
  public:
    Gpt2(SafeTensors& file, const std::filesystem::path& path, const ModelConfig& config,
         CpuBackend& backend);
    std::unique_ptr<Workspace> workspace(std::size_t positions, std::size_t block) const override {
        return std::make_unique<Gpt2Workspace>(config, positions, block);
    }
    void execute_batch(Workspace& scratch, const EvaluationBatch& batch) const override;

  private:
    const ModelConfig& config;
    const CpuBackend& backend;
    TensorView embedding, positions, output;
    Affine final_norm;
    std::vector<LayerWeights> layers;
};

Gpt2::Gpt2(SafeTensors& file, const std::filesystem::path& path, const ModelConfig& c, CpuBackend& cpu)
    : config(c), backend(cpu) {
    const auto h = config.hidden, ff = config.intermediate;
    const auto qkv = product({3, h}, "GPT-2 QKV width");
    const std::string prefix = file.contains("transformer.wte.weight") ? "transformer." : "";
    const auto take = [&](std::string_view name, std::initializer_list<std::size_t> shape) {
        return cpu.bind(file.take(name, shape), path, name);
    };
    const auto norm = [&](const std::string& base) {
        return Affine{take(base + ".weight", {h}), take(base + ".bias", {h})};
    };
    const auto projection = [&](const std::string& base, std::size_t in, std::size_t out) {
        const auto name = base + ".weight";
        return Affine{cpu.bind(file.take_transposed_matrix(name, out, in), path, name),
                      take(base + ".bias", {out})};
    };
    embedding = take(prefix + "wte.weight", {config.vocabulary, h});
    positions = take(prefix + "wpe.weight", {config.context, h});
    final_norm = norm(prefix + "ln_f");
    output = embedding;
    if (!config.tied_embeddings) {
        output = take("lm_head.weight", {config.vocabulary, h});
    } else if (file.contains("lm_head.weight")) {
        const auto stored = take("lm_head.weight", {config.vocabulary, h});
        if (stored.desc.dtype != embedding.desc.dtype || !std::ranges::equal(stored.bytes, embedding.bytes))
            throw Error(ErrorCode::malformed, path.string() + ": tied lm_head differs from embedding");
    }
    layers.reserve(config.layers);
    for (std::size_t i = 0; i < config.layers; ++i) {
        const auto layer = prefix + "h." + std::to_string(i) + ".";
        const auto mask = layer + "attn.bias";
        if (file.contains(mask))
            validate_mask(file.take(mask, {1, 1, config.context, config.context}), config.context, path,
                          mask);
        layers.push_back({norm(layer + "ln_1"), projection(layer + "attn.c_attn", h, qkv),
                          projection(layer + "attn.c_proj", h, h), norm(layer + "ln_2"),
                          projection(layer + "mlp.c_fc", h, ff), projection(layer + "mlp.c_proj", ff, h)});
    }
    if (const auto name = file.first_unbound(); !name.empty())
        throw Error(ErrorCode::unsupported, path.string() + ": unexpected tensor " + std::string(name));
}

void Gpt2::execute_batch(Workspace& scratch, const EvaluationBatch& batch) const {
    auto& w = static_cast<Gpt2Workspace&>(scratch);
    const auto h = config.hidden, ff = config.intermediate;
    const auto count = batch.size();
    const auto hidden = w.hidden.first(count * h);
    const auto norm = w.norm.first(count * h);
    const auto qkv = w.qkv.first(count * 3 * h);
    const auto keys = w.keys.first(count * h);
    const auto values = w.values.first(count * h);
    const auto attention = w.attention.first(count * h);
    const auto projection = w.projection.first(count * h);
    const auto ff_values = w.ff.first(count * ff);
    const auto affine = [&](const Affine& weight, std::span<const float> input, std::span<float> result) {
        backend.project(weight.weight, input, result, count, w.packed);
        const auto width = result.size() / count;
        for (std::size_t i = 0; i < count; ++i)
            Scalar::add_bias(row(result, width, i), weight.bias);
    };
    const auto normalize = [&](const Affine& weights, std::span<const float> input, std::span<float> result) {
        Scalar::layer_norm(weights.weight, weights.bias, input, result, config.rms_epsilon);
    };
    batch.for_each([&](const Evaluation& e, std::size_t i, std::size_t index, std::size_t) {
        auto activation = row(hidden, h, index);
        auto position = row(norm, h, index);
        Scalar::embedding(embedding, e.tokens[i], activation);
        Scalar::embedding(positions, e.start + i, position);
        Scalar::add(activation, position);
        e.record("embedding", TraceEvent::model_level, i, activation);
    });

    for (std::size_t layer = 0; layer < layers.size(); ++layer) {
        const auto& weights = layers[layer];
        batch.for_each([&](const Evaluation& e, std::size_t i, std::size_t index, std::size_t) {
            auto activation = row(hidden, h, index);
            auto norm_row = row(norm, h, index);
            e.record("layer_input", layer, i, activation);
            normalize(weights.input_norm, activation, norm_row);
            e.record("input_norm", layer, i, norm_row);
        });
        affine(weights.qkv, norm, qkv);
        batch.for_each([&](const Evaluation& e, std::size_t i, std::size_t index, std::size_t) {
            const auto combined = row(qkv, 3 * h, index);
            const auto key = combined.subspan(h, h);
            const auto value = combined.last(h);
            e.record("query", layer, i, combined.first(h));
            e.record("key", layer, i, key);
            e.record("value", layer, i, value);
            if (e.cache) {
                if (!e.replay_cached)
                    Scalar::store(*e.cache, layer, e.start + i, key, value);
            } else {
                std::ranges::copy(key, row(keys, h, index).begin());
                std::ranges::copy(value, row(values, h, index).begin());
            }
        });
        batch.for_each([&](const Evaluation& e, std::size_t i, std::size_t index, std::size_t offset) {
            auto result = row(attention, h, index);
            Scalar::attention(config, layer, e.start + i, e.cache,
                              keys.subspan(offset * h, e.tokens.size() * h),
                              values.subspan(offset * h, e.tokens.size() * h),
                              row(qkv, 3 * h, index).first(h), result, w.scores);
            e.record("attention", layer, i, result);
        });
        affine(weights.attention_output, attention, projection);
        batch.for_each([&](const Evaluation& e, std::size_t i, std::size_t index, std::size_t) {
            auto activation = row(hidden, h, index);
            auto result = row(projection, h, index);
            auto norm_row = row(norm, h, index);
            e.record("attention_projection", layer, i, result);
            Scalar::add(activation, result);
            e.record("attention_residual", layer, i, activation);
            normalize(weights.post_attention_norm, activation, norm_row);
            e.record("post_attention_norm", layer, i, norm_row);
        });
        affine(weights.ff_in, norm, ff_values);
        batch.for_each([&](const Evaluation& e, std::size_t i, std::size_t index, std::size_t) {
            auto result = row(ff_values, ff, index);
            e.record("ffn", layer, i, result);
            Scalar::gelu(result);
            e.record("gelu", layer, i, result);
        });
        affine(weights.ff_out, ff_values, projection);
        batch.for_each([&](const Evaluation& e, std::size_t i, std::size_t index, std::size_t) {
            auto result = row(projection, h, index);
            auto activation = row(hidden, h, index);
            e.record("down", layer, i, result);
            Scalar::add(activation, result);
            e.record("layer_output", layer, i, activation);
        });
    }
    project_logits(w, batch, backend, output, hidden, norm.first(h),
                   [&](auto input, auto result) { normalize(final_norm, input, result); });
}

} // namespace

std::unique_ptr<Decoder> load_gpt2(SafeTensors& file, const std::filesystem::path& path,
                                   const ModelConfig& config, CpuBackend& backend) {
    return std::make_unique<Gpt2>(file, path, config, backend);
}

} // namespace nathr::detail
