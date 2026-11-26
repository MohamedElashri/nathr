#include "models/llama/config.hpp"

#include "formats/gguf.hpp"

#include <algorithm>
#include <cmath>

namespace nathr::detail {
namespace {

void validate_geometry(const ModelConfig& c, const std::string& path) {
    if (!c.layers || c.layers > 56 || !c.hidden || !c.intermediate || !c.query_heads || !c.kv_heads ||
        c.query_heads % c.kv_heads || c.hidden % c.query_heads || !c.context || !c.vocabulary ||
        c.vocabulary > max_vocabulary) {
        throw Error(ErrorCode::unsupported, path + ": unsupported Llama dimensions or head geometry");
    }
}

void validate_config(const ModelConfig& c, const std::string& path) {
    if (c.head_dim != c.hidden / c.query_heads || c.head_dim % 2)
        throw Error(ErrorCode::unsupported, path + ": unsupported rotary head width");
    if (!(c.rope_theta > 0) || !std::isfinite(c.rope_theta) || !(c.rms_epsilon > 0) ||
        !std::isfinite(c.rms_epsilon)) {
        throw Error(ErrorCode::malformed, path + ": invalid RMSNorm epsilon or RoPE theta");
    }
}

void set_boundary_tokens(ModelConfig& c, const Json& bos_id, const Json& eos_id, const std::string& path) {
    const auto bos = natural(bos_id, path + ": BOS token ID");
    const auto eos = natural(eos_id, path + ": EOS token ID");
    if (bos >= c.vocabulary || eos >= c.vocabulary)
        throw Error(ErrorCode::malformed, path + ": special token ID exceeds vocabulary");
    c.bos = static_cast<Token>(bos);
    c.eos = static_cast<Token>(eos);
}

} // namespace

ModelConfig load_llama_config(const Json& json, const std::filesystem::path& config_path) {
    const auto path = config_path.string();
    try {
        const auto field = [&](std::string_view name) {
            return natural(json.at(name), path + ": " + std::string(name));
        };
        if (json.at("model_type") != "llama" || json.at("hidden_act") != "silu" ||
            (json.contains("architectures") &&
             json.at("architectures") != Json::array({"LlamaForCausalLM"})) ||
            json.value("attention_bias", false) || json.value("mlp_bias", false) ||
            json.value("rope_interleaved", false) ||
            (json.contains("rope_scaling") && !json.at("rope_scaling").is_null()) ||
            (json.contains("rope_parameters") && !json.at("rope_parameters").is_null()) ||
            (json.contains("pretraining_tp") && field("pretraining_tp") != 1)) {
            throw Error(ErrorCode::unsupported, path + ": unsupported Llama configuration");
        }
        ModelConfig c{};
        c.layers = field("num_hidden_layers");
        c.hidden = field("hidden_size");
        c.intermediate = field("intermediate_size");
        c.query_heads = field("num_attention_heads");
        c.kv_heads = json.contains("num_key_value_heads") && !json.at("num_key_value_heads").is_null()
                         ? field("num_key_value_heads")
                         : c.query_heads;
        c.vocabulary = field("vocab_size");
        c.context = field("max_position_embeddings");
        validate_geometry(c, path);
        c.head_dim = json.contains("head_dim") && !json.at("head_dim").is_null() ? field("head_dim")
                                                                                 : c.hidden / c.query_heads;
        c.rope_theta = json.value("rope_theta", 10000.0f);
        c.rms_epsilon = json.at("rms_norm_eps").get<float>();
        c.tied_embeddings = json.value("tie_word_embeddings", false);
        validate_config(c, path);
        set_boundary_tokens(c, json.at("bos_token_id"), json.at("eos_token_id"), path);
        return c;
    } catch (const Json::exception& error) {
        throw Error(ErrorCode::malformed, path + ": " + error.what());
    }
}

ModelConfig load_llama_config(const Gguf& file) {
    const auto path = file.path().string();
    const auto& architecture = file.architecture();
    if (!file.split().empty())
        throw Error(ErrorCode::unsupported, path + ": split GGUF execution is unsupported");
    if (!architecture.contains("general.architecture") || architecture.at("general.architecture") != "llama")
        throw Error(ErrorCode::unsupported, path + ": expected a GGUF Llama architecture");
    try {
        const auto field = [&](std::string_view key) {
            return natural(architecture.at(key), path + ": " + std::string(key));
        };
        constexpr std::string_view supported[] = {"general.architecture",
                                                  "llama.block_count",
                                                  "llama.context_length",
                                                  "llama.embedding_length",
                                                  "llama.feed_forward_length",
                                                  "llama.attention.head_count",
                                                  "llama.attention.head_count_kv",
                                                  "llama.attention.key_length",
                                                  "llama.attention.value_length",
                                                  "llama.rope.freq_base",
                                                  "llama.attention.layer_norm_rms_epsilon",
                                                  "llama.vocab_size",
                                                  "llama.rope.dimension_count"};
        for (const auto& [key, ignored] : architecture.items()) {
            if (std::ranges::find(supported, key) == std::end(supported))
                throw Error(ErrorCode::unsupported, path + ": unsupported architecture metadata " + key);
        }
        ModelConfig c{};
        c.layers = field("llama.block_count");
        c.hidden = field("llama.embedding_length");
        c.intermediate = field("llama.feed_forward_length");
        c.query_heads = field("llama.attention.head_count");
        c.kv_heads = architecture.contains("llama.attention.head_count_kv")
                         ? field("llama.attention.head_count_kv")
                         : c.query_heads;
        const auto embedding = file.records().find("token_embd.weight");
        if (embedding == file.records().end() || embedding->second.view.desc.rank != 2)
            throw Error(ErrorCode::malformed, path + ": missing or invalid GGUF token embedding");
        c.vocabulary = architecture.contains("llama.vocab_size") ? field("llama.vocab_size")
                                                                 : embedding->second.view.desc.shape[0];
        c.context = field("llama.context_length");
        validate_geometry(c, path);
        c.head_dim = architecture.contains("llama.rope.dimension_count") ? field("llama.rope.dimension_count")
                                                                         : c.hidden / c.query_heads;
        c.rope_theta = architecture.value("llama.rope.freq_base", 10000.0f);
        c.rms_epsilon = architecture.at("llama.attention.layer_norm_rms_epsilon").get<float>();
        c.tied_embeddings = !file.records().contains("output.weight");
        validate_config(c, path);
        if ((architecture.contains("llama.attention.key_length") &&
             field("llama.attention.key_length") != c.head_dim) ||
            (architecture.contains("llama.attention.value_length") &&
             field("llama.attention.value_length") != c.head_dim)) {
            throw Error(ErrorCode::unsupported, path + ": unsupported GGUF key or value head width");
        }
        const auto& tokenizer = file.tokenizer();
        set_boundary_tokens(c, tokenizer.at("tokenizer.ggml.bos_token_id"),
                            tokenizer.at("tokenizer.ggml.eos_token_id"), path);
        return c;
    } catch (const Json::exception& error) {
        throw Error(ErrorCode::malformed, path + ": " + error.what());
    }
}

} // namespace nathr::detail
