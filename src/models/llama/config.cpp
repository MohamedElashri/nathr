#include "models/llama/config.hpp"

#include "formats/gguf.hpp"

#include <algorithm>
#include <cmath>

namespace nathr::detail {
namespace {

void validate_geometry(const ModelConfig& c, const std::string& path, bool independent_head_width = false) {
    if (!c.layers || c.layers > 56 || !c.hidden || !c.intermediate || !c.query_heads || !c.kv_heads ||
        c.query_heads % c.kv_heads || (!independent_head_width && c.hidden % c.query_heads) || !c.context ||
        !c.vocabulary || c.vocabulary > max_vocabulary) {
        throw Error(ErrorCode::unsupported, path + ": unsupported decoder dimensions or head geometry");
    }
}

void validate_config(const ModelConfig& c, const std::string& path, bool independent_head_width = false) {
    if (!c.head_dim || c.head_dim % 2 || (!independent_head_width && c.head_dim != c.hidden / c.query_heads))
        throw Error(ErrorCode::unsupported, path + ": unsupported rotary head width");
    if (!(c.rope_theta > 0) || !std::isfinite(c.rope_theta) || !(c.rms_epsilon > 0) ||
        !std::isfinite(c.rms_epsilon)) {
        throw Error(ErrorCode::malformed, path + ": invalid RMSNorm epsilon or RoPE theta");
    }
}

void set_boundary_tokens(ModelConfig& c, const Json& bos_id, const Json& eos_id, const std::string& path) {
    const auto bos = natural(bos_id, path + ": BOS token ID");
    if (bos >= c.vocabulary)
        throw Error(ErrorCode::malformed, path + ": special token ID exceeds vocabulary");
    c.bos = static_cast<Token>(bos);
    c.eos = primary_eos_id(eos_id, c.vocabulary, path + ": EOS token ID");
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

ModelConfig load_gguf_decoder_config(const Gguf& file) {
    const auto path = file.path().string();
    const auto& architecture = file.architecture();
    if (!file.split().empty())
        throw Error(ErrorCode::unsupported, path + ": split GGUF execution is unsupported");
    try {
        const auto name = architecture.value("general.architecture", std::string{});
        const bool qwen3 = name == "qwen3";
        if (!qwen3 && name != "qwen2" && name != "llama")
            throw Error(ErrorCode::unsupported,
                        path + ": expected a GGUF Llama, Qwen2, or Qwen3 architecture");
        const auto prefix = name + ".";
        const auto field = [&](std::string_view key) {
            const auto qualified = prefix + std::string(key);
            return natural(architecture.at(qualified), path + ": " + qualified);
        };
        const auto contains = [&](std::string_view key) {
            return architecture.contains(prefix + std::string(key));
        };
        constexpr std::string_view supported[] = {"block_count",          "context_length",
                                                  "embedding_length",     "feed_forward_length",
                                                  "attention.head_count", "attention.head_count_kv",
                                                  "attention.key_length", "attention.value_length",
                                                  "rope.freq_base",       "attention.layer_norm_rms_epsilon",
                                                  "vocab_size",           "rope.dimension_count"};
        for (const auto& [key, ignored] : architecture.items()) {
            if (key == "general.architecture")
                continue;
            if (!key.starts_with(prefix) ||
                std::ranges::find(supported, std::string_view(key).substr(prefix.size())) ==
                    std::end(supported))
                throw Error(ErrorCode::unsupported, path + ": unsupported architecture metadata " + key);
        }
        ModelConfig c{};
        c.layers = field("block_count");
        c.hidden = field("embedding_length");
        c.intermediate = field("feed_forward_length");
        c.query_heads = field("attention.head_count");
        c.kv_heads = contains("attention.head_count_kv") ? field("attention.head_count_kv") : c.query_heads;
        const auto embedding = file.records().find("token_embd.weight");
        if (embedding == file.records().end() || embedding->second.view.desc.rank != 2)
            throw Error(ErrorCode::malformed, path + ": missing or invalid GGUF token embedding");
        c.vocabulary = contains("vocab_size") ? field("vocab_size") : embedding->second.view.desc.shape[0];
        c.context = field("context_length");
        validate_geometry(c, path, qwen3);
        c.head_dim = contains("rope.dimension_count") ? field("rope.dimension_count")
                     : qwen3                          ? field("attention.key_length")
                                                      : c.hidden / c.query_heads;
        c.rope_theta = architecture.value(prefix + "rope.freq_base", 10000.0f);
        c.rms_epsilon = architecture.at(prefix + "attention.layer_norm_rms_epsilon").get<float>();
        c.tied_embeddings = !file.records().contains("output.weight");
        validate_config(c, path, qwen3);
        if ((contains("attention.key_length") && field("attention.key_length") != c.head_dim) ||
            (contains("attention.value_length") && field("attention.value_length") != c.head_dim)) {
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
