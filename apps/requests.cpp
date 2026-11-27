#include "requests.hpp"

#include "backend/cpu/scalar.hpp"
#include "core/sha256.hpp"

#include <algorithm>
#include <limits>

namespace nathr::cli {
namespace {
using detail::Json;

Json optional_config(const std::filesystem::path& path) {
    if (std::filesystem::exists(path))
        return detail::read_json(path, detail::mib);
    return Json::object();
}

} // namespace

void add_capabilities(Json& info, const Model& model) {
    Json outputs = Json::array();
    for (const auto& output : model.outputs())
        outputs.push_back({{"name", output.name},
                           {"width", output.width},
                           {"input_limit", output.input_limit},
                           {"sequence", output.sequence}});
    info["outputs"] = std::move(outputs);
    if (!info.contains("architecture"))
        info["architecture"] = model.architecture();
    info["context"] = model.config().context;
    info["vocabulary"] = model.config().vocabulary;
    info["generation"] = model.supports_generation();
    info["fused_decoder_batch"] = model.supports_fused_batch();
    info["inputs"] = Json::array({"tokens"});
    if (model.architecture() == "bert")
        info["inputs"].push_back("token_type_ids");
    if (model.architecture() == "t5") {
        info["inputs"].push_back("decoder_tokens");
        info["decoder_start_id"] = model.config().bos;
    }
}

Tokenizer load_tokenizer(const Model& model, const std::filesystem::path& path,
                         const std::filesystem::path& sidecar) {
    auto tokenizer = Tokenizer::load_model(path, sidecar);
    const auto vocabulary = model.config().vocabulary;
    if (tokenizer.vocabulary() > vocabulary ||
        (model.architecture() != "t5" && tokenizer.vocabulary() != vocabulary))
        throw Error(ErrorCode::invalid_argument, "model and tokenizer vocabulary differ");
    return tokenizer;
}

std::vector<Token> token_array(const Json& input, std::string_view name) {
    if (!input.is_array())
        throw Error(ErrorCode::invalid_argument, std::string(name) + " must be an array");
    std::vector<Token> ids;
    ids.reserve(input.size());
    for (const auto& item : input) {
        const auto id = detail::natural(item, name);
        if (id > std::numeric_limits<Token>::max())
            throw Error(ErrorCode::invalid_argument, std::string(name) + " ID exceeds token range");
        ids.push_back(static_cast<Token>(id));
    }
    return ids;
}

std::vector<std::string> stop_strings(const Json& value) {
    if (value.is_string())
        return {value.get<std::string>()};
    if (!value.is_array() || value.size() > 16)
        throw Error(ErrorCode::invalid_argument, "stop must be a string or an array of up to 16 strings");
    return value.get<std::vector<std::string>>();
}

std::optional<ChatFormat> chat_format(const std::filesystem::path& path, const Model& model,
                                      std::string_view selection) {
    if (selection == "none")
        return {};
    if (selection != "auto") {
        if (!model.supports_generation() || model.architecture() == "t5")
            throw Error(ErrorCode::unsupported, "a chat format requires a decoder-only checkpoint");
        if (selection == "chatml")
            return ChatFormat{ChatStyle::chatml};
        if (selection == "qwen3")
            return ChatFormat{ChatStyle::qwen3};
        if (selection == "zephyr")
            return ChatFormat{ChatStyle::zephyr};
        throw Error(ErrorCode::invalid_argument, "chat format must be auto, none, chatml, qwen3 or zephyr");
    }
    if (!model.supports_generation())
        return {};

    std::string format;
    if (detail::is_model_directory(path)) {
        const auto config = optional_config(path / "tokenizer_config.json");
        const auto field = config.find("chat_template");
        if (field == config.end() || !field->is_string())
            return {};
        format = field->get<std::string>();
    } else {
        format = model.embedded_chat_template();
    }
    if (format.empty())
        return {};

    const auto digest = detail::sha256(format);
    // These Qwen3 templates render the same plain-text prompt with thinking disabled.
    if ((model.architecture() == "qwen3" &&
         (digest == "57f1fd00f0013a2be96aa79b857391f27e23df5b5f847072b524c897e24d0361" ||
          digest == "87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5")) ||
        (model.architecture() == "qwen3moe" &&
         digest == "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"))
        return ChatFormat{ChatStyle::qwen3};
    // These Qwen2 templates differ only in the tool-call example.
    if (model.architecture() == "qwen2" &&
        (digest == "d5495a1e5db0611132a97e46a65dbb64a642a499421228b9c8b93229097fa9a4" ||
         digest == "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"))
        return ChatFormat{ChatStyle::qwen2,
                          "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."};
    if (digest == "66291cf0045c2425a3a667cf3cbb7af2b11f09e025c02f97245323ab79119362")
        return ChatFormat{ChatStyle::zephyr};

    // Recognize the SmolLM2 ChatML template and read its default system message.
    // The checkpoint's Jinja source is data and is never executed.
    constexpr std::string_view prefix =
        R"({% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system
)";
    constexpr std::string_view suffix =
        R"(<|im_end|>
' }}{% endif %}{{'<|im_start|>' + message['role'] + '
' + message['content'] + '<|im_end|>' + '
'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant
' }}{% endif %})";
    if (format.size() < prefix.size() + suffix.size() || !format.starts_with(prefix) ||
        !format.ends_with(suffix))
        return {};
    const auto system = format.substr(prefix.size(), format.size() - prefix.size() - suffix.size());
    if (system.find_first_of("'\\\r\n") != system.npos || system.find("<|") != system.npos)
        return {};
    return ChatFormat{ChatStyle::chatml, system, true};
}

std::string render_chat(const Json& messages, const std::optional<ChatFormat>& format) {
    if (!format)
        throw Error(ErrorCode::unsupported, "checkpoint has no supported chat template");
    if (!messages.is_array() || messages.empty())
        throw Error(ErrorCode::invalid_argument, "messages must be a nonempty array");
    const bool qwen3 = format->style == ChatStyle::qwen3;
    const bool qwen = qwen3 || format->style == ChatStyle::qwen2;
    const bool zephyr = format->style == ChatStyle::zephyr;
    constexpr std::string_view zephyr_delimiters[] = {"<s>", "</s>", "<|system|>", "<|user|>",
                                                      "<|assistant|>"};
    std::string prompt;
    const auto append = [&](std::string_view role, std::string_view text) {
        constexpr std::string_view qwen_delimiters[] = {"<|endoftext|>",   "<think>",      "</think>",
                                                        "<tool_call>",     "</tool_call>", "<tool_response>",
                                                        "</tool_response>"};
        const auto contains = [&](std::string_view tag) { return text.find(tag) != text.npos; };
        if ((zephyr && std::ranges::any_of(zephyr_delimiters, contains)) ||
            (!zephyr && (contains("<|im_start|>") || contains("<|im_end|>"))) ||
            (qwen && std::ranges::any_of(qwen_delimiters, contains)))
            throw Error(ErrorCode::invalid_argument, "message text contains a reserved chat delimiter");
        const auto size = detail::sum(text.size(), role.size() + 24, "chat prompt");
        if (size > detail::mib - prompt.size())
            throw Error(ErrorCode::resource_limit, "rendered chat prompt exceeds 1 MiB");
        prompt += zephyr ? "<|" : "<|im_start|>";
        prompt += role;
        if (zephyr)
            prompt += "|>";
        prompt += '\n';
        prompt += text;
        prompt += zephyr ? "</s>\n" : "<|im_end|>\n";
    };
    bool user_turn = true;
    for (std::size_t i = 0; i < messages.size(); ++i) {
        const auto& message = messages[i];
        const auto role = message.at("role").get<std::string>();
        const auto text = message.at("content").get<std::string>();
        if (qwen && (!message.value("tool_calls", Json{}).is_null() ||
                     !message.value("reasoning_content", Json{}).is_null()))
            throw Error(ErrorCode::unsupported, "Qwen chat supports plain-text messages");
        if (!i && role == "system") {
            append(role, text);
            continue;
        }
        if (!i && format->default_system)
            append("system", *format->default_system);
        if (role != (user_turn ? "user" : "assistant"))
            throw Error(ErrorCode::invalid_argument, "expected alternating user and assistant messages");
        append(role, text);
        user_turn = !user_turn;
    }
    if (user_turn)
        throw Error(ErrorCode::invalid_argument, "the last message must have the user role");
    const std::string_view assistant = zephyr  ? "<|assistant|>\n"
                                       : qwen3 ? "<|im_start|>assistant\n<think>\n\n</think>\n\n"
                                               : "<|im_start|>assistant\n";
    if (assistant.size() > detail::mib - prompt.size())
        throw Error(ErrorCode::resource_limit, "rendered chat prompt exceeds 1 MiB");
    return prompt + std::string(assistant);
}

std::vector<Token> encode_prompt(const Tokenizer& tokenizer, const Model& model, std::string_view prompt,
                                 bool add_bos) {
    auto ids = tokenizer.encode(prompt);
    const auto bos = model.config().bos;
    if (tokenizer.adds_bos() ? (ids.empty() || ids.front() != bos) : add_bos)
        ids.insert(ids.begin(), bos);
    return ids;
}

RequestDefaults request_defaults(const std::filesystem::path& path, const Model& model, bool greedy,
                                 std::string_view chat_override) {
    RequestDefaults result;
    result.sampling.greedy = greedy;
    result.chat = chat_format(path, model, chat_override);
    if (result.chat && ((result.chat->style == ChatStyle::qwen3 &&
                         (model.architecture() == "qwen3" || model.architecture() == "qwen3moe")) ||
                        result.chat->style == ChatStyle::qwen2))
        result.stop_ids.push_back(
            model.config().bos); // <|endoftext|> is an extra EOS in these Qwen templates.
    if (!model.supports_generation() || !detail::is_model_directory(path))
        return result;

    const auto config = optional_config(path / "config.json");
    const auto generation = optional_config(path / "generation_config.json");
    const auto value = [&](std::string_view key) -> Json {
        if (const auto field = generation.find(key); field != generation.end() && !field->is_null())
            return *field;
        return config.value(key, Json{});
    };
    if (const auto field = value("do_sample"); !field.is_null())
        result.sampling.greedy = !field.get<bool>();
    if (const auto field = value("temperature"); !field.is_null())
        result.sampling.temperature = field.get<double>();
    if (const auto field = value("top_p"); !field.is_null())
        result.sampling.top_p = field.get<double>();
    if (const auto field = value("top_k"); !field.is_null())
        result.sampling.top_k = detail::natural(field, "top_k");
    if (const auto field = value("max_new_tokens"); !field.is_null())
        result.max_new_tokens = detail::natural(field, "max_new_tokens");
    if (const auto field = value("eos_token_id"); !field.is_null())
        result.stop_ids = token_array(field.is_array() ? field : Json::array({field}), "eos_token_id");
    if (std::ranges::any_of(result.stop_ids, [&](Token id) { return id >= model.config().vocabulary; }))
        throw Error(ErrorCode::malformed, "checkpoint stop ID exceeds model vocabulary");
    return result;
}

KvPrecision default_cache_precision(const Model& model, std::size_t context, std::size_t kv_budget) {
    const auto target = std::min(context ? context : detail::default_context_tokens, model.config().context);
    if (model.supports_fused_batch() &&
        target > detail::Cache::fit_tokens(model.config(), kv_budget, KvPrecision::f32))
        return KvPrecision::f16;
    return KvPrecision::f32;
}

std::size_t fit_generation_budget(const Model& model, std::size_t prompt, std::size_t context,
                                  std::size_t desired, KvPrecision precision, std::size_t kv_budget) {
    const bool causal = model.supports_fused_batch();
    const auto input_limit = causal ? context : model.config().context;
    if (prompt > input_limit)
        throw Error(ErrorCode::invalid_argument, "prompt exceeds context");
    if (!desired)
        return 0;
    const auto cache = detail::Cache::fit_tokens(model.config(), kv_budget, precision);
    const auto capacity = causal ? std::min(context, cache) : cache;
    if (prompt > capacity || (!causal && prompt == capacity))
        throw Error(ErrorCode::resource_limit, "prompt leaves no room for generation within the KV budget");
    const auto remaining = capacity - prompt;
    return std::min(desired, causal ? remaining + 1 : std::min(context, remaining));
}

detail::Json logprobs_json(const TokenLogprobs& scores, const Tokenizer* tokenizer, bool first) {
    const auto record = [&](const TokenLogprob& score) {
        Json value{
            {"token_id", score.id}, {"logprob", score.logprob}, {"token", nullptr}, {"bytes", nullptr}};
        if (tokenizer && score.id < tokenizer->vocabulary()) {
            const auto piece = tokenizer->decode_piece(score.id, first, false);
            Utf8Stream utf8;
            auto text = utf8.push(piece);
            text += utf8.finish();
            value["token"] = std::move(text);
            if (!tokenizer->token_bytes(score.id, true).empty()) {
                std::vector<unsigned> bytes;
                bytes.reserve(piece.size());
                for (unsigned char byte : piece)
                    bytes.push_back(byte);
                value["bytes"] = std::move(bytes);
            }
        }
        return value;
    };
    auto result = record(scores.chosen);
    auto alternatives = Json::array();
    for (const auto& alternative : scores.alternatives)
        alternatives.push_back(record(alternative));
    result["top_logprobs"] = std::move(alternatives);
    return result;
}

} // namespace nathr::cli
