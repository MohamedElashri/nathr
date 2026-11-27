#include "requests.hpp"

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

std::optional<ChatFormat> chat_format(const std::filesystem::path& path, const Model& model) {
    if (!model.supports_generation())
        return {};
    if (model.architecture() == "qwen3") {
        // Official Qwen3-0.6B GGUF template. Implement its plain-text, non-thinking form.
        constexpr std::string_view qwen3_template =
            "57f1fd00f0013a2be96aa79b857391f27e23df5b5f847072b524c897e24d0361";
        if (detail::sha256(model.embedded_chat_template()) == qwen3_template)
            return ChatFormat{{}, true};
        return {};
    }
    if (!detail::is_model_directory(path))
        return {};
    const auto config = optional_config(path / "tokenizer_config.json");
    const auto field = config.find("chat_template");
    if (field == config.end() || !field->is_string())
        return {};
    const auto& format = field->get_ref<const std::string&>();
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
    return ChatFormat{system};
}

std::string render_chat(const Json& messages, const std::optional<ChatFormat>& format) {
    if (!format)
        throw Error(ErrorCode::unsupported, "checkpoint has no supported chat template");
    if (!messages.is_array() || messages.empty())
        throw Error(ErrorCode::invalid_argument, "messages must be a nonempty array");
    std::string prompt;
    const auto append = [&](std::string_view role, std::string_view text) {
        constexpr std::string_view qwen_delimiters[] = {"<|endoftext|>",   "<think>",      "</think>",
                                                        "<tool_call>",     "</tool_call>", "<tool_response>",
                                                        "</tool_response>"};
        if (text.find("<|im_start|>") != text.npos || text.find("<|im_end|>") != text.npos ||
            (format->qwen3 &&
             std::ranges::any_of(qwen_delimiters, [&](auto tag) { return text.find(tag) != text.npos; })))
            throw Error(ErrorCode::invalid_argument, "message text contains a reserved chat delimiter");
        const auto size = detail::sum(text.size(), role.size() + 24, "chat prompt");
        if (size > detail::mib - prompt.size())
            throw Error(ErrorCode::resource_limit, "rendered chat prompt exceeds 1 MiB");
        prompt += "<|im_start|>";
        prompt += role;
        prompt += '\n';
        prompt += text;
        prompt += "<|im_end|>\n";
    };
    bool user_turn = true;
    for (std::size_t i = 0; i < messages.size(); ++i) {
        const auto& message = messages[i];
        const auto role = message.at("role").get<std::string>();
        const auto text = message.at("content").get<std::string>();
        if (format->qwen3 && (!message.value("tool_calls", Json{}).is_null() ||
                              !message.value("reasoning_content", Json{}).is_null()))
            throw Error(ErrorCode::unsupported, "Qwen3 chat supports plain-text messages");
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
    const std::string_view assistant =
        format->qwen3 ? "<|im_start|>assistant\n<think>\n\n</think>\n\n" : "<|im_start|>assistant\n";
    if (assistant.size() > detail::mib - prompt.size())
        throw Error(ErrorCode::resource_limit, "rendered chat prompt exceeds 1 MiB");
    return prompt + std::string(assistant);
}

RequestDefaults request_defaults(const std::filesystem::path& path, const Model& model, bool greedy) {
    RequestDefaults result;
    result.sampling.greedy = greedy;
    result.chat = chat_format(path, model);
    if (result.chat && result.chat->qwen3)
        result.stop_ids.push_back(
            model.config().bos); // The pinned checkpoint uses <|endoftext|> as an extra EOS.
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
