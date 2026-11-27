#pragma once

#include "core/input.hpp"

#include <nathr/generation.hpp>

namespace nathr::cli {

enum class ChatStyle { chatml, qwen3, zephyr };

struct ChatFormat {
    ChatStyle style;
    std::optional<std::string> default_system{};
    bool function_calls = false;
};

struct RequestDefaults {
    SamplingOptions sampling;
    std::size_t max_new_tokens = 32;
    std::vector<Token> stop_ids;
    std::optional<ChatFormat> chat;
};

RequestDefaults request_defaults(const std::filesystem::path& path, const Model& model, bool greedy,
                                 std::string_view chat_override = "auto");
void add_capabilities(detail::Json& info, const Model& model);
std::optional<ChatFormat> chat_format(const std::filesystem::path& path, const Model& model,
                                      std::string_view selection = "auto");
std::string render_chat(const detail::Json& messages, const std::optional<ChatFormat>& format);
std::vector<Token> encode_prompt(const Tokenizer& tokenizer, const Model& model, std::string_view prompt,
                                 bool add_bos = false);
std::vector<Token> token_array(const detail::Json& value, std::string_view name);
std::vector<std::string> stop_strings(const detail::Json& value);
detail::Json logprobs_json(const TokenLogprobs& scores, const Tokenizer* tokenizer, bool first);
Tokenizer load_tokenizer(const Model& model, const std::filesystem::path& path,
                         const std::filesystem::path& sidecar);

} // namespace nathr::cli
