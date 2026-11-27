#pragma once

#include "core/input.hpp"

#include <nathr/generation.hpp>

namespace nathr::cli {

struct ChatFormat {
    std::optional<std::string> default_system;
    bool qwen3 = false;
};

struct RequestDefaults {
    SamplingOptions sampling;
    std::size_t max_new_tokens = 32;
    std::vector<Token> stop_ids;
    std::optional<ChatFormat> chat;
};

RequestDefaults request_defaults(const std::filesystem::path& path, const Model& model, bool greedy);
void add_capabilities(detail::Json& info, const Model& model);
std::optional<ChatFormat> chat_format(const std::filesystem::path& path, const Model& model);
std::string render_chat(const detail::Json& messages, const std::optional<ChatFormat>& format);
std::vector<Token> token_array(const detail::Json& value, std::string_view name);
std::vector<std::string> stop_strings(const detail::Json& value);
Tokenizer load_tokenizer(const Model& model, const std::filesystem::path& path,
                         const std::filesystem::path& sidecar);

} // namespace nathr::cli
