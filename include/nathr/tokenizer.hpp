#pragma once

#include <nathr/nathr.hpp>

#include <string>
#include <vector>

namespace nathr {

struct TokenizedInput {
    std::vector<Token> ids, token_type_ids;
};

class Tokenizer {
  public:
    static Tokenizer load(const std::filesystem::path& tokenizer_json);
    // A GGUF sidecar must match the embedded token IDs and merge order.
    static Tokenizer load_model(const std::filesystem::path& model,
                                const std::filesystem::path& sidecar = {});
    [[nodiscard]] std::size_t vocabulary() const noexcept;
    [[nodiscard]] bool byte_level() const noexcept;
    [[nodiscard]] std::vector<Token> encode(std::string_view text) const;
    [[nodiscard]] TokenizedInput encode_pair(std::string_view text, std::string_view second) const;
    // Returns the pretokenized pieces with added tokens treated as ordinary text.
    [[nodiscard]] std::vector<std::string> pretokenize(std::string_view text) const;
    // Subword pipelines return vocabulary pieces here. decode() applies their spacing rules.
    [[nodiscard]] std::string_view token_bytes(Token token, bool skip_special = false) const;
    [[nodiscard]] std::string decode_piece(Token token, bool first, bool skip_special = false) const;
    [[nodiscard]] std::string decode_bytes(std::span<const Token> tokens, bool skip_special = false) const;
    [[nodiscard]] std::string decode(std::span<const Token> tokens, bool skip_special = false) const;

  private:
    struct Impl;
    explicit Tokenizer(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}
    std::shared_ptr<const Impl> impl_;
};

// Converts token bytes to UTF-8 text, retaining a potentially incomplete final code point.
class Utf8Stream {
  public:
    std::string push(std::string_view bytes);
    std::string finish();

  private:
    std::string pending_;
};

} // namespace nathr
