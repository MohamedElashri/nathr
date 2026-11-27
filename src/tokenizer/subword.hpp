#pragma once

#include "core/input.hpp"

#include <nathr/tokenizer.hpp>

#include <array>
#include <map>
#include <optional>
#include <unordered_map>

namespace nathr::detail {

class SubwordTokenizer {
  public:
    explicit SubwordTokenizer(const Json& json);
    [[nodiscard]] TokenizedInput encode(std::string_view first, std::optional<std::string_view> second) const;
    [[nodiscard]] std::vector<std::string> pieces(std::string_view text) const;
    [[nodiscard]] std::string_view piece(Token token, bool skip_special) const;
    [[nodiscard]] std::string decode_piece(Token token, bool first, bool skip_special) const;
    [[nodiscard]] std::string decode(std::span<const Token> tokens, bool skip_special) const;
    [[nodiscard]] std::size_t vocabulary() const noexcept { return vocab_.size(); }

  private:
    struct TrieNode {
        std::map<unsigned char, std::size_t> edges;
        std::optional<Token> token;
    };
    struct TemplateItem {
        int sequence;
        Token token, type;
    };

    std::vector<Token> encode_sequence(std::string_view text) const;
    void wordpiece(std::string_view text, std::vector<Token>& out) const;
    void unigram(std::string_view text, std::vector<Token>& out) const;
    std::string normalize(std::string_view text) const;
    std::optional<std::string_view> normalize_lookup(std::string_view text) const;

    bool wordpiece_, lower_ = false, accents_ = false, clean_ = false, chinese_ = false, cleanup_ = false;
    bool split_whitespace_ = true, collapse_spaces_ = false;
    std::string prefix_, metaspace_;
    std::size_t max_word_ = 0;
    std::optional<std::size_t> max_length_;
    Token unknown_ = 0;
    double unknown_score_ = 0;
    std::vector<std::string> vocab_;
    std::vector<bool> special_;
    std::unordered_map<std::string, Token> ids_;
    std::vector<Token> added_;
    std::array<bool, 256> added_starts_{};
    std::vector<double> scores_;
    std::vector<TrieNode> trie_;
    std::vector<std::uint32_t> norm_trie_;
    std::string rewrites_;
    std::vector<TemplateItem> single_, pair_;
};

} // namespace nathr::detail
