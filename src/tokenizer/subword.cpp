#include "tokenizer/subword.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unicode/brkiter.h>
#include <unicode/locid.h>
#include <unicode/normalizer2.h>
#include <unicode/uchar.h>
#include <unicode/utext.h>
#include <unicode/utf8.h>

namespace nathr::detail {
namespace {

struct Character {
    UChar32 value;
    std::size_t next;
};

Character character(std::string_view text, std::size_t offset) {
    auto next = static_cast<std::int32_t>(offset);
    UChar32 cp;
    U8_NEXT(reinterpret_cast<const std::uint8_t*>(text.data()), next, static_cast<std::int32_t>(text.size()),
            cp);
    if (cp < 0)
        throw Error(ErrorCode::invalid_argument, "invalid UTF-8 in tokenizer input");
    return {cp, static_cast<std::size_t>(next)};
}

void validate_text(std::string_view text) {
    if (text.size() > mib)
        throw Error(ErrorCode::resource_limit, "tokenizer input exceeds 1 MiB");
    for (std::size_t i = 0; i < text.size();)
        i = character(text, i).next;
}

bool type(const Json& json, std::string_view name) {
    return json.is_object() && json.value("type", std::string{}) == name;
}

[[noreturn]] void unsupported(std::string_view what) {
    throw Error(ErrorCode::unsupported, "unsupported " + std::string(what));
}

[[noreturn]] void malformed(std::string_view what) {
    throw Error(ErrorCode::malformed, "invalid " + std::string(what));
}

bool chinese(UChar32 cp) {
    return (cp >= 0x4e00 && cp <= 0x9fff) || (cp >= 0x3400 && cp <= 0x4dbf) ||
           (cp >= 0x20000 && cp <= 0x2a6df) || (cp >= 0x2a700 && cp <= 0x2b73f) ||
           (cp >= 0x2b740 && cp <= 0x2b81f) || (cp >= 0x2b920 && cp <= 0x2ceaf) ||
           (cp >= 0xf900 && cp <= 0xfaff) || (cp >= 0x2f800 && cp <= 0x2fa1f);
}

bool punctuation(UChar32 cp) {
    return (cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) ||
           (cp >= 123 && cp <= 126) || u_ispunct(cp);
}

bool control(UChar32 cp) {
    const auto kind = u_charType(cp);
    return kind == U_CONTROL_CHAR || kind == U_FORMAT_CHAR || kind == U_UNASSIGNED ||
           kind == U_PRIVATE_USE_CHAR;
}

std::vector<std::byte> base64(std::string_view encoded) {
    constexpr std::string_view alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    const auto end = encoded.find('=');
    const auto digits = end == encoded.npos ? encoded.size() : end;
    if (encoded.size() % 4 || encoded.size() - digits > 2 ||
        encoded.substr(digits).find_first_not_of('=') != encoded.npos)
        malformed("SentencePiece normalization encoding");
    std::vector<std::byte> bytes;
    bytes.reserve(digits * 3 / 4);
    unsigned residue = 0, bits = 0;
    for (const auto digit : encoded.substr(0, digits)) {
        const auto value = alphabet.find(digit);
        if (value == alphabet.npos)
            malformed("SentencePiece normalization encoding");
        residue = (residue << 6) | static_cast<unsigned>(value);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            bytes.push_back(static_cast<std::byte>((residue >> bits) & 255));
        }
    }
    if (residue & ((1u << bits) - 1))
        malformed("SentencePiece normalization padding");
    return bytes;
}

std::uint32_t trie_offset(std::uint32_t unit) {
    return (unit >> 10) << ((unit & 0x200u) ? 8 : 0);
}

void replace(std::string& text, std::string_view before, std::string_view after) {
    for (std::size_t pos = 0; (pos = text.find(before, pos)) != text.npos; pos += after.size())
        text.replace(pos, before.size(), after);
}

} // namespace

SubwordTokenizer::SubwordTokenizer(const Json& json) : wordpiece_(json.at("model").at("vocab").is_object()) {
    if (json.at("version") != "1.0")
        unsupported("subword tokenizer configuration");
    const auto& truncation = json.at("truncation");
    if (!truncation.is_null()) {
        if (truncation.at("strategy") != "LongestFirst" ||
            truncation.value("direction", std::string("Right")) != "Right" || truncation.at("stride") != 0)
            unsupported("tokenizer truncation");
        max_length_ = natural(truncation.at("max_length"), "truncation length");
    }
    const auto& padding = json.at("padding");
    if (!padding.is_null() &&
        (padding.at("strategy") != "BatchLongest" || !padding.at("pad_to_multiple_of").is_null()))
        unsupported("tokenizer padding");
    const auto& model = json.at("model");
    if (model.value("type", std::string(wordpiece_ ? "WordPiece" : "Unigram")) !=
        (wordpiece_ ? "WordPiece" : "Unigram"))
        unsupported("subword vocabulary type");
    const auto& vocab = model.at("vocab");
    if (vocab.empty() || vocab.size() > max_vocabulary)
        malformed("subword vocabulary size");
    vocab_.resize(vocab.size());
    const auto add = [&](std::size_t id, std::string piece) {
        if (piece.empty() || piece.size() > 1024 || !vocab_[id].empty() ||
            !ids_.emplace(piece, static_cast<Token>(id)).second)
            malformed("subword vocabulary entry");
        vocab_[id] = std::move(piece);
    };
    const auto& normalizer = json.at("normalizer");
    const auto& pre = json.at("pre_tokenizer");
    const auto& decoder = json.at("decoder");
    if (wordpiece_) {
        if (!type(normalizer, "BertNormalizer") || !type(pre, "BertPreTokenizer") ||
            !type(decoder, "WordPiece"))
            unsupported("WordPiece pipeline");
        clean_ = normalizer.value("clean_text", true);
        chinese_ = normalizer.value("handle_chinese_chars", true);
        lower_ = normalizer.value("lowercase", true);
        const auto accents = normalizer.value("strip_accents", Json{});
        accents_ = accents.is_null() ? lower_ : accents.get<bool>();
        prefix_ = model.at("continuing_subword_prefix").get<std::string>();
        if (prefix_.empty() || decoder.at("prefix") != prefix_)
            unsupported("WordPiece affix");
        cleanup_ = decoder.value("cleanup", true);
        max_word_ = natural(model.at("max_input_chars_per_word"), "WordPiece length");
        if (!max_word_ || max_word_ > 1024)
            unsupported("WordPiece length");
        for (const auto& [piece, value] : vocab.items()) {
            const auto id = natural(value, "WordPiece ID");
            if (id >= vocab_.size())
                malformed("WordPiece ID");
            add(id, piece);
        }
        const auto found = ids_.find(model.at("unk_token").get<std::string>());
        if (found == ids_.end())
            malformed("WordPiece unknown ID");
        unknown_ = found->second;
    } else {
        if (!vocab.is_array() || model.value("byte_fallback", false) || !type(decoder, "Metaspace"))
            unsupported("Unigram pipeline");
        const Json* metaspace = &pre;
        split_whitespace_ = type(pre, "Sequence");
        if (split_whitespace_) {
            const auto& stages = pre.at("pretokenizers");
            if (!stages.is_array() || stages.size() != 2 || !type(stages[0], "WhitespaceSplit"))
                unsupported("Unigram pretokenization");
            metaspace = &stages[1];
        }
        if (!type(*metaspace, "Metaspace") || !metaspace->value("split", true))
            unsupported("Unigram pretokenization");
        const auto always = [](const Json& stage) {
            return stage.value("add_prefix_space", true) &&
                   stage.value("prepend_scheme", std::string("always")) == "always";
        };
        metaspace_ = metaspace->at("replacement").get<std::string>();
        if (metaspace_.empty() || character(metaspace_, 0).next != metaspace_.size() ||
            decoder.at("replacement") != metaspace_ || !always(*metaspace) || !always(decoder))
            unsupported("Unigram metaspace configuration");
        const auto unk = natural(model.at("unk_id"), "Unigram unknown ID");
        if (unk >= vocab_.size())
            malformed("Unigram unknown ID");
        unknown_ = static_cast<Token>(unk);
        const Json* table = &normalizer;
        if (type(normalizer, "Sequence")) {
            const auto& stages = normalizer.at("normalizers");
            if (!stages.is_array() || stages.size() != 2 || !type(stages[0], "Precompiled") ||
                !type(stages[1], "Replace") || stages[1].at("pattern") != Json{{"Regex", " {2,}"}} ||
                stages[1].at("content") != " ")
                unsupported("Unigram normalization");
            table = &stages[0];
            collapse_spaces_ = true;
        }
        if (!table->is_null()) {
            if (!type(*table, "Precompiled"))
                unsupported("Unigram normalization");
            const auto bytes = base64(table->at("precompiled_charsmap").get<std::string>());
            if (bytes.size() < 9)
                malformed("SentencePiece normalization table");
            const auto data = std::span(bytes);
            const auto size = little_integer(data.first(4));
            if (!size || size % 4 || size >= bytes.size() - 4)
                malformed("SentencePiece normalization trie");
            norm_trie_.reserve(size / 4);
            for (std::size_t i = 4; i < 4 + size; i += 4)
                norm_trie_.push_back(static_cast<std::uint32_t>(little_integer(data.subspan(i, 4))));
            rewrites_.assign(reinterpret_cast<const char*>(bytes.data() + 4 + size), bytes.size() - 4 - size);
            validate_text(rewrites_);
        }
        trie_.emplace_back();
        for (std::size_t id = 0; id < vocab.size(); ++id) {
            if (!vocab[id].is_array() || vocab[id].size() != 2)
                malformed("Unigram entry");
            add(id, vocab[id][0].get<std::string>());
            const auto score = vocab[id][1].get<double>();
            if (!std::isfinite(score))
                malformed("Unigram score");
            scores_.push_back(score);
            std::size_t node = 0;
            for (unsigned char byte : vocab_[id]) {
                auto [edge, inserted] = trie_[node].edges.try_emplace(byte, trie_.size());
                node = edge->second;
                if (inserted) {
                    if (trie_.size() == mib)
                        throw Error(ErrorCode::resource_limit, "Unigram trie is too large");
                    trie_.emplace_back();
                }
            }
            trie_[node].token = static_cast<Token>(id);
        }
        unknown_score_ = *std::ranges::min_element(scores_) - 10.0;
    }
    special_.resize(vocab_.size());
    const auto& added = json.at("added_tokens");
    if (!added.is_array() || added.size() > 256)
        malformed("added-token list");
    for (const auto& entry : added) {
        if (!entry.at("special").get<bool>() || entry.at("normalized").get<bool>() ||
            entry.at("single_word").get<bool>() || entry.at("lstrip").get<bool>() ||
            entry.at("rstrip").get<bool>())
            unsupported("subword added-token flags");
        const auto text = entry.at("content").get<std::string>();
        const auto id = natural(entry.at("id"), "added ID");
        if (id >= max_vocabulary || text.empty() || text.size() > 128)
            malformed("subword special token");
        if (id >= vocab_.size()) {
            vocab_.resize(id + 1);
            special_.resize(id + 1);
        }
        if (vocab_[id].empty())
            add(id, text);
        if (vocab_[id] != text || special_[id])
            malformed("subword special token");
        special_[id] = true;
        added_starts_[static_cast<unsigned char>(text.front())] = true;
        added_.push_back(static_cast<Token>(id));
    }
    if (std::ranges::any_of(vocab_, [](const std::string& piece) { return piece.empty(); }))
        malformed("subword vocabulary ID gap");
    std::ranges::sort(added_, [&](Token a, Token b) { return vocab_[a].size() > vocab_[b].size(); });

    const auto& post = json.at("post_processor");
    if (!type(post, "TemplateProcessing"))
        unsupported("subword postprocessor");
    const auto parse_template = [&](const Json& entries, bool pair) {
        if (!entries.is_array() || entries.empty() || entries.size() > 16)
            malformed("tokenizer template");
        std::vector<TemplateItem> result;
        std::array<std::size_t, 2> seen{};
        for (const auto& entry : entries) {
            if (entry.size() != 1)
                malformed("tokenizer template entry");
            const bool sequence = entry.contains("Sequence");
            const auto& value = entry.at(sequence ? "Sequence" : "SpecialToken");
            const auto type_id = natural(value.at("type_id"), "template type ID");
            if (type_id > std::numeric_limits<Token>::max())
                malformed("template type ID");
            const auto item = value.at("id").get<std::string>();
            if (sequence) {
                const int index = item == "A" ? 0 : item == "B" && pair ? 1 : -1;
                if (index < 0 || seen[index]++)
                    malformed("tokenizer template sequence");
                result.push_back({index, 0, static_cast<Token>(type_id)});
            } else {
                const auto& ids = post.at("special_tokens").at(item).at("ids");
                if (!ids.is_array() || ids.size() != 1)
                    unsupported("multi-token template entries");
                const auto id = natural(ids[0], "template token ID");
                if (id >= vocab_.size() || !special_[id])
                    malformed("template special token");
                result.push_back({-1, static_cast<Token>(id), static_cast<Token>(type_id)});
            }
        }
        if (seen[0] != 1 || seen[1] != static_cast<std::size_t>(pair))
            malformed("incomplete tokenizer template");
        return result;
    };
    single_ = parse_template(post.at("single"), false);
    if (post.contains("pair") && !post.at("pair").is_null())
        pair_ = parse_template(post.at("pair"), true);
}

std::optional<std::string_view> SubwordTokenizer::normalize_lookup(std::string_view text) const {
    auto node = trie_offset(norm_trie_.front());
    for (const unsigned char byte : text) {
        if (!byte)
            break;
        node ^= byte;
        if (node >= norm_trie_.size())
            malformed("SentencePiece normalization transition");
        const auto unit = norm_trie_[node];
        if ((unit & 0x800000ffu) != byte)
            break;
        node ^= trie_offset(unit);
        if (unit & 0x100u) {
            if (node >= norm_trie_.size())
                malformed("SentencePiece normalization leaf");
            const auto offset = norm_trie_[node] & 0x7fffffffu;
            if (offset >= rewrites_.size())
                malformed("SentencePiece rewrite offset");
            const auto end = rewrites_.find('\0', offset);
            if (end == rewrites_.npos)
                malformed("SentencePiece rewrite");
            return std::string_view(rewrites_).substr(offset, end - offset);
        }
    }
    return {};
}

std::string SubwordTokenizer::normalize(std::string_view text) const {
    if (wordpiece_) {
        icu::UnicodeString cleaned;
        for (std::size_t i = 0; i < text.size();) {
            const auto [cp, next] = character(text, i);
            i = next;
            if (clean_ && (!cp || cp == 0xfffd || (control(cp) && cp != 9 && cp != 10 && cp != 13)))
                continue;
            if (clean_ && u_isUWhiteSpace(cp)) {
                cleaned.append(u' ');
            } else {
                const bool separate = chinese_ && chinese(cp);
                if (separate)
                    cleaned.append(u' ');
                cleaned.append(cp);
                if (separate)
                    cleaned.append(u' ');
            }
        }
        if (accents_) {
            UErrorCode error = U_ZERO_ERROR;
            const auto* nfd = icu::Normalizer2::getNFDInstance(error);
            icu::UnicodeString decomposed;
            if (U_SUCCESS(error))
                nfd->normalize(cleaned, decomposed, error);
            if (U_FAILURE(error))
                throw Error(ErrorCode::backend, "could not normalize WordPiece accents");
            cleaned.remove();
            for (std::int32_t i = 0; i < decomposed.length();) {
                const auto cp = decomposed.char32At(i);
                i += U16_LENGTH(cp);
                if (u_charType(cp) != U_NON_SPACING_MARK)
                    cleaned.append(cp);
            }
        }
        std::string result;
        if (lower_) {
            for (std::int32_t i = 0; i < cleaned.length();) {
                const auto cp = cleaned.char32At(i);
                i += U16_LENGTH(cp);
                icu::UnicodeString letter(cp);
                letter.toLower(icu::Locale::getRoot()).toUTF8String(result);
            }
        } else {
            cleaned.toUTF8String(result);
        }
        return result;
    }
    if (text.empty() || norm_trie_.empty())
        return std::string(text);
    UErrorCode error = U_ZERO_ERROR;
    std::unique_ptr<UText, decltype(&utext_close)> source(
        utext_openUTF8(nullptr, text.data(), static_cast<std::int64_t>(text.size()), &error), utext_close);
    std::unique_ptr<UBreakIterator, decltype(&ubrk_close)> graphemes(
        ubrk_open(UBRK_CHARACTER, "root", nullptr, 0, &error), ubrk_close);
    if (U_SUCCESS(error))
        ubrk_setUText(graphemes.get(), source.get(), &error);
    if (U_FAILURE(error))
        throw Error(ErrorCode::backend, "could not segment SentencePiece input");
    std::string result;
    const auto append = [&](std::string_view piece) {
        if (piece.size() > mib - result.size())
            throw Error(ErrorCode::resource_limit, "normalized input exceeds 1 MiB");
        if (collapse_spaces_) {
            for (const char byte : piece) {
                if (byte != ' ' || result.empty() || result.back() != ' ')
                    result.push_back(byte);
            }
        } else {
            result.append(piece);
        }
    };
    auto start = ubrk_first(graphemes.get());
    for (auto end = ubrk_next(graphemes.get()); end != UBRK_DONE; end = ubrk_next(graphemes.get())) {
        const auto cluster = text.substr(start, end - start);
        start = end;
        if (cluster.size() < 6) {
            if (const auto mapped = normalize_lookup(cluster)) {
                append(*mapped);
                continue;
            }
        }
        for (std::size_t i = 0; i < cluster.size();) {
            const auto next = character(cluster, i).next;
            const auto cp = cluster.substr(i, next - i);
            append(normalize_lookup(cp).value_or(cp));
            i = next;
        }
    }
    return result;
}

std::vector<std::string> SubwordTokenizer::pieces(std::string_view text) const {
    validate_text(text);
    const auto normalized = normalize(text);
    if (normalized.size() > mib)
        throw Error(ErrorCode::resource_limit, "normalized input exceeds 1 MiB");
    std::vector<std::string> parts;
    const auto add = [&](std::string_view word) {
        if (word.empty())
            return;
        if (wordpiece_) {
            parts.emplace_back(word);
        } else {
            std::string marked =
                word.starts_with(metaspace_) ? std::string(word) : metaspace_ + std::string(word);
            for (std::size_t i = 0; i < marked.size();) {
                const auto end = marked.find(metaspace_, i + metaspace_.size());
                parts.push_back(marked.substr(i, end == marked.npos ? end : end - i));
                i = end == marked.npos ? marked.size() : end;
            }
        }
    };
    if (!split_whitespace_) {
        std::string marked;
        marked.reserve(normalized.size());
        for (const char byte : normalized) {
            if (byte == ' ')
                marked += metaspace_;
            else
                marked.push_back(byte);
        }
        add(marked);
        return parts;
    }
    const auto source = std::string_view(normalized);
    std::size_t start = 0;
    for (std::size_t i = 0; i < source.size();) {
        const auto [cp, next] = character(source, i);
        if (u_isUWhiteSpace(cp) || (wordpiece_ && punctuation(cp))) {
            add(source.substr(start, i - start));
            if (!u_isUWhiteSpace(cp))
                add(source.substr(i, next - i));
            start = next;
        }
        i = next;
    }
    add(source.substr(start));
    return parts;
}

void SubwordTokenizer::wordpiece(std::string_view text, std::vector<Token>& out) const {
    std::vector<std::size_t> boundaries{0};
    while (boundaries.back() < text.size()) {
        if (boundaries.size() > max_word_) {
            out.push_back(unknown_);
            return;
        }
        boundaries.push_back(character(text, boundaries.back()).next);
    }
    const auto initial = out.size();
    for (std::size_t begin = 0; begin + 1 < boundaries.size();) {
        auto end = boundaries.size() - 1;
        for (; end > begin; --end) {
            const auto segment = text.substr(boundaries[begin], boundaries[end] - boundaries[begin]);
            auto key = begin ? prefix_ : std::string();
            key += segment;
            if (const auto found = ids_.find(key); found != ids_.end()) {
                out.push_back(found->second);
                break;
            }
        }
        if (end == begin) {
            out.resize(initial);
            out.push_back(unknown_);
            return;
        }
        begin = end;
    }
}

void SubwordTokenizer::unigram(std::string_view text, std::vector<Token>& out) const {
    struct Path {
        double score = -std::numeric_limits<double>::infinity();
        std::size_t from = 0;
        Token id = 0;
    };
    std::vector<Path> paths(text.size() + 1);
    paths[0].score = 0;
    for (std::size_t begin = 0; begin < text.size();) {
        const auto next = character(text, begin).next;
        const auto offer = [&](std::size_t end, Token id, double score) {
            const auto candidate = paths[begin].score + score;
            if (candidate > paths[end].score)
                paths[end] = {candidate, begin, id};
        };
        std::size_t node = 0;
        bool single = false;
        for (std::size_t end = begin; end < text.size(); ++end) {
            const auto edge = trie_[node].edges.find(static_cast<unsigned char>(text[end]));
            if (edge == trie_[node].edges.end())
                break;
            node = edge->second;
            if (const auto id = trie_[node].token) {
                offer(end + 1, *id, scores_[*id]);
                single |= end + 1 == next;
            }
        }
        if (!single)
            offer(next, unknown_, unknown_score_);
        begin = next;
    }
    const auto initial = out.size();
    for (auto end = text.size(); end > 0; end = paths[end].from) {
        const auto id = paths[end].id;
        if (id != unknown_ || out.size() == initial || out.back() != unknown_)
            out.push_back(id);
    }
    std::ranges::reverse(std::span(out).subspan(initial));
}

std::vector<Token> SubwordTokenizer::encode_sequence(std::string_view text) const {
    validate_text(text);
    std::vector<Token> result;
    const auto process = [&](std::string_view chunk) {
        for (const auto& part : pieces(chunk)) {
            if (wordpiece_)
                wordpiece(part, result);
            else
                unigram(part, result);
        }
    };
    std::size_t start = 0;
    for (std::size_t i = 0; i < text.size();) {
        const auto rest = text.substr(i);
        if (added_starts_[static_cast<unsigned char>(rest.front())]) {
            const auto found =
                std::ranges::find_if(added_, [&](Token id) { return rest.starts_with(vocab_[id]); });
            if (found != added_.end()) {
                process(text.substr(start, i - start));
                result.push_back(*found);
                i += vocab_[*found].size();
                start = i;
                continue;
            }
        }
        i = character(text, i).next;
    }
    process(text.substr(start));
    return result;
}

TokenizedInput SubwordTokenizer::encode(std::string_view first,
                                        std::optional<std::string_view> second) const {
    if (second && (pair_.empty() || !wordpiece_))
        unsupported("text pairs for this tokenizer");
    std::array<std::vector<Token>, 2> sequences{encode_sequence(first),
                                                second ? encode_sequence(*second) : std::vector<Token>{}};
    const auto& format = second ? pair_ : single_;
    if (max_length_) {
        const auto special = format.size() - (second ? 2 : 1);
        if (*max_length_ < special)
            malformed("truncation length smaller than tokenizer template");
        const auto limit = *max_length_ - special;
        if (sequences[0].size() + sequences[1].size() > limit) {
            if (second) {
                const auto shorter = sequences[0].size() <= sequences[1].size() ? 0u : 1u;
                const auto keep = std::min(sequences[shorter].size(), limit / 2);
                sequences[shorter].resize(keep);
                sequences[1 - shorter].resize(limit - keep);
            } else {
                sequences[0].resize(limit);
            }
        }
    }
    TokenizedInput result;
    for (const auto& item : format) {
        if (item.sequence < 0) {
            result.ids.push_back(item.token);
            result.token_type_ids.push_back(item.type);
        } else {
            const auto& tokens = sequences[item.sequence];
            result.ids.insert(result.ids.end(), tokens.begin(), tokens.end());
            result.token_type_ids.insert(result.token_type_ids.end(), tokens.size(), item.type);
        }
    }
    return result;
}

std::string_view SubwordTokenizer::piece(Token token, bool skip_special) const {
    if (token >= vocab_.size())
        throw Error(ErrorCode::invalid_argument, "token ID exceeds tokenizer vocabulary");
    return skip_special && special_[token] ? std::string_view{} : vocab_[token];
}

std::string SubwordTokenizer::decode_piece(Token token, bool first, bool skip_special) const {
    std::string decoded(piece(token, skip_special));
    if (decoded.empty())
        return decoded;
    if (!wordpiece_) {
        replace(decoded, metaspace_, first ? "" : " ");
    } else {
        if (!first) {
            if (decoded.starts_with(prefix_))
                decoded.erase(0, prefix_.size());
            else
                decoded.insert(0, 1, ' ');
        }
        if (cleanup_) {
            for (const auto word : {" .", " ?", " !", " ,"})
                replace(decoded, word, std::string_view(word).substr(1));
            replace(decoded, " ' ", "'");
            for (const auto word : {" n't", " 'm", " 's", " 've", " 're"})
                replace(decoded, word, std::string_view(word).substr(1));
            replace(decoded, " do not", " don't");
        }
    }
    return decoded;
}

std::string SubwordTokenizer::decode(std::span<const Token> tokens, bool skip_special) const {
    std::string text;
    bool first = true;
    for (const auto id : tokens) {
        auto decoded = decode_piece(id, first, skip_special);
        first = first && decoded.empty() && piece(id, skip_special).empty();
        if (decoded.size() > mib - text.size())
            throw Error(ErrorCode::resource_limit, "decoded text exceeds 1 MiB");
        text += decoded;
    }
    return text;
}

} // namespace nathr::detail
