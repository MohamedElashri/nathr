#include <nathr/tokenizer.hpp>

#include "core/input.hpp"
#include "formats/gguf.hpp"
#include "tokenizer/subword.hpp"
#include "tokenizer/unicode_tables.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <queue>
#include <unordered_map>

namespace nathr {
namespace {

constexpr std::size_t input_limit = detail::mib;
constexpr Token missing = std::numeric_limits<Token>::max();
constexpr std::string_view replacement = "\xef\xbf\xbd";

enum class Pretokenizer { byte_level, digits_byte_level, unsupported };
enum class GgufTokenType : std::size_t { normal = 1, control = 3 };

enum class Utf8Status { valid, invalid, incomplete };
struct Codepoint {
    char32_t value;
    std::size_t bytes;
    Utf8Status status;
};

Codepoint read_codepoint(std::string_view text) {
    const auto lead = static_cast<unsigned char>(text.front());
    if (lead < 128)
        return {lead, 1, Utf8Status::valid};
    if (lead < 0xc2 || lead > 0xf4)
        return {0, 1, Utf8Status::invalid};
    const std::size_t length = lead < 0xe0 ? 2 : lead < 0xf0 ? 3 : 4;
    char32_t cp = lead & ((1u << (7 - length)) - 1);
    for (std::size_t i = 1; i < length; ++i) {
        if (i == text.size())
            return {0, i, Utf8Status::incomplete};
        const auto byte = static_cast<unsigned char>(text[i]);
        const unsigned low = i == 1 && lead == 0xe0 ? 0xa0 : i == 1 && lead == 0xf0 ? 0x90 : 0x80;
        const unsigned high = i == 1 && lead == 0xed ? 0x9f : i == 1 && lead == 0xf4 ? 0x8f : 0xbf;
        if (byte < low || byte > high)
            return {0, i, Utf8Status::invalid};
        cp = (cp << 6) | (byte & 0x3f);
    }
    return {cp, length, Utf8Status::valid};
}

struct Unit {
    char32_t cp;
    std::size_t byte;
};
std::vector<Unit> units(std::string_view text) {
    if (text.size() > input_limit)
        throw Error(ErrorCode::resource_limit, "tokenizer input exceeds 1 MiB");
    std::vector<Unit> result;
    result.reserve(text.size() + 1);
    for (std::size_t i = 0; i < text.size();) {
        const auto cp = read_codepoint(text.substr(i));
        if (cp.status != Utf8Status::valid) {
            throw Error(ErrorCode::invalid_argument, "invalid UTF-8 at input byte " + std::to_string(i));
        }
        result.push_back({cp.value, i});
        i += cp.bytes;
    }
    result.push_back({0, text.size()});
    return result;
}

constexpr std::array<char32_t, 256> byte_alphabet() {
    std::array<char32_t, 256> alphabet{};
    char32_t extra = 256;
    for (unsigned b = 0; b < 256; ++b) {
        const bool direct = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174;
        alphabet[b] = direct ? b : extra++;
    }
    return alphabet;
}
constexpr auto alphabet = byte_alphabet();

std::string as_alphabet(std::string_view raw) {
    std::string text;
    text.reserve(2 * raw.size());
    for (const unsigned char byte : raw) {
        const auto cp = alphabet[byte];
        if (cp < 128)
            text.push_back(static_cast<char>(cp));
        else {
            text.push_back(static_cast<char>(0xc0 | (cp >> 6)));
            text.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
        }
    }
    return text;
}

std::string as_bytes(std::string_view encoded, std::string_view context) {
    constexpr auto inverse = [] {
        std::array<int, 512> table{};
        table.fill(-1);
        for (unsigned i = 0; i < alphabet.size(); ++i)
            table[alphabet[i]] = static_cast<int>(i);
        return table;
    }();
    std::string bytes;
    while (!encoded.empty()) {
        const auto cp = read_codepoint(encoded);
        if (cp.status != Utf8Status::valid || cp.value >= inverse.size() || inverse[cp.value] < 0) {
            throw Error(ErrorCode::malformed, std::string(context) + ": non-ByteLevel vocabulary entry");
        }
        bytes.push_back(static_cast<char>(inverse[cp.value]));
        encoded.remove_prefix(cp.bytes);
    }
    return bytes;
}

bool in_ranges(char32_t cp, std::span<const detail::UnicodeRange> ranges) {
    const auto next = std::ranges::lower_bound(ranges, cp, {}, &detail::UnicodeRange::last);
    return next != ranges.end() && cp >= next->first;
}

enum class Category { letter, number, space, other };
Category category(char32_t cp) {
    if ((cp >= 9 && cp <= 13) || cp == 0x20 || cp == 0x85 || cp == 0xa0 || cp == 0x1680 ||
        (cp >= 0x2000 && cp <= 0x200a) || cp == 0x2028 || cp == 0x2029 || cp == 0x202f || cp == 0x205f ||
        cp == 0x3000)
        return Category::space;
    if (in_ranges(cp, detail::letters_16))
        return Category::letter;
    if (in_ranges(cp, detail::numbers_16))
        return Category::number;
    return Category::other;
}

std::string_view slice(std::string_view text, std::span<const Unit> codepoints, std::size_t begin,
                       std::size_t end) {
    return text.substr(codepoints[begin].byte, codepoints[end].byte - codepoints[begin].byte);
}

void byte_level(std::string_view text, std::span<const Unit> cp, std::size_t begin, std::size_t end,
                const auto& emit) {
    constexpr std::array<std::string_view, 7> contractions{"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
    while (begin < end) {
        auto next = begin;
        if (cp[begin].cp == U'\'') {
            const auto rest = slice(text, cp, begin, end);
            for (const auto contraction : contractions) {
                if (rest.starts_with(contraction)) {
                    next = begin + contraction.size();
                    break;
                }
            }
        }
        if (next == begin) {
            const auto first = cp[begin].cp;
            if (first == U' ' && begin + 1 < end && category(cp[begin + 1].cp) != Category::space)
                ++next;
            const auto kind = category(cp[next++].cp);
            while (next < end && category(cp[next].cp) == kind)
                ++next;
            // The first whitespace alternative backtracks one code point before non-whitespace.
            if (kind == Category::space && next < end && next - begin > 1)
                --next;
        }
        emit(slice(text, cp, begin, next));
        begin = next;
    }
}

void split(std::string_view text, std::span<const Unit> cp, std::size_t begin, std::size_t end,
           Pretokenizer pre, const auto& emit) {
    if (pre == Pretokenizer::byte_level) {
        byte_level(text, cp, begin, end, emit);
        return;
    }
    auto chunk = begin;
    for (auto i = begin; i < end; ++i) {
        if (in_ranges(cp[i].cp, detail::numbers_17)) {
            byte_level(text, cp, chunk, i, emit);
            byte_level(text, cp, i, i + 1, emit);
            chunk = i + 1;
        }
    }
    byte_level(text, cp, chunk, end, emit);
}

std::uint64_t pair_id(Token a, Token b) {
    return (std::uint64_t(a) << 32) | b;
}

detail::Json byte_level_config(bool add_prefix_space) {
    using detail::Json;
    return Json{{"type", "ByteLevel"},
                {"add_prefix_space", add_prefix_space},
                {"trim_offsets", true},
                {"use_regex", true}};
}

detail::Json pretokenizer_config(Pretokenizer pre) {
    using detail::Json;
    const auto bytes = byte_level_config(false);
    if (pre == Pretokenizer::byte_level)
        return bytes;
    return {{"type", "Sequence"},
            {"pretokenizers", Json::array({{{"type", "Digits"}, {"individual_digits", true}}, bytes})}};
}

bool has_type(const detail::Json& config, std::string_view type) {
    const auto field = config.find("type");
    return field != config.end() && *field == type;
}

bool byte_level_pretokenizer(const detail::Json& config) {
    return has_type(config, "ByteLevel") && !config.value("add_prefix_space", true) &&
           config.value("use_regex", true);
}

Pretokenizer read_pretokenizer(const detail::Json& config) {
    if (byte_level_pretokenizer(config))
        return Pretokenizer::byte_level;
    if (has_type(config, "Sequence")) {
        const auto& stages = config.at("pretokenizers");
        if (stages.is_array() && stages.size() == 2 && has_type(stages[0], "Digits") &&
            stages[0].value("individual_digits", false) && byte_level_pretokenizer(stages[1]))
            return Pretokenizer::digits_byte_level;
    }
    throw Error(ErrorCode::unsupported, "unsupported byte-level BPE pretokenizer");
}

bool empty_affix(const detail::Json& value) {
    return value.is_null() || value == "";
}

std::pair<std::string_view, std::string_view> merge_pair(const detail::Json& entry) {
    if (entry.is_string()) {
        const std::string_view pair = entry.get_ref<const std::string&>();
        const auto space = pair.find(' ');
        if (space == pair.npos)
            throw Error(ErrorCode::malformed, "BPE merge has no delimiter");
        return {pair.substr(0, space), pair.substr(space + 1)};
    }
    if (entry.is_array() && entry.size() == 2)
        return {entry[0].get_ref<const std::string&>(), entry[1].get_ref<const std::string&>()};
    throw Error(ErrorCode::malformed, "invalid BPE merge pair");
}

} // namespace

struct Tokenizer::Impl {
    struct Merge {
        Token merged;
        std::size_t rank;
    };
    struct Added {
        std::string text;
        Token id;
    };

    explicit Impl(const detail::Json& json) {
        const auto& model = json.at("model");
        if (model.at("vocab").is_array() || has_type(json.at("decoder"), "WordPiece")) {
            subword = std::make_unique<detail::SubwordTokenizer>(json);
            return;
        }
        if (json.at("version") != "1.0" || !json.at("normalizer").is_null() ||
            !json.at("truncation").is_null() || !json.at("padding").is_null() ||
            (!json.at("post_processor").is_null() && !has_type(json.at("post_processor"), "ByteLevel")) ||
            !has_type(json.at("decoder"), "ByteLevel"))
            throw Error(ErrorCode::unsupported, "unsupported tokenizer pipeline");
        pre = read_pretokenizer(json.at("pre_tokenizer"));
        if (model.value("type", std::string("BPE")) != "BPE" || !model.at("dropout").is_null() ||
            !model.at("unk_token").is_null() || !empty_affix(model.at("continuing_subword_prefix")) ||
            !empty_affix(model.at("end_of_word_suffix")) || model.value("fuse_unk", false) ||
            model.value("byte_fallback", false) || model.value("ignore_merges", false)) {
            throw Error(ErrorCode::unsupported, "unsupported BPE options");
        }
        const auto& vocab = model.at("vocab");
        if (!vocab.is_object() || vocab.empty() || vocab.size() > detail::max_vocabulary) {
            throw Error(ErrorCode::malformed, "invalid BPE vocabulary size");
        }
        raw.resize(vocab.size());
        special.resize(vocab.size());
        byte_tokens.fill(missing);
        for (const auto& [text, value] : vocab.items()) {
            const auto id = checked_id(value);
            if (text.empty() || text.size() > 1024 || !raw[id].empty())
                throw Error(ErrorCode::malformed, "empty, oversized or duplicate BPE token");
            raw[id] = text;
        }
        const auto& adds = json.at("added_tokens");
        if (!adds.is_array() || adds.size() > 64)
            throw Error(ErrorCode::malformed, "invalid added-token list");
        for (const auto& entry : adds) {
            if (!entry.at("special").get<bool>() || !entry.at("normalized").is_boolean() ||
                entry.at("single_word").get<bool>() || entry.at("lstrip").get<bool>() ||
                entry.at("rstrip").get<bool>()) {
                throw Error(ErrorCode::unsupported, "unsupported added-token flags");
            }
            auto text = entry.at("content").get<std::string>();
            const auto id = checked_id(entry.at("id"));
            if (text.empty() || text.size() > 128 || special[id] || raw[id] != text) {
                throw Error(ErrorCode::malformed, "invalid or duplicate added token");
            }
            special[id] = true;
            added_starts[static_cast<unsigned char>(text.front())] = true;
            added.push_back({std::move(text), id});
        }
        std::ranges::sort(added,
                          [](const Added& a, const Added& b) { return a.text.size() > b.text.size(); });

        for (std::size_t id = 0; id < raw.size(); ++id) {
            if (special[id])
                continue;
            raw[id] = as_bytes(raw[id], "BPE vocabulary");
            if (raw[id].size() == 1)
                byte_tokens[static_cast<unsigned char>(raw[id][0])] = static_cast<Token>(id);
        }

        const auto& pairs = model.at("merges");
        if (!pairs.is_array() || pairs.size() > vocab.size())
            throw Error(ErrorCode::malformed, "invalid BPE merge list");
        merges.reserve(pairs.size());
        for (std::size_t rank = 0; rank < pairs.size(); ++rank) {
            const auto [first, second] = merge_pair(pairs[rank]);
            const auto left = checked_id(vocab.at(first));
            const auto right = checked_id(vocab.at(second));
            const auto result = checked_id(vocab.at(std::string(first) + std::string(second)));
            if (!merges.emplace(pair_id(left, right), Merge{result, rank}).second) {
                throw Error(ErrorCode::malformed, "duplicate BPE merge");
            }
        }
    }

    Token checked_id(const detail::Json& value) const {
        const auto id = detail::natural(value, "token ID");
        if (id >= raw.size())
            throw Error(ErrorCode::malformed, "token ID exceeds BPE vocabulary");
        return static_cast<Token>(id);
    }

    void encode_piece(std::string_view input, std::vector<Token>& output) const {
        if (input.size() == 1) {
            const auto token = byte_tokens[static_cast<unsigned char>(input.front())];
            if (token != missing)
                output.push_back(token);
            return;
        }
        struct Node {
            Token id;
            std::size_t previous, next, generation{};
        };
        struct Candidate {
            std::size_t rank, left, left_generation, right_generation;
            Token merged;
            bool operator<(const Candidate& other) const {
                return rank == other.rank ? left > other.left : rank > other.rank;
            }
        };
        std::vector<Node> nodes;
        nodes.reserve(input.size());
        // This BPE subset uses no unknown token or byte fallback.
        for (const unsigned char byte : input) {
            const auto token = byte_tokens[byte];
            if (token != missing) {
                const auto index = nodes.size();
                nodes.push_back({token, index == 0 ? 0 : index - 1, index + 1});
            }
        }
        if (nodes.empty())
            return;
        const auto end = nodes.size();
        nodes.front().previous = end;
        std::priority_queue<Candidate> pending;
        const auto schedule = [&](std::size_t pos) {
            if (pos == end || nodes[pos].next == end)
                return;
            const auto& left = nodes[pos];
            const auto& right = nodes[left.next];
            const auto found = merges.find(pair_id(left.id, right.id));
            if (found != merges.end())
                pending.push(
                    {found->second.rank, pos, left.generation, right.generation, found->second.merged});
        };
        for (std::size_t i = 0; i + 1 < end; ++i)
            schedule(i);
        while (!pending.empty()) {
            const auto top = pending.top();
            pending.pop();
            auto& left = nodes[top.left];
            if (left.next == end || left.generation != top.left_generation)
                continue;
            auto& right = nodes[left.next];
            if (right.generation != top.right_generation)
                continue;
            left.id = top.merged;
            left.next = right.next;
            ++left.generation;
            ++right.generation;
            right.next = end;
            if (left.next != end)
                nodes[left.next].previous = top.left;
            schedule(left.previous);
            schedule(top.left);
        }
        for (std::size_t i = 0; i < end; i = nodes[i].next)
            output.push_back(nodes[i].id);
    }

    std::unique_ptr<detail::SubwordTokenizer> subword;
    std::vector<std::string> raw;
    std::vector<bool> special;
    std::array<Token, 256> byte_tokens;
    std::array<bool, 256> added_starts{};
    Pretokenizer pre = Pretokenizer::unsupported;
    std::unordered_map<std::uint64_t, Merge> merges;
    std::vector<Added> added;
};

Tokenizer Tokenizer::load(const std::filesystem::path& path) {
    try {
        return Tokenizer(std::make_shared<Impl>(detail::read_json(path, 8 * detail::mib)));
    } catch (const detail::Json::exception& error) {
        throw Error(ErrorCode::malformed, path.string() + ": " + error.what());
    } catch (const Error& error) {
        throw Error(error.code(), path.string() + ": " + error.what());
    }
}

Tokenizer Tokenizer::load_model(const std::filesystem::path& model, const std::filesystem::path& sidecar) {
    using detail::Json;
    if (detail::is_model_directory(model)) {
        if (!sidecar.empty())
            throw Error(ErrorCode::invalid_argument,
                        "a separate tokenizer sidecar is only supported for GGUF");
        return load(model / "tokenizer.json");
    }
    const detail::Gguf container(model);
    try {
        const auto& metadata = container.tokenizer();
        if (metadata.value("tokenizer.ggml.model", std::string{}) != "gpt2")
            throw Error(ErrorCode::unsupported, "expected a GGUF byte-level BPE tokenizer");
        const auto& tokens = metadata.at("tokenizer.ggml.tokens");
        const auto& merges = metadata.at("tokenizer.ggml.merges");
        if (!tokens.is_array() || tokens.empty() || tokens.size() > detail::max_vocabulary ||
            !merges.is_array() || merges.size() > tokens.size())
            throw Error(ErrorCode::malformed, "invalid GGUF BPE vocabulary or merge list");
        const auto types = metadata.find("tokenizer.ggml.token_type");
        if (types != metadata.end() && (!types->is_array() || types->size() != tokens.size()))
            throw Error(ErrorCode::malformed, "GGUF token types must match vocabulary length");

        const auto name = metadata.value("tokenizer.ggml.pre", std::string{});
        auto pre = Pretokenizer::unsupported;
        if (name == "smollm")
            pre = Pretokenizer::digits_byte_level;
        else if (name == "gpt-2")
            pre = Pretokenizer::byte_level;
        if (metadata.value("tokenizer.ggml.add_space_prefix", false))
            throw Error(ErrorCode::unsupported, "GGUF tokenizer prefix spaces are unsupported");

        Json vocabulary = Json::object();
        std::vector<bool> special(tokens.size());
        for (std::size_t id = 0; id < tokens.size(); ++id) {
            const auto& text = tokens[id].get_ref<const std::string&>();
            if (vocabulary.contains(text))
                throw Error(ErrorCode::malformed, "duplicate GGUF vocabulary token");
            vocabulary[text] = id;
            if (types != metadata.end()) {
                const auto type =
                    static_cast<GgufTokenType>(detail::natural((*types)[id], "GGUF token type"));
                if (type != GgufTokenType::normal && type != GgufTokenType::control)
                    throw Error(ErrorCode::unsupported, "expected normal or control GGUF BPE tokens");
                special[id] = type == GgufTokenType::control;
            }
        }

        if (!sidecar.empty()) {
            const auto json = detail::read_json(sidecar, 8 * detail::mib);
            const auto& bpe = json.at("model");
            if (bpe.at("vocab") != vocabulary)
                throw Error(ErrorCode::malformed, "GGUF vocabulary differs from the tokenizer sidecar");
            const auto& sidecar_merges = bpe.at("merges");
            const auto same_pair = [](const Json& a, const Json& b) {
                return merge_pair(a) == merge_pair(b);
            };
            if (!sidecar_merges.is_array() || !std::ranges::equal(merges, sidecar_merges, same_pair))
                throw Error(ErrorCode::malformed, "GGUF merges differ from the tokenizer sidecar");
            auto impl = std::make_shared<Impl>(json);
            if (impl->subword)
                throw Error(ErrorCode::unsupported, "GGUF requires a byte-level BPE sidecar");
            if (types != metadata.end() && impl->special != special)
                throw Error(ErrorCode::malformed, "GGUF token types differ from the tokenizer sidecar");
            if (pre != Pretokenizer::unsupported && impl->pre != pre)
                throw Error(ErrorCode::malformed, "GGUF pretokenizer differs from the tokenizer sidecar");
            return Tokenizer(std::move(impl));
        }

        if (pre == Pretokenizer::unsupported || types == metadata.end())
            throw Error(ErrorCode::unsupported,
                        "unsupported or incomplete GGUF pretokenizer; supply a matching tokenizer.json");
        auto added = Json::array();
        for (std::size_t id = 0; id < tokens.size(); ++id) {
            if (special[id]) {
                added.push_back({{"content", tokens[id]},
                                 {"id", id},
                                 {"special", true},
                                 {"normalized", false},
                                 {"single_word", false},
                                 {"lstrip", false},
                                 {"rstrip", false}});
            }
        }
        Json json = {{"version", "1.0"},
                     {"normalizer", nullptr},
                     {"post_processor", nullptr},
                     {"truncation", nullptr},
                     {"padding", nullptr},
                     {"pre_tokenizer", pretokenizer_config(pre)},
                     {"decoder", byte_level_config(true)}};
        json["model"] = {{"type", "BPE"},
                         {"vocab", std::move(vocabulary)},
                         {"merges", merges},
                         {"dropout", nullptr},
                         {"unk_token", nullptr},
                         {"continuing_subword_prefix", nullptr},
                         {"end_of_word_suffix", nullptr}};
        json["added_tokens"] = std::move(added);
        return Tokenizer(std::make_shared<Impl>(json));
    } catch (const Json::exception& e) {
        throw Error(ErrorCode::malformed, model.string() + ": " + e.what());
    } catch (const Error& e) {
        throw Error(e.code(), model.string() + ": " + e.what());
    }
}

std::size_t Tokenizer::vocabulary() const noexcept {
    return impl_->subword ? impl_->subword->vocabulary() : impl_->raw.size();
}

std::vector<Token> Tokenizer::encode(std::string_view text) const {
    if (impl_->subword)
        return impl_->subword->encode(text, {}).ids;
    const auto cp = units(text);
    const auto count = cp.size() - 1;
    std::vector<Token> result;
    const auto emit = [&](std::string_view piece) { impl_->encode_piece(piece, result); };
    std::size_t begin = 0;
    for (std::size_t i = 0; i < count;) {
        const auto rest = text.substr(cp[i].byte);
        const auto match =
            impl_->added_starts[static_cast<unsigned char>(rest.front())]
                ? std::ranges::find_if(impl_->added,
                                       [&](const auto& token) { return rest.starts_with(token.text); })
                : impl_->added.end();
        if (match == impl_->added.end()) {
            ++i;
            continue;
        }
        split(text, cp, begin, i, impl_->pre, emit);
        result.push_back(match->id);
        const auto finish = cp[i].byte + match->text.size();
        while (cp[i].byte < finish)
            ++i;
        begin = i;
    }
    split(text, cp, begin, count, impl_->pre, emit);
    return result;
}

TokenizedInput Tokenizer::encode_pair(std::string_view text, std::string_view second) const {
    if (!impl_->subword)
        throw Error(ErrorCode::unsupported, "this tokenizer has no text-pair template");
    return impl_->subword->encode(text, second);
}

std::vector<std::string> Tokenizer::pretokenize(std::string_view text) const {
    if (impl_->subword)
        return impl_->subword->pieces(text);
    const auto cp = units(text);
    std::vector<std::string> result;
    split(text, cp, 0, cp.size() - 1, impl_->pre,
          [&](std::string_view part) { result.push_back(as_alphabet(part)); });
    return result;
}

std::string_view Tokenizer::token_bytes(Token token, bool skip_special) const {
    if (impl_->subword)
        return impl_->subword->piece(token, skip_special);
    if (token >= vocabulary())
        throw Error(ErrorCode::invalid_argument, "token ID exceeds tokenizer vocabulary");
    return skip_special && impl_->special[token] ? std::string_view{} : impl_->raw[token];
}

std::string Tokenizer::decode_piece(Token token, bool first, bool skip_special) const {
    if (impl_->subword)
        return impl_->subword->decode_piece(token, first, skip_special);
    return std::string(token_bytes(token, skip_special));
}

std::string Tokenizer::decode_bytes(std::span<const Token> tokens, bool skip_special) const {
    if (impl_->subword)
        return impl_->subword->decode(tokens, skip_special);
    std::string bytes;
    for (const auto token : tokens) {
        auto piece = token_bytes(token, skip_special);
        if (piece.size() > input_limit - bytes.size())
            throw Error(ErrorCode::resource_limit, "decoded text exceeds 1 MiB");
        bytes.append(piece);
    }
    return bytes;
}

std::string Tokenizer::decode(std::span<const Token> tokens, bool skip_special) const {
    Utf8Stream display;
    auto text = display.push(decode_bytes(tokens, skip_special));
    text += display.finish();
    return text;
}

std::string Utf8Stream::push(std::string_view bytes) {
    pending_.append(bytes);
    std::string text;
    auto remaining = std::string_view(pending_);
    while (!remaining.empty()) {
        const auto cp = read_codepoint(remaining);
        if (cp.status == Utf8Status::incomplete)
            break;
        text += cp.status == Utf8Status::valid ? remaining.substr(0, cp.bytes) : replacement;
        remaining.remove_prefix(cp.bytes);
    }
    pending_.erase(0, pending_.size() - remaining.size());
    return text;
}

std::string Utf8Stream::finish() {
    if (pending_.empty())
        return {};
    pending_.clear();
    return std::string(replacement);
}

} // namespace nathr
