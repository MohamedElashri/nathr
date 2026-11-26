#include <nathr/tokenizer.hpp>

#include "core/input.hpp"
#include "core/sha256.hpp"
#include "formats/gguf.hpp"
#include "tokenizer/unicode_tables.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <queue>
#include <unordered_map>

namespace nathr {
namespace {

constexpr std::size_t input_limit = detail::mib;
constexpr std::size_t vocab_limit = 65536;
constexpr Token missing = std::numeric_limits<Token>::max();
constexpr std::string_view replacement = "\xef\xbf\xbd";

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
           const auto& emit) {
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

detail::Json expected_pipeline() {
    using detail::Json;
    const Json pre = {{"type", "Sequence"},
                      {"pretokenizers", Json::array({{{"type", "Digits"}, {"individual_digits", true}},
                                                     {{"type", "ByteLevel"},
                                                      {"add_prefix_space", false},
                                                      {"trim_offsets", true},
                                                      {"use_regex", true}}})}};
    const Json decoder = {
        {"type", "ByteLevel"}, {"add_prefix_space", true}, {"trim_offsets", true}, {"use_regex", true}};
    return {{"version", "1.0"},      {"normalizer", nullptr}, {"post_processor", nullptr},
            {"truncation", nullptr}, {"padding", nullptr},    {"pre_tokenizer", pre},
            {"decoder", decoder}};
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
        using detail::Json;
        const auto pipeline = expected_pipeline();
        for (const auto& [key, value] : pipeline.items()) {
            if (json.at(key) != value)
                throw Error(ErrorCode::unsupported, "unsupported tokenizer pipeline");
        }
        const auto& model = json.at("model");
        if (model.at("type") != "BPE" || !model.at("dropout").is_null() || !model.at("unk_token").is_null() ||
            !model.at("continuing_subword_prefix").is_null() || !model.at("end_of_word_suffix").is_null() ||
            model.value("fuse_unk", false) || model.value("byte_fallback", false) ||
            model.value("ignore_merges", false)) {
            throw Error(ErrorCode::unsupported, "unsupported BPE options");
        }
        const auto& vocab = model.at("vocab");
        if (!vocab.is_object() || vocab.size() < 256 || vocab.size() > vocab_limit) {
            throw Error(ErrorCode::malformed, "invalid BPE vocabulary size");
        }
        raw.resize(vocab.size());
        special.resize(vocab.size());
        byte_tokens.fill(missing);
        for (const auto& [text, value] : vocab.items()) {
            const auto id = checked_id(value);
            if (text.empty() || text.size() > 1024 || !raw[id].empty())
                throw Error(ErrorCode::malformed, "empty, oversized or duplicate BPE token");
            raw[id] = as_bytes(text, "BPE vocabulary");
            if (raw[id].size() == 1)
                byte_tokens[static_cast<unsigned char>(raw[id][0])] = id;
        }
        const auto& adds = json.at("added_tokens");
        if (!adds.is_array() || adds.size() > 64)
            throw Error(ErrorCode::malformed, "invalid added-token list");
        for (const auto& entry : adds) {
            if (!entry.at("special").get<bool>() || entry.at("normalized").get<bool>() ||
                entry.at("single_word").get<bool>() || entry.at("lstrip").get<bool>() ||
                entry.at("rstrip").get<bool>()) {
                throw Error(ErrorCode::unsupported, "unsupported added-token flags");
            }
            auto text = entry.at("content").get<std::string>();
            const auto id = checked_id(entry.at("id"));
            if (text.size() < 2 || text.size() > 128 || !text.starts_with('<') || !text.ends_with('>') ||
                special[id] || !vocab.contains(text) || checked_id(vocab.at(text)) != id) {
                throw Error(ErrorCode::malformed, "invalid or duplicate added token");
            }
            special[id] = true;
            added.push_back({std::move(text), id});
        }
        std::ranges::sort(added,
                          [](const Added& a, const Added& b) { return a.text.size() > b.text.size(); });

        const auto& pairs = model.at("merges");
        if (!pairs.is_array() || pairs.size() > vocab.size())
            throw Error(ErrorCode::malformed, "invalid BPE merge list");
        merges.reserve(pairs.size());
        for (std::size_t rank = 0; rank < pairs.size(); ++rank) {
            const auto& entry = pairs[rank];
            std::string first, second;
            if (entry.is_string()) {
                const auto& pair = entry.get_ref<const std::string&>();
                const auto space = pair.find(' ');
                if (space == pair.npos)
                    throw Error(ErrorCode::malformed, "BPE merge has no delimiter");
                first = pair.substr(0, space);
                second = pair.substr(space + 1);
            } else if (entry.is_array() && entry.size() == 2) {
                first = entry[0].get<std::string>();
                second = entry[1].get<std::string>();
            } else {
                throw Error(ErrorCode::malformed, "invalid BPE merge pair");
            }
            const auto left = checked_id(vocab.at(first));
            const auto right = checked_id(vocab.at(second));
            const auto result = checked_id(vocab.at(first + second));
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
        // The pinned BPE has neither an unknown token nor byte fallback.
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

    std::vector<std::string> raw;
    std::vector<bool> special;
    std::array<Token, 256> byte_tokens;
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
    std::error_code error;
    if (std::filesystem::is_directory(model, error)) {
        if (!sidecar.empty())
            throw Error(ErrorCode::invalid_argument,
                        "a separate tokenizer sidecar is only supported for GGUF");
        return load(model / "tokenizer.json");
    }
    if (error)
        throw Error(ErrorCode::io, model.string() + ": " + error.message());
    const detail::Gguf container(model);
    try {
        const auto& metadata = container.tokenizer();
        const auto& tokens = metadata.at("tokenizer.ggml.tokens");
        const auto& merges = metadata.at("tokenizer.ggml.merges");
        if (!tokens.is_array() || tokens.size() != 49152 || !merges.is_array() || merges.size() != 48900 ||
            metadata.at("tokenizer.ggml.bos_token_id") != 0 ||
            metadata.at("tokenizer.ggml.eos_token_id") != 0)
            throw Error(ErrorCode::unsupported,
                        "GGUF is missing the canonical SmolLM2 vocabulary or boundary IDs");

        if (!sidecar.empty()) {
            auto bytes = detail::read_file(sidecar, 8 * detail::mib);
            if (detail::sha256(bytes) != "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c")
                throw Error(ErrorCode::unsupported,
                            sidecar.string() + ": expected the pinned SmolLM2 tokenizer.json");
            auto json = detail::parse_json(bytes, sidecar);
            const auto& bpe = json.at("model");
            const auto& vocab = bpe.at("vocab");
            for (std::size_t id = 0; id < tokens.size(); ++id) {
                const auto& text = tokens[id].get_ref<const std::string&>();
                if (!vocab.contains(text) || vocab.at(text) != id)
                    throw Error(ErrorCode::malformed, "GGUF vocabulary differs from the tokenizer sidecar");
            }
            if (bpe.at("merges") != merges)
                throw Error(ErrorCode::malformed, "GGUF merges differ from the tokenizer sidecar");
            return Tokenizer(std::make_shared<Impl>(json));
        }

        const auto fingerprint = container.tokenizer_fingerprint();
        if (fingerprint != "46646ba36ecae43de6f9f649d217774b889e0fd405af92205319b882927493fc" &&
            fingerprint != "8183bac60a960542a17ba286ecdd3063046e57c91c3bbf4b1b4b3e797d193a41")
            throw Error(ErrorCode::unsupported,
                        "unknown GGUF tokenizer fingerprint; supply a pinned tokenizer sidecar");
        auto json = expected_pipeline();
        Json vocabulary = Json::object(), added = Json::array();
        for (std::size_t id = 0; id < tokens.size(); ++id) {
            const auto& text = tokens[id].get_ref<const std::string&>();
            vocabulary[text] = id;
            if (id < 17) {
                added.push_back({{"content", text},
                                 {"id", id},
                                 {"special", true},
                                 {"normalized", false},
                                 {"single_word", false},
                                 {"lstrip", false},
                                 {"rstrip", false}});
            }
        }
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
    return impl_->raw.size();
}

std::vector<Token> Tokenizer::encode(std::string_view text) const {
    const auto cp = units(text);
    const auto count = cp.size() - 1;
    std::vector<Token> result;
    const auto emit = [&](std::string_view piece) { impl_->encode_piece(piece, result); };
    std::size_t begin = 0;
    for (std::size_t i = 0; i < count;) {
        const auto rest = text.substr(cp[i].byte);
        const auto match =
            cp[i].cp == U'<'
                ? std::ranges::find_if(impl_->added,
                                       [&](const auto& token) { return rest.starts_with(token.text); })
                : impl_->added.end();
        if (match == impl_->added.end()) {
            ++i;
            continue;
        }
        split(text, cp, begin, i, emit);
        result.push_back(match->id);
        const auto finish = cp[i].byte + match->text.size();
        while (cp[i].byte < finish)
            ++i;
        begin = i;
    }
    split(text, cp, begin, count, emit);
    return result;
}

std::vector<std::string> Tokenizer::pretokenize(std::string_view text) const {
    const auto cp = units(text);
    std::vector<std::string> result;
    split(text, cp, 0, cp.size() - 1, [&](std::string_view part) { result.push_back(as_alphabet(part)); });
    return result;
}

std::string_view Tokenizer::token_bytes(Token token, bool skip_special) const {
    if (token >= vocabulary())
        throw Error(ErrorCode::invalid_argument, "token ID exceeds tokenizer vocabulary");
    return skip_special && impl_->special[token] ? std::string_view{} : impl_->raw[token];
}

std::string Tokenizer::decode_bytes(std::span<const Token> tokens, bool skip_special) const {
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
