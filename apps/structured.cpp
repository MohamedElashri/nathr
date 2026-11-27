#include "structured.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <limits>

namespace nathr::cli {

class StructuredVocabulary {
  public:
    explicit StructuredVocabulary(const Tokenizer& source) : tokenizer(source) {
        if (!source.byte_level())
            throw Error(ErrorCode::unsupported, "structured output requires a byte-level tokenizer");
        pieces.reserve(source.vocabulary());
        for (Token id = 0; id < source.vocabulary(); ++id) {
            const auto bytes = tokenizer.token_bytes(id, true);
            if (!bytes.empty() && bytes.size() <= 256)
                pieces.push_back({bytes, id});
        }
        std::ranges::sort(pieces);
    }

    Tokenizer tokenizer;
    std::vector<std::pair<std::string_view, Token>> pieces;
};

namespace {
using detail::Json;

constexpr std::size_t max_properties = 32;
constexpr std::size_t max_depth = 8;
constexpr std::uint32_t bit(std::size_t index) {
    return std::uint32_t{1} << index;
}

enum class Kind { any, object, array, string, integer, number, boolean, null, choices };

struct Schema {
    Kind kind = Kind::any;
    bool open = false;
    std::uint32_t required = 0;
    std::vector<std::string> keys;
    std::vector<Schema> fields;
    std::vector<std::string> literals;
};

const Schema anything;
const Schema open_object = [] {
    Schema schema;
    schema.kind = Kind::object;
    schema.open = true;
    return schema;
}();

[[noreturn]] void invalid(std::string message) {
    throw Error(ErrorCode::invalid_argument, std::move(message));
}

Kind schema_kind(const Json& type) {
    if (type == "object")
        return Kind::object;
    if (type == "array")
        return Kind::array;
    if (type == "string")
        return Kind::string;
    if (type == "integer")
        return Kind::integer;
    if (type == "number")
        return Kind::number;
    if (type == "boolean")
        return Kind::boolean;
    if (type == "null")
        return Kind::null;
    invalid("unsupported JSON Schema type");
}

bool matches_kind(const Json& value, Kind kind) {
    switch (kind) {
    case Kind::string:
        return value.is_string();
    case Kind::integer:
        return value.is_number_integer();
    case Kind::number:
        return value.is_number();
    case Kind::boolean:
        return value.is_boolean();
    case Kind::null:
        return value.is_null();
    default:
        return kind == Kind::any;
    }
}

Schema compile(const Json& spec, std::size_t depth, std::size_t& nodes) {
    if (depth > max_depth || ++nodes > 256)
        throw Error(ErrorCode::resource_limit, "JSON Schema exceeds the nesting or node limit");
    if (!spec.is_object())
        invalid("JSON Schema must be an object");
    Schema schema;
    if (spec.contains("type"))
        schema.kind = schema_kind(spec.at("type"));

    for (const auto& [key, value] : spec.items()) {
        if (key == "type" || key == "description" || key == "title" || key == "default" || key == "enum" ||
            key == "const")
            continue;
        if (schema.kind == Kind::object &&
            (key == "properties" || key == "required" || key == "additionalProperties"))
            continue;
        if (schema.kind == Kind::array && key == "items")
            continue;
        invalid("unsupported JSON Schema keyword: " + key);
    }

    if (spec.contains("enum") || spec.contains("const")) {
        if (schema.kind == Kind::object || schema.kind == Kind::array ||
            (spec.contains("enum") && spec.contains("const")))
            invalid("enum and const support scalar values and cannot be combined");
        const auto choices = spec.contains("enum") ? spec.at("enum") : Json::array({spec.at("const")});
        if (!choices.is_array() || choices.empty() || choices.size() > max_properties)
            invalid("enum requires between one and 32 scalar values");
        for (const auto& value : choices) {
            if (value.is_structured() || !matches_kind(value, schema.kind))
                invalid("enum or const value does not match its scalar type");
            auto literal = value.dump();
            if (literal.size() > 256)
                throw Error(ErrorCode::resource_limit, "enum or const literal exceeds 256 bytes");
            schema.literals.push_back(std::move(literal));
        }
        schema.kind = Kind::choices;
    } else if (schema.kind == Kind::object) {
        const auto properties = spec.value("properties", Json::object());
        if (!properties.is_object() || properties.size() > max_properties)
            invalid("properties must be an object with at most 32 fields");
        if (spec.contains("additionalProperties") && spec.at("additionalProperties") != false)
            invalid("additionalProperties must be false when specified");
        for (const auto& [key, field] : properties.items()) {
            auto literal = Json(key).dump();
            if (literal.size() > 256)
                throw Error(ErrorCode::resource_limit, "property name exceeds 256 encoded bytes");
            schema.keys.push_back(std::move(literal));
            schema.fields.push_back(compile(field, depth + 1, nodes));
        }
        const auto required = spec.value("required", Json::array());
        if (!required.is_array() || required.size() > schema.keys.size())
            invalid("required must list declared properties");
        for (const auto& key : required) {
            const auto found = std::ranges::find(schema.keys, key.dump());
            if (!key.is_string() || found == schema.keys.end())
                invalid("required must list declared properties");
            const auto flag = bit(static_cast<std::size_t>(found - schema.keys.begin()));
            if (schema.required & flag)
                invalid("duplicate required property");
            schema.required |= flag;
        }
    } else if (schema.kind == Kind::array && spec.contains("items")) {
        schema.fields.push_back(compile(spec.at("items"), depth + 1, nodes));
    } else if (schema.kind == Kind::boolean) {
        schema.literals = {"false", "true"};
    } else if (schema.kind == Kind::null) {
        schema.literals = {"null"};
    }
    return schema;
}

const std::vector<std::string> booleans{"false", "true"};
const std::vector<std::string> nulls{"null"};

std::uint32_t bits(std::size_t count) {
    return count == max_properties ? std::numeric_limits<std::uint32_t>::max() : bit(count) - 1;
}

int hex_digit(unsigned char byte) {
    if (byte >= '0' && byte <= '9')
        return byte - '0';
    const auto lower = byte | 0x20;
    return lower >= 'a' && lower <= 'f' ? lower - 'a' + 10 : -1;
}

class Parser {
  public:
    explicit Parser(const Schema& schema) : expected_(&schema) {}

    bool complete() const { return mode_ == Mode::done; }

    bool push(unsigned char byte) {
        switch (mode_) {
        case Mode::value:
            return value(byte);
        case Mode::object_start:
        case Mode::object_key:
            if (byte == '}' && mode_ == Mode::object_start && can_close_object())
                return close();
            if (byte != '"')
                return false;
            if (frame().schema->open) {
                start_string(Mode::open_key);
                return true;
            }
            if (auto remaining = frame().remaining) {
                start_literal(Mode::fixed_key, frame().schema->keys, remaining);
                return literal(byte);
            }
            return false;
        case Mode::open_key:
        case Mode::string:
            return string(byte);
        case Mode::fixed_key:
        case Mode::literal:
            return literal(byte);
        case Mode::colon:
            if (byte != ':')
                return false;
            mode_ = Mode::value;
            return true;
        case Mode::object_next:
            if (byte == '}' && can_close_object())
                return close();
            if (byte != ',' || (!frame().schema->open && !frame().remaining))
                return false;
            mode_ = Mode::object_key;
            return true;
        case Mode::array_start:
            if (byte == ']')
                return close();
            return value(byte);
        case Mode::array_next:
            if (byte == ']')
                return close();
            if (byte != ',')
                return false;
            mode_ = Mode::value;
            expected_ = items();
            return true;
        case Mode::number:
            return number(byte);
        case Mode::done:
            return byte == ' ' || byte == '\n' || byte == '\t' || byte == '\r';
        }
        return false;
    }

  private:
    enum class Mode {
        value,
        object_start,
        object_key,
        open_key,
        fixed_key,
        colon,
        object_next,
        array_start,
        array_next,
        string,
        number,
        literal,
        done,
    };

    enum class Escape { none, backslash, hex, low_slash, low_u, low_hex };
    enum class Number { sign, zero, integer, decimal, fraction };

    struct Frame {
        const Schema* schema;
        std::uint32_t remaining;
        bool array;
    };

    Frame& frame() { return stack_[size_ - 1]; }

    const Schema* items() {
        const auto& fields = frame().schema->fields;
        return fields.empty() ? &anything : &fields.front();
    }

    bool can_close_object() { return !(frame().remaining & frame().schema->required); }

    bool close() {
        --size_;
        finish_value();
        return true;
    }

    void finish_value() {
        mode_ = !size_ ? Mode::done : frame().array ? Mode::array_next : Mode::object_next;
    }

    bool value(unsigned char byte) {
        const auto kind = expected_->kind;
        if (kind == Kind::choices || kind == Kind::boolean || kind == Kind::null) {
            start_literal(Mode::literal, expected_->literals, bits(expected_->literals.size()));
            return literal(byte);
        }
        if (byte == '{' && (kind == Kind::any || kind == Kind::object)) {
            if (size_ == stack_.size())
                return false;
            const auto* schema = kind == Kind::object ? expected_ : &open_object;
            stack_[size_++] = {schema, bits(schema->keys.size()), false};
            mode_ = Mode::object_start;
            return true;
        }
        if (byte == '[' && (kind == Kind::any || kind == Kind::array)) {
            if (size_ == stack_.size())
                return false;
            stack_[size_++] = {expected_, 0, true};
            expected_ = items();
            mode_ = Mode::array_start;
            return true;
        }
        if (byte == '"' && (kind == Kind::any || kind == Kind::string)) {
            start_string(Mode::string);
            return true;
        }
        if (kind == Kind::any && (byte == 'f' || byte == 't' || byte == 'n')) {
            const auto& choices = byte == 'n' ? nulls : booleans;
            start_literal(Mode::literal, choices, bits(choices.size()));
            return literal(byte);
        }
        if (kind != Kind::any && kind != Kind::number && kind != Kind::integer)
            return false;
        if (byte != '-' && !(byte >= '0' && byte <= '9'))
            return false;
        mode_ = Mode::number;
        number_ = byte == '-' ? Number::sign : byte == '0' ? Number::zero : Number::integer;
        integer_only_ = kind == Kind::integer;
        digits_ = 1;
        return true;
    }

    bool number(unsigned char byte) {
        if (byte >= '0' && byte <= '9') {
            if (number_ == Number::zero || digits_ >= 18)
                return false;
            if (number_ == Number::sign)
                number_ = byte == '0' ? Number::zero : Number::integer;
            else if (number_ == Number::decimal)
                number_ = Number::fraction;
            ++digits_;
            return true;
        }
        if (byte == '.' && !integer_only_ && (number_ == Number::zero || number_ == Number::integer)) {
            number_ = Number::decimal;
            digits_ = 0;
            return true;
        }
        if (number_ == Number::sign || number_ == Number::decimal)
            return false;
        finish_value();
        return push(byte);
    }

    void start_literal(Mode mode, const std::vector<std::string>& choices, std::uint32_t matches) {
        mode_ = mode;
        choices_ = &choices;
        matches_ = matches;
        offset_ = 0;
    }

    bool literal(unsigned char byte) {
        std::uint32_t next = 0;
        bool previous_valid = false, extendible = false;
        for (auto choices = matches_; choices; choices &= choices - 1) {
            const auto index = static_cast<std::size_t>(std::countr_zero(choices));
            const auto& text = (*choices_)[index];
            if (text.size() == offset_)
                previous_valid = true;
            else if (static_cast<unsigned char>(text[offset_]) == byte) {
                next |= bit(index);
                if (text.size() > offset_ + 1)
                    extendible = true;
            }
        }
        if (!next) {
            if (!previous_valid || mode_ == Mode::fixed_key)
                return false;
            finish_value();
            return push(byte);
        }
        if (extendible) {
            ++offset_;
            matches_ = next;
        } else if (mode_ == Mode::fixed_key) {
            const auto property = static_cast<std::size_t>(std::countr_zero(next));
            frame().remaining &= ~next;
            expected_ = &frame().schema->fields[property];
            mode_ = Mode::colon;
        } else {
            finish_value();
        }
        return true;
    }

    void start_string(Mode mode) {
        mode_ = mode;
        escape_ = Escape::none;
        utf8_left_ = 0;
    }

    bool escaped(unsigned char byte) {
        switch (escape_) {
        case Escape::backslash:
            if (byte == 'u') {
                code_ = digits_ = 0;
                escape_ = Escape::hex;
                return true;
            }
            if (std::string_view("\"\\/bfnrt").find(static_cast<char>(byte)) == std::string_view::npos)
                return false;
            escape_ = Escape::none;
            return true;
        case Escape::low_slash:
            if (byte != '\\')
                return false;
            escape_ = Escape::low_u;
            return true;
        case Escape::low_u:
            if (byte != 'u')
                return false;
            code_ = digits_ = 0;
            escape_ = Escape::low_hex;
            return true;
        case Escape::hex:
        case Escape::low_hex: {
            const auto digit = hex_digit(byte);
            if (digit < 0)
                return false;
            const bool low = escape_ == Escape::low_hex;
            code_ = static_cast<std::uint16_t>((code_ << 4) + digit);
            ++digits_;
            if (low && ((digits_ == 1 && digit != 0xd) || (digits_ == 2 && digit < 0xc)))
                return false;
            if (!low && digits_ == 2 && code_ >= 0xdc && code_ <= 0xdf)
                return false;
            if (digits_ == 4)
                escape_ = code_ >= 0xd800 && code_ <= 0xdbff ? Escape::low_slash : Escape::none;
            return true;
        }
        case Escape::none:
            return false;
        }
        return false;
    }

    bool string(unsigned char byte) {
        if (escape_ != Escape::none)
            return escaped(byte);
        if (utf8_left_) {
            if (byte < utf8_min_ || byte > utf8_max_)
                return false;
            --utf8_left_;
            utf8_min_ = 0x80;
            utf8_max_ = 0xbf;
            return true;
        }
        if (byte == '"') {
            if (mode_ == Mode::open_key) {
                expected_ = &anything;
                mode_ = Mode::colon;
            } else {
                finish_value();
            }
            return true;
        }
        if (byte == '\\') {
            escape_ = Escape::backslash;
            return true;
        }
        if (byte < 0x80)
            return byte >= 0x20;
        if (byte < 0xc2 || byte > 0xf4)
            return false;
        utf8_left_ = byte < 0xe0 ? 1 : byte < 0xf0 ? 2 : 3;
        utf8_min_ = byte == 0xe0 ? 0xa0 : byte == 0xf0 ? 0x90 : 0x80;
        utf8_max_ = byte == 0xed ? 0x9f : byte == 0xf4 ? 0x8f : 0xbf;
        return true;
    }

    const Schema* expected_;
    std::array<Frame, max_depth> stack_{};
    std::size_t size_ = 0;
    Mode mode_ = Mode::value;
    const std::vector<std::string>* choices_ = nullptr;
    std::uint32_t matches_ = 0;
    std::size_t offset_ = 0;
    Escape escape_ = Escape::none;
    Number number_ = Number::zero;
    bool integer_only_ = false;
    std::uint16_t code_ = 0;
    std::uint8_t digits_ = 0, utf8_left_ = 0, utf8_min_ = 0, utf8_max_ = 0;
};

using Pieces = decltype(StructuredVocabulary::pieces);

void allowed_tokens(Pieces::const_iterator first, Pieces::const_iterator last, std::size_t offset,
                    const Parser& parser, std::span<std::uint8_t> mask) {
    while (first != last) {
        const auto& [piece, id] = *first;
        if (piece.size() == offset) {
            mask[id] = 1;
            ++first;
            continue;
        }
        const auto byte = piece[offset];
        // The sorted vocabulary groups all tokens with the same byte prefix.
        const auto end =
            std::partition_point(first, last, [&](const auto& entry) { return entry.first[offset] == byte; });
        auto next = parser;
        if (next.push(static_cast<unsigned char>(byte)))
            allowed_tokens(first, end, offset + 1, next, mask);
        first = end;
    }
}

struct Constraint {
    Constraint(Schema schema, std::shared_ptr<const StructuredVocabulary> vocabulary,
               std::vector<Token> stop_ids, Token eos)
        : schema(std::move(schema)), parser(this->schema), vocabulary(std::move(vocabulary)),
          stop_ids(std::move(stop_ids)), mask(this->vocabulary->tokenizer.vocabulary()) {
        this->stop_ids.push_back(eos);
    }

    ConstrainedToken sample(Sampler& sampler, std::span<const float> logits, std::mt19937_64& random) {
        std::ranges::fill(mask, 0);
        allowed_tokens(vocabulary->pieces.begin(), vocabulary->pieces.end(), 0, parser, mask);
        for (const auto id : stop_ids) {
            if (id < mask.size())
                mask[id] = 0;
        }
        const auto token = sampler.select_masked(logits, random, mask);
        const auto bytes = vocabulary->tokenizer.token_bytes(token, true);
        if (bytes.size() > detail::mib - generated.size())
            throw Error(ErrorCode::resource_limit, "structured output exceeds 1 MiB");
        generated.append(bytes);
        for (unsigned char byte : bytes) {
            if (!parser.push(byte))
                throw Error(ErrorCode::backend, "selected token violates the structured output");
        }
        if (parser.complete() && !Json::accept(generated))
            throw Error(ErrorCode::backend, "structured output failed final JSON validation");
        return {token, parser.complete()};
    }

    Schema schema;
    Parser parser;
    std::shared_ptr<const StructuredVocabulary> vocabulary;
    std::vector<Token> stop_ids;
    std::vector<std::uint8_t> mask;
    std::string generated;
};
} // namespace

std::shared_ptr<const StructuredVocabulary> structured_vocabulary(const Tokenizer& tokenizer) {
    return std::make_shared<StructuredVocabulary>(tokenizer);
}

TokenConstraint structured_output(const Json& requested,
                                  std::shared_ptr<const StructuredVocabulary> vocabulary,
                                  std::vector<Token> stop_ids, Token eos) {
    std::size_t nodes = 0;
    auto schema = requested.is_null() ? open_object : compile(requested, 1, nodes);
    if (schema.kind != Kind::object)
        invalid("structured output requires an object schema");
    auto constraint =
        std::make_shared<Constraint>(std::move(schema), std::move(vocabulary), std::move(stop_ids), eos);
    return [constraint = std::move(constraint)](auto& sampler, auto logits, auto& random) {
        return constraint->sample(sampler, logits, random);
    };
}

} // namespace nathr::cli
