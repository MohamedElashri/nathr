#include "core/input.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <unordered_set>

namespace nathr::detail {

bool is_model_directory(const std::filesystem::path& path) {
    std::error_code error;
    const bool directory = std::filesystem::is_directory(path, error);
    if (error)
        throw Error(ErrorCode::io, path.string() + ": " + error.message());
    return directory;
}

std::size_t product(std::initializer_list<std::size_t> factors, std::string_view context) {
    std::size_t result = 1;
    for (const auto factor : factors) {
        if (factor && result > std::numeric_limits<std::size_t>::max() / factor) {
            throw Error(ErrorCode::resource_limit, std::string(context) + ": size overflow");
        }
        result *= factor;
    }
    return result;
}

std::size_t sum(std::size_t a, std::size_t b, std::string_view context) {
    if (a > std::numeric_limits<std::size_t>::max() - b) {
        throw Error(ErrorCode::resource_limit, std::string(context) + ": size overflow");
    }
    return a + b;
}

std::vector<std::byte> read_file(const std::filesystem::path& path, std::size_t maximum) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    const auto size = stream.tellg();
    if (size < 0) {
        throw Error(ErrorCode::io, path.string() + ": cannot open file");
    }
    if (static_cast<std::uint64_t>(size) > maximum) {
        throw Error(ErrorCode::resource_limit, path.string() + ": file exceeds size limit");
    }
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    stream.seekg(0);
    if (!stream.read(reinterpret_cast<char*>(bytes.data()), size) ||
        stream.peek() != std::char_traits<char>::eof() || stream.bad()) {
        throw Error(ErrorCode::io, path.string() + ": file changed or could not be read");
    }
    return bytes;
}

Json parse_json(std::span<const std::byte> bytes, const std::filesystem::path& path) {
    using Event = Json::parse_event_t;
    std::vector<std::unordered_set<std::string>> objects;
    const auto callback = [&](int depth, Event event, Json& value) {
        if (depth > 32) {
            throw Error(ErrorCode::resource_limit, path.string() + ": JSON nesting exceeds limit");
        }
        if (event == Event::object_start) {
            objects.emplace_back();
        } else if (event == Event::object_end) {
            objects.pop_back();
        } else if (event == Event::key) {
            const auto& key = value.get_ref<const std::string&>();
            if (!objects.back().insert(key).second)
                throw Error(ErrorCode::malformed, path.string() + ": duplicate JSON key " + key);
        }
        return true;
    };
    if (bytes.empty() || std::ranges::find(bytes, std::byte{0}) != bytes.end()) {
        throw Error(ErrorCode::malformed, path.string() + ": empty JSON or embedded NUL");
    }
    const auto* text = reinterpret_cast<const char*>(bytes.data());
    try {
        return Json::parse(text, text + bytes.size(), callback);
    } catch (const Json::exception& error) {
        throw Error(ErrorCode::malformed, path.string() + ": " + error.what());
    }
}

Json read_json(const std::filesystem::path& path, std::size_t maximum) {
    return parse_json(read_file(path, maximum), path);
}

std::size_t natural(const Json& value, std::string_view context) {
    if (!value.is_number_unsigned() && !(value.is_number_integer() && value.get<std::int64_t>() >= 0)) {
        throw Error(ErrorCode::malformed, std::string(context) + ": expected a non-negative integer");
    }
    return value.get<std::size_t>();
}

std::uint64_t little_integer(std::span<const std::byte> bytes) {
    if (bytes.size() > sizeof(std::uint64_t)) {
        throw Error(ErrorCode::malformed, "integer exceeds 64 bits");
    }
    std::uint64_t result = 0;
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        result |= std::uint64_t{std::to_integer<unsigned char>(bytes[i])} << (8 * i);
    }
    return result;
}

} // namespace nathr::detail
