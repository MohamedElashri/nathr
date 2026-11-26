#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <string_view>

namespace nathr::detail {
std::string sha256(std::span<const std::byte> bytes);
inline std::string sha256(std::string_view text) {
    return sha256(std::as_bytes(std::span(text)));
}
} // namespace nathr::detail
