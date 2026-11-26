#pragma once

#include "core/limits.hpp"

#include <json.hpp>
#include <nathr/nathr.hpp>

#include <initializer_list>
#include <vector>

namespace nathr::detail {

using Json = nlohmann::json;
bool is_model_directory(const std::filesystem::path& path);
std::size_t product(std::initializer_list<std::size_t> factors, std::string_view context);
std::size_t sum(std::size_t a, std::size_t b, std::string_view context);
std::vector<std::byte> read_file(const std::filesystem::path& path, std::size_t maximum);
Json parse_json(std::span<const std::byte> bytes, const std::filesystem::path& path);
Json read_json(const std::filesystem::path& path, std::size_t maximum);
std::size_t natural(const Json& value, std::string_view context);
std::uint64_t little_integer(std::span<const std::byte> bytes);

} // namespace nathr::detail
