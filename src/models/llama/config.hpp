#pragma once

#include "core/input.hpp"

namespace nathr::detail {

class Gguf;
ModelConfig load_llama_config(const Json& json, const std::filesystem::path& path);
ModelConfig load_llama_config(const Gguf& file);

} // namespace nathr::detail
