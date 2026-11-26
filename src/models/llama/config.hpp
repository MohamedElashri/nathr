#pragma once

#include <nathr/nathr.hpp>

namespace nathr::detail {

class Gguf;
ModelConfig load_llama_config(const std::filesystem::path& path);
ModelConfig load_llama_config(const Gguf& file);

} // namespace nathr::detail
