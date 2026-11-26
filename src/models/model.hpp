#pragma once

#include "formats/gguf.hpp"
#include "formats/safetensors.hpp"
#include "models/decoder.hpp"

#include <variant>

namespace nathr::detail {

enum class Checkpoint { llama_safetensors, llama_gguf, gpt2_safetensors };
void validate_tokens(std::span<const Token> tokens, const ModelConfig& config, std::size_t limit);
void validate_mode(LogitsMode mode, const LogitsSink& sink);

} // namespace nathr::detail

struct nathr::Model::Impl {
    using WeightFile = std::variant<detail::SafeTensors, detail::Gguf>;
    Impl(const std::filesystem::path& path, detail::Checkpoint checkpoint, ModelConfig config,
         BackendOptions options, CpuFeatures features);

    ModelConfig config;
    WeightFile tensors;
    detail::CpuBackend backend;
    std::unique_ptr<detail::Decoder> decoder;
    const std::string_view architecture;
};
