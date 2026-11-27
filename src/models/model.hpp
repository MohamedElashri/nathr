#pragma once

#include "formats/gguf.hpp"
#include "formats/safetensors.hpp"
#include "models/bert/model.hpp"
#include "models/decoder.hpp"
#include "models/t5/model.hpp"

#include <variant>

namespace nathr::detail {

enum class Checkpoint { llama_safetensors, decoder_gguf, gpt2_safetensors, bert_safetensors, t5_safetensors };
void validate_tokens(std::span<const Token> tokens, const ModelConfig& config, std::size_t limit);
void validate_mode(LogitsMode mode, const LogitsSink& sink);

} // namespace nathr::detail

struct nathr::Model::Impl {
    using WeightFile = std::variant<detail::SafeTensors, detail::Gguf>;
    Impl(const std::filesystem::path& path, detail::Checkpoint checkpoint, ModelConfig config,
         BackendOptions options, CpuFeatures features, const detail::Json& json = {});
    [[nodiscard]] const detail::Decoder& decoder() const;

    ModelConfig config;
    WeightFile tensors;
    detail::CpuBackend backend;
    std::variant<std::unique_ptr<detail::Decoder>, std::unique_ptr<detail::Bert>, std::unique_ptr<detail::T5>>
        execution;
    std::string architecture;
    std::string chat_template;
};
