#pragma once

#include "core/input.hpp"

namespace nathr::detail {

class Gguf;
inline constexpr std::size_t max_moe_experts = 256;

struct MoeConfig {
    std::size_t experts = 0, active = 0;
    float weight_scale = 1.f;
};

struct RotaryScales {
    float embedding = 1.f;
    float residual = 1.f;
    float attention = 0.f;
    float logits = 1.f;
};

ModelConfig load_llama_config(const Json& json, const std::filesystem::path& path);
ModelConfig load_gguf_decoder_config(const Gguf& file);
MoeConfig load_gguf_moe_config(const Gguf& file);
RotaryScales load_gguf_granite_scales(const Gguf& file);

} // namespace nathr::detail
