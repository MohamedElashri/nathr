#pragma once

#include "models/decoder.hpp"

namespace nathr::detail {

class Gguf;
class SafeTensors;

std::unique_ptr<Decoder> load_llama(SafeTensors& file, const std::filesystem::path& path,
                                    const ModelConfig& config, CpuBackend& backend);
std::unique_ptr<Decoder> load_gguf_decoder(Gguf& file, const std::filesystem::path& path,
                                           const ModelConfig& config, CpuBackend& backend, bool qwen3);

} // namespace nathr::detail
