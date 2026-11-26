#pragma once

#include "models/decoder.hpp"

namespace nathr::detail {

class Gguf;
class SafeTensors;

std::unique_ptr<Decoder> load_llama(SafeTensors& file, const std::filesystem::path& path,
                                    const ModelConfig& config, CpuBackend& backend);
std::unique_ptr<Decoder> load_llama(Gguf& file, const std::filesystem::path& path, const ModelConfig& config,
                                    CpuBackend& backend);

} // namespace nathr::detail
