#pragma once

#include "models/decoder.hpp"

namespace nathr::detail {

class SafeTensors;
ModelConfig gpt2_config(const Json& json, const std::filesystem::path& path);
std::unique_ptr<Decoder> load_gpt2(SafeTensors& file, const std::filesystem::path& path,
                                   const ModelConfig& config, CpuBackend& backend);

} // namespace nathr::detail
