#pragma once

#include "formats/safetensors.hpp"
#include "models/decoder.hpp"

namespace nathr::detail {

ModelConfig bert_config(const Json& json, const std::filesystem::path& path);

class Bert {
  public:
    Bert(SafeTensors& file, const std::filesystem::path& path, const ModelConfig& config, const Json& json,
         CpuBackend& backend);
    [[nodiscard]] std::vector<float> encode(std::span<const Token> tokens) const;

  private:
    struct Affine {
        TensorView weight, bias;
    };
    struct Layer {
        Affine query, key, value, attention, attention_norm, ff_in, ff_out, ff_norm;
    };

    const ModelConfig& config_;
    const CpuBackend& backend_;
    TensorView tokens_, positions_, segments_;
    Affine embedding_norm_;
    std::vector<Layer> layers_;
};

} // namespace nathr::detail
