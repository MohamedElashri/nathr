#pragma once

#include "formats/safetensors.hpp"
#include "models/decoder.hpp"

#include <optional>

namespace nathr::detail {

ModelConfig bert_config(const Json& json, const std::filesystem::path& path);

class Bert {
  public:
    Bert(SafeTensors& file, const std::filesystem::path& path, const ModelConfig& config, const Json& json,
         CpuBackend& backend);
    [[nodiscard]] std::vector<float> encode(std::span<const Token> tokens,
                                            std::span<const Token> token_types = {}) const;
    [[nodiscard]] std::vector<OutputInfo> outputs() const;
    [[nodiscard]] TensorResult infer(InferenceInput input, const OutputInfo& output) const;

  private:
    struct Affine {
        TensorView weight, bias;
    };
    struct Layer {
        Affine query, key, value, attention, attention_norm, ff_in, ff_out, ff_norm;
    };
    enum class Pooling { none, mean, cls, maximum };

    const ModelConfig& config_;
    const CpuBackend& backend_;
    TensorView tokens_, positions_, segments_;
    Affine embedding_norm_;
    std::vector<Layer> layers_;
    std::optional<Affine> pooler_, classifier_;
    std::size_t segments_count_ = 0, label_count_ = 0, embedding_context_ = 0;
    Pooling pooling_ = Pooling::none;
    bool normalize_embedding_ = false;
};

} // namespace nathr::detail
