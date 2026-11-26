#pragma once

#include "formats/safetensors.hpp"
#include "models/decoder.hpp"

namespace nathr::detail {

ModelConfig t5_config(const Json& json, const std::filesystem::path& path);

class T5 {
  public:
    struct State;

    T5(SafeTensors& file, const std::filesystem::path& path, const ModelConfig& config, const Json& json,
       CpuBackend& backend);
    [[nodiscard]] std::unique_ptr<State> start(std::span<const Token> tokens, std::size_t capacity) const;
    void step(State& state, Token token) const;

  private:
    struct Attention {
        TensorView norm, query, key, value, output;
    };
    struct FeedForward {
        TensorView norm, input, output;
    };
    struct Layer {
        Attention self, cross;
        FeedForward ff;
    };
    struct Scratch;

    void normalize(Scratch& w, TensorView gain) const;
    void feed_forward(Scratch& w, const FeedForward& weights) const;
    void project(Scratch& w, TensorView weight, std::span<const float> input, std::span<float> output) const;
    void encode(std::span<const Token> tokens, Scratch& w) const;
    void position_bias(Scratch& w, std::size_t query, std::size_t keys, bool bidirectional) const;

    const ModelConfig& config_;
    const CpuBackend& backend_;
    std::size_t buckets_, max_distance_;
    TensorView embedding_, output_, encoder_norm_, decoder_norm_;
    std::vector<float> encoder_bias_, decoder_bias_;
    std::vector<Layer> encoder_, decoder_;
};

} // namespace nathr::detail
