#pragma once

#include "formats/safetensors.hpp"
#include "models/decoder.hpp"

namespace nathr::detail {

ModelConfig t5_config(const Json& json, const std::filesystem::path& path);

class T5 {
  public:
    using Source = std::shared_ptr<const Cache>;

    struct State {
        State(const ModelConfig& config, std::size_t capacity, Source source, CachePagePool* pool)
            : self(config, capacity, KvPrecision::f32, pool), source(std::move(source)) {}

        Cache self;
        Source source;
        std::size_t length = 0;
    };

    struct Step {
        State& state;
        Token token;
        std::span<float> logits;
    };

    T5(SafeTensors& file, const std::filesystem::path& path, const ModelConfig& config, const Json& json,
       CpuBackend& backend);
    [[nodiscard]] std::vector<Source> prepare_sources(std::span<const std::span<const Token>> inputs,
                                                      CachePagePool* pool = nullptr) const;
    [[nodiscard]] std::unique_ptr<Workspace> workspace(std::size_t rows, std::size_t positions) const;
    void step(Workspace& scratch, std::span<const Step> steps) const;

  private:
    struct Attention {
        TensorView norm, query, key, value, output;
    };
    struct FeedForward {
        TensorView norm, input, output, up;
    };
    struct Layer {
        Attention self, cross;
        FeedForward ff;
    };
    struct Scratch;

    void normalize(Scratch& w, TensorView gain) const;
    void project_attention(Scratch& w, const Attention& weights) const;
    void feed_forward(Scratch& w, const FeedForward& weights) const;
    void project(Scratch& w, TensorView weight, std::span<const float> input, std::span<float> output) const;
    void encode(std::span<const std::span<const Token>> inputs, Scratch& w) const;
    void position_bias(std::span<float> bias, std::size_t query, std::size_t keys, bool bidirectional) const;

    const ModelConfig& config_;
    const CpuBackend& backend_;
    std::size_t buckets_, max_distance_;
    bool gated_;
    TensorView embedding_, output_, encoder_norm_, decoder_norm_;
    std::vector<float> encoder_bias_, decoder_bias_;
    std::vector<Layer> encoder_, decoder_;
};

} // namespace nathr::detail
