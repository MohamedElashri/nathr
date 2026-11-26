#pragma once

#include "backend/cpu/scalar.hpp"
#include "backend/cpu/thread_pool.hpp"
#include "formats/gguf.hpp"
#include "formats/safetensors.hpp"

#include <variant>

namespace nathr::detail {

struct LayerWeights {
    TensorView input_norm, q, k, v, o, post_attention_norm, gate, up, down;
};

struct Scratch {
    Scratch(const ModelConfig& config, std::size_t positions, std::size_t block);
    [[nodiscard]] std::size_t bytes() const noexcept;
    std::vector<float> storage;
    std::span<float> hidden, query, keys, values, norm, attention, projection, gate, up, packed, scores,
        logits;
};

void validate_tokens(std::span<const Token> tokens, const ModelConfig& config, std::size_t limit);
void validate_mode(LogitsMode mode, const LogitsSink& sink);

} // namespace nathr::detail

struct nathr::Model::Impl {
    Impl(const std::filesystem::path& path, bool hf_directory, BackendOptions options);
    using WeightFile = std::variant<detail::SafeTensors, detail::Gguf>;
    static WeightFile open(const std::filesystem::path& path, bool hf_directory);
    template <class File> void bind(File& file, const std::filesystem::path& path);
    void execute(std::span<const Token> tokens, std::size_t start, detail::Cache* cache,
                 detail::Scratch& scratch, LogitsMode mode, const LogitsSink& sink,
                 const TraceSink& trace) const;

    ModelConfig config;
    WeightFile tensors;
    std::array<detail::KernelChoice, 31> kernels{};
    std::string_view active_backend;
    detail::TensorView embedding, output, norm;
    std::vector<detail::LayerWeights> layers;
    std::vector<float> frequency;
    const unsigned thread_count;
    detail::ThreadPool pool;
};
