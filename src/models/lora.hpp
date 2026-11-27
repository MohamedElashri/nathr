#pragma once

#include "formats/tensor.hpp"

#include <nathr/nathr.hpp>

#include <map>

namespace nathr::detail {

struct LoraTarget {
    std::string name;
    TensorView weight;
    bool embedding = false;
};

class LoraWeights {
  public:
    LoraWeights(const std::filesystem::path& directory, std::span<const LoraTarget> targets,
                std::size_t byte_limit);
    [[nodiscard]] std::size_t memory_bytes() const noexcept { return bytes_; }
    [[nodiscard]] std::uint64_t cache_key() const noexcept { return cache_key_; }

    void project(TensorView weight, std::span<const float> input, std::span<float> output,
                 std::size_t rows) const;
    void embed(TensorView weight, Token token, std::span<float> output) const;

  private:
    struct Matrix {
        std::vector<float> a, b, bias;
        std::size_t rank;
        double scale;

        void add(std::span<const double> low, std::span<float> result) const;
    };
    std::map<const std::byte*, Matrix> projections_, embeddings_;
    std::size_t bytes_ = 0;
    const std::uint64_t cache_key_;
};

} // namespace nathr::detail
