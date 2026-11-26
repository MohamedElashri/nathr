#pragma once

#include "backend/cpu/scalar.hpp"
#include "backend/cpu/thread_pool.hpp"

#include <initializer_list>
#include <utility>

namespace nathr::detail {

class CpuBackend {
  public:
    CpuBackend(BackendOptions options, CpuFeatures features);
    TensorView bind(TensorView view, const std::filesystem::path& path, std::string_view name);
    void project(TensorView weight, std::span<const float> input, std::span<float> output, std::size_t tokens,
                 std::span<float> packed) const;
    [[nodiscard]] std::string_view kernel(std::uint32_t id) const noexcept;
    [[nodiscard]] std::string_view name() const noexcept;
    [[nodiscard]] unsigned threads() const noexcept { return options_.threads; }

  private:
    BackendOptions options_;
    CpuFeatures features_;
    ThreadPool pool_;
    std::array<KernelChoice, static_cast<std::size_t>(DType::id_limit)> kernels_{};
};

struct Workspace {
    virtual ~Workspace() = default;
    [[nodiscard]] std::size_t bytes() const noexcept;
    std::span<float> packed, scores, logits;

  protected:
    Workspace() = default;
    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;
    void allocate(std::initializer_list<std::pair<std::span<float>*, std::size_t>> buffers);

  private:
    std::vector<float> storage_;
};

struct Evaluation {
    std::span<const Token> tokens;
    std::size_t start;
    Cache* cache;
    LogitsMode mode;
    const LogitsSink& sink;
    const TraceSink& trace;

    [[nodiscard]] bool wants_logits(std::size_t index) const;
    void record(std::string_view name, std::size_t layer, std::size_t index,
                std::span<const float> values) const;
    void publish_logits(std::size_t index, std::span<const float> values) const;
};

class Decoder {
  public:
    virtual ~Decoder() = default;
    [[nodiscard]] virtual std::unique_ptr<Workspace> workspace(std::size_t positions,
                                                               std::size_t block) const = 0;
    virtual void execute(Workspace& scratch, const Evaluation& evaluation) const = 0;
};

inline std::span<float> row(std::span<float> values, std::size_t width, std::size_t index) {
    return values.subspan(width * index, width);
}

} // namespace nathr::detail
