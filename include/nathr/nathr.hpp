#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace nathr {

enum class ErrorCode { invalid_argument, unsupported, malformed, io, resource_limit, backend };

class Error : public std::runtime_error {
  public:
    Error(ErrorCode code, const std::string& message) : std::runtime_error(message), code_(code) {}
    [[nodiscard]] ErrorCode code() const noexcept { return code_; }

  private:
    ErrorCode code_;
};

using Token = std::uint32_t;

struct ModelConfig {
    std::size_t layers, hidden, intermediate, query_heads, kv_heads, head_dim, vocabulary, context;
    float rope_theta, rms_epsilon;
    Token bos, eos;
    bool tied_embeddings;
};

struct ModelMemory {
    std::size_t file_bytes, encoded_weight_bytes, stored_parameters, tensors;
    std::size_t f32_tensors, f16_tensors, bf16_tensors;
    std::size_t mapped_file_bytes = 0;
};

enum class ExecutionPolicy { scalar, automatic, avx2 };

struct CpuFeatures {
    bool avx2{}, fma{}, f16c{};
};
[[nodiscard]] CpuFeatures cpu_features() noexcept;
struct BackendOptions {
    ExecutionPolicy policy = ExecutionPolicy::scalar;
    unsigned threads = 1;
};

enum class LogitsMode { none, last_token, all_tokens };
using LogitsSink = std::function<void(std::size_t position, std::span<const float>)>;

struct TraceEvent {
    static constexpr std::size_t model_level = static_cast<std::size_t>(-1);
    std::string_view operation;
    std::size_t layer, position;
    std::span<const float> values;
};
using TraceSink = std::function<void(const TraceEvent&)>;

class Model {
  public:
    static Model load(const std::filesystem::path& path, BackendOptions options = {});
    [[nodiscard]] const ModelConfig& config() const noexcept;
    [[nodiscard]] ModelMemory memory() const noexcept;
    [[nodiscard]] std::string_view backend() const noexcept;
    [[nodiscard]] unsigned threads() const noexcept;
    [[nodiscard]] std::string_view kernel(std::uint32_t encoding_id) const noexcept;
    // Uncached, absolute positions starting at zero. At most 128 tokens.
    // Last-token mode writes last_logits and all-token mode streams to sink.
    void forward(std::span<const Token> tokens, LogitsMode mode, std::span<float> last_logits = {},
                 const LogitsSink& sink = {}, const TraceSink& trace = {}) const;

  private:
    struct Impl;
    explicit Model(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}
    std::shared_ptr<const Impl> impl_;
    friend class Session;
};

enum class SessionState { empty, ready, invalid };
enum class KvPrecision { f32, f16 };
struct SessionOptions {
    KvPrecision kv_precision = KvPrecision::f32;
    std::size_t prefill_chunk = 16;
};
struct SessionMemory {
    std::size_t cache_bytes, workspace_bytes, history_bytes;
};

class Sampler;

class Session {
  public:
    static Session create(const Model& model, std::size_t capacity = 0, std::uint64_t seed = 0,
                          SessionOptions options = {});
    ~Session();
    Session(Session&&) noexcept;
    Session& operator=(Session&&) noexcept;

    // The callback values are borrowed only for the duration of the callback.
    // Callbacks must not reenter or mutate the same session.
    void prefill(std::span<const Token> tokens, LogitsMode mode = LogitsMode::last_token,
                 const LogitsSink& sink = {}, const TraceSink& trace = {});
    void append(std::span<const Token> tokens, LogitsMode mode = LogitsMode::last_token,
                const LogitsSink& sink = {}, const TraceSink& trace = {});
    // A view remains valid until the next mutating operation or destruction.
    [[nodiscard]] std::span<const float> logits() const;
    Token sample(Sampler& sampler);
    [[nodiscard]] std::span<const Token> history() const noexcept;
    [[nodiscard]] SessionState state() const noexcept;
    [[nodiscard]] std::size_t capacity() const noexcept;
    [[nodiscard]] SessionMemory memory() const noexcept;
    void reset() noexcept;

  private:
    struct Impl;
    explicit Session(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace nathr
