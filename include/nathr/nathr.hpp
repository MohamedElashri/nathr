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
#include <vector>

namespace nathr {

namespace detail {
class LoraWeights;
} // namespace detail

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
    // For T5, layers describes the decoder and context bounds each sequence independently.
    std::size_t layers, hidden, intermediate, query_heads, kv_heads, head_dim, vocabulary, context;
    float rope_theta, rms_epsilon; // rms_epsilon also carries the LayerNorm epsilon for GPT-2 and BERT.
    Token bos, eos;                // For T5, bos is the decoder start ID.
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

struct InferenceInput {
    std::span<const Token> tokens;
    std::span<const Token> token_type_ids{};
    std::span<const Token> decoder_tokens{};
};

struct OutputInfo {
    std::string_view name;
    std::size_t width, input_limit;
    bool sequence;
};

struct TensorResult {
    std::vector<std::size_t> shape;
    std::vector<float> values;
};

class LoraAdapter {
  public:
    static constexpr std::size_t max_bytes = 128 * 1024 * 1024;
    [[nodiscard]] std::size_t memory_bytes() const noexcept;

  private:
    LoraAdapter(std::shared_ptr<const void> model, std::shared_ptr<const detail::LoraWeights> weights)
        : model_(std::move(model)), weights_(std::move(weights)) {}
    std::shared_ptr<const void> model_;
    std::shared_ptr<const detail::LoraWeights> weights_;
    friend class Model;
    friend class Session;
    friend class BatchScheduler;
};

class Model {
  public:
    static Model load(const std::filesystem::path& path, BackendOptions options = {});
    [[nodiscard]] const ModelConfig& config() const noexcept;
    [[nodiscard]] ModelMemory memory() const noexcept;
    [[nodiscard]] std::string_view architecture() const noexcept;
    [[nodiscard]] std::string_view embedded_chat_template() const noexcept;
    [[nodiscard]] std::string_view backend() const noexcept;
    [[nodiscard]] unsigned threads() const noexcept;
    [[nodiscard]] std::string_view kernel(std::uint32_t encoding_id) const noexcept;
    [[nodiscard]] std::vector<OutputInfo> outputs() const;
    [[nodiscard]] bool supports_generation() const noexcept;
    [[nodiscard]] bool supports_fused_batch() const noexcept;
    [[nodiscard]] LoraAdapter load_lora(const std::filesystem::path& directory,
                                        std::size_t byte_limit = LoraAdapter::max_bytes) const;
    [[nodiscard]] TensorResult infer(InferenceInput input, std::string_view output,
                                     const LoraAdapter* lora = nullptr) const;
    // Returns results in input order. Encoders may share projections across inputs.
    [[nodiscard]] std::vector<TensorResult> infer_batch(std::span<const InferenceInput> inputs,
                                                        std::string_view output) const;
    // Uncached, absolute positions starting at zero. At most 128 tokens.
    // Last-token mode writes last_logits and all-token mode streams to sink.
    void forward(std::span<const Token> tokens, LogitsMode mode, std::span<float> last_logits = {},
                 const LogitsSink& sink = {}, const TraceSink& trace = {},
                 const LoraAdapter* lora = nullptr) const;
    // BERT only. Returns one hidden-size row per input token, in input order.
    [[nodiscard]] std::vector<float> encode(std::span<const Token> tokens) const;

  private:
    struct Impl;
    explicit Model(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}
    std::shared_ptr<const Impl> impl_;
    friend class Session;
    friend class T5Session;
    friend class BatchScheduler;
};

enum class SessionState { empty, ready, invalid };
enum class KvPrecision { f32, f16 };
struct SessionOptions {
    KvPrecision kv_precision = KvPrecision::f32;
    std::size_t prefill_chunk = 16;
};
struct SessionMemory {
    // cache_bytes counts allocated KV page payloads, including pages retained after reset.
    std::size_t cache_bytes, workspace_bytes, history_bytes;
};

class Sampler;

class Session {
  public:
    static Session create(const Model& model, std::size_t capacity = 0, std::uint64_t seed = 0,
                          SessionOptions options = {}, const LoraAdapter* lora = nullptr);
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

class T5Session {
  public:
    // Encodes one unpadded source sequence and evaluates the decoder's start token.
    static T5Session create(const Model& model, std::span<const Token> source, std::size_t capacity = 0,
                            std::uint64_t seed = 0);
    ~T5Session();
    T5Session(T5Session&&) noexcept;
    T5Session& operator=(T5Session&&) noexcept;

    void append(std::span<const Token> tokens);
    // A view remains valid until the next mutating operation or destruction.
    [[nodiscard]] std::span<const float> logits() const;
    Token sample(Sampler& sampler);
    [[nodiscard]] std::size_t length() const noexcept;
    [[nodiscard]] std::size_t capacity() const noexcept;

  private:
    struct Impl;
    explicit T5Session(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace nathr
