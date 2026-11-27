#pragma once

#include <nathr/generation.hpp>

namespace nathr {

struct BatchOptions {
    std::size_t context = 0;
    std::size_t max_active = 4;
    std::size_t max_queued = 32;
    std::size_t max_queued_tokens = 16384;
    std::size_t step_tokens = 32;
    std::size_t kv_budget_bytes = 512 * 1024 * 1024;
    SessionOptions session;
    bool prefix_cache = true;
    std::size_t max_queue_bypasses = 0;
};

using RequestId = std::uint64_t;

struct BatchRequest {
    RequestId id;
    std::vector<Token> tokens;
    std::size_t max_new_tokens = 32;
    std::uint64_t seed = 0;
    SamplingOptions sampling;
    std::vector<Token> stop_ids{};
    std::optional<Tokenizer> tokenizer{};
    std::vector<std::string> stop_strings{};
};

enum class BatchEventKind { token, eos, stop, length, cancelled, error };

struct BatchEvent {
    RequestId id;
    BatchEventKind kind;
    std::optional<Token> token;
    std::size_t emitted;
    double elapsed_ms;
    std::optional<double> inter_token_ms;
    std::string message;
    std::optional<std::string> text{};
    std::size_t prompt_tokens{};
    std::size_t cached_prompt_tokens{};
    double queue_ms{};
};

struct BatchMetrics {
    std::size_t active_requests{}, queued_requests{}, queued_tokens{};
    std::size_t kv_used_bytes{}, kv_resident_bytes{}, kv_budget_bytes{};
    std::size_t prefix_lookups{}, prefix_hits{}, cached_prompt_tokens{};

    bool operator==(const BatchMetrics&) const = default;
};

// Call from one thread. Submit and cancel requests between calls to step().
class BatchScheduler {
  public:
    explicit BatchScheduler(const Model& model, BatchOptions options = {});
    ~BatchScheduler();
    BatchScheduler(BatchScheduler&&) noexcept;
    BatchScheduler& operator=(BatchScheduler&&) noexcept;

    // IDs must be unique among pending requests. A rejected request never enters the queue.
    void submit(BatchRequest request);
    [[nodiscard]] bool contains(RequestId id) const noexcept;
    [[nodiscard]] std::optional<BatchEvent> cancel(RequestId id);
    [[nodiscard]] std::vector<BatchEvent> step();
    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] std::size_t context() const noexcept;
    [[nodiscard]] std::size_t peak_kv_bytes() const noexcept;
    [[nodiscard]] std::size_t resident_kv_bytes() const noexcept;
    [[nodiscard]] std::size_t kv_page_reuses() const noexcept;
    [[nodiscard]] std::size_t queue_bypasses() const noexcept;
    [[nodiscard]] BatchMetrics metrics() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace nathr
