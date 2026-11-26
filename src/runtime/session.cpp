#include "models/model.hpp"

#include <nathr/generation.hpp>

#include <algorithm>

namespace nathr {

struct Session::Impl {
    Impl(const Model& source, std::size_t limit, std::uint64_t start_seed, SessionOptions options)
        : model(source), decoder(model.impl_->decoder()), cache(model.config(), limit, options.kv_precision),
          chunk(std::min(limit, options.prefill_chunk)), scratch(decoder.workspace(limit, chunk)),
          seed(start_seed), rng(seed) {
        tokens.reserve(limit);
    }

    void run(std::span<const Token> ids, LogitsMode mode, const LogitsSink& sink, const TraceSink& trace) {
        detail::validate_tokens(ids, model.config(), cache.capacity - tokens.size());
        detail::validate_mode(mode, sink);
        const auto start = tokens.size();
        state = SessionState::invalid;
        has_logits = false;
        for (std::size_t i = 0; i < ids.size();) {
            const auto count = std::min(chunk, ids.size() - i);
            const auto output =
                (mode == LogitsMode::last_token && i + count != ids.size()) ? LogitsMode::none : mode;
            decoder.execute(*scratch, {ids.subspan(i, count), start + i, &cache, output, sink, trace});
            i += count;
        }
        // The input may be a view of the already-consumed prefix.
        tokens.resize(start + ids.size());
        std::ranges::copy(ids, tokens.begin() + static_cast<std::ptrdiff_t>(start));
        has_logits = mode != LogitsMode::none;
        state = SessionState::ready;
    }

    Model model;
    const detail::Decoder& decoder;
    detail::Cache cache;
    const std::size_t chunk;
    std::unique_ptr<detail::Workspace> scratch;
    std::vector<Token> tokens;
    SessionState state = SessionState::empty;
    bool has_logits = false;
    std::uint64_t seed;
    std::mt19937_64 rng;
};

Session::Session(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Session::~Session() = default;
Session::Session(Session&&) noexcept = default;
Session& Session::operator=(Session&&) noexcept = default;

Session Session::create(const Model& model, std::size_t capacity, std::uint64_t seed,
                        SessionOptions options) {
    if (capacity == 0)
        capacity = std::min<std::size_t>(2048, model.config().context);
    if (capacity > model.config().context)
        throw Error(ErrorCode::invalid_argument, "context exceeds model capacity");
    if (!options.prefill_chunk || options.prefill_chunk > 32)
        throw Error(ErrorCode::invalid_argument, "prefill chunk must be between 1 and 32 tokens");
    if (options.kv_precision != KvPrecision::f32 && options.kv_precision != KvPrecision::f16)
        throw Error(ErrorCode::invalid_argument, "unknown KV cache precision");
    return Session(std::make_unique<Impl>(model, capacity, seed, options));
}

void Session::prefill(std::span<const Token> tokens, LogitsMode mode, const LogitsSink& sink,
                      const TraceSink& trace) {
    if (state() != SessionState::empty)
        throw Error(ErrorCode::invalid_argument, "prefill requires an empty session");
    impl_->run(tokens, mode, sink, trace);
}

void Session::append(std::span<const Token> tokens, LogitsMode mode, const LogitsSink& sink,
                     const TraceSink& trace) {
    if (state() != SessionState::ready)
        throw Error(ErrorCode::invalid_argument, "append requires a ready session");
    impl_->run(tokens, mode, sink, trace);
}

std::span<const float> Session::logits() const {
    if (state() != SessionState::ready || !impl_->has_logits)
        throw Error(ErrorCode::invalid_argument, "logits are unavailable");
    return impl_->scratch->logits;
}

Token Session::sample(Sampler& sampler) {
    return sampler.select(logits(), impl_->rng);
}

std::span<const Token> Session::history() const noexcept {
    return impl_->tokens;
}
SessionState Session::state() const noexcept {
    return impl_->state;
}
std::size_t Session::capacity() const noexcept {
    return impl_->cache.capacity;
}
SessionMemory Session::memory() const noexcept {
    return {impl_->cache.bytes(), impl_->scratch->bytes(), impl_->tokens.capacity() * sizeof(Token)};
}

void Session::reset() noexcept {
    impl_->tokens.clear();
    impl_->rng.seed(impl_->seed);
    impl_->has_logits = false;
    impl_->state = SessionState::empty;
}

} // namespace nathr
