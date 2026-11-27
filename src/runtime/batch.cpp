#include <nathr/batch.hpp>

#include "models/model.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <iterator>
#include <list>
#include <map>

namespace nathr {
namespace {
using Clock = std::chrono::steady_clock;
using detail::Cache;
using detail::Evaluation;
constexpr std::size_t max_draft_tokens = 8;

double milliseconds(Clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

// A page is identified by its parent prefix and the exact tokens on the page.
class PrefixCache {
    using Key = std::pair<std::size_t, std::array<Token, Cache::page_tokens>>;

  public:
    struct Entry {
        Key key;
        Cache::Page page;
        Entry* parent;
        std::size_t id;
        std::size_t children = 0;
    };

    Entry* find(Entry* parent, std::span<const Token> tokens) {
        const auto it = index_.find(key(parent, tokens));
        if (it == index_.end())
            return nullptr;
        pages_.splice(pages_.end(), pages_, it->second);
        return &*it->second;
    }

    Entry* remember(Entry* parent, std::span<const Token> tokens, const Cache::Page& page) {
        auto [it, inserted] = index_.try_emplace(key(parent, tokens), pages_.end());
        if (!inserted) {
            pages_.splice(pages_.end(), pages_, it->second);
            return &*it->second;
        }
        try {
            pages_.push_back({it->first, page, parent, ++next_id_});
        } catch (...) {
            index_.erase(it);
            throw;
        }
        it->second = std::prev(pages_.end());
        if (parent)
            ++parent->children;
        return &pages_.back();
    }

    bool reclaim() {
        const auto it = std::ranges::find_if(
            pages_, [](const Entry& entry) { return !entry.children && entry.page.use_count() == 1; });
        if (it == pages_.end())
            return false;
        if (it->parent)
            --it->parent->children;
        index_.erase(it->key);
        pages_.erase(it);
        return true;
    }

  private:
    static Key key(const Entry* parent, std::span<const Token> tokens) {
        Key result{parent ? parent->id : 0, {}};
        std::ranges::copy(tokens.first(Cache::page_tokens), result.second.begin());
        return result;
    }

    std::list<Entry> pages_;
    std::map<Key, std::list<Entry>::iterator> index_;
    std::size_t next_id_ = 0;
};

struct Speculation {
    static constexpr std::size_t ngram = 3;
    static constexpr std::size_t lookup_window = 2048;

    Speculation(std::span<const Token> prompt, Sampler& sampler, std::mt19937_64& rng)
        : verify([this, &sampler, &rng](std::size_t position, std::span<const float> logits) {
              predictions[position - start] = sampler.select(logits, rng);
          }) {
        std::ranges::copy(prompt.last(ngram), suffix.begin());
    }
    Speculation(const Speculation&) = delete;
    Speculation& operator=(const Speculation&) = delete;

    void advance(Token token) {
        std::move(suffix.begin() + 1, suffix.end(), suffix.begin());
        suffix.back() = token;
    }

    std::span<const Token> propose(std::span<const Token> prompt, std::size_t position, std::size_t limit) {
        const auto first = prompt.size() - std::min(prompt.size(), lookup_window);
        std::size_t continuation = 0;
        for (auto end = prompt.size() - 1; end >= first + ngram; --end) {
            if (!std::ranges::equal(prompt.subspan(end - ngram, ngram), suffix))
                continue;
            continuation = end;
            if (prompt.size() - end >= limit)
                break;
        }
        if (!continuation)
            return {};
        start = position;
        const auto count = std::min(limit, prompt.size() - continuation);
        inputs[0] = suffix.back();
        std::ranges::copy(prompt.subspan(continuation, count), inputs.begin() + 1);
        return std::span(inputs).first(1 + count);
    }

    std::array<Token, ngram> suffix;
    std::array<Token, max_draft_tokens + 1> inputs, predictions;
    std::size_t start = 0;
    LogitsSink verify;
};

struct Request {
    Request(BatchRequest&& input, std::size_t capacity, std::size_t reservation)
        : input(std::move(input)), capacity(capacity), reservation(reservation) {
        if (this->input.tokenizer)
            text.emplace(std::move(*this->input.tokenizer), std::move(this->input.stop_strings));
    }

    BatchRequest input;
    std::size_t capacity, reservation;
    Clock::time_point arrival = Clock::now();
    std::optional<Clock::time_point> admitted;
    std::optional<Clock::time_point> last_token;
    std::optional<Cache> cache;
    std::optional<T5Session> seq2seq;
    std::optional<Sampler> sampler;
    std::optional<Token> pending;
    std::optional<GenerationText> text;
    std::mt19937_64 rng{input.seed};
    std::optional<Speculation> speculation;
    std::vector<float> logits;
    std::size_t emitted = 0, position = 0;
    PrefixCache::Entry* prefix = nullptr;
    std::size_t cached_prompt_tokens = 0, bypasses = 0;
    std::size_t draft_tokens = 0, accepted_draft_tokens = 0;

    BatchEvent event(BatchEventKind kind, Clock::time_point now = Clock::now()) const {
        return {input.id,
                kind,
                {},
                emitted,
                milliseconds(now - arrival),
                {},
                {},
                {},
                input.tokens.size(),
                cached_prompt_tokens,
                milliseconds(admitted.value_or(now) - arrival),
                draft_tokens,
                accepted_draft_tokens};
    }
};

using Requests = std::list<Request>;
} // namespace

struct BatchScheduler::Impl {
    Impl(const Model& source, BatchOptions settings)
        : model(source), options(settings),
          pool(settings.kv_budget_bytes, [this] { return prefixes.reclaim(); }) {
        if (const auto* loaded = std::get_if<std::unique_ptr<detail::Decoder>>(&model.impl_->execution))
            decoder = loaded->get();
        if (!model.supports_generation())
            throw Error(ErrorCode::unsupported, "checkpoint does not support generation");
        if (!options.context)
            options.context = std::min<std::size_t>(2048, model.config().context);
        if (!options.max_active || !options.max_queued || !options.max_queued_tokens ||
            options.step_tokens < options.max_active || !options.kv_budget_bytes ||
            options.context > model.config().context || !options.session.prefill_chunk ||
            options.session.prefill_chunk > 32 ||
            (options.session.kv_precision != KvPrecision::f32 &&
             options.session.kv_precision != KvPrecision::f16))
            throw Error(ErrorCode::invalid_argument, "invalid batch context, scheduling, or cache settings");
        if (options.speculative_tokens > max_draft_tokens)
            throw Error(ErrorCode::invalid_argument, "speculative tokens must be between 0 and 8");
        if (decoder)
            scratch = decoder->workspace(options.context, options.step_tokens);
        else if (options.session.kv_precision != KvPrecision::f32)
            throw Error(ErrorCode::unsupported, "T5 batches use an FP32 KV cache");
    }

    void remove_active(Requests::iterator request) {
        reserved -= request->reservation;
        active.erase(request);
    }

    void admit(std::vector<BatchEvent>& events) {
        while (!queued.empty() && active.size() < options.max_active) {
            const auto available = options.kv_budget_bytes - reserved;
            auto candidate = queued.begin();
            if (candidate->reservation > available) {
                if (candidate->bypasses >= options.max_queue_bypasses)
                    break;
                candidate = std::find_if(std::next(candidate), queued.end(),
                                         [=](const Request& item) { return item.reservation <= available; });
                if (candidate == queued.end())
                    break;
            }
            auto& request = *candidate;
            request.admitted = Clock::now();
            try {
                request.sampler.emplace(request.input.sampling, model.config().vocabulary);
                if (!request.input.max_new_tokens) {
                    events.push_back(request.event(BatchEventKind::length));
                    queued_tokens -= request.input.tokens.size();
                    queued.erase(candidate);
                    continue;
                }
                if (decoder) {
                    request.cache.emplace(model.config(), request.capacity, options.session.kv_precision,
                                          &pool);
                    request.logits.resize(model.config().vocabulary);
                    if (options.prefix_cache)
                        reuse_prefix(request);
                    if (options.speculative_tokens && request.input.sampling.greedy &&
                        request.input.tokens.size() > Speculation::ngram)
                        request.speculation.emplace(request.input.tokens, *request.sampler, request.rng);
                }
            } catch (const Error& error) {
                auto event = request.event(BatchEventKind::error);
                event.message = error.what();
                events.push_back(std::move(event));
                queued_tokens -= request.input.tokens.size();
                queued.erase(candidate);
                continue;
            }
            if (candidate != queued.begin()) {
                ++queued.front().bypasses;
                ++queue_bypasses;
            }
            reserved += request.reservation;
            queued_tokens -= request.input.tokens.size();
            active.splice(active.end(), queued, candidate);
        }
    }

    void reuse_prefix(Request& request) {
        const auto tokens = std::span(request.input.tokens);
        while (tokens.size() - request.position >= Cache::page_tokens) {
            auto* page = prefixes.find(request.prefix, tokens.subspan(request.position));
            if (!page)
                break;
            request.cache->use_prefix_page(request.position / Cache::page_tokens, page->page);
            request.prefix = page;
            request.position += Cache::page_tokens;
        }
        request.cached_prompt_tokens = request.position;
        // Replay the last token for its logits while reading the already complete page.
        if (request.position == tokens.size())
            --request.position;
    }

    bool emit_token(Request& request, std::vector<BatchEvent>& events) {
        const auto token = request.seq2seq ? request.seq2seq->sample(*request.sampler)
                                           : request.sampler->select(request.logits, request.rng);
        return emit_token(request, token, events);
    }

    bool emit_token(Request& request, Token token, std::vector<BatchEvent>& events) {
        const auto now = Clock::now();
        const bool eos = token == model.config().eos;
        const bool stop_id = std::ranges::find(request.input.stop_ids, token) != request.input.stop_ids.end();
        ++request.emitted;
        auto event = request.event(BatchEventKind::token, now);
        event.token = token;
        if (request.last_token)
            event.inter_token_ms = milliseconds(now - *request.last_token);
        if (request.text)
            event.text = eos || stop_id ? std::string() : request.text->push(token);
        request.last_token = now;
        events.push_back(std::move(event));
        const bool stop = stop_id || (request.text && request.text->stopped());
        if (eos || stop || request.emitted == request.input.max_new_tokens) {
            auto done = request.event(eos    ? BatchEventKind::eos
                                      : stop ? BatchEventKind::stop
                                             : BatchEventKind::length);
            if (request.text)
                done.text = request.text->finish();
            events.push_back(std::move(done));
            return false;
        }
        request.pending = token;
        if (request.speculation)
            request.speculation->advance(token);
        return true;
    }

    void finish(Requests::iterator request, const Evaluation& eval, std::vector<BatchEvent>& events) {
        if (eval.mode == LogitsMode::all_tokens) {
            finish_verification(request, eval.tokens, events);
            return;
        }
        const auto previous_position = request->position;
        request->position += eval.tokens.size();
        if (options.prefix_cache && !request->pending && !eval.replay_cached) {
            const auto tokens = std::span(request->input.tokens);
            const auto complete = request->position / Cache::page_tokens;
            for (auto index = previous_position / Cache::page_tokens; index < complete; ++index) {
                request->prefix = prefixes.remember(
                    request->prefix, tokens.subspan(index * Cache::page_tokens), request->cache->page(index));
                request->cache->use_prefix_page(index, request->prefix->page);
            }
        }
        if (eval.mode == LogitsMode::last_token && !emit_token(*request, events))
            remove_active(request);
    }

    void finish_verification(Requests::iterator request, std::span<const Token> inputs,
                             std::vector<BatchEvent>& events) {
        request->draft_tokens += inputs.size() - 1;
        for (std::size_t index = 0; index < inputs.size(); ++index) {
            const auto token = request->speculation->predictions[index];
            ++request->position;
            const bool matches = index + 1 < inputs.size() && token == inputs[index + 1];
            if (matches)
                ++request->accepted_draft_tokens;
            if (!emit_token(*request, token, events)) {
                remove_active(request);
                return;
            }
            // Leave rows after a mismatch outside the visible cache. Future steps overwrite them.
            if (!matches)
                return;
        }
    }

    void extend_decodes(std::size_t decodes, std::size_t budget) {
        const auto eligible = [](const Request& request) {
            return request.speculation && request.input.max_new_tokens - request.emitted > 1;
        };
        auto ready = std::span(scheduled).first(decodes);
        auto remaining = static_cast<std::size_t>(
            std::ranges::count_if(ready, [&](auto request) { return eligible(*request); }));
        for (std::size_t index = 0; index < decodes && budget; ++index) {
            auto& request = *ready[index];
            if (!eligible(request))
                continue;
            const auto share = budget / remaining + (budget % remaining != 0);
            --remaining;
            // Keep the final budget token as an output, without allocating a cache row for it.
            const auto limit = std::min(
                {share, options.speculative_tokens, request.input.max_new_tokens - request.emitted - 1});
            const auto draft = request.speculation->propose(request.input.tokens, request.position, limit);
            if (draft.empty())
                continue;
            try {
                request.cache->ensure(request.position + draft.size());
            } catch (const Error&) {
                // The one-token evaluation already has the pages it needs.
                continue;
            }
            auto& eval = evaluations[index];
            eval.tokens = draft;
            eval.mode = LogitsMode::all_tokens;
            eval.last_logits = {};
            budget -= draft.size() - 1;
        }
    }

    void fail(Requests::iterator request, const Error& error, std::vector<BatchEvent>& events) {
        auto event = request->event(BatchEventKind::error);
        event.message = error.what();
        events.push_back(std::move(event));
        remove_active(request);
    }

    void step_seq2seq(std::vector<BatchEvent>& events) {
        // A bidirectional source is encoded as a whole. Admit at most one new source per step.
        auto waiting = active.end();
        for (auto next = active.begin(); next != active.end();) {
            const auto it = next++;
            if (!it->seq2seq) {
                if (waiting == active.end())
                    waiting = it;
                continue;
            }
            try {
                const auto token = *it->pending;
                it->seq2seq->append(std::span(&token, 1));
                if (!emit_token(*it, events))
                    remove_active(it);
            } catch (const Error& error) {
                fail(it, error, events);
            }
        }
        if (waiting == active.end())
            return;
        try {
            waiting->seq2seq.emplace(T5Session::create_pooled(model, waiting->input.tokens, waiting->capacity,
                                                              waiting->input.seed, &pool));
            if (!emit_token(*waiting, events))
                remove_active(waiting);
        } catch (const Error& error) {
            fail(waiting, error, events);
        }
    }

    Model model;
    BatchOptions options;
    const detail::Decoder* decoder = nullptr;
    detail::CachePagePool pool;
    PrefixCache prefixes;
    std::unique_ptr<detail::Workspace> scratch;
    Requests queued, active;
    std::vector<Evaluation> evaluations;
    std::vector<Requests::iterator> scheduled;
    std::size_t reserved = 0, queued_tokens = 0, queue_bypasses = 0;
};

BatchScheduler::BatchScheduler(const Model& model, BatchOptions options)
    : impl_(std::make_unique<Impl>(model, options)) {}
BatchScheduler::~BatchScheduler() = default;
BatchScheduler::BatchScheduler(BatchScheduler&&) noexcept = default;
BatchScheduler& BatchScheduler::operator=(BatchScheduler&&) noexcept = default;

void BatchScheduler::submit(BatchRequest request) {
    auto& s = *impl_;
    if (contains(request.id))
        throw Error(ErrorCode::invalid_argument, "duplicate request ID");
    if (s.queued.size() >= s.options.max_queued)
        throw Error(ErrorCode::resource_limit, "batch queue is full");
    if (request.tokens.size() > s.options.max_queued_tokens - s.queued_tokens)
        throw Error(ErrorCode::resource_limit, "batch queued-token budget exceeded");
    if (!request.tokenizer && !request.stop_strings.empty())
        throw Error(ErrorCode::invalid_argument, "stop strings require a tokenizer");
    if (std::ranges::any_of(request.stop_ids, [&](Token id) { return id >= s.model.config().vocabulary; }))
        throw Error(ErrorCode::invalid_argument, "stop token ID exceeds model vocabulary");
    detail::validate_tokens(request.tokens, s.model.config(), s.options.context);
    const auto append = request.max_new_tokens ? request.max_new_tokens - 1 : 0;
    if ((s.decoder && append > s.options.context - request.tokens.size()) ||
        (!s.decoder && request.max_new_tokens > s.options.context))
        throw Error(ErrorCode::invalid_argument, "prompt and generation budget exceed context");
    const auto capacity =
        s.decoder ? request.tokens.size() + append : std::max<std::size_t>(1, request.max_new_tokens);
    const auto positions = s.decoder ? capacity : detail::sum(request.tokens.size(), capacity, "T5 cache");
    const auto reservation = request.max_new_tokens ? Cache::required_bytes(s.model.config(), positions,
                                                                            s.options.session.kv_precision)
                                                    : 0;
    if (reservation > std::min(s.options.kv_budget_bytes, detail::max_kv_cache_bytes))
        throw Error(ErrorCode::resource_limit, "request exceeds KV budget");
    s.queued.emplace_back(std::move(request), capacity, reservation);
    s.queued_tokens += s.queued.back().input.tokens.size();
}

bool BatchScheduler::contains(RequestId id) const noexcept {
    const auto matches = [=](const Request& item) { return item.input.id == id; };
    return std::ranges::any_of(impl_->active, matches) || std::ranges::any_of(impl_->queued, matches);
}

std::optional<BatchEvent> BatchScheduler::cancel(RequestId id) {
    auto& s = *impl_;
    const auto matches = [=](const Request& item) { return item.input.id == id; };
    const auto running = std::ranges::find_if(s.active, matches);
    if (running != s.active.end()) {
        auto event = running->event(BatchEventKind::cancelled);
        s.remove_active(running);
        return event;
    }
    const auto waiting = std::ranges::find_if(s.queued, matches);
    if (waiting == s.queued.end())
        return {};
    auto event = waiting->event(BatchEventKind::cancelled);
    s.queued_tokens -= waiting->input.tokens.size();
    s.queued.erase(waiting);
    return event;
}

std::vector<BatchEvent> BatchScheduler::step() {
    auto& s = *impl_;
    std::vector<BatchEvent> events;
    s.admit(events);
    events.reserve(events.size() + 2 * s.active.size());
    if (!s.decoder) {
        s.step_seq2seq(events);
        return events;
    }
    auto& evaluations = s.evaluations;
    auto& requests = s.scheduled;
    evaluations.clear();
    requests.clear();
    evaluations.reserve(s.active.size());
    requests.reserve(s.active.size());
    static const LogitsSink sink;
    static const TraceSink trace;
    const auto schedule = [&](Requests::iterator it, std::span<const Token> tokens, LogitsMode mode) {
        try {
            it->cache->ensure(it->position + tokens.size());
            const auto& logits_sink = it->speculation ? it->speculation->verify : sink;
            evaluations.push_back({tokens, it->position, &*it->cache, mode, logits_sink, trace, it->logits,
                                   it->position < it->cached_prompt_tokens});
            requests.push_back(it);
        } catch (const Error& error) {
            s.fail(it, error, events);
        }
    };
    auto budget = s.options.step_tokens;
    std::size_t prefills = 0;
    for (auto next = s.active.begin(); next != s.active.end();) {
        const auto it = next++;
        if (!it->pending)
            ++prefills;
        else {
            --budget;
            schedule(it, std::span(&*it->pending, 1), LogitsMode::last_token);
        }
    }
    const auto decodes = evaluations.size();
    for (auto next = s.active.begin(); next != s.active.end();) {
        const auto it = next++;
        if (it->pending)
            continue;
        const auto remaining = std::span(it->input.tokens).subspan(it->position);
        const auto count = std::min({remaining.size(), s.options.session.prefill_chunk, budget / prefills});
        --prefills;
        budget -= count;
        const auto mode = count == remaining.size() ? LogitsMode::last_token : LogitsMode::none;
        schedule(it, remaining.first(count), mode);
    }
    if (evaluations.empty())
        return events;
    if (s.options.speculative_tokens && budget && decodes)
        s.extend_decodes(decodes, budget);
    bool retry_individually = false;
    try {
        s.decoder->execute_batch(*s.scratch, {evaluations});
    } catch (const Error&) {
        // Recompute the uncommitted rows individually to isolate a failing request.
        retry_individually = true;
    }
    for (std::size_t i = 0; i < evaluations.size(); ++i) {
        try {
            auto& eval = evaluations[i];
            if (retry_individually) {
                if (eval.mode == LogitsMode::all_tokens) {
                    // A rejected speculative row may overflow or produce nonfinite logits.
                    eval.tokens = eval.tokens.first(1);
                    eval.mode = LogitsMode::last_token;
                    eval.last_logits = requests[i]->logits;
                }
                s.decoder->execute(*s.scratch, eval);
            }
            s.finish(requests[i], eval, events);
        } catch (const Error& error) {
            s.fail(requests[i], error, events);
        }
    }
    return events;
}

bool BatchScheduler::empty() const noexcept {
    return impl_->active.empty() && impl_->queued.empty();
}

std::size_t BatchScheduler::context() const noexcept {
    return impl_->options.context;
}

std::size_t BatchScheduler::peak_kv_bytes() const noexcept {
    return impl_->pool.peak_bytes();
}

std::size_t BatchScheduler::resident_kv_bytes() const noexcept {
    return impl_->pool.resident_bytes();
}
std::size_t BatchScheduler::kv_page_reuses() const noexcept {
    return impl_->pool.reuses();
}

std::size_t BatchScheduler::queue_bypasses() const noexcept {
    return impl_->queue_bypasses;
}
} // namespace nathr
