#include <nathr/batch.hpp>

#include "models/model.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <iterator>
#include <list>
#include <map>
#include <tuple>
#include <utility>

namespace nathr {
namespace {
using Clock = std::chrono::steady_clock;
using detail::Cache;
using detail::Evaluation;
using detail::T5;
constexpr std::size_t max_draft_tokens = 8;
constexpr std::size_t retained_t5_sources = 32;

double milliseconds(Clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

// A page is identified by its adapter, parent prefix, and exact tokens.
class PrefixCache {
    using Key = std::tuple<std::uint64_t, std::size_t, std::array<Token, Cache::page_tokens>>;

  public:
    struct Entry {
        Key key;
        Cache::Page page;
        Entry* parent;
        std::size_t id;
        std::size_t children = 0;
        std::size_t active_references = 0;
    };

    using Iterator = std::list<Entry>::iterator;

    std::optional<Iterator> find(std::uint64_t adapter, Entry* parent, std::span<const Token> tokens) {
        const auto it = index_.find(key(adapter, parent, tokens));
        if (it == index_.end())
            return {};
        return it->second;
    }

    void touch(Iterator page) { pages_.splice(pages_.end(), pages_, page); }

    Entry* remember(std::uint64_t adapter, Entry* parent, std::span<const Token> tokens,
                    const Cache::Page& page) {
        auto [it, inserted] = index_.try_emplace(key(adapter, parent, tokens), pages_.end());
        if (!inserted) {
            touch(it->second);
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
    static Key key(std::uint64_t adapter, const Entry* parent, std::span<const Token> tokens) {
        std::array<Token, Cache::page_tokens> page;
        std::ranges::copy(tokens.first(Cache::page_tokens), page.begin());
        return {adapter, parent ? parent->id : 0, page};
    }

    std::list<Entry> pages_;
    std::map<Key, Iterator> index_;
    std::size_t next_id_ = 0;
};

struct Speculation {
    static constexpr std::size_t ngram = 3;
    static constexpr std::size_t lookup_window = 2048;

    Speculation(std::span<const Token> prompt, Sampler& sampler, std::mt19937_64& rng,
                std::optional<std::size_t> top_logprobs)
        : verify([this, &sampler, &rng, top_logprobs](std::size_t position, std::span<const float> logits) {
              if (!verifying)
                  return;
              const auto index = position - start;
              predictions[index] = sampler.select(logits, rng);
              if (top_logprobs)
                  scores[index] = token_logprobs(logits, predictions[index], *top_logprobs);
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
    std::array<std::optional<TokenLogprobs>, max_draft_tokens + 1> scores;
    std::size_t start = 0;
    bool verifying = false;
    LogitsSink verify;
};

struct T5Source {
    std::vector<Token> tokens;
    std::size_t bytes;
    T5::Source cross;
    std::size_t references = 0;
    bool used = false;
};

struct Request {
    Request(BatchRequest&& input, std::size_t capacity, std::size_t reservation,
            const detail::LoraWeights* lora)
        : input(std::move(input)), capacity(capacity), reservation(reservation), lora(lora) {
        if (this->input.tokenizer)
            text.emplace(std::move(*this->input.tokenizer), std::move(this->input.stop_strings));
    }

    BatchRequest input;
    std::size_t capacity, reservation;
    const detail::LoraWeights* lora;
    std::uint64_t lora_key() const { return lora ? lora->cache_key() : 0; }
    Clock::time_point arrival = Clock::now();
    std::optional<Clock::time_point> admitted;
    std::optional<Clock::time_point> first_token;
    std::optional<Clock::time_point> last_token;
    std::optional<Cache> cache;
    std::optional<T5::State> seq2seq;
    std::optional<Sampler> sampler;
    std::optional<Token> pending;
    std::optional<GenerationText> text;
    std::mt19937_64 rng{input.seed};
    std::optional<Speculation> speculation;
    std::vector<float> logits;
    std::size_t emitted = 0, position = 0;
    PrefixCache::Entry* prefix = nullptr;
    T5Source* source = nullptr;
    std::size_t cached_prompt_tokens = 0, bypasses = 0;
    std::size_t draft_tokens = 0, accepted_draft_tokens = 0;

    BatchEvent event(BatchEventKind kind, Clock::time_point now = Clock::now()) const {
        BatchEvent result{input.id,
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
                          {},
                          draft_tokens,
                          accepted_draft_tokens};
        if (kind != BatchEventKind::token && first_token && last_token) {
            result.first_token_ms = milliseconds(*first_token - arrival);
            result.last_token_ms = milliseconds(*last_token - arrival);
        }
        return result;
    }
};

using Requests = std::list<Request>;
} // namespace

struct BatchScheduler::Impl {
    Impl(const Model& source, BatchOptions settings)
        : model(source), options(settings),
          pool(settings.kv_budget_bytes, [this] { return prefixes.reclaim() || reclaim_source(); }) {
        if (const auto* loaded = std::get_if<std::unique_ptr<detail::Decoder>>(&model.impl_->execution))
            decoder = loaded->get();
        if (const auto* loaded = std::get_if<std::unique_ptr<T5>>(&model.impl_->execution))
            seq2seq = loaded->get();
        if (!model.supports_generation())
            throw Error(ErrorCode::unsupported, "checkpoint does not support generation");
        if (!options.max_active || !options.max_queued || !options.max_queued_tokens ||
            options.step_tokens < options.max_active || !options.kv_budget_bytes ||
            options.context > model.config().context || !options.session.prefill_chunk ||
            options.session.prefill_chunk > 32 ||
            (options.session.kv_precision != KvPrecision::f32 &&
             options.session.kv_precision != KvPrecision::f16))
            throw Error(ErrorCode::invalid_argument, "invalid batch context, scheduling, or cache settings");
        if (!options.context) {
            const auto slots =
                Cache::fit_tokens(model.config(), options.kv_budget_bytes, options.session.kv_precision);
            if (slots <= (seq2seq ? 1u : 0u))
                throw Error(ErrorCode::resource_limit, "KV budget cannot hold a generation request");
            options.context =
                std::min({detail::default_context_tokens, model.config().context, slots - (seq2seq ? 1 : 0)});
        }
        if (options.speculative_tokens > max_draft_tokens)
            throw Error(ErrorCode::invalid_argument, "speculative tokens must be between 0 and 8");
        if (decoder) {
            scratch = decoder->workspace(options.context, options.step_tokens);
            if (options.prefix_cache)
                prefix_page_bytes =
                    Cache::required_bytes(model.config(), Cache::page_tokens, options.session.kv_precision);
        } else {
            if (options.session.kv_precision != KvPrecision::f32)
                throw Error(ErrorCode::unsupported, "T5 batches use an FP32 KV cache");
            decode_rows = std::min(options.max_active, detail::batch_tile);
            for (;;) {
                try {
                    scratch = seq2seq->workspace(decode_rows, options.context);
                    break;
                } catch (const Error& error) {
                    if (error.code() != ErrorCode::resource_limit || decode_rows == 1)
                        throw;
                    decode_rows /= 2;
                }
            }
        }
    }

    bool reclaim_source() {
        const auto idle = std::ranges::find(sources, 0, &T5Source::references);
        if (idle == sources.end())
            return false;
        sources.erase(idle);
        return true;
    }

    void remove_active(Requests::iterator request) {
        reserved -= request->reservation;
        // A shared page remains reserved until its last active reader leaves.
        for (auto* page = request->prefix; page; page = page->parent) {
            if (--page->active_references)
                reserved += prefix_page_bytes;
        }
        auto* source = request->source;
        active.erase(request);
        if (source) {
            if (--source->references) {
                reserved += source->bytes;
            } else if (!options.prefix_cache || !source->used) {
                std::erase_if(sources, [&](const auto& entry) { return &entry == source; });
            } else if (sources.size() > retained_t5_sources) {
                (void)reclaim_source();
            }
        }
    }

    std::size_t admission_cost(const Request& request) {
        admission_pages.clear();
        admission_source = sources.end();
        auto cost = request.reservation;
        if (seq2seq && options.prefix_cache && cost) {
            admission_source = std::ranges::find(sources, request.input.tokens, &T5Source::tokens);
            if (admission_source != sources.end() && admission_source->references)
                cost -= admission_source->bytes;
        }
        if (!prefix_page_bytes || !cost)
            return cost;

        const auto tokens = std::span(request.input.tokens);
        PrefixCache::Entry* parent = nullptr;
        for (std::size_t start = 0; tokens.size() - start >= Cache::page_tokens;
             start += Cache::page_tokens) {
            auto page = prefixes.find(request.lora_key(), parent, tokens.subspan(start));
            if (!page)
                break;
            parent = &**page;
            admission_pages.push_back(*page);
            if (parent->active_references)
                cost -= prefix_page_bytes;
        }
        return cost;
    }

    void admit(std::vector<BatchEvent>& events) {
        while (!queued.empty() && active.size() < options.max_active) {
            const auto available = options.kv_budget_bytes - reserved;
            std::size_t cost = 0;
            const auto fits = [&](const Request& item) {
                cost = admission_cost(item);
                return cost <= available;
            };
            auto candidate = queued.begin();
            if (!fits(*candidate)) {
                if (candidate->bypasses >= options.max_queue_bypasses)
                    break;
                candidate = std::find_if(std::next(candidate), queued.end(), fits);
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
                request.logits.resize(model.config().vocabulary);
                if (decoder) {
                    request.cache.emplace(model.config(), request.capacity, options.session.kv_precision,
                                          &pool);
                    if (options.speculative_tokens && request.input.sampling.greedy &&
                        !request.input.constraint && request.input.tokens.size() > Speculation::ngram)
                        request.speculation.emplace(request.input.tokens, *request.sampler, request.rng,
                                                    request.input.top_logprobs);
                    reuse_prefix(request);
                } else {
                    if (admission_source == sources.end()) {
                        const auto bytes = Cache::required_bytes(model.config(), request.input.tokens.size(),
                                                                 KvPrecision::f32);
                        sources.push_back({request.input.tokens, bytes, {}});
                    } else {
                        sources.splice(sources.end(), sources, admission_source);
                    }
                    request.source = &sources.back();
                    ++request.source->references;
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
            reserved += cost;
            queued_tokens -= request.input.tokens.size();
            active.splice(active.end(), queued, candidate);
        }
    }

    void reuse_prefix(Request& request) {
        for (std::size_t index = 0; index < admission_pages.size(); ++index)
            request.cache->use_prefix_page(index, admission_pages[index]->page);
        for (auto page : admission_pages) {
            ++page->active_references;
            prefixes.touch(page);
            request.prefix = &*page;
        }
        request.position = admission_pages.size() * Cache::page_tokens;
        request.cached_prompt_tokens = request.position;
        if (options.prefix_cache && request.input.tokens.size() >= Cache::page_tokens) {
            ++prefix_lookups;
            prefix_hits += !admission_pages.empty();
            cached_prompt_tokens += request.position;
        }
        // Replay the last token for its logits while reading the already complete page.
        if (request.position == request.input.tokens.size())
            --request.position;
    }

    bool emit_token(Request& request, std::vector<BatchEvent>& events) {
        const auto logits = std::span<const float>(request.logits);
        const auto [token, complete] =
            request.input.constraint ? request.input.constraint(*request.sampler, logits, request.rng)
                                     : ConstrainedToken{request.sampler->select(logits, request.rng), false};
        auto scores = request.input.top_logprobs
                          ? std::optional(token_logprobs(logits, token, *request.input.top_logprobs))
                          : std::nullopt;
        return emit_token(request, token, complete, std::move(scores), events);
    }

    bool emit_token(Request& request, Token token, bool complete, std::optional<TokenLogprobs> scores,
                    std::vector<BatchEvent>& events) {
        const auto now = Clock::now();
        const bool eos = token == model.config().eos;
        const bool stop_id = std::ranges::find(request.input.stop_ids, token) != request.input.stop_ids.end();
        ++request.emitted;
        auto event = request.event(BatchEventKind::token, now);
        event.token = token;
        event.logprobs = std::move(scores);
        if (request.last_token)
            event.inter_token_ms = milliseconds(now - *request.last_token);
        else
            request.first_token = now;
        if (request.text)
            event.text = eos || stop_id ? std::string() : request.text->push(token);
        request.last_token = now;
        events.push_back(std::move(event));
        const bool stop = complete || stop_id || (request.text && request.text->stopped());
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
                request->prefix = prefixes.remember(request->lora_key(), request->prefix,
                                                    tokens.subspan(index * Cache::page_tokens),
                                                    request->cache->page(index));
                request->cache->use_prefix_page(index, request->prefix->page);
                if (request->prefix->active_references++)
                    reserved -= prefix_page_bytes;
            }
        }
        if (eval.mode == LogitsMode::last_token && !emit_token(*request, events))
            remove_active(request);
    }

    void finish_verification(Requests::iterator request, std::span<const Token> inputs,
                             std::vector<BatchEvent>& events) {
        request->speculation->verifying = false;
        request->draft_tokens += inputs.size() - 1;
        for (std::size_t index = 0; index < inputs.size(); ++index) {
            const auto token = request->speculation->predictions[index];
            ++request->position;
            const bool matches = index + 1 < inputs.size() && token == inputs[index + 1];
            if (matches)
                ++request->accepted_draft_tokens;
            if (!emit_token(*request, token, false, std::move(request->speculation->scores[index]), events)) {
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
            request.speculation->verifying = true;
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

    void prepare_sources(std::vector<BatchEvent>& events) {
        std::vector<T5Source*> waiting;
        std::vector<std::span<const Token>> inputs;
        const auto budget = std::max(options.context, options.step_tokens);
        std::size_t rows = 0;
        for (const auto& request : active) {
            auto* source = request.source;
            if (source->cross || std::ranges::find(waiting, source) != waiting.end())
                continue;
            if (source->tokens.size() > budget - rows)
                break;
            waiting.push_back(source);
            inputs.emplace_back(source->tokens);
            rows += source->tokens.size();
        }
        if (waiting.empty())
            return;
        try {
            auto encoded = seq2seq->prepare_sources(inputs, &pool);
            for (std::size_t i = 0; i < waiting.size(); ++i)
                waiting[i]->cross = std::move(encoded[i]);
            return;
        } catch (const Error&) {
            // A source fitting the individual workspace can exceed it as part of a batch.
        }
        for (std::size_t i = 0; i < waiting.size(); ++i) {
            auto* source = waiting[i];
            try {
                source->cross = seq2seq->prepare_sources(std::span(inputs).subspan(i, 1), &pool).front();
            } catch (const Error& error) {
                for (auto next = active.begin(); next != active.end();) {
                    const auto it = next++;
                    if (it->source != source)
                        continue;
                    const bool last = source->references == 1;
                    fail(it, error, events);
                    if (last)
                        break;
                }
            }
        }
    }

    void step_seq2seq(std::vector<BatchEvent>& events) {
        prepare_sources(events);
        scheduled.clear();
        for (auto next = active.begin(); next != active.end();) {
            const auto it = next++;
            if (!it->source->cross)
                continue;
            try {
                if (!it->seq2seq)
                    it->seq2seq.emplace(model.config(), it->capacity, it->source->cross, &pool);
                it->seq2seq->self.ensure(it->seq2seq->length + 1);
                scheduled.push_back(it);
            } catch (const Error& error) {
                fail(it, error, events);
            }
        }
        const auto ready = std::span(scheduled);
        for (std::size_t start = 0; start < ready.size(); start += decode_rows) {
            const auto tile = ready.subspan(start, std::min(decode_rows, ready.size() - start));
            t5_steps.clear();
            for (const auto it : tile)
                t5_steps.push_back({*it->seq2seq, it->pending.value_or(model.config().bos), it->logits});
            bool retry = false;
            try {
                seq2seq->step(*scratch, t5_steps);
            } catch (const Error&) {
                retry = true;
            }
            for (std::size_t i = 0; i < tile.size(); ++i) {
                const auto it = tile[i];
                try {
                    if (retry)
                        seq2seq->step(*scratch, std::span(t5_steps).subspan(i, 1));
                    if (it->seq2seq->length == 1 && options.prefix_cache) {
                        ++prefix_lookups;
                        if (std::exchange(it->source->used, true)) {
                            ++prefix_hits;
                            it->cached_prompt_tokens = it->source->tokens.size();
                            cached_prompt_tokens += it->cached_prompt_tokens;
                        }
                    }
                    if (!emit_token(*it, events))
                        remove_active(it);
                } catch (const Error& error) {
                    fail(it, error, events);
                }
            }
        }
    }

    Model model;
    BatchOptions options;
    const detail::Decoder* decoder = nullptr;
    const T5* seq2seq = nullptr;
    detail::CachePagePool pool;
    PrefixCache prefixes;
    std::list<T5Source> sources;
    std::list<T5Source>::iterator admission_source;
    std::vector<PrefixCache::Iterator> admission_pages;
    std::unique_ptr<detail::Workspace> scratch;
    Requests queued, active;
    std::vector<Evaluation> evaluations;
    std::vector<T5::Step> t5_steps;
    std::vector<Requests::iterator> scheduled;
    std::size_t reserved = 0, queued_tokens = 0, queue_bypasses = 0;
    std::size_t prefix_page_bytes = 0, decode_rows = 0;
    std::size_t prefix_lookups = 0, prefix_hits = 0, cached_prompt_tokens = 0;
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
    if (request.lora && request.lora->model_ != s.model.impl_)
        throw Error(ErrorCode::invalid_argument, "LoRA adapter belongs to a different base model");
    if (request.top_logprobs && *request.top_logprobs > max_top_logprobs)
        throw Error(ErrorCode::invalid_argument, "top logprobs must be between 0 and 20");
    if (request.constraint && (!s.decoder || !request.tokenizer || !request.stop_strings.empty()))
        throw Error(ErrorCode::invalid_argument,
                    "structured output requires a decoder tokenizer and no stop strings");
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
    const auto* lora = request.lora ? request.lora->weights_.get() : nullptr;
    s.queued.emplace_back(std::move(request), capacity, reservation, lora);
    s.queued_tokens += s.queued.back().input.tokens.size();
}

bool BatchScheduler::contains(RequestId id) const noexcept {
    const auto matches = [=](const Request& item) { return item.input.id == id; };
    return std::ranges::any_of(impl_->active, matches) || std::ranges::any_of(impl_->queued, matches);
}

bool BatchScheduler::uses_lora(const LoraAdapter& adapter) const noexcept {
    const auto matches = [&](const Request& item) { return item.lora == adapter.weights_.get(); };
    return adapter.weights_ &&
           (std::ranges::any_of(impl_->active, matches) || std::ranges::any_of(impl_->queued, matches));
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
                                   it->position < it->cached_prompt_tokens, it->lora});
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
                    requests[i]->speculation->verifying = false;
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

BatchMetrics BatchScheduler::metrics() const noexcept {
    const auto& s = *impl_;
    return {s.active.size(),     s.queued.size(),         s.queued_tokens,
            s.pool.used_bytes(), s.pool.resident_bytes(), s.options.kv_budget_bytes,
            s.prefix_lookups,    s.prefix_hits,           s.cached_prompt_tokens};
}
} // namespace nathr
