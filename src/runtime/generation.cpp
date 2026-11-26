#include <nathr/generation.hpp>

#include "core/limits.hpp"

#include <algorithm>
#include <cmath>

namespace nathr {

Sampler::Sampler(SamplingOptions options, std::size_t vocabulary)
    : options_(options), vocabulary_(vocabulary) {
    if (!vocabulary || vocabulary > detail::max_vocabulary || options.top_k > vocabulary ||
        !std::isfinite(options.temperature) || options.temperature <= 0 || !std::isfinite(options.top_p) ||
        options.top_p <= 0 || options.top_p > 1) {
        throw Error(ErrorCode::invalid_argument, "invalid sampling parameters");
    }
    if (!options.greedy)
        candidates_.resize(vocabulary);
}

Token Sampler::select(std::span<const float> logits, std::mt19937_64& generator) {
    constexpr auto invalid_logits = "sampler requires vocabulary-sized finite logits";
    if (logits.size() != vocabulary_)
        throw Error(ErrorCode::invalid_argument, invalid_logits);
    if (options_.greedy) {
        Token best = 0;
        for (Token i = 0; i < logits.size(); ++i) {
            if (!std::isfinite(logits[i]))
                throw Error(ErrorCode::invalid_argument, invalid_logits);
            if (logits[i] > logits[best])
                best = i;
        }
        return best;
    }
    for (Token i = 0; i < logits.size(); ++i) {
        if (!std::isfinite(logits[i]))
            throw Error(ErrorCode::invalid_argument, invalid_logits);
        candidates_[i] = {i, logits[i]};
    }
    auto allowed = std::span(candidates_);
    if (options_.top_k)
        allowed = allowed.first(options_.top_k);
    std::partial_sort(candidates_.begin(), candidates_.begin() + static_cast<std::ptrdiff_t>(allowed.size()),
                      candidates_.end(), [](const auto& a, const auto& b) {
                          return a.value == b.value ? a.id < b.id : a.value > b.value;
                      });
    const auto maximum = allowed.front().value;
    double sum = 0.0;
    for (auto& candidate : allowed) {
        candidate.value = std::exp((candidate.value - maximum) / options_.temperature);
        sum += candidate.value;
    }
    const auto threshold = options_.top_p * sum;
    double retained = 0.0;
    std::size_t count = 0;
    do {
        retained += allowed[count++].value;
    } while (count < allowed.size() && retained < threshold);
    const double uniform = static_cast<double>(generator() >> 11) * 0x1.0p-53;
    const auto draw = uniform * retained;
    double cumulative = 0.0;
    for (const auto& candidate : allowed.first(count)) {
        cumulative += candidate.value;
        if (draw < cumulative)
            return candidate.id;
    }
    return allowed[count - 1].id;
}

namespace {
GenerationResult generate_tokens(auto& session, std::size_t length, Sampler& sampler, Token eos,
                                 std::span<const Token> stop_ids, std::size_t max_new_tokens,
                                 const GenerationSink& emit) {
    GenerationResult result;
    if (!max_new_tokens)
        return result;
    const auto vocabulary = session.logits().size();
    if (eos >= vocabulary || !emit || max_new_tokens - 1 > session.capacity() - length ||
        std::ranges::any_of(stop_ids, [&](Token id) { return id >= vocabulary; })) {
        throw Error(ErrorCode::invalid_argument, "invalid EOS, token sink or generation budget");
    }
    while (result.emitted < max_new_tokens) {
        const auto token = session.sample(sampler);
        const bool eos_token = token == eos;
        const bool stop_id = std::ranges::find(stop_ids, token) != stop_ids.end();
        const bool keep_going = emit(token, eos_token || stop_id);
        ++result.emitted;
        result.stopped_at_eos = eos_token;
        result.stopped_at_stop = !eos_token && (stop_id || !keep_going);
        if (result.stopped_at_eos || result.stopped_at_stop || result.emitted == max_new_tokens) {
            result.pending = token;
            break;
        }
        session.append(std::span(&token, 1));
        ++result.consumed;
    }
    return result;
}
} // namespace

GenerationText::GenerationText(Tokenizer tokenizer, std::vector<std::string> stops)
    : tokenizer_(std::move(tokenizer)), stops_(std::move(stops)) {
    if (stops_.size() > 16)
        throw Error(ErrorCode::resource_limit, "at most 16 stop strings are supported");
    for (const auto& stop : stops_) {
        Utf8Stream check;
        if (stop.empty() || stop.size() > 256 || check.push(stop) != stop || !check.finish().empty())
            throw Error(ErrorCode::invalid_argument,
                        "stop strings must be nonempty UTF-8 of at most 256 bytes");
    }
}

std::string GenerationText::push_text(std::string text) {
    if (stopped_)
        return {};
    if (stops_.empty())
        return text;
    pending_.append(text);
    auto end = std::string::npos;
    std::size_t hold = 0;
    for (const auto& stop : stops_) {
        end = std::min(end, pending_.find(stop));
        auto size = std::min(stop.size() - 1, pending_.size());
        while (size > hold && !std::string_view(pending_).ends_with(std::string_view(stop).substr(0, size)))
            --size;
        hold = std::max(hold, size);
    }
    if (end != std::string::npos) {
        stopped_ = true;
        auto output = pending_.substr(0, end);
        pending_.clear();
        return output;
    }
    const auto bytes = pending_.size() - hold;
    auto output = pending_.substr(0, bytes);
    pending_.erase(0, bytes);
    return output;
}

std::string GenerationText::push(Token token) {
    if (stopped_ || token >= tokenizer_.vocabulary())
        return {};
    const auto bytes = tokenizer_.decode_piece(token, first_, true);
    first_ = first_ && bytes.empty() && tokenizer_.token_bytes(token, true).empty();
    if (bytes.empty())
        return {};
    return push_text(utf8_.push(bytes));
}

std::string GenerationText::finish() {
    auto text = push_text(utf8_.finish());
    text += std::exchange(pending_, {});
    return text;
}

GenerationResult generate_until(Session& session, Sampler& sampler, Token eos,
                                std::span<const Token> stop_ids, std::size_t max_new_tokens,
                                const GenerationSink& emit) {
    return generate_tokens(session, session.history().size(), sampler, eos, stop_ids, max_new_tokens, emit);
}

GenerationResult generate_until(T5Session& session, Sampler& sampler, Token eos,
                                std::span<const Token> stop_ids, std::size_t max_new_tokens,
                                const GenerationSink& emit) {
    return generate_tokens(session, session.length(), sampler, eos, stop_ids, max_new_tokens, emit);
}

GenerationResult generate(Session& session, Sampler& sampler, Token eos, std::size_t max_new_tokens,
                          const TokenSink& emit) {
    if (max_new_tokens && !emit)
        throw Error(ErrorCode::invalid_argument, "missing token sink");
    return generate_until(session, sampler, eos, {}, max_new_tokens, [&](Token token, bool) {
        emit(token);
        return true;
    });
}

GenerationResult generate(T5Session& session, Sampler& sampler, Token eos, std::size_t max_new_tokens,
                          const TokenSink& emit) {
    if (max_new_tokens && !emit)
        throw Error(ErrorCode::invalid_argument, "missing token sink");
    return generate_until(session, sampler, eos, {}, max_new_tokens, [&](Token token, bool) {
        emit(token);
        return true;
    });
}

} // namespace nathr
