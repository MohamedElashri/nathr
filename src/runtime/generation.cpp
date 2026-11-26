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
    if (logits.size() != vocabulary_ ||
        !std::ranges::all_of(logits, [](float value) { return std::isfinite(value); })) {
        throw Error(ErrorCode::invalid_argument, "sampler requires vocabulary-sized finite logits");
    }
    if (options_.greedy)
        return static_cast<Token>(std::ranges::max_element(logits) - logits.begin());
    for (std::size_t i = 0; i < logits.size(); ++i)
        candidates_[i] = {static_cast<Token>(i), logits[i]};
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

GenerationResult generate(Session& session, Sampler& sampler, Token eos, std::size_t max_new_tokens,
                          const TokenSink& emit) {
    GenerationResult result;
    if (!max_new_tokens)
        return result;
    if (eos >= session.logits().size() || !emit ||
        max_new_tokens - 1 > session.capacity() - session.history().size()) {
        throw Error(ErrorCode::invalid_argument, "invalid EOS, token sink or generation budget");
    }
    while (result.emitted < max_new_tokens) {
        const auto token = session.sample(sampler);
        emit(token);
        ++result.emitted;
        result.stopped_at_eos = token == eos;
        if (result.stopped_at_eos || result.emitted == max_new_tokens) {
            result.pending = token;
            break;
        }
        session.append(std::span(&token, 1));
        ++result.consumed;
    }
    return result;
}

} // namespace nathr
