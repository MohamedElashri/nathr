#pragma once

#include "core/input.hpp"

#include <algorithm>
#include <optional>

namespace nathr::cli {

struct InferenceTiming {
    std::size_t prompt_tokens, cached_prompt_tokens, output_tokens;
    double elapsed_ms, queue_ms;
    std::optional<double> first_token_ms, last_token_ms;
};

inline detail::Json inference_stats(const InferenceTiming& timing) {
    using detail::Json;
    const auto measure = [](std::optional<double> value) { return value ? Json(*value) : Json(nullptr); };
    const auto rate = [&](std::size_t tokens, std::optional<double> ms) {
        return measure(tokens && ms && *ms > 0 ? std::optional(1000.0 * tokens / *ms) : std::nullopt);
    };
    const auto uncached = timing.prompt_tokens - std::min(timing.prompt_tokens, timing.cached_prompt_tokens);
    const auto prefill = timing.first_token_ms
                             ? std::optional(std::max(0.0, *timing.first_token_ms - timing.queue_ms))
                             : std::nullopt;
    std::optional<double> decode, tpot;
    std::size_t intervals = 0;
    if (timing.output_tokens > 1 && timing.first_token_ms && timing.last_token_ms) {
        intervals = timing.output_tokens - 1;
        decode = std::max(0.0, *timing.last_token_ms - *timing.first_token_ms);
        tpot = *decode / intervals;
    }
    return {{"prompt_tokens", timing.prompt_tokens},
            {"cached_prompt_tokens", timing.cached_prompt_tokens},
            {"output_tokens", timing.output_tokens},
            {"elapsed_ms", timing.elapsed_ms},
            {"queue_ms", timing.queue_ms},
            {"ttft_ms", measure(timing.first_token_ms)},
            {"prefill_ms", measure(prefill)},
            {"decode_ms", measure(decode)},
            {"tpot_ms", measure(tpot)},
            {"prefill_tokens_s", rate(uncached, prefill)},
            {"decode_tokens_s", rate(intervals, decode)},
            {"output_tokens_s", rate(timing.output_tokens, timing.elapsed_ms)}};
}

} // namespace nathr::cli
