#pragma once

#include <nathr/nathr.hpp>

#include <optional>
#include <random>
#include <vector>

namespace nathr {

struct SamplingOptions {
    bool greedy = true;
    double temperature = 1.0;
    std::size_t top_k = 0;
    double top_p = 1.0;
};

class Sampler {
  public:
    Sampler(SamplingOptions options, std::size_t vocabulary);
    Token select(std::span<const float> logits, std::mt19937_64& generator);

  private:
    struct Candidate {
        Token id;
        double value;
    };
    SamplingOptions options_;
    std::size_t vocabulary_;
    std::vector<Candidate> candidates_;
};

struct GenerationResult {
    std::size_t emitted{}, consumed{};
    bool stopped_at_eos{};
    // EOS and the final budget token are emitted without being appended.
    std::optional<Token> pending;
};

using TokenSink = std::function<void(Token)>;
GenerationResult generate(Session& session, Sampler& sampler, Token eos, std::size_t max_new_tokens,
                          const TokenSink& emit);

} // namespace nathr
