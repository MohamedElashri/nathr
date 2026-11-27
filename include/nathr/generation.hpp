#pragma once

#include <nathr/nathr.hpp>
#include <nathr/tokenizer.hpp>

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
    Token select_masked(std::span<const float> logits, std::mt19937_64& generator,
                        std::span<const std::uint8_t> allowed);

  private:
    struct Candidate {
        Token id;
        double value;
    };
    SamplingOptions options_;
    std::size_t vocabulary_;
    std::vector<Candidate> candidates_;
};

inline constexpr std::size_t max_top_logprobs = 20;

struct TokenLogprob {
    Token id;
    double logprob;
};

struct TokenLogprobs {
    TokenLogprob chosen;
    std::vector<TokenLogprob> alternatives;
};

// Scores use the model distribution before sampling transforms.
[[nodiscard]] TokenLogprobs token_logprobs(std::span<const float> logits, Token chosen,
                                           std::size_t top_count);
struct ConstrainedToken {
    Token id;
    bool complete;
};

// Advances the constraint state for one request. Complete stops generation after this token.
using TokenConstraint = std::function<ConstrainedToken(Sampler&, std::span<const float>, std::mt19937_64&)>;

struct GenerationResult {
    std::size_t emitted{}, consumed{};
    bool stopped_at_eos{};
    // EOS and the final budget token are emitted without being appended.
    std::optional<Token> pending;
    bool stopped_at_stop{};
};

// Holds back any suffix that might be the start of a stop string.
class GenerationText {
  public:
    explicit GenerationText(Tokenizer tokenizer, std::vector<std::string> stops = {});
    [[nodiscard]] std::string push(Token token);
    [[nodiscard]] std::string finish();
    [[nodiscard]] bool stopped() const noexcept { return stopped_; }

  private:
    std::string push_text(std::string text);
    Tokenizer tokenizer_;
    std::vector<std::string> stops_;
    Utf8Stream utf8_;
    std::string pending_;
    bool first_ = true, stopped_ = false;
};

using TokenSink = std::function<void(Token)>;
// The second argument identifies a terminating token ID. Return false to stop on text.
using GenerationSink = std::function<bool(Token, bool)>;
GenerationResult generate_until(Session& session, Sampler& sampler, Token eos,
                                std::span<const Token> stop_ids, std::size_t max_new_tokens,
                                const GenerationSink& emit);
GenerationResult generate_until(T5Session& session, Sampler& sampler, Token eos,
                                std::span<const Token> stop_ids, std::size_t max_new_tokens,
                                const GenerationSink& emit);
GenerationResult generate(Session& session, Sampler& sampler, Token eos, std::size_t max_new_tokens,
                          const TokenSink& emit);
GenerationResult generate(T5Session& session, Sampler& sampler, Token eos, std::size_t max_new_tokens,
                          const TokenSink& emit);

} // namespace nathr
