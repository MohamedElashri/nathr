#pragma once

#include "core/input.hpp"

#include <nathr/generation.hpp>

namespace nathr::cli {

class StructuredVocabulary;

std::shared_ptr<const StructuredVocabulary> structured_vocabulary(const Tokenizer& tokenizer);
TokenConstraint structured_output(const detail::Json& schema,
                                  std::shared_ptr<const StructuredVocabulary> vocabulary,
                                  std::vector<Token> stop_ids, Token eos);

} // namespace nathr::cli
