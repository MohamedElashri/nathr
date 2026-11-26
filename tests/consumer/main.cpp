#include <nathr/generation.hpp>
#include <nathr/nathr.hpp>
#include <nathr/tokenizer.hpp>

#include <algorithm>
#include <array>
#include <cmath>

int main(int argc, char** argv) {
    if (argc != 2)
        return 1;
    const auto fixtures = std::filesystem::path(argv[1]);
    const auto model = nathr::Model::load(fixtures / "f32", {nathr::ExecutionPolicy::scalar, 2});
    auto session = nathr::Session::create(model, 8, 0, {nathr::KvPrecision::f16, 3});
    session.prefill(std::array<nathr::Token, 3>{1, 7, 2});
    const auto tokenizer = nathr::Tokenizer::load(fixtures / "tokenizer.json");
    const auto logits = session.logits();
    return model.threads() == 2 && model.config().vocabulary == logits.size() &&
                   std::ranges::all_of(logits, [](float x) { return std::isfinite(x); }) &&
                   tokenizer.decode(tokenizer.encode("ab")) == "ab"
               ? 0
               : 1;
}
