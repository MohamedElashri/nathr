// Phase 0 tokenizer and timing probe for the pinned external llama.cpp API.
#include <llama.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
constexpr int prompt_length = 128;
constexpr int decode_length = 32;
constexpr int context = 512;
constexpr int warmups = 2;
constexpr int repetitions = 7;
using clock_type = std::chrono::steady_clock;

llama_token fixed_token(int position) {
    return static_cast<llama_token>(17 + (position * 7919 + 12345) % (49152 - 17));
}

void decode(llama_context * ctx, llama_batch batch) {
    if (llama_decode(ctx, batch) != 0 || !llama_get_logits_ith(ctx, -1)) {
        throw std::runtime_error("decode failed");
    }
}

void benchmark(llama_model * model, int threads) {
    auto params = llama_context_default_params();
    params.n_ctx = context;
    params.n_batch = prompt_length;
    params.n_ubatch = prompt_length;
    params.n_outputs_max = 1;
    params.n_threads = threads;
    params.n_threads_batch = threads;
    params.type_k = params.type_v = GGML_TYPE_F32;
    params.offload_kqv = false;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    params.no_perf = true;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model, params), llama_free);
    if (!ctx || llama_n_ctx(ctx.get()) != context) {
        throw std::runtime_error("context setup failed");
    }
    std::array<llama_token, prompt_length> prompt{};
    for (int i = 0; i < prompt_length; ++i) {
        prompt[i] = fixed_token(i);
    }
    std::cout << "{\"prompt_tokens\":" << prompt_length << ",\"decode_tokens\":" << decode_length
              << ",\"context\":" << context << ",\"threads\":" << threads
              << ",\"warmups\":" << warmups << ",\"prefill_and_decode_ms\":[";
    for (int repetition = -warmups; repetition < repetitions; ++repetition) {
        llama_memory_clear(llama_get_memory(ctx.get()), false);
        const auto start = clock_type::now();
        decode(ctx.get(), llama_batch_get_one(prompt.data(), static_cast<int>(prompt.size())));
        const auto prefetched = clock_type::now();
        for (int i = 0; i < decode_length; ++i) {
            auto token = fixed_token(prompt_length + i);
            decode(ctx.get(), llama_batch_get_one(&token, 1));
        }
        const auto finished = clock_type::now();
        if (repetition >= 0) {
            if (repetition > 0) {
                std::cout << ',';
            }
            using ms = std::chrono::duration<double, std::milli>;
            std::cout << '[' << ms(prefetched - start).count() << ',' << ms(finished - prefetched).count() << ']';
        }
    }
    std::cout << "]}\n";
}

void tokenize(const llama_vocab * vocab, std::string_view text) {
    std::vector<llama_token> ids(text.size() + 2);
    const int count = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()),
                                     ids.data(), static_cast<int32_t>(ids.size()), false, true);
    if (count < 0) {
        throw std::runtime_error("token buffer exhausted");
    }
    std::cout << '[';
    for (int i = 0; i < count; ++i) {
        if (i > 0) {
            std::cout << ',';
        }
        std::cout << ids[i];
    }
    std::cout << "]\n";
}

void forward(llama_model * model, int argc, char ** argv) {
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    auto params = llama_context_default_params();
    params.n_ctx = 128;
    params.n_batch = 128;
    params.n_ubatch = 128;
    params.n_outputs_max = 1;
    params.n_threads = params.n_threads_batch = 1;
    params.type_k = params.type_v = GGML_TYPE_F32;
    params.offload_kqv = false;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    params.no_perf = true;
    if (argc > 128)
        throw std::runtime_error("too many forward tokens");
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model, params), llama_free);
    if (!ctx)
        throw std::runtime_error("context setup failed");
    std::cout << '[';
    for (int i = 0; i < argc; ++i) {
        std::size_t used = 0;
        auto token = std::stoi(argv[i], &used);
        if (used != std::string_view(argv[i]).size() || token < 0 || token >= vocab)
            throw std::runtime_error("invalid forward token ID");
        decode(ctx.get(), llama_batch_get_one(&token, 1));
        const auto* logits = llama_get_logits_ith(ctx.get(), -1);
        if (i)
            std::cout << ',';
        std::cout << std::max_element(logits, logits + vocab) - logits;
    }
    std::cout << "]\n";
}
}  // namespace

int main(int argc, char ** argv) try {
    if (argc < 4) {
        throw std::runtime_error("usage: llama-probe bench MODEL THREADS | tokenize MODEL TEXT... | forward MODEL IDS...");
    }
    const std::string_view command(argv[1]);
    if (command != "bench" && command != "tokenize" && command != "forward") {
        throw std::runtime_error("unknown command");
    }
    llama_backend_init();
    auto params = llama_model_default_params();
    params.vocab_only = command == "tokenize";
    params.n_gpu_layers = 0;
    const auto start = clock_type::now();
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(argv[2], params), llama_model_free);
    if (!model) {
        throw std::runtime_error("model setup failed");
    }
    std::cerr << "load_ms=" << std::chrono::duration<double, std::milli>(clock_type::now() - start).count()
              << " ggml=" << ggml_version() << '@' << ggml_commit() << '\n';
    if (command == "bench") {
        const int threads = std::stoi(argv[3]);
        if (threads < 1) {
            throw std::runtime_error("threads must be positive");
        }
        benchmark(model.get(), threads);
    } else if (command == "forward") {
        forward(model.get(), argc - 3, argv + 3);
    } else {
        for (int i = 3; i < argc; ++i) {
            tokenize(llama_model_get_vocab(model.get()), argv[i]);
        }
    }
    return 0;
} catch (const std::exception & error) {
    std::cerr << error.what() << '\n';
    return 1;
}
