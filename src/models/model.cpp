#include "models/model.hpp"

#include "models/gpt2/model.hpp"
#include "models/llama/config.hpp"
#include "models/llama/model.hpp"

#include <algorithm>
#include <cmath>
#include <system_error>

namespace nathr::detail {

CpuBackend::CpuBackend(BackendOptions options, CpuFeatures features)
    : options_(options), features_(features), pool_(options.threads) {}

TensorView CpuBackend::bind(TensorView view, const std::filesystem::path& path, std::string_view name) {
    const auto type = view.desc.dtype;
    const auto format = encoding(type);
    auto& kernel = kernels_[static_cast<std::size_t>(type)];
    if (!kernel.dot)
        kernel = select_kernel(type, options_.policy, features_);
    if (!kernel.dot || (view.desc.rank == 1 && format->elements != 1)) {
        throw Error(ErrorCode::unsupported, path.string() + ": no execution kernel for " + std::string(name) +
                                                " (" + std::string(format->name) + ")");
    }
    return view;
}

void CpuBackend::project(TensorView weight, std::span<const float> input, std::span<float> output,
                         std::size_t tokens, std::span<float> packed) const {
    detail::project(kernels_[static_cast<std::size_t>(weight.desc.dtype)], pool_, weight, input, output,
                    tokens, packed);
}

std::string_view CpuBackend::kernel(std::uint32_t id) const noexcept {
    return id < kernels_.size() ? kernels_[id].name : std::string_view{};
}

std::string_view CpuBackend::name() const noexcept {
    return options_.policy != ExecutionPolicy::scalar && features_.avx2 ? "avx2" : "scalar";
}

void Workspace::allocate(std::initializer_list<std::pair<std::span<float>*, std::size_t>> buffers) {
    std::size_t count = 0;
    for (const auto& buffer : buffers)
        count = sum(count, buffer.second, "workspace");
    if (count > max_workspace_bytes / sizeof(float))
        throw Error(ErrorCode::resource_limit, "workspace exceeds 32 MiB");
    storage_.resize(count);
    auto remaining = std::span(storage_);
    for (const auto& [buffer, width] : buffers) {
        *buffer = remaining.first(width);
        remaining = remaining.subspan(width);
    }
}

std::size_t Workspace::bytes() const noexcept {
    return storage_.capacity() * sizeof(float);
}

bool Evaluation::wants_logits(std::size_t index) const {
    return mode == LogitsMode::all_tokens || (mode == LogitsMode::last_token && index + 1 == tokens.size());
}

void Evaluation::record(std::string_view name, std::size_t layer, std::size_t index,
                        std::span<const float> values) const {
    if (trace)
        trace({name, layer, start + index, values});
}

void Evaluation::publish_logits(std::size_t index, std::span<const float> values) const {
    if (!std::ranges::all_of(values, [](float value) { return std::isfinite(value); }))
        throw Error(ErrorCode::backend, "output projection produced non-finite logits");
    record("logits", TraceEvent::model_level, index, values);
    if (mode == LogitsMode::all_tokens)
        sink(start + index, values);
}

void validate_tokens(std::span<const Token> tokens, const ModelConfig& config, std::size_t limit) {
    if (tokens.empty())
        throw Error(ErrorCode::invalid_argument, "token span must be nonempty");
    if (tokens.size() > limit)
        throw Error(ErrorCode::invalid_argument, "token span exceeds capacity");
    if (std::ranges::any_of(tokens, [&](Token token) { return token >= config.vocabulary; }))
        throw Error(ErrorCode::invalid_argument, "token ID exceeds model vocabulary");
}

void validate_mode(LogitsMode mode, const LogitsSink& sink) {
    if ((mode != LogitsMode::all_tokens && mode != LogitsMode::last_token && mode != LogitsMode::none) ||
        ((mode == LogitsMode::all_tokens) != static_cast<bool>(sink))) {
        throw Error(ErrorCode::invalid_argument,
                    "all-token logits mode requires a sink, other modes take no sink");
    }
}

} // namespace nathr::detail

namespace nathr {
using namespace detail;

namespace {
std::filesystem::path weights_path(const std::filesystem::path& directory) {
    std::error_code error;
    if (std::filesystem::exists(directory / "model.safetensors.index.json", error))
        throw Error(ErrorCode::unsupported, directory.string() + ": sharded Safetensors");
    if (error)
        throw Error(ErrorCode::io, directory.string() + ": " + error.message());
    return directory / "model.safetensors";
}
} // namespace

Model::Impl::Impl(const std::filesystem::path& path, Checkpoint checkpoint, ModelConfig native_config,
                  BackendOptions options, CpuFeatures features)
    : config(native_config), tensors(checkpoint == Checkpoint::llama_gguf
                                         ? WeightFile(std::in_place_type<Gguf>, path)
                                         : WeightFile(std::in_place_type<SafeTensors>, weights_path(path))),
      backend(options, features),
      architecture(checkpoint == Checkpoint::gpt2_safetensors ? "gpt2" : "llama") {
    switch (checkpoint) {
    case Checkpoint::llama_gguf: {
        auto& file = std::get<Gguf>(tensors);
        config = load_llama_config(file);
        decoder = load_llama(file, path, config, backend);
        break;
    }
    case Checkpoint::llama_safetensors:
        decoder = load_llama(std::get<SafeTensors>(tensors), path, config, backend);
        break;
    case Checkpoint::gpt2_safetensors:
        decoder = load_gpt2(std::get<SafeTensors>(tensors), path, config, backend);
        break;
    }
}

Model Model::load(const std::filesystem::path& path, BackendOptions options) {
    validate_thread_count(options.threads);
    if (options.policy != ExecutionPolicy::scalar && options.policy != ExecutionPolicy::automatic &&
        options.policy != ExecutionPolicy::avx2)
        throw Error(ErrorCode::unsupported, "unknown execution policy");
    const auto features = cpu_features();
    if (options.policy == ExecutionPolicy::avx2 && !features.avx2)
        throw Error(ErrorCode::unsupported, "AVX2 was requested but CPU or OS AVX2 state is unavailable");
    if (!is_model_directory(path))
        return Model(std::make_shared<Impl>(path, Checkpoint::llama_gguf, ModelConfig{}, options, features));
    const auto config_path = path / "config.json";
    const auto json = read_json(config_path, mib);
    const auto type = json.find("model_type");
    const bool gpt2 = type != json.end() && *type == "gpt2";
    const auto config = gpt2 ? gpt2_config(json, config_path) : load_llama_config(json, config_path);
    const auto checkpoint = gpt2 ? Checkpoint::gpt2_safetensors : Checkpoint::llama_safetensors;
    return Model(std::make_shared<Impl>(path, checkpoint, config, options, features));
}

const ModelConfig& Model::config() const noexcept {
    return impl_->config;
}

ModelMemory Model::memory() const noexcept {
    return std::visit([](const auto& file) { return file.memory(); }, impl_->tensors);
}

std::string_view Model::architecture() const noexcept {
    return impl_->architecture;
}

std::string_view Model::backend() const noexcept {
    return impl_->backend.name();
}

unsigned Model::threads() const noexcept {
    return impl_->backend.threads();
}

std::string_view Model::kernel(std::uint32_t id) const noexcept {
    return impl_->backend.kernel(id);
}

void Model::forward(std::span<const Token> tokens, LogitsMode mode, std::span<float> last_logits,
                    const LogitsSink& sink, const TraceSink& trace) const {
    validate_tokens(tokens, config(), std::min(config().context, max_uncached_tokens));
    validate_mode(mode, sink);
    if ((mode == LogitsMode::last_token && last_logits.size() != config().vocabulary) ||
        (mode != LogitsMode::last_token && !last_logits.empty())) {
        throw Error(ErrorCode::invalid_argument, "last-token mode requires a vocabulary-sized output span");
    }
    auto scratch = impl_->decoder->workspace(tokens.size(), tokens.size());
    impl_->decoder->execute(*scratch, {tokens, 0, nullptr, mode, sink, trace});
    if (mode == LogitsMode::last_token)
        std::ranges::copy(scratch->logits, last_logits.begin());
}

} // namespace nathr
