#include "models/lora.hpp"

#include "backend/cpu/scalar.hpp"
#include "formats/safetensors.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>

namespace nathr::detail {
namespace {

constexpr std::size_t max_rank = 256;

std::uint64_t next_cache_key() {
    static std::atomic<std::uint64_t> next{1};
    return next.fetch_add(1, std::memory_order_relaxed);
}

std::size_t rank_value(const Json& value, const std::string& context) {
    const auto rank = natural(value, context);
    if (!rank || rank > max_rank)
        throw Error(ErrorCode::unsupported, context + ": LoRA rank must be between 1 and 256");
    return rank;
}

double alpha_value(const Json& value, const std::string& context) {
    const auto alpha = value.get<double>();
    if (!value.is_number() || !std::isfinite(alpha) || alpha < 0 || alpha > std::numeric_limits<float>::max())
        throw Error(ErrorCode::malformed, context + ": LoRA alpha must be finite and nonnegative");
    return alpha;
}

template <class T, class Parse>
std::map<std::string, T> patterns(const Json& config, std::string_view field, Parse parse,
                                  const std::string& context) {
    std::map<std::string, T> values;
    const auto entry = config.find(field);
    if (entry == config.end())
        return values;
    if (!entry->is_object() || entry->size() > 512)
        throw Error(ErrorCode::malformed, context + ": invalid " + std::string(field));
    for (const auto& [key, value] : entry->items()) {
        auto name = std::string_view(key);
        if (name.starts_with('^'))
            name.remove_prefix(1);
        if (name.empty() || !std::ranges::all_of(name, [](char ch) {
                return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') ||
                       ch == '.' || ch == '_';
            }))
            throw Error(ErrorCode::unsupported, context + ": LoRA patterns require dotted module names");
        values.emplace(key, parse(value, context));
    }
    return values;
}

template <class T>
T scale_setting(std::string_view module, const std::map<std::string, T>& values, T fallback,
                const std::string& context) {
    std::optional<T> matched;
    for (const auto& [pattern, value] : values) {
        const bool matches =
            pattern.starts_with('^')
                ? module == std::string_view(pattern).substr(1)
                : (module == pattern || (module.size() > pattern.size() && module.ends_with(pattern) &&
                                         module[module.size() - pattern.size() - 1] == '.'));
        if (matches) {
            if (matched && *matched != value)
                throw Error(ErrorCode::unsupported,
                            context + ": conflicting LoRA patterns for " + std::string(module));
            matched = value;
        }
    }
    return matched.value_or(fallback);
}

std::vector<float> matrix(TensorView view, const std::string& name) {
    const auto type = view.desc.dtype;
    if (type != DType::f32 && type != DType::f16 && type != DType::bf16)
        throw Error(ErrorCode::unsupported, name + ": LoRA weights require F32, F16, or BF16");
    std::vector<float> values(view.desc.elements);
    if (type == DType::f32) {
        std::memcpy(values.data(), view.bytes.data(), view.bytes.size());
    } else {
        const auto decode = type == DType::f16 ? half_to_float : bf16_to_float;
        for (std::size_t i = 0; i < values.size(); ++i) {
            std::uint16_t bits;
            std::memcpy(&bits, view.bytes.data() + i * sizeof(bits), sizeof(bits));
            values[i] = decode(bits);
        }
    }
    if (!std::ranges::all_of(values, [](float value) { return std::isfinite(value); }))
        throw Error(ErrorCode::malformed, name + ": nonfinite LoRA weight");
    return values;
}

double dot(std::span<const float> weights, std::span<const float> input) {
    double value = 0;
    for (std::size_t i = 0; i < input.size(); ++i)
        value += static_cast<double>(weights[i]) * input[i];
    return value;
}

} // namespace

LoraWeights::LoraWeights(const std::filesystem::path& directory, std::span<const LoraTarget> targets,
                         std::size_t byte_limit)
    : cache_key_(next_cache_key()) {
    if (targets.empty())
        throw Error(ErrorCode::unsupported, "this model does not support LoRA");
    const auto config_path = directory / "adapter_config.json";
    const auto weights_path = directory / "adapter_model.safetensors";
    for (const auto& path : {config_path, weights_path}) {
        if (!std::filesystem::is_regular_file(path))
            throw Error(ErrorCode::io, path.string() + ": expected a regular file");
    }
    const auto context = config_path.string();
    const auto config = read_json(config_path, 64 * 1024);
    try {
        if (config.value("peft_type", std::string{}) != "LORA" ||
            config.value("bias", std::string("none")) != "none" || config.value("use_dora", false) ||
            config.value("use_qalora", false))
            throw Error(ErrorCode::unsupported,
                        context + ": expected a PEFT LoRA adapter without base bias changes");
        for (const auto field :
             {"modules_to_save", "layer_replication", "target_parameters", "trainable_token_indices",
              "alora_invocation_tokens", "megatron_config", "arrow_config", "monteclora_config"}) {
            const auto value = config.find(field);
            if (value != config.end() && !value->is_null() && !value->empty())
                throw Error(ErrorCode::unsupported, context + ": unsupported LoRA field " + field);
        }
        if (const auto bd = config.find("use_bdlora"); bd != config.end() && !bd->is_null() && *bd != false)
            throw Error(ErrorCode::unsupported, context + ": unsupported LoRA field use_bdlora");
        if (const auto initial = config.find("init_lora_weights");
            initial != config.end() && !initial->is_boolean()) {
            constexpr std::string_view supported[] = {"gaussian", "eva", "orthogonal", "nora"};
            if (!initial->is_string() ||
                std::ranges::find(supported, initial->get_ref<const std::string&>()) == std::end(supported))
                throw Error(ErrorCode::unsupported,
                            context + ": export this initialization as standard LoRA");
        }
        const auto rank = rank_value(config.at("r"), context);
        const auto alpha = alpha_value(config.at("lora_alpha"), context);
        const auto ranks = patterns<std::size_t>(config, "rank_pattern", rank_value, context);
        const auto alphas = patterns<double>(config, "alpha_pattern", alpha_value, context);
        const auto rslora = config.value("use_rslora", false);
        const auto lora_bias = config.value("lora_bias", false);
        const auto limit = std::min(byte_limit, LoraAdapter::max_bytes);
        SafeTensors file(weights_path, sum(limit, mib + 8, "LoRA file"));
        for (const auto& target : targets) {
            const auto a_suffix = target.embedding ? ".lora_embedding_A" : ".lora_A.weight";
            const auto b_suffix = target.embedding ? ".lora_embedding_B" : ".lora_B.weight";
            const auto has_pair = [&](const std::string& module) {
                return file.contains(module + a_suffix) || file.contains(module + b_suffix);
            };
            auto module = "base_model.model." + target.name;
            if (!has_pair(module)) {
                module = target.name;
                if (!has_pair(module))
                    continue;
            }
            const auto r = scale_setting(target.name, ranks, rank, context);
            const auto a = scale_setting(target.name, alphas, alpha, context);
            const auto [out, in] = target.embedding
                                       ? std::pair(target.weight.desc.shape[1], target.weight.desc.shape[0])
                                       : std::pair(target.weight.desc.shape[0], target.weight.desc.shape[1]);
            const bool adapted_bias = lora_bias && !target.embedding;
            const auto elements = sum(product({sum(in, out, "LoRA dimensions"), r}, "LoRA dimensions"),
                                      adapted_bias ? out : 0, "LoRA dimensions");
            bytes_ = sum(bytes_, product({elements, sizeof(float)}, "LoRA adapter"), "LoRA adapter");
            if (bytes_ > limit)
                throw Error(ErrorCode::resource_limit,
                            context + ": decoded LoRA weights exceed the adapter budget");
            const auto a_name = module + a_suffix;
            const auto b_name = module + b_suffix;
            Matrix update{matrix(file.take(a_name, {r, in}), a_name),
                          matrix(file.take(b_name, {out, r}), b_name),
                          {},
                          r,
                          a / (rslora ? std::sqrt(static_cast<double>(r)) : static_cast<double>(r))};
            if (adapted_bias) {
                const auto name = module + ".lora_B.bias";
                update.bias = matrix(file.take(name, {out}), name);
            }
            auto& updates = target.embedding ? embeddings_ : projections_;
            updates.emplace(target.weight.bytes.data(), std::move(update));
        }
        if (const auto name = file.first_unbound(); !name.empty())
            throw Error(ErrorCode::unsupported, context + ": unsupported LoRA tensor " + std::string(name));
        if (!bytes_)
            throw Error(ErrorCode::malformed, context + ": adapter has no weights for this model");
    } catch (const Json::exception& error) {
        throw Error(ErrorCode::malformed, context + ": " + error.what());
    }
}

void LoraWeights::Matrix::add(std::span<const double> low, std::span<float> result) const {
    for (std::size_t i = 0; i < result.size(); ++i) {
        double value = bias.empty() ? 0 : bias[i];
        for (std::size_t j = 0; j < rank; ++j)
            value += low[j] * b[i * rank + j];
        result[i] += static_cast<float>(scale * value);
    }
}

void LoraWeights::project(TensorView weight, std::span<const float> input, std::span<float> output,
                          std::size_t rows) const {
    const auto found = projections_.find(weight.bytes.data());
    if (found == projections_.end())
        return;
    const auto& update = found->second;
    const auto in = weight.desc.shape[1], out = weight.desc.shape[0];
    std::array<double, max_rank> low;
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < update.rank; ++j)
            low[j] = dot(std::span(update.a).subspan(j * in, in), input.subspan(i * in, in));
        update.add(std::span(low).first(update.rank), output.subspan(i * out, out));
    }
}

void LoraWeights::embed(TensorView weight, Token token, std::span<float> output) const {
    const auto found = embeddings_.find(weight.bytes.data());
    if (found == embeddings_.end())
        return;
    const auto& update = found->second;
    std::array<double, max_rank> low;
    for (std::size_t j = 0; j < update.rank; ++j)
        low[j] = update.a[j * weight.desc.shape[0] + token];
    update.add(std::span(low).first(update.rank), output);
}

} // namespace nathr::detail
