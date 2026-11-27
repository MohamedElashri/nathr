#include "models/t5/model.hpp"

#include "models/model.hpp"

#include <nathr/generation.hpp>

#include <algorithm>
#include <cmath>
#include <iterator>

namespace nathr::detail {
namespace {

std::size_t field(const Json& json, std::string_view name) {
    return natural(json.at(name), name);
}

std::size_t buckets(const Json& json) {
    return natural(json.value("relative_attention_num_buckets", Json(32)), "T5 attention buckets");
}

std::size_t max_distance(const Json& json) {
    return natural(json.value("relative_attention_max_distance", Json(128)), "T5 attention distance");
}

} // namespace

ModelConfig t5_config(const Json& json, const std::filesystem::path& path) {
    try {
        const auto projection = json.value("feed_forward_proj", std::string("relu"));
        const bool gated = projection == "gated-gelu";
        const std::string activation = gated ? "gelu_new" : "relu";
        if (!json.value("is_encoder_decoder", true) || (!gated && projection != "relu") ||
            json.value("dense_act_fn", activation) != activation ||
            json.value("is_gated_act", gated) != gated ||
            !json.value("pruned_heads", Json::object()).empty()) {
            throw Error(ErrorCode::unsupported, path.string() + ": expected a ReLU or gated-GELU T5 model");
        }
        const auto encoder_layers = field(json, "num_layers");
        const auto decoder_layers = json.find("num_decoder_layers");
        ModelConfig c{};
        c.layers = decoder_layers == json.end() || decoder_layers->is_null()
                       ? encoder_layers
                       : natural(*decoder_layers, "num_decoder_layers");
        c.hidden = field(json, "d_model");
        c.intermediate = field(json, "d_ff");
        c.head_dim = field(json, "d_kv");
        c.query_heads = c.kv_heads = field(json, "num_heads");
        c.vocabulary = field(json, "vocab_size");
        c.context = std::min<std::size_t>(512, natural(json.value("n_positions", Json(512)), "n_positions"));
        c.rms_epsilon = json.value("layer_norm_epsilon", 1e-6f);
        c.tied_embeddings = json.value("tie_word_embeddings", true);
        const auto count = buckets(json);
        if (!encoder_layers || encoder_layers > 24 || !c.layers || c.layers > 24 || !c.hidden ||
            !c.intermediate || !c.head_dim || !c.query_heads || !c.context || !c.vocabulary ||
            c.vocabulary > max_vocabulary || count < 4 || count % 4 || max_distance(json) <= count / 2 ||
            !(c.rms_epsilon > 0) || !std::isfinite(c.rms_epsilon)) {
            throw Error(ErrorCode::unsupported, path.string() + ": unsupported T5 dimensions or epsilon");
        }
        const auto start = field(json, "decoder_start_token_id");
        if (start >= c.vocabulary)
            throw Error(ErrorCode::malformed, path.string() + ": T5 special token ID exceeds vocabulary");
        c.bos = static_cast<Token>(start);
        c.eos = primary_eos_id(json.at("eos_token_id"), c.vocabulary, path.string() + ": eos_token_id");
        return c;
    } catch (const Json::exception& error) {
        throw Error(ErrorCode::malformed, path.string() + ": " + error.what());
    }
}

T5::T5(SafeTensors& file, const std::filesystem::path& path, const ModelConfig& c, const Json& json,
       CpuBackend& backend)
    : config_(c), backend_(backend), buckets_(buckets(json)), max_distance_(max_distance(json)),
      gated_(json.value("feed_forward_proj", std::string("relu")) == "gated-gelu") {
    const auto h = c.hidden, f = c.intermediate;
    const auto inner = product({c.query_heads, c.head_dim}, "T5 attention width");
    const auto take = [&](std::string_view name, std::initializer_list<std::size_t> shape) {
        return backend.bind(file.take(name, shape), path, name);
    };
    embedding_ = take("shared.weight", {c.vocabulary, h});
    const auto tied = [&](std::string_view name) {
        if (file.contains(name)) {
            const auto alias = take(name, {c.vocabulary, h});
            if (alias.desc.dtype != embedding_.desc.dtype ||
                !std::ranges::equal(alias.bytes, embedding_.bytes))
                throw Error(ErrorCode::unsupported,
                            path.string() + ": untied T5 tensor " + std::string(name));
        }
    };
    tied("encoder.embed_tokens.weight");
    tied("decoder.embed_tokens.weight");
    output_ = embedding_;
    if (c.tied_embeddings)
        tied("lm_head.weight");
    else
        output_ = take("lm_head.weight", {c.vocabulary, h});
    const auto attention = [&](const std::string& layer, std::string_view kind) {
        const auto prefix = layer + std::string(kind) + ".";
        return Attention{take(layer + "layer_norm.weight", {h}), take(prefix + "q.weight", {inner, h}),
                         take(prefix + "k.weight", {inner, h}), take(prefix + "v.weight", {inner, h}),
                         take(prefix + "o.weight", {h, inner})};
    };
    const auto stack = [&](const std::string& name, std::size_t count, bool decoder,
                           std::vector<Layer>& layers, std::vector<float>& bias) {
        const auto relative = take(name + ".block.0.layer.0.SelfAttention.relative_attention_bias.weight",
                                   {buckets_, c.query_heads});
        bias.resize(product({buckets_, c.query_heads}, "T5 position bias table"));
        for (std::size_t i = 0; i < buckets_; ++i)
            Scalar::embedding(relative, i, row(bias, c.query_heads, i));
        layers.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            const auto layer = name + ".block." + std::to_string(i) + ".layer.";
            const auto ff = layer + (decoder ? "2." : "1.");
            Layer weights{};
            weights.self = attention(layer + "0.", "SelfAttention");
            if (decoder)
                weights.cross = attention(layer + "1.", "EncDecAttention");
            const auto dense = ff + "DenseReluDense.";
            weights.ff = {take(ff + "layer_norm.weight", {h}),
                          take(dense + (gated_ ? "wi_0.weight" : "wi.weight"), {f, h}),
                          take(dense + "wo.weight", {h, f}),
                          gated_ ? take(dense + "wi_1.weight", {f, h}) : TensorView{}};
            layers.push_back(weights);
        }
    };
    stack("encoder", field(json, "num_layers"), false, encoder_, encoder_bias_);
    stack("decoder", c.layers, true, decoder_, decoder_bias_);
    encoder_norm_ = take("encoder.final_layer_norm.weight", {h});
    decoder_norm_ = take("decoder.final_layer_norm.weight", {h});
    // Original published checkpoints may store a legacy cross-attention bias table.
    const std::string_view legacy_bias =
        "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight";
    if (file.contains(legacy_bias))
        (void)file.take(legacy_bias, {buckets_, c.query_heads});
    if (const auto name = file.first_unbound(); !name.empty())
        throw Error(ErrorCode::unsupported, path.string() + ": unexpected T5 tensor " + std::string(name));
}

struct T5::Scratch final : Workspace {
    Scratch(const ModelConfig& c, std::size_t rows, std::size_t positions, bool gated, bool decode)
        : count(rows), inner(product({c.query_heads, c.head_dim}, "T5 attention width")),
          bias_stride(product({c.query_heads, positions}, "T5 position bias row")) {
        const auto hidden_rows = product({rows, c.hidden}, "T5 hidden workspace");
        const auto attention_rows = product({rows, inner}, "T5 attention workspace");
        const auto ff_rows = product({rows, c.intermediate}, "T5 feed-forward workspace");
        allocate({{&hidden, hidden_rows},
                  {&norm, hidden_rows},
                  {&projected, hidden_rows},
                  {&query, attention_rows},
                  {&key, attention_rows},
                  {&value, attention_rows},
                  {&attended, attention_rows},
                  {&ff, ff_rows},
                  {&up, gated ? ff_rows : 0},
                  {&packed, rows == 1 ? 0
                                      : product({batch_tile, std::max({c.hidden, c.intermediate, inner})},
                                                "T5 matrix tile workspace")},
                  {&bias, product({decode ? rows : 1, bias_stride}, "T5 position bias workspace")},
                  {&scores, positions},
                  {&logits, decode && rows > 1 ? product({rows, c.vocabulary}, "T5 logits workspace") : 0}});
    }

    void resize(const ModelConfig& c, std::size_t rows) {
        for (auto* buffer : {&hidden, &norm, &projected})
            *buffer = std::span(buffer->data(), rows * c.hidden);
        for (auto* buffer : {&query, &key, &value, &attended})
            *buffer = std::span(buffer->data(), rows * inner);
        ff = std::span(ff.data(), rows * c.intermediate);
        if (!up.empty())
            up = std::span(up.data(), rows * c.intermediate);
        count = rows;
    }

    std::size_t count;
    const std::size_t inner, bias_stride;
    std::span<float> hidden, norm, projected, query, key, value, attended, ff, up, bias;
};

void T5::project(Scratch& w, TensorView weight, std::span<const float> input, std::span<float> output) const {
    backend_.project(weight, input, output, w.count, w.packed);
}

void T5::normalize(Scratch& w, TensorView gain) const {
    const auto h = config_.hidden;
    for (std::size_t i = 0; i < w.count; ++i)
        Scalar::rms_norm(gain, row(w.hidden, h, i), row(w.norm, h, i), config_.rms_epsilon);
}

void T5::project_attention(Scratch& w, const Attention& weights) const {
    normalize(w, weights.norm);
    backend_.project({{weights.query, w.query}, {weights.key, w.key}, {weights.value, w.value}}, w.norm,
                     w.count, w.packed);
}

void T5::feed_forward(Scratch& w, const FeedForward& weights) const {
    normalize(w, weights.norm);
    if (gated_) {
        backend_.project({{weights.input, w.ff}, {weights.up, w.up}}, w.norm, w.count, w.packed);
        Scalar::gelu(w.ff);
        for (std::size_t i = 0; i < w.ff.size(); ++i)
            w.ff[i] *= w.up[i];
    } else {
        project(w, weights.input, w.norm, w.ff);
        for (auto& value : w.ff)
            value = std::max(value, 0.f);
    }
    project(w, weights.output, w.ff, w.projected);
    Scalar::add(w.hidden, w.projected);
}

void T5::position_bias(std::span<float> bias, std::size_t query, std::size_t keys, bool bidirectional) const {
    const auto count = bidirectional ? buckets_ / 2 : buckets_;
    const auto exact = count / 2;
    const auto heads = config_.query_heads;
    const auto& table = bidirectional ? encoder_bias_ : decoder_bias_;
    const auto log_range = std::log(static_cast<double>(max_distance_) / exact);
    for (std::size_t i = 0; i < keys; ++i) {
        const auto distance = i > query ? i - query : query - i;
        auto bucket = distance;
        if (distance >= exact) {
            const auto bounded = std::min(distance, max_distance_);
            const auto fraction = std::log(static_cast<double>(bounded) / exact) / log_range;
            bucket = std::min(count - 1, exact + static_cast<std::size_t>(fraction * (count - exact)));
        }
        if (bidirectional && i > query)
            bucket += count;
        for (std::size_t head = 0; head < heads; ++head)
            bias[head * keys + i] = table[bucket * heads + head];
    }
}

void T5::encode(std::span<const std::span<const Token>> inputs, Scratch& w) const {
    std::size_t index = 0;
    for (const auto tokens : inputs) {
        for (const auto token : tokens)
            Scalar::embedding(embedding_, token, row(w.hidden, config_.hidden, index++));
    }
    for (const auto& layer : encoder_) {
        const auto& weights = layer.self;
        project_attention(w, weights);
        std::size_t start = 0;
        // Rows share projections, while bidirectional attention stays inside each source.
        for (const auto tokens : inputs) {
            const auto size = tokens.size();
            const auto keys = w.key.subspan(start * w.inner, size * w.inner);
            const auto values = w.value.subspan(start * w.inner, size * w.inner);
            for (std::size_t i = 0; i < size; ++i) {
                position_bias(w.bias, i, size, true);
                Scalar::attention_range(config_, 0, nullptr, keys, values, row(w.query, w.inner, start + i),
                                        row(w.attended, w.inner, start + i), w.scores.first(size), 1.0,
                                        w.bias);
            }
            start += size;
        }
        project(w, weights.output, w.attended, w.projected);
        Scalar::add(w.hidden, w.projected);
        feed_forward(w, layer.ff);
    }
    normalize(w, encoder_norm_);
}

std::vector<T5::Source> T5::prepare_sources(std::span<const std::span<const Token>> inputs,
                                            CachePagePool* pool) const {
    std::size_t rows = 0, positions = 0;
    for (const auto tokens : inputs) {
        rows = sum(rows, tokens.size(), "T5 source batch rows");
        positions = std::max(positions, tokens.size());
    }
    Scratch encoder(config_, rows, positions, gated_, false);
    encode(inputs, encoder);
    std::vector<std::shared_ptr<Cache>> caches;
    caches.reserve(inputs.size());
    for (const auto tokens : inputs) {
        auto cache = std::make_shared<Cache>(config_, tokens.size(), KvPrecision::f32, pool);
        cache->ensure(tokens.size());
        caches.push_back(std::move(cache));
    }
    for (std::size_t i = 0; i < decoder_.size(); ++i) {
        const auto& weights = decoder_[i].cross;
        backend_.project({{weights.key, encoder.key}, {weights.value, encoder.value}}, encoder.norm,
                         encoder.count, encoder.packed);
        std::size_t index = 0;
        for (auto& cache : caches) {
            for (std::size_t pos = 0; pos < cache->capacity; ++pos, ++index)
                Scalar::store(*cache, i, pos, row(encoder.key, encoder.inner, index),
                              row(encoder.value, encoder.inner, index));
        }
    }
    return {std::make_move_iterator(caches.begin()), std::make_move_iterator(caches.end())};
}

std::unique_ptr<Workspace> T5::workspace(std::size_t rows, std::size_t positions) const {
    return std::make_unique<Scratch>(config_, rows, positions, gated_, true);
}

void T5::step(Workspace& scratch, std::span<const Step> steps) const {
    auto& w = static_cast<Scratch&>(scratch);
    w.resize(config_, steps.size());
    for (std::size_t slot = 0; slot < steps.size(); ++slot) {
        const auto& [state, token, output] = steps[slot];
        Scalar::embedding(embedding_, token, row(w.hidden, config_.hidden, slot));
        position_bias(row(w.bias, w.bias_stride, slot), state.length, state.length + 1, false);
    }
    for (std::size_t i = 0; i < decoder_.size(); ++i) {
        const auto& layer = decoder_[i];
        project_attention(w, layer.self);
        for (std::size_t slot = 0; slot < steps.size(); ++slot) {
            auto& state = steps[slot].state;
            Scalar::store(state.self, i, state.length, row(w.key, w.inner, slot),
                          row(w.value, w.inner, slot));
            Scalar::attention_range(config_, i, &state.self, {}, {}, row(w.query, w.inner, slot),
                                    row(w.attended, w.inner, slot), w.scores.first(state.length + 1), 1.0,
                                    row(w.bias, w.bias_stride, slot));
        }
        project(w, layer.self.output, w.attended, w.projected);
        Scalar::add(w.hidden, w.projected);
        normalize(w, layer.cross.norm);
        project(w, layer.cross.query, w.norm, w.query);
        for (std::size_t slot = 0; slot < steps.size(); ++slot) {
            const auto& source = steps[slot].state.source;
            Scalar::attention_range(config_, i, source.get(), {}, {}, row(w.query, w.inner, slot),
                                    row(w.attended, w.inner, slot), w.scores.first(source->capacity), 1.0);
        }
        project(w, layer.cross.output, w.attended, w.projected);
        Scalar::add(w.hidden, w.projected);
        feed_forward(w, layer.ff);
    }
    normalize(w, decoder_norm_);
    if (config_.tied_embeddings) {
        const auto scale = 1.0 / std::sqrt(static_cast<double>(config_.hidden));
        for (auto& value : w.norm)
            value = static_cast<float>(value * scale);
    }
    const bool single = steps.size() == 1;
    auto logits = single ? steps.front().logits : w.logits.first(steps.size() * config_.vocabulary);
    project(w, output_, w.norm, logits);
    if (!std::ranges::all_of(logits, [](float value) { return std::isfinite(value); }))
        throw Error(ErrorCode::backend, "T5 output projection produced non-finite logits");
    // Publish only once all rows can commit. Retry can overwrite the same self-attention positions.
    for (std::size_t slot = 0; slot < steps.size(); ++slot) {
        if (!single)
            std::ranges::copy(row(logits, config_.vocabulary, slot), steps[slot].logits.begin());
        ++steps[slot].state.length;
    }
}

} // namespace nathr::detail

namespace nathr {
using namespace detail;

struct T5Session::Impl {
    Impl(const Model& model, const T5& decoder, std::span<const Token> source, std::size_t capacity,
         std::uint64_t seed)
        : model(model), decoder(decoder),
          state(model.config(), capacity, decoder.prepare_sources(std::span(&source, 1)).front(), nullptr),
          scratch(decoder.workspace(1, std::max(source.size(), capacity))), logits(model.config().vocabulary),
          rng(seed) {
        state.self.ensure(1);
        step(model.config().bos);
    }

    void step(Token token) {
        const T5::Step input{state, token, logits};
        decoder.step(*scratch, std::span(&input, 1));
    }

    Model model;
    const T5& decoder;
    T5::State state;
    std::unique_ptr<Workspace> scratch;
    std::vector<float> logits;
    std::mt19937_64 rng;
    bool valid = true;
};

T5Session::T5Session(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
T5Session::~T5Session() = default;
T5Session::T5Session(T5Session&&) noexcept = default;
T5Session& T5Session::operator=(T5Session&&) noexcept = default;

T5Session T5Session::create(const Model& model, std::span<const Token> source, std::size_t capacity,
                            std::uint64_t seed) {
    const auto* decoder = std::get_if<std::unique_ptr<T5>>(&model.impl_->execution);
    if (!decoder)
        throw Error(ErrorCode::unsupported, "T5Session requires a T5 model");
    const auto& config = model.config();
    validate_tokens(source, config, config.context);
    if (!capacity)
        capacity = config.context;
    if (capacity > config.context)
        throw Error(ErrorCode::invalid_argument, "T5 decoder capacity exceeds model limit");
    const auto bytes =
        Cache::required_bytes(config, sum(source.size(), capacity, "T5 cache positions"), KvPrecision::f32);
    if (bytes > max_kv_cache_bytes)
        throw Error(ErrorCode::resource_limit, "T5 self-attention and cross-attention caches exceed 1 GiB");
    return T5Session(std::make_unique<Impl>(model, **decoder, source, capacity, seed));
}

void T5Session::append(std::span<const Token> tokens) {
    if (!impl_->valid)
        throw Error(ErrorCode::invalid_argument, "T5 session is invalid");
    validate_tokens(tokens, impl_->model.config(), capacity() - length());
    impl_->state.self.ensure(length() + tokens.size());
    impl_->valid = false;
    for (const auto token : tokens)
        impl_->step(token);
    impl_->valid = true;
}

std::span<const float> T5Session::logits() const {
    if (!impl_->valid)
        throw Error(ErrorCode::invalid_argument, "T5 logits are unavailable");
    return impl_->logits;
}

Token T5Session::sample(Sampler& sampler) {
    return sampler.select(logits(), impl_->rng);
}

std::size_t T5Session::length() const noexcept {
    return impl_->state.length;
}

std::size_t T5Session::capacity() const noexcept {
    return impl_->state.self.capacity;
}

} // namespace nathr
