#include "batch.hpp"

#include "backend/cpu/scalar.hpp"
#include "core/input.hpp"
#include "models/t5/model.hpp"
#include "requests.hpp"
#include "structured.hpp"

#include <nathr/tokenizer.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <iostream>
#include <map>
#include <poll.h>
#include <unistd.h>

namespace nathr::cli {
namespace {
using detail::Json;
using Clock = std::chrono::steady_clock;

struct Line {
    std::string text;
    bool too_long = false;
};

// Read at most one command per scheduler iteration. A partial command can wait
// for more input while an admitted request continues generating.
class BatchInput {
  public:
    std::optional<Line> next(bool wait, int timeout_ms = 0) {
        bool read_buffer = false;
        while (!eof_) {
            while (cursor_ < end_) {
                const auto begin = buffer_.begin() + static_cast<std::ptrdiff_t>(cursor_);
                const auto end = buffer_.begin() + static_cast<std::ptrdiff_t>(end_);
                const auto newline = std::find(begin, end, '\n');
                const auto count = static_cast<std::size_t>(newline - begin);
                if (count > detail::mib - line_.text.size())
                    line_.too_long = true;
                if (!line_.too_long)
                    line_.text.append(begin, count);
                cursor_ += count;
                if (newline != end) {
                    ++cursor_;
                    return std::exchange(line_, {});
                }
            }
            if (!wait && read_buffer)
                return {};
            pollfd input{STDIN_FILENO, POLLIN, 0};
            const auto ready = poll(&input, 1, wait ? -1 : timeout_ms);
            if (ready == 0)
                return {};
            if (ready < 0 && errno == EINTR)
                continue;
            if (ready < 0 || (input.revents & POLLNVAL))
                throw Error(ErrorCode::io, "cannot poll batch input");
            const auto bytes = read(STDIN_FILENO, buffer_.data(), buffer_.size());
            if (bytes < 0 && (errno == EINTR || errno == EAGAIN))
                continue;
            if (bytes < 0)
                throw Error(ErrorCode::io, "cannot read batch input");
            cursor_ = 0;
            end_ = static_cast<std::size_t>(bytes);
            eof_ = end_ == 0;
            read_buffer = true;
        }
        if (!line_.text.empty() || line_.too_long)
            return std::exchange(line_, {});
        return {};
    }

    bool done() const noexcept { return eof_; }

  private:
    std::array<char, 4096> buffer_{};
    std::size_t cursor_ = 0, end_ = 0;
    Line line_;
    bool eof_ = false;
};

void write_output(const Json& value) {
    std::cout << value.dump() << '\n' << std::flush;
    if (!std::cout)
        throw Error(ErrorCode::io, "cannot write batch output");
}

std::size_t integer(const Json& value, std::string_view key, std::size_t fallback) {
    const auto found = value.find(key);
    return found == value.end() ? fallback : detail::natural(*found, key);
}

constexpr auto lora_budget_bytes = 256 * detail::mib;

class LoraRegistry {
  public:
    LoraRegistry(const Model& model, const BatchIoOptions& io)
        : model_(model),
          root_(io.lora_root.empty() ? io.lora_root : std::filesystem::canonical(io.lora_root)) {
        if (!root_.empty() && (!model.supports_fused_batch() || !std::filesystem::is_directory(root_)))
            throw Error(ErrorCode::invalid_argument,
                        "LoRA reloads require a decoder-only model and a directory");
        for (const auto& spec : io.lora_adapters) {
            const auto eq = spec.find('=');
            if (eq == spec.npos || eq + 1 == spec.size())
                throw Error(ErrorCode::invalid_argument, "batch --lora requires NAME=DIR");
            load(spec.substr(0, eq), spec.substr(eq + 1));
        }
    }

    void reload(std::string name, const std::filesystem::path& path) {
        require_root();
        const auto received = root_ / path;
        check_root(received.lexically_normal());
        const auto directory = std::filesystem::canonical(received);
        check_root(directory);
        for (const auto filename : {"adapter_config.json", "adapter_model.safetensors"})
            check_root(std::filesystem::canonical(directory / filename));
        load(std::move(name), directory);
    }

    void unload(const std::string& name, const BatchScheduler& scheduler) {
        require_root();
        const auto found = adapters_.find(name);
        if (found == adapters_.end())
            throw Error(ErrorCode::invalid_argument, "LoRA adapter is not loaded");
        if (scheduler.uses_lora(found->second))
            throw Error(ErrorCode::resource_limit, "LoRA adapter is in use");
        bytes_ -= found->second.memory_bytes();
        adapters_.erase(found);
    }

    [[nodiscard]] const LoraAdapter* get(const Json& request) const {
        const auto choice = request.find("lora");
        if (choice == request.end())
            return nullptr;
        const auto found = adapters_.find(choice->get<std::string>());
        if (found == adapters_.end())
            throw Error(ErrorCode::invalid_argument, "LoRA adapter is not loaded");
        return &found->second;
    }

    [[nodiscard]] Json names() const {
        auto names = Json::array();
        for (const auto& [name, adapter] : adapters_)
            names.push_back(name);
        return names;
    }

  private:
    void require_root() const {
        if (root_.empty())
            throw Error(ErrorCode::unsupported, "loading or unloading adapters requires --lora-root");
    }

    void check_root(const std::filesystem::path& resolved) const {
        const auto parent = std::mismatch(root_.begin(), root_.end(), resolved.begin(), resolved.end());
        if (parent.first != root_.end())
            throw Error(ErrorCode::invalid_argument, "LoRA directory and files must be inside --lora-root");
    }

    void load(std::string name, const std::filesystem::path& directory) {
        if (name.empty() || name.size() > 128 || !std::ranges::all_of(name, [](char ch) {
                return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') ||
                       ch == '_' || ch == '-' || ch == '.';
            }))
            throw Error(ErrorCode::invalid_argument, "invalid LoRA adapter name");
        if (adapters_.contains(name))
            throw Error(ErrorCode::invalid_argument, "LoRA adapter name is already loaded");
        if (adapters_.size() == 16)
            throw Error(ErrorCode::resource_limit, "LoRA adapter count limit reached");
        auto adapter = model_.load_lora(directory, lora_budget_bytes - bytes_);
        const auto size = adapter.memory_bytes();
        adapters_.emplace(std::move(name), std::move(adapter));
        bytes_ += size;
    }

    const Model& model_;
    const std::filesystem::path root_;
    std::map<std::string, LoraAdapter, std::less<>> adapters_;
    std::size_t bytes_ = 0;
};

struct BatchOutput {
    std::optional<Tokenizer> tokenizer;
    std::shared_ptr<const StructuredVocabulary> structured;
    std::size_t output_tokens = 0;
    std::size_t draft_tokens = 0, accepted_draft_tokens = 0;
    const BatchScheduler* live_stats = nullptr;

    void publish(const BatchEvent& event) {
        Json value = {{"id", event.id},
                      {"emitted", event.emitted},
                      {"elapsed_ms", event.elapsed_ms},
                      {"prompt_tokens", event.prompt_tokens}};
        if (event.cached_prompt_tokens)
            value["cached_prompt_tokens"] = event.cached_prompt_tokens;
        if (event.text)
            value["text"] = *event.text;
        if (event.kind == BatchEventKind::token) {
            ++output_tokens;
            value["event"] = "token";
            value["token_id"] = *event.token;
            if (event.logprobs)
                value["logprobs"] =
                    logprobs_json(*event.logprobs, tokenizer ? &*tokenizer : nullptr, event.emitted == 1);
            if (event.inter_token_ms)
                value["inter_token_ms"] = *event.inter_token_ms;
        } else {
            value["event"] = "done";
            if (event.draft_tokens) {
                value["draft_tokens"] = event.draft_tokens;
                value["accepted_draft_tokens"] = event.accepted_draft_tokens;
                draft_tokens += event.draft_tokens;
                accepted_draft_tokens += event.accepted_draft_tokens;
            }
            if (live_stats) {
                value["queue_ms"] = event.queue_ms;
                value["kv"] = {{"peak_bytes", live_stats->peak_kv_bytes()},
                               {"resident_bytes", live_stats->resident_kv_bytes()}};
                value["queue_bypasses"] = live_stats->queue_bypasses();
            }
            switch (event.kind) {
            case BatchEventKind::eos:
                value["reason"] = "eos";
                break;
            case BatchEventKind::stop:
                value["reason"] = "stop";
                break;
            case BatchEventKind::length:
                value["reason"] = "length";
                break;
            case BatchEventKind::cancelled:
                value["reason"] = "cancelled";
                break;
            default:
                value["reason"] = "error";
                value["message"] = event.message;
                break;
            }
        }
        write_output(value);
    }
};

TokenizedInput input_tokens(const Json& json, BatchOutput& output, const Model& model,
                            const std::filesystem::path& path, const std::filesystem::path& sidecar,
                            const RequestDefaults& defaults, bool decode) {
    const auto prompt = json.find("prompt");
    const auto tokens = json.find("tokens");
    const auto messages = json.find("messages");
    const auto pair = json.find("text_pair");
    const auto types = json.find("token_type_ids");
    if (static_cast<int>(prompt != json.end()) + (tokens != json.end()) + (messages != json.end()) != 1 ||
        (pair != json.end() && prompt == json.end()) || (types != json.end() && tokens == json.end()))
        throw Error(ErrorCode::invalid_argument, "batch needs exactly one of prompt, messages or tokens");

    if ((decode || tokens == json.end()) && !output.tokenizer)
        output.tokenizer = load_tokenizer(model, path, sidecar);
    if (tokens != json.end())
        return {token_array(*tokens, "tokens"),
                types == json.end() ? std::vector<Token>{} : token_array(*types, "token_type_ids")};
    const auto text =
        messages == json.end() ? prompt->get<std::string>() : render_chat(*messages, defaults.chat);
    if (pair != json.end())
        return output.tokenizer->encode_pair(text, pair->get<std::string>());
    return {encode_prompt(*output.tokenizer, model, text), {}};
}

struct TensorRequest {
    RequestId id;
    std::string name;
    std::vector<TokenizedInput> inputs;
    bool multiple;
};

void publish_error(std::optional<RequestId> id, const std::exception& error) {
    const auto* typed = dynamic_cast<const Error*>(&error);
    const auto code =
        typed && typed->code() == ErrorCode::resource_limit ? "resource_limit" : "invalid_request";
    write_output({{"event", "error"},
                  {"id", id ? Json(*id) : Json(nullptr)},
                  {"message", error.what()},
                  {"code", code}});
}

void publish_tensor(RequestId id, std::string_view name, TensorResult result, std::size_t tokens,
                    std::optional<std::size_t> index = {}) {
    Json value = {{"event", index ? "tensor" : "result"},
                  {"id", id},
                  {"output", name},
                  {"prompt_tokens", tokens},
                  {"shape", std::move(result.shape)},
                  {"values", std::move(result.values)}};
    if (index)
        value["index"] = *index;
    write_output(value);
}

void publish_group(const TensorRequest& request, std::span<TensorResult> results) {
    for (std::size_t i = 0; i < results.size(); ++i)
        publish_tensor(request.id, request.name, std::move(results[i]), request.inputs[i].ids.size(),
                       request.multiple ? std::optional{i} : std::nullopt);
    if (request.multiple)
        write_output({{"event", "done"}, {"id", request.id}, {"reason", "result"}});
}

void flush_tensors(const Model& model, std::vector<TensorRequest>& pending) {
    std::ranges::stable_sort(pending, {}, &TensorRequest::name);
    std::vector<InferenceInput> inputs;
    const auto append = [&](const TensorRequest& request) {
        for (const auto& item : request.inputs)
            inputs.emplace_back(item.ids, item.token_type_ids);
    };
    for (auto first = pending.begin(); first != pending.end();) {
        const auto last = std::find_if(std::next(first), pending.end(),
                                       [&](const auto& request) { return request.name != first->name; });
        inputs.clear();
        for (auto it = first; it != last; ++it)
            append(*it);
        std::vector<TensorResult> results;
        try {
            results = model.infer_batch(inputs, first->name);
        } catch (const Error&) {
            // Retry independent requests so a bad input cannot fail another client's batch.
            for (auto it = first; it != last; ++it) {
                inputs.clear();
                append(*it);
                try {
                    auto individual = model.infer_batch(inputs, it->name);
                    publish_group(*it, individual);
                } catch (const Error& error) {
                    publish_error(it->id, error);
                }
            }
            first = last;
            continue;
        }
        auto remaining = std::span(results);
        for (auto it = first; it != last; ++it) {
            publish_group(*it, remaining.first(it->inputs.size()));
            remaining = remaining.subspan(it->inputs.size());
        }
        first = last;
    }
    pending.clear();
}

void command(const Line& line, const Model& model, std::optional<BatchScheduler>& scheduler,
             BatchOutput& output, const std::filesystem::path& path, const std::filesystem::path& sidecar,
             const RequestDefaults& defaults, const BatchOptions& options, const BatchIoOptions& io,
             LoraRegistry& lora, std::vector<TensorRequest>* pending) {
    std::optional<RequestId> id;
    try {
        if (line.too_long)
            throw Error(ErrorCode::resource_limit, "batch input line exceeds 1 MiB");
        if (line.text.find_first_not_of(" \r\t") == std::string::npos)
            return;
        const auto json = detail::parse_json(std::as_bytes(std::span(line.text)), "<batch stdin>");
        if (json.contains("cancel")) {
            id = detail::natural(json.at("cancel"), "cancel");
            if (json.size() != 1)
                throw Error(ErrorCode::invalid_argument, "a cancel command takes only a request ID");
            if (pending) {
                const auto found = std::ranges::find(*pending, *id, &TensorRequest::id);
                if (found != pending->end()) {
                    pending->erase(found);
                    write_output({{"event", "done"}, {"id", *id}, {"reason", "cancelled"}});
                    return;
                }
            }
            const auto cancellation = scheduler ? scheduler->cancel(*id) : std::optional<BatchEvent>{};
            if (!cancellation)
                throw Error(ErrorCode::invalid_argument, "request ID is not pending");
            output.publish(*cancellation);
            return;
        }
        id = detail::natural(json.at("id"), "id");
        const auto load_lora = json.find("load_lora");
        const auto unload_lora = json.find("unload_lora");
        if (load_lora != json.end() || unload_lora != json.end()) {
            if (!scheduler || json.size() != 2 || scheduler->contains(*id))
                throw Error(ErrorCode::invalid_argument, "invalid LoRA control request");
            if (load_lora != json.end()) {
                if (!load_lora->is_object() || load_lora->size() != 2)
                    throw Error(ErrorCode::invalid_argument, "load_lora requires name and path");
                lora.reload(load_lora->at("name").get<std::string>(),
                            load_lora->at("path").get<std::string>());
            } else {
                lora.unload(unload_lora->get<std::string>(), *scheduler);
            }
            Json result = {{"event", "result"}, {"id", *id}, {"lora_adapters", lora.names()}};
            if (load_lora != json.end())
                result["lora_loaded"] = *load_lora;
            write_output(result);
            return;
        }
        const auto* adapter = lora.get(json);
        const auto tensor = json.find("output");
        const auto group = json.find("inputs");
        if ((tensor != json.end() && scheduler && scheduler->contains(*id)) ||
            (pending && std::ranges::find(*pending, *id, &TensorRequest::id) != pending->end()))
            throw Error(ErrorCode::invalid_argument, "duplicate request ID");
        if (group != json.end()) {
            if (!pending || tensor == json.end() || json.size() != 3 || !group->is_array() ||
                group->empty() || group->size() > 16)
                throw Error(ErrorCode::invalid_argument, "expected a BERT output and 1 to 16 encoder inputs");
            auto name = tensor->get<std::string>();
            const auto outputs = model.outputs();
            const auto info = std::ranges::find(outputs, name, &OutputInfo::name);
            if (info == outputs.end() || info->sequence)
                throw Error(ErrorCode::unsupported, "batched encoder inputs require a vector output");
            if (io.max_result_values && info->width > io.max_result_values)
                throw Error(ErrorCode::resource_limit, "named tensor exceeds response limit");
            TensorRequest request{*id, std::move(name), {}, true};
            request.inputs.reserve(group->size());
            for (const auto& item : *group) {
                if (!item.is_object())
                    throw Error(ErrorCode::invalid_argument, "invalid grouped encoder input");
                for (const auto& [key, value] : item.items())
                    if (key != "prompt" && key != "tokens" && key != "text_pair" && key != "token_type_ids")
                        throw Error(ErrorCode::invalid_argument, "invalid grouped encoder input");
                request.inputs.push_back(input_tokens(item, output, model, path, sidecar, defaults, false));
            }
            pending->push_back(std::move(request));
            return;
        }
        constexpr std::string_view common[] = {"id", "prompt", "tokens", "lora"};
        constexpr std::string_view tensors[] = {"output", "text_pair", "token_type_ids", "decoder_tokens"};
        constexpr std::string_view generation[] = {"max_new_tokens", "seed",   "greedy",       "temperature",
                                                   "top_k",          "top_p",  "messages",     "stop",
                                                   "stop_ids",       "decode", "top_logprobs", "json_schema"};
        const auto extra = tensor != json.end() ? std::span<const std::string_view>(tensors)
                                                : std::span<const std::string_view>(generation);
        for (const auto& [key, value] : json.items()) {
            if (std::ranges::find(common, key) == std::end(common) &&
                std::ranges::find(extra, key) == extra.end())
                throw Error(ErrorCode::invalid_argument, "unknown batch request field: " + key);
        }
        const bool decode = tensor == json.end() &&
                            (json.contains("prompt") || json.contains("messages") || json.contains("stop") ||
                             json.contains("json_schema") || json.value("decode", false));
        auto input = input_tokens(json, output, model, path, sidecar, defaults, decode);
        if (tensor != json.end()) {
            auto name = tensor->get<std::string>();
            const auto prefix = json.contains("decoder_tokens")
                                    ? token_array(json.at("decoder_tokens"), "decoder_tokens")
                                    : std::vector<Token>{};
            const auto outputs = model.outputs();
            const auto shape = std::ranges::find(outputs, name, &OutputInfo::name);
            if (shape != outputs.end()) {
                if (io.max_result_values) {
                    const auto rows = !shape->sequence               ? 1
                                      : model.architecture() == "t5" ? std::max<std::size_t>(1, prefix.size())
                                                                     : input.ids.size();
                    if (rows > io.max_result_values / shape->width)
                        throw Error(ErrorCode::resource_limit, "named tensor exceeds response limit");
                }
                if (pending && !shape->sequence && prefix.empty()) {
                    TensorRequest request{*id, std::move(name), {}, false};
                    request.inputs.push_back(std::move(input));
                    pending->push_back(std::move(request));
                    return;
                }
            }
            auto result = model.infer({input.ids, input.token_type_ids, prefix}, name, adapter);
            publish_tensor(*id, name, std::move(result), input.ids.size());
            return;
        }
        if (!scheduler)
            throw Error(ErrorCode::unsupported, "this checkpoint requires an output name in batch requests");
        const auto maximum = json.find("max_new_tokens");
        const auto max_tokens =
            maximum != json.end()
                ? detail::natural(*maximum, "max_new_tokens")
                : fit_generation_budget(model, input.ids.size(), scheduler->context(),
                                        defaults.max_new_tokens, options.session.kv_precision,
                                        options.kv_budget_bytes);
        if (io.max_output_tokens && max_tokens > io.max_output_tokens)
            throw Error(ErrorCode::invalid_argument, "max_new_tokens exceeds the output limit");
        auto stops = defaults.stop_ids;
        if (const auto field = json.find("stop_ids"); field != json.end()) {
            const auto ids = token_array(*field, "stop_ids");
            stops.insert(stops.end(), ids.begin(), ids.end());
        }
        TokenConstraint constraint;
        if (const auto schema = json.find("json_schema"); schema != json.end()) {
            if (!model.supports_fused_batch())
                throw Error(ErrorCode::unsupported, "structured output requires a decoder-only checkpoint");
            if (!output.structured)
                output.structured = structured_vocabulary(*output.tokenizer);
            constraint = structured_output(*schema, output.structured, stops, model.config().eos);
        }
        scheduler->submit(
            {*id,
             std::move(input.ids),
             max_tokens,
             integer(json, "seed", 0),
             {json.value("greedy", defaults.sampling.greedy),
              json.value("temperature", defaults.sampling.temperature),
              integer(json, "top_k", defaults.sampling.top_k), json.value("top_p", defaults.sampling.top_p)},
             std::move(stops),
             decode ? output.tokenizer : std::optional<Tokenizer>{},
             json.contains("stop") ? stop_strings(json.at("stop")) : std::vector<std::string>{},
             adapter ? std::optional(*adapter) : std::nullopt,
             json.contains("top_logprobs")
                 ? std::optional(detail::natural(json.at("top_logprobs"), "top_logprobs"))
                 : std::nullopt,
             std::move(constraint)});
    } catch (const std::exception& error) {
        publish_error(id, error);
    }
}
} // namespace

void print_batch_memory_plan(const std::filesystem::path& path, BatchOptions options, BatchIoOptions io) {
    const bool directory = detail::is_model_directory(path);
    const auto weights_path = directory ? path / "model.safetensors" : path;
    const auto file_bytes = std::filesystem::file_size(weights_path);
    if (file_bytes > detail::max_checkpoint_bytes)
        throw Error(ErrorCode::resource_limit, "checkpoint exceeds 1 GiB");
    const auto weights = static_cast<std::size_t>(file_bytes);

    // Generation scratch can coexist with a source encoder or a tensor inference workspace.
    // BERT's hidden-state buffer can be as large as its scratch.
    auto kv = options.kv_budget_bytes;
    std::size_t result_count = 1;
    auto vocabulary = detail::max_vocabulary;
    if (directory) {
        const auto config_path = path / "config.json";
        const auto config = detail::read_json(config_path, detail::mib);
        const auto type = config.find("model_type");
        if (const auto value = config.find("vocab_size"); value != config.end())
            vocabulary = detail::natural(*value, "vocab_size");
        if (type != config.end() && *type == "bert") {
            kv = 0;
            result_count = 16;
        } else if (type != config.end() && *type == "t5") {
            const auto c = detail::t5_config(config, config_path);
            const auto tensor_kv = detail::Cache::required_bytes(c, 2 * c.context, KvPrecision::f32);
            kv = detail::sum(kv, std::min(tensor_kv, detail::max_kv_cache_bytes), "worker KV capacity");
        }
    }
    auto workspace = 2 * detail::max_workspace_bytes;
    if (kv) {
        // Cover retained logits and sampler candidates for each active request.
        const auto sampling =
            detail::product({options.max_active, vocabulary, 32}, "worker sampling buffers");
        workspace = detail::sum(workspace, sampling, "worker workspace total");
    }
    workspace = detail::sum(
        workspace, detail::product({result_count, io.max_result_values, sizeof(float)}, "tensor results"),
        "worker workspace total");
    const auto lora = io.lora_adapters.empty() && io.lora_root.empty() ? 0 : lora_budget_bytes;
    const auto total = detail::sum(detail::sum(weights, workspace, "worker memory"),
                                   detail::sum(kv, lora, "worker memory"), "worker memory");
    write_output({{"weights_bytes", weights},
                  {"workspace_bytes", workspace},
                  {"kv_bytes", kv},
                  {"lora_bytes", lora},
                  {"vocabulary_limit", vocabulary},
                  {"total_bytes", total}});
}

void run_batch(const Model& model, const std::filesystem::path& path, const std::filesystem::path& sidecar,
               BatchOptions options, BatchIoOptions io) {
    LoraRegistry lora(model, io);
    std::optional<BatchScheduler> scheduler;
    if (model.supports_generation())
        scheduler.emplace(model, options);
    BatchOutput output;
    if (io.ready && scheduler)
        output.live_stats = &*scheduler;
    BatchInput input;
    auto defaults = request_defaults(path, model, true, io.chat_format);
    if (io.max_output_tokens)
        defaults.max_new_tokens = std::min(defaults.max_new_tokens, io.max_output_tokens);
    std::optional<BatchMetrics> last_metrics;
    const auto publish_metrics = [&] {
        if (!io.metrics || !scheduler)
            return;
        const auto values = scheduler->metrics();
        if (last_metrics == values)
            return;
        last_metrics = values;
        write_output({{"event", "metrics"},
                      {"values",
                       {{"requests_active", values.active_requests},
                        {"requests_queued", values.queued_requests},
                        {"queued_tokens", values.queued_tokens},
                        {"kv_used_bytes", values.kv_used_bytes},
                        {"kv_resident_bytes", values.kv_resident_bytes},
                        {"kv_budget_bytes", values.kv_budget_bytes},
                        {"prefix_cache_lookups_total", values.prefix_lookups},
                        {"prefix_cache_hits_total", values.prefix_hits},
                        {"prefix_cache_tokens_total", values.cached_prompt_tokens}}}});
    };
    publish_metrics();
    if (io.ready) {
        try {
            output.tokenizer = load_tokenizer(model, path, sidecar);
        } catch (const Error&) {
            // Checkpoints with external tokenizers can still accept raw IDs.
        }
        Json ready = {{"event", "ready"},
                      {"text", output.tokenizer.has_value()},
                      {"chat", model.supports_generation() && defaults.chat.has_value() && output.tokenizer},
                      {"max_new_tokens", defaults.max_new_tokens},
                      {"lora_adapters", lora.names()}};
        add_capabilities(ready, model);
        ready["structured_outputs"] =
            model.supports_fused_batch() && output.tokenizer && output.tokenizer->byte_level();
        ready["function_calls"] =
            ready["structured_outputs"].get<bool>() && defaults.chat && defaults.chat->function_calls;
        ready["weight_bytes"] = model.memory().file_bytes;
        if (scheduler)
            ready["generation_context"] = scheduler->context();
        if (model.architecture() == "bert")
            ready["encoder_batch"] = true;
        write_output(ready);
    }
    const auto start = Clock::now();
    std::vector<TensorRequest> tensors;
    auto* pending = model.architecture() == "bert" ? &tensors : nullptr;
    const auto process = [&](const Line& line) {
        command(line, model, scheduler, output, path, sidecar, defaults, options, io, lora, pending);
    };
    const auto empty = [&] { return !scheduler || scheduler->empty(); };
    while (!input.done() || !empty()) {
        if (auto line = input.next(empty())) {
            process(*line);
            publish_metrics();
        }
        if (!tensors.empty()) {
            if (io.ready) {
                // Wait at most one millisecond to coalesce concurrent encoder requests.
                for (int i = 0; i < 7; ++i) {
                    const auto line = input.next(false, i == 0 ? 1 : 0);
                    if (!line)
                        break;
                    process(*line);
                }
            }
            flush_tensors(model, tensors);
        }
        if (!empty()) {
            const auto events = scheduler->step();
            publish_metrics();
            for (const auto& event : events)
                output.publish(event);
        }
    }
    if (io.stats) {
        const auto seconds = std::chrono::duration<double>(Clock::now() - start).count();
        write_output({{"event", "stats"},
                      {"output_tokens", output.output_tokens},
                      {"elapsed_ms", 1000 * seconds},
                      {"output_tokens_s", seconds > 0 ? output.output_tokens / seconds : 0.0},
                      {"draft_tokens", output.draft_tokens},
                      {"accepted_draft_tokens", output.accepted_draft_tokens},
                      {"peak_kv_bytes", scheduler ? scheduler->peak_kv_bytes() : 0},
                      {"resident_kv_bytes", scheduler ? scheduler->resident_kv_bytes() : 0},
                      {"kv_page_reuses", scheduler ? scheduler->kv_page_reuses() : 0},
                      {"queue_bypasses", scheduler ? scheduler->queue_bypasses() : 0}});
    }
}
} // namespace nathr::cli
