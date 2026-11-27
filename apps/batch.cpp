#include "batch.hpp"

#include "core/input.hpp"
#include "requests.hpp"

#include <nathr/tokenizer.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <iostream>
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
    std::optional<Line> next(bool wait) {
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
            const auto ready = poll(&input, 1, wait ? -1 : 0);
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

struct BatchOutput {
    std::optional<Tokenizer> tokenizer;
    std::size_t output_tokens = 0;
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
        messages == json.end() ? prompt->get<std::string>() : render_chat(*messages, defaults.chat_system);
    if (pair != json.end())
        return output.tokenizer->encode_pair(text, pair->get<std::string>());
    return {output.tokenizer->encode(text), {}};
}

void command(const Line& line, const Model& model, std::optional<BatchScheduler>& scheduler,
             BatchOutput& output, const std::filesystem::path& path, const std::filesystem::path& sidecar,
             const RequestDefaults& defaults, std::size_t max_result_values) {
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
            const auto cancellation = scheduler ? scheduler->cancel(*id) : std::optional<BatchEvent>{};
            if (!cancellation)
                throw Error(ErrorCode::invalid_argument, "request ID is not pending");
            output.publish(*cancellation);
            return;
        }
        id = detail::natural(json.at("id"), "id");
        const auto tensor = json.find("output");
        if (tensor != json.end() && scheduler && scheduler->contains(*id))
            throw Error(ErrorCode::invalid_argument, "duplicate request ID");
        constexpr std::string_view common[] = {"id", "prompt", "tokens"};
        constexpr std::string_view tensors[] = {"output", "text_pair", "token_type_ids", "decoder_tokens"};
        constexpr std::string_view generation[] = {"max_new_tokens", "seed",   "greedy",      "temperature",
                                                   "top_k",          "top_p",  "messages",    "stop",
                                                   "stop_ids",       "decode", "top_logprobs"};
        const auto extra = tensor != json.end() ? std::span<const std::string_view>(tensors)
                                                : std::span<const std::string_view>(generation);
        for (const auto& [key, value] : json.items()) {
            if (std::ranges::find(common, key) == std::end(common) &&
                std::ranges::find(extra, key) == extra.end())
                throw Error(ErrorCode::invalid_argument, "unknown batch request field: " + key);
        }
        const bool decode = tensor == json.end() && (json.contains("prompt") || json.contains("messages") ||
                                                     json.contains("stop") || json.value("decode", false));
        auto input = input_tokens(json, output, model, path, sidecar, defaults, decode);
        if (tensor != json.end()) {
            const auto name = tensor->get<std::string>();
            const auto prefix = json.contains("decoder_tokens")
                                    ? token_array(json.at("decoder_tokens"), "decoder_tokens")
                                    : std::vector<Token>{};
            if (max_result_values) {
                const auto outputs = model.outputs();
                const auto shape = std::ranges::find(outputs, name, &OutputInfo::name);
                if (shape != outputs.end()) {
                    const auto rows = !shape->sequence               ? 1
                                      : model.architecture() == "t5" ? std::max<std::size_t>(1, prefix.size())
                                                                     : input.ids.size();
                    if (rows > max_result_values / shape->width)
                        throw Error(ErrorCode::resource_limit, "named tensor exceeds response limit");
                }
            }
            auto result = model.infer({input.ids, input.token_type_ids, prefix}, name);
            write_output({{"event", "result"},
                          {"id", *id},
                          {"output", name},
                          {"prompt_tokens", input.ids.size()},
                          {"shape", std::move(result.shape)},
                          {"values", std::move(result.values)}});
            return;
        }
        if (!scheduler)
            throw Error(ErrorCode::unsupported, "this checkpoint requires an output name in batch requests");
        auto stops = defaults.stop_ids;
        if (const auto field = json.find("stop_ids"); field != json.end()) {
            const auto ids = token_array(*field, "stop_ids");
            stops.insert(stops.end(), ids.begin(), ids.end());
        }
        scheduler->submit(
            {*id,
             std::move(input.ids),
             integer(json, "max_new_tokens", defaults.max_new_tokens),
             integer(json, "seed", 0),
             {json.value("greedy", defaults.sampling.greedy),
              json.value("temperature", defaults.sampling.temperature),
              integer(json, "top_k", defaults.sampling.top_k), json.value("top_p", defaults.sampling.top_p)},
             std::move(stops),
             decode ? output.tokenizer : std::optional<Tokenizer>{},
             json.contains("stop") ? stop_strings(json.at("stop")) : std::vector<std::string>{},
             json.contains("top_logprobs")
                 ? std::optional(detail::natural(json.at("top_logprobs"), "top_logprobs"))
                 : std::nullopt});
    } catch (const std::exception& error) {
        const auto* typed = dynamic_cast<const Error*>(&error);
        const auto code =
            typed && typed->code() == ErrorCode::resource_limit ? "resource_limit" : "invalid_request";
        write_output({{"event", "error"},
                      {"id", id ? Json(*id) : Json(nullptr)},
                      {"message", error.what()},
                      {"code", code}});
    }
}
} // namespace

void run_batch(const Model& model, const std::filesystem::path& path, const std::filesystem::path& sidecar,
               BatchOptions options, BatchIoOptions io) {
    std::optional<BatchScheduler> scheduler;
    if (model.supports_generation())
        scheduler.emplace(model, options);
    BatchOutput output;
    if (io.ready && scheduler)
        output.live_stats = &*scheduler;
    BatchInput input;
    const auto defaults = request_defaults(path, model, true);
    if (io.ready) {
        try {
            output.tokenizer = load_tokenizer(model, path, sidecar);
        } catch (const Error&) {
            // Checkpoints with external tokenizers can still accept raw IDs.
        }
        Json ready = {
            {"event", "ready"},
            {"text", output.tokenizer.has_value()},
            {"chat", model.supports_generation() && defaults.chat_system.has_value() && output.tokenizer},
            {"max_new_tokens", defaults.max_new_tokens}};
        add_capabilities(ready, model);
        if (scheduler)
            ready["generation_context"] = scheduler->context();
        write_output(ready);
    }
    const auto start = Clock::now();
    const auto empty = [&] { return !scheduler || scheduler->empty(); };
    while (!input.done() || !empty()) {
        if (auto line = input.next(empty()))
            command(*line, model, scheduler, output, path, sidecar, defaults, io.max_result_values);
        if (!empty()) {
            for (const auto& event : scheduler->step())
                output.publish(event);
        }
    }
    if (io.stats) {
        const auto seconds = std::chrono::duration<double>(Clock::now() - start).count();
        write_output({{"event", "stats"},
                      {"output_tokens", output.output_tokens},
                      {"elapsed_ms", 1000 * seconds},
                      {"output_tokens_s", seconds > 0 ? output.output_tokens / seconds : 0.0},
                      {"peak_kv_bytes", scheduler ? scheduler->peak_kv_bytes() : 0},
                      {"resident_kv_bytes", scheduler ? scheduler->resident_kv_bytes() : 0},
                      {"kv_page_reuses", scheduler ? scheduler->kv_page_reuses() : 0},
                      {"queue_bypasses", scheduler ? scheduler->queue_bypasses() : 0}});
    }
}
} // namespace nathr::cli
