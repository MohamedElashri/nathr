#include <nathr/generation.hpp>
#include <nathr/tokenizer.hpp>

#include "backend/cpu/kernels.hpp"
#include "backend/cpu/thread_pool.hpp"
#include "batch.hpp"
#include "core/input.hpp"
#include "formats/gguf.hpp"
#include "requests.hpp"

#include <algorithm>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <sys/resource.h>
#include <unistd.h>

namespace {
using namespace nathr;
using nathr::detail::Json;

[[noreturn]] void invalid(const std::string& message) {
    throw Error(ErrorCode::invalid_argument, message);
}

class Arguments {
  public:
    Arguments(std::span<char*> arguments, bool accepts_prompt) {
        constexpr std::string_view flags[] = {"--bos",        "--cached", "--greedy", "--json",
                                              "--logits-f32", "--ready",  "--sample", "--skip-special",
                                              "--stages",     "--stats",  "--stdin",  "--summary"};
        bool positional_only = false;
        for (std::size_t i = 0; i < arguments.size(); ++i) {
            std::string key(arguments[i]);
            if (!positional_only && key == "--") {
                positional_only = true;
                continue;
            }
            if (positional_only || !key.starts_with("--")) {
                if (!has("--model"))
                    values_.emplace("--model", std::move(key));
                else if (accepts_prompt && !has("--prompt"))
                    values_.emplace("--prompt", std::move(key));
                else
                    invalid("unexpected argument: " + key);
                continue;
            }
            const auto equals = key.find('=');
            const auto name = key.substr(0, equals);
            const auto has_next_value =
                i + 1 < arguments.size() && !std::string_view(arguments[i + 1]).starts_with("--");
            if (name == "--stop") {
                if (equals != key.npos)
                    stops_.push_back(key.substr(equals + 1));
                else if (has_next_value)
                    stops_.emplace_back(arguments[++i]);
                else
                    invalid("missing value for --stop");
                continue;
            }
            auto [entry, added] = values_.try_emplace(name);
            if (!added)
                invalid("duplicate option: " + name);
            if (equals != key.npos) {
                entry->second = key.substr(equals + 1);
            } else if (has_next_value && std::ranges::find(flags, name) == std::end(flags)) {
                entry->second = arguments[++i];
            }
        }
    }

    bool flag(const std::string& key) {
        const auto item = values_.find(key);
        if (item == values_.end())
            return false;
        if (item->second)
            invalid("unexpected value for " + key);
        values_.erase(item);
        return true;
    }

    std::optional<std::string> option(const std::string& key) {
        const auto item = values_.find(key);
        if (item == values_.end())
            return {};
        if (!item->second)
            invalid("missing value for " + key);
        auto value = std::move(item->second);
        values_.erase(item);
        return value;
    }

    std::string required(const std::string& key) {
        auto value = option(key);
        if (!value)
            invalid("missing " + key);
        return std::move(*value);
    }

    bool has(const std::string& key) const { return values_.contains(key); }
    std::vector<std::string> stops() { return std::exchange(stops_, {}); }

    void finish() const {
        if (!values_.empty())
            invalid("unknown option: " + values_.begin()->first);
        if (!stops_.empty())
            invalid("unknown option: --stop");
    }

  private:
    std::map<std::string, std::optional<std::string>> values_;
    std::vector<std::string> stops_;
};

template <class Number> Number parse_number(std::string_view value, std::string_view option) {
    Number result{};
    const auto error = std::from_chars(value.data(), value.data() + value.size(), result);
    if (error.ec != std::errc{} || error.ptr != value.data() + value.size()) {
        invalid("invalid number for " + std::string(option));
    }
    return result;
}

template <class Number> Number number(Arguments& args, const std::string& option, Number fallback) {
    const auto value = args.option(option);
    return value ? parse_number<Number>(*value, option) : fallback;
}

std::vector<Token> parse_tokens(std::string_view text) {
    std::vector<Token> tokens;
    while (true) {
        const auto comma = text.find(',');
        const auto field = text.substr(0, comma);
        constexpr std::string_view whitespace = " \t\r\n";
        const auto first = field.find_first_not_of(whitespace);
        const auto trimmed = first == field.npos
                                 ? field.substr(0, 0)
                                 : field.substr(first, field.find_last_not_of(whitespace) - first + 1);
        tokens.push_back(parse_number<Token>(trimmed, "--tokens"));
        if (comma == text.npos)
            break;
        text.remove_prefix(comma + 1);
    }
    return tokens;
}

std::string read_stdin() {
    std::string text(detail::mib + 1, '\0');
    std::cin.read(text.data(), static_cast<std::streamsize>(text.size()));
    const auto count = static_cast<std::size_t>(std::cin.gcount());
    if (count > detail::mib)
        throw Error(ErrorCode::resource_limit, "standard input exceeds 1 MiB");
    if (std::cin.bad() || !std::cin.eof())
        throw Error(ErrorCode::io, "cannot read standard input");
    text.resize(count);
    return text;
}

BackendOptions backend_options(Arguments& args) {
    const auto backend = args.option("--backend").value_or("scalar");
    ExecutionPolicy policy;
    if (backend == "scalar")
        policy = ExecutionPolicy::scalar;
    else if (backend == "auto")
        policy = ExecutionPolicy::automatic;
    else if (backend == "avx2")
        policy = ExecutionPolicy::avx2;
    else
        throw Error(ErrorCode::unsupported, "unsupported backend: " + backend);
    const auto threads = number(args, "--threads", 1u);
    detail::validate_thread_count(threads);
    return {policy, threads};
}

SessionOptions session_options(Arguments& args, bool cached = true) {
    const auto cache = args.option("--kv-cache");
    const auto chunk = args.option("--prefill-chunk");
    if (!cached && (cache || chunk))
        invalid("--kv-cache and --prefill-chunk require --cached for forward");
    SessionOptions options;
    if (cache) {
        if (*cache == "f16")
            options.kv_precision = KvPrecision::f16;
        else if (*cache != "f32")
            invalid("--kv-cache must be f32 or f16");
    }
    if (chunk)
        options.prefill_chunk = parse_number<std::size_t>(*chunk, "--prefill-chunk");
    return options;
}

Json kernel_info(const Model& model) {
    Json kernels = Json::object();
    for (const auto& format : detail::encodings) {
        auto name = model.kernel(static_cast<std::uint32_t>(format.type));
        if (!name.empty())
            kernels[format.name] = name;
    }
    return kernels;
}

Json capabilities() {
    const auto detected = cpu_features();
    return {{"avx2", detected.avx2}, {"fma", detected.fma}, {"f16c", detected.f16c}};
}

Json gguf_info(const std::filesystem::path& path, BackendOptions options) {
    const auto features = cpu_features();
    if (options.policy == ExecutionPolicy::avx2 && !features.avx2)
        throw Error(ErrorCode::unsupported, "requested CPU policy is unavailable");
    const detail::Gguf file(path);
    const auto memory = file.memory();
    Json tensors = Json::object(), histogram = Json::object(), kernels = Json::object();
    bool supported = true;
    for (const auto& [name, item] : file.records()) {
        const auto& view = item.view;
        const auto& format = *detail::encoding(view.desc.dtype);
        const auto dimensions = view.desc.dimensions();
        auto shape = std::vector<std::size_t>(dimensions.rbegin(), dimensions.rend());
        tensors[name] = {{"shape", shape},
                         {"encoding", format.name},
                         {"id", static_cast<std::uint32_t>(format.type)},
                         {"encoded_bytes", view.bytes.size()},
                         {"offset", item.offset}};
        auto& totals = histogram[format.name];
        if (totals.is_null())
            totals = {{"tensors", 0}, {"parameters", 0}, {"bytes", 0}};
        totals["tensors"] = totals["tensors"].get<std::size_t>() + 1;
        totals["parameters"] = totals["parameters"].get<std::size_t>() + view.desc.elements;
        totals["bytes"] = totals["bytes"].get<std::size_t>() + view.bytes.size();
        const auto kernel = detail::select_kernel(format.type, options.policy, features);
        if (kernel.dot)
            kernels[format.name] = kernel.name;
        else
            supported = false;
    }
    const auto backend = options.policy != ExecutionPolicy::scalar && features.avx2 ? "avx2" : "scalar";
    return {{"format", "gguf"},
            {"version", 3},
            {"byte_order", "little"},
            {"file_bytes", memory.file_bytes},
            {"alignment", file.alignment()},
            {"data_offset", file.data_offset()},
            {"split", file.split()},
            {"architecture", file.architecture()},
            {"tokenizer_fingerprint", file.tokenizer_fingerprint()},
            {"output_tensor", file.records().contains("output.weight")},
            {"tensors", memory.tensors},
            {"tensor_table", tensors},
            {"dtype_histogram", histogram},
            {"encoded_weight_bytes", memory.encoded_weight_bytes},
            {"stored_parameters", memory.stored_parameters},
            {"effective_bits_per_parameter",
             memory.stored_parameters ? 8.0 * memory.encoded_weight_bytes / memory.stored_parameters : 0.0},
            {"mapped_file_bytes", memory.mapped_file_bytes},
            {"mapped_weight_bytes", memory.encoded_weight_bytes},
            {"owned_file_bytes", 0},
            {"all_encodings_executable", supported},
            {"linear_kernels", kernels},
            {"backend", backend},
            {"threads", options.threads},
            {"cpu_features", capabilities()}};
}

Json model_info(const Model& model) {
    const auto& c = model.config();
    const auto m = model.memory();
    Json info = {
        {"format", "safetensors"},
        {"backend", model.backend()},
        {"linear_kernels", kernel_info(model)},
        {"cpu_features", capabilities()},
        {"threads", model.threads()},
        {"layers", c.layers},
        {"hidden", c.hidden},
        {"intermediate", c.intermediate},
        {"query_heads", c.query_heads},
        {"kv_heads", c.kv_heads},
        {"head_dim", c.head_dim},
        {"tied_embeddings", c.tied_embeddings},
        {"bos_id", c.bos},
        {"eos_id", c.eos},
        {"owned_file_bytes", m.file_bytes},
        {"mapped_weight_bytes", 0},
        {"encoded_weight_bytes", m.encoded_weight_bytes},
        {"stored_parameters", m.stored_parameters},
        {"tensors", m.tensors},
        {"dtype_histogram", {{"F32", m.f32_tensors}, {"F16", m.f16_tensors}, {"BF16", m.bf16_tensors}}}};
    if (model.architecture() == "bert") {
        info["layer_norm_epsilon"] = c.rms_epsilon;
        info.erase("bos_id");
        info.erase("eos_id");
        info.erase("tied_embeddings");
    } else if (model.architecture() == "t5") {
        info["rms_epsilon"] = c.rms_epsilon;
        info.erase("bos_id");
    } else if (model.architecture() == "gpt2") {
        info["layer_norm_epsilon"] = c.rms_epsilon;
    } else {
        info["rope_theta"] = c.rope_theta;
        info["rms_epsilon"] = c.rms_epsilon;
    }
    cli::add_capabilities(info, model);
    return info;
}

void encode(Arguments& args, const std::filesystem::path& path, BackendOptions options) {
    const auto prompt = args.option("--prompt");
    const auto tokens = args.option("--tokens");
    if (prompt.has_value() == tokens.has_value())
        invalid("encode needs exactly one of --prompt or --tokens");
    (void)args.flag("--json");
    args.finish();
    const auto model = Model::load(path, options);
    const auto ids = prompt ? Tokenizer::load_model(path).encode(*prompt) : parse_tokens(*tokens);
    const auto activations = model.encode(ids);
    const auto width = model.config().hidden;
    Json rows = Json::array();
    for (std::size_t i = 0; i < ids.size(); ++i)
        rows.push_back(std::span(activations).subspan(i * width, width));
    std::cout << Json({{"tokens", ids}, {"hidden_size", width}, {"hidden_states", std::move(rows)}}).dump()
              << '\n';
}

void infer(Arguments& args, const std::filesystem::path& path, BackendOptions options) {
    const auto output = args.required("--output");
    auto prompt = args.option("--prompt");
    const auto stdin_prompt = args.flag("--stdin");
    const auto tokens = args.option("--tokens");
    const auto pair = args.option("--text-pair");
    const auto types = args.option("--token-type-ids");
    const auto decoder = args.option("--decoder-tokens");
    const auto sidecar = args.option("--tokenizer").value_or("");
    if (static_cast<int>(prompt.has_value()) + stdin_prompt + tokens.has_value() != 1 || (pair && tokens) ||
        (types && !tokens))
        invalid("infer needs a prompt, --stdin or --tokens IDS");
    args.finish();
    if (stdin_prompt)
        prompt = read_stdin();
    const auto model = Model::load(path, options);
    TokenizedInput input;
    if (prompt) {
        const auto tokenizer = cli::load_tokenizer(model, path, sidecar);
        if (pair)
            input = tokenizer.encode_pair(*prompt, *pair);
        else
            input.ids = tokenizer.encode(*prompt);
    } else {
        input.ids = parse_tokens(*tokens);
        if (types)
            input.token_type_ids = parse_tokens(*types);
    }
    const auto prefix = decoder ? parse_tokens(*decoder) : std::vector<Token>{};
    auto result = model.infer({input.ids, input.token_type_ids, prefix}, output);
    std::cout << Json({{"output", output},
                       {"shape", std::move(result.shape)},
                       {"values", std::move(result.values)}})
                     .dump()
              << '\n';
}

class TraceFile {
  public:
    void record(const TraceEvent& event) {
        const auto bytes = event.values.size_bytes();
        if (bytes > 63 * detail::mib - bytes_) {
            throw Error(ErrorCode::resource_limit, "trace tensor data exceeds 63 MiB");
        }
        bytes_ += bytes;
        auto name = std::string(event.operation);
        if (event.layer != TraceEvent::model_level) {
            name = "layer." + std::to_string(event.layer) + "." + name;
        }
        auto& result = rows_[name];
        if (event.position != result.positions || (result.positions && event.values.size() != result.width)) {
            throw Error(ErrorCode::backend, "inconsistent trace layout for " + name);
        }
        result.width = event.values.size();
        ++result.positions;
        result.values.insert(result.values.end(), event.values.begin(), event.values.end());
    }

    void write(const std::filesystem::path& path, std::span<const Token> tokens) const {
        Json header = {
            {"__metadata__",
             {{"schema_version", "1"}, {"axes", "position,feature"}, {"token_ids", Json(tokens).dump()}}}};
        std::size_t begin = 0;
        for (const auto& [name, rows] : rows_) {
            const auto end = begin + rows.values.size() * sizeof(float);
            header[name] = {
                {"dtype", "F32"}, {"shape", {rows.positions, rows.width}}, {"data_offsets", {begin, end}}};
            begin = end;
        }
        auto encoded = header.dump();
        encoded.append((8 - encoded.size() % 8) % 8, ' ');
        if (encoded.size() > detail::mib - 8)
            throw Error(ErrorCode::resource_limit, "trace header exceeds limit");
        std::ofstream file(path, std::ios::binary);
        if (!file)
            throw Error(ErrorCode::io, "cannot open trace " + path.string());
        static_assert(std::endian::native == std::endian::little);
        const std::uint64_t length = encoded.size();
        file.write(reinterpret_cast<const char*>(&length), sizeof(length));
        file.write(encoded.data(), static_cast<std::streamsize>(encoded.size()));
        for (const auto& [name, rows] : rows_) {
            file.write(reinterpret_cast<const char*>(rows.values.data()),
                       static_cast<std::streamsize>(rows.values.size() * sizeof(float)));
        }
        file.close();
        if (!file)
            throw Error(ErrorCode::io, "cannot write trace " + path.string());
    }

  private:
    struct Rows {
        std::size_t positions{}, width{};
        std::vector<float> values;
    };
    std::map<std::string, Rows> rows_;
    std::size_t bytes_{};
};

void forward(Arguments& args, const std::filesystem::path& path, BackendOptions options) {
    const auto ids = parse_tokens(args.required("--tokens"));
    const auto trace_path = args.option("--trace");
    const auto cached = args.flag("--cached");
    const auto settings = session_options(args, cached);
    const auto stream = args.flag("--logits-f32");
    const auto full = args.flag("--json");
    const auto context = number(args, "--context", std::size_t{0});
    if (!cached && context)
        invalid("--context requires --cached");
    if (stream && (trace_path || full))
        invalid("--logits-f32 cannot be combined with --trace or --json");
    args.finish();

    const auto model = Model::load(path, options);
    const auto mode = (stream || trace_path) ? LogitsMode::all_tokens : LogitsMode::last_token;
    std::vector<float> logits(stream ? 0 : model.config().vocabulary);
    TraceFile trace_file;
    TraceSink trace;
    LogitsSink sink;
    if (stream) {
        static_assert(std::endian::native == std::endian::little && sizeof(float) == 4);
        sink = [](std::size_t, std::span<const float> values) {
            std::cout.write(reinterpret_cast<const char*>(values.data()),
                            static_cast<std::streamsize>(values.size_bytes()));
            if (!std::cout)
                throw Error(ErrorCode::io, "cannot write FP32 logits to stdout");
        };
    } else if (trace_path) {
        trace = [&](const TraceEvent& event) { trace_file.record(event); };
        sink = [&](std::size_t pos, std::span<const float> output) {
            if (pos + 1 == ids.size())
                std::ranges::copy(output, logits.begin());
        };
    }
    if (cached) {
        auto session = Session::create(model, context, 0, settings);
        session.prefill(ids, mode, sink, trace);
        if (mode == LogitsMode::last_token)
            std::ranges::copy(session.logits(), logits.begin());
    } else {
        const auto last = mode == LogitsMode::last_token ? std::span<float>(logits) : std::span<float>{};
        model.forward(ids, mode, last, sink, trace);
    }
    if (stream) {
        std::cout.flush();
        if (!std::cout)
            throw Error(ErrorCode::io, "cannot flush FP32 logits to stdout");
        return;
    }
    if (trace_path)
        trace_file.write(*trace_path, ids);
    const auto greedy = std::ranges::max_element(logits) - logits.begin();
    Json result = {{"tokens", ids}, {"greedy_id", greedy}, {"backend", model.backend()}};
    if (full)
        result["logits"] = logits;
    std::cout << result.dump() << '\n';
}

std::string hex_bytes(std::string_view bytes) {
    constexpr std::string_view digits = "0123456789abcdef";
    std::string hex;
    hex.reserve(bytes.size() * 2);
    for (const unsigned char byte : bytes) {
        hex.push_back(digits[byte >> 4]);
        hex.push_back(digits[byte & 15]);
    }
    return hex;
}

Json text_tokens(const Tokenizer& tokenizer, std::string_view text, bool stages) {
    const auto ids = tokenizer.encode(text);
    Json result = {{"ids", ids},
                   {"decoded", tokenizer.decode(ids)},
                   {"decoded_skip_special", tokenizer.decode(ids, true)}};
    if (stages)
        result["pretokenized"] = tokenizer.pretokenize(text);
    return result;
}

void tokenize(Arguments& args, const std::filesystem::path& path) {
    const auto prompt = args.option("--prompt");
    const auto pair = args.option("--text-pair");
    const auto tokens = args.option("--tokens");
    const auto stdin_mode = args.flag("--stdin");
    const auto skip_special = args.flag("--skip-special");
    const auto stages = args.flag("--stages");
    if (static_cast<int>(prompt.has_value()) + tokens.has_value() + stdin_mode != 1 || (stages && tokens) ||
        (skip_special && !tokens) || (pair && (!prompt || stages))) {
        invalid("tokenize needs --prompt, --tokens or --stdin");
    }
    const auto sidecar = args.option("--tokenizer").value_or("");
    args.finish();
    const auto tokenizer = Tokenizer::load_model(path, sidecar);
    if (stdin_mode) {
        // JSON escaping can expand each input byte into six characters.
        const auto maximum = 6 * detail::mib + 2;
        std::vector<char> line(maximum + 2);
        while (std::cin.getline(line.data(), static_cast<std::streamsize>(line.size()))) {
            const auto length = static_cast<std::size_t>(std::cin.gcount()) - (std::cin.eof() ? 0 : 1);
            if (length > maximum)
                invalid("tokenizer input line exceeds limit");
            const auto text = detail::parse_json(std::as_bytes(std::span(line).first(length)), "<stdin>")
                                  .get<std::string>();
            std::cout << text_tokens(tokenizer, text, stages).dump() << '\n';
        }
        if (!std::cin.eof() || std::cin.bad())
            invalid("failed to read tokenizer JSON line within the size limit");
    } else if (pair) {
        auto encoded = tokenizer.encode_pair(*prompt, *pair);
        std::cout << Json({{"ids", std::move(encoded.ids)},
                           {"token_type_ids", std::move(encoded.token_type_ids)}})
                         .dump()
                  << '\n';
    } else if (prompt) {
        std::cout << text_tokens(tokenizer, *prompt, stages).dump() << '\n';
    } else {
        const auto ids = parse_tokens(*tokens);
        const auto bytes = tokenizer.decode_bytes(ids, skip_special);
        Utf8Stream decoder;
        auto decoded = decoder.push(bytes);
        decoded += decoder.finish();
        std::cout << Json({{"ids", ids}, {"decoded", decoded}, {"raw_hex", hex_bytes(bytes)}}).dump() << '\n';
    }
}

void emit_generation(auto& session, const Model& model, const SamplingOptions& sampling,
                     std::span<const Token> input, std::span<const Token> stop_ids, std::size_t max_tokens,
                     std::optional<GenerationText>& decoder, bool as_json) {
    Sampler sampler(sampling, model.config().vocabulary);
    std::vector<Token> emitted;
    const bool json = as_json || !decoder;
    std::string text;
    if (json)
        emitted.reserve(max_tokens);
    const auto append_text = [&](std::string_view piece) {
        if (json)
            text += piece;
        else
            std::cout << piece << std::flush;
    };
    const auto result = generate_until(session, sampler, model.config().eos, stop_ids, max_tokens,
                                       [&](Token token, bool stop_id) {
                                           if (json)
                                               emitted.push_back(token);
                                           if (decoder && !stop_id)
                                               append_text(decoder->push(token));
                                           return !decoder || !decoder->stopped();
                                       });
    if (decoder)
        append_text(decoder->finish());
    if (json) {
        Json output = {{model.architecture() == "t5" ? "source_ids" : "prompt_ids", input},
                       {"generated_ids", emitted},
                       {"emitted", result.emitted},
                       {"consumed", result.consumed},
                       {"stopped_at_eos", result.stopped_at_eos},
                       {"stopped_at_stop", result.stopped_at_stop},
                       {"finish_reason", result.stopped_at_eos || result.stopped_at_stop ? "stop" : "length"},
                       {"pending_id", result.pending ? Json(*result.pending) : Json(nullptr)}};
        if (decoder)
            output["text"] = text;
        std::cout << output.dump() << '\n';
    } else {
        std::cout << '\n';
    }
}

void generate_text(Arguments& args, const std::filesystem::path& path, BackendOptions options) {
    auto prompt = args.option("--prompt");
    const auto stdin_prompt = args.flag("--stdin");
    const auto tokens = args.option("--tokens");
    const auto messages = args.option("--messages");
    if (static_cast<int>(prompt.has_value()) + tokens.has_value() + messages.has_value() + stdin_prompt != 1)
        invalid("generate needs exactly one of --prompt or --tokens or --messages or --stdin");
    const auto model = Model::load(path, options);
    if (!model.supports_generation())
        invalid("BERT uses the infer or encode command");
    auto defaults = cli::request_defaults(path, model, messages.has_value());
    const auto bos = args.flag("--bos");
    const auto as_json = args.flag("--json");
    const auto max_tokens = number(args, "--max-new-tokens", defaults.max_new_tokens);
    const auto context = number(args, "--context", std::size_t{0});
    const auto seed = number(args, "--seed", std::uint64_t{0});
    const auto decoder_flags = args.has("--kv-cache") || args.has("--prefill-chunk");
    const auto settings = session_options(args);
    const bool greedy = args.flag("--greedy"), sample = args.flag("--sample");
    if (greedy && sample)
        invalid("choose --greedy or --sample");
    auto sampling = defaults.sampling;
    sampling.greedy = greedy || (!sample && sampling.greedy);
    sampling.temperature = number(args, "--temperature", sampling.temperature);
    sampling.top_k = number(args, "--top-k", sampling.top_k);
    sampling.top_p = number(args, "--top-p", sampling.top_p);
    auto stops = args.stops();
    if (const auto ids = args.option("--stop-ids")) {
        const auto additional = parse_tokens(*ids);
        defaults.stop_ids.insert(defaults.stop_ids.end(), additional.begin(), additional.end());
    }
    const auto sidecar = args.option("--tokenizer").value_or("");
    args.finish();
    const bool seq2seq = model.architecture() == "t5";
    if (seq2seq && (bos || decoder_flags || !sidecar.empty()))
        invalid("T5 does not accept --bos, --kv-cache, --prefill-chunk or --tokenizer");
    if (stdin_prompt)
        prompt = read_stdin();
    if (messages) {
        if (bos)
            invalid("--messages provides its own start token");
        if (*messages == "-") {
            const auto input = read_stdin();
            prompt = cli::render_chat(detail::parse_json(std::as_bytes(std::span(input)), "<stdin>"),
                                      defaults.chat_system);
        } else {
            prompt = cli::render_chat(detail::read_json(*messages, detail::mib), defaults.chat_system);
        }
    }
    std::optional<Tokenizer> tokenizer;
    if (prompt || (!seq2seq && !as_json) || !sidecar.empty() || !stops.empty())
        tokenizer = cli::load_tokenizer(model, path, sidecar);
    auto ids = prompt ? tokenizer->encode(*prompt) : parse_tokens(*tokens);
    if (bos)
        ids.insert(ids.begin(), model.config().bos);
    std::optional<GenerationText> decoder;
    if (tokenizer)
        decoder.emplace(std::move(*tokenizer), std::move(stops));
    if (seq2seq) {
        const auto capacity = context ? context : std::max<std::size_t>(1, max_tokens);
        if (max_tokens > capacity)
            invalid("T5 generation budget exceeds decoder context");
        auto session = T5Session::create(model, ids, capacity, seed);
        emit_generation(session, model, sampling, ids, defaults.stop_ids, max_tokens, decoder, as_json);
        return;
    }
    auto session = Session::create(model, context, seed, settings);
    if (ids.empty() || ids.size() > session.capacity() ||
        (max_tokens && max_tokens - 1 > session.capacity() - ids.size()))
        invalid("prompt and generation budget exceed context, or prompt is empty");
    session.prefill(ids);
    emit_generation(session, model, sampling, ids, defaults.stop_ids, max_tokens, decoder, as_json);
}

void batch(Arguments& args, const std::filesystem::path& path, BackendOptions options) {
    BatchOptions batch_options;
    batch_options.context = number(args, "--context", batch_options.context);
    batch_options.max_active = number(args, "--max-active", batch_options.max_active);
    batch_options.max_queued = number(args, "--max-queued", batch_options.max_queued);
    batch_options.max_queued_tokens = number(args, "--max-queued-tokens", batch_options.max_queued_tokens);
    batch_options.max_queue_bypasses = number(args, "--max-queue-bypasses", batch_options.max_queue_bypasses);
    batch_options.speculative_tokens = number(args, "--speculative-tokens", batch_options.speculative_tokens);
    batch_options.prefix_cache = !args.flag("--no-prefix-cache");
    batch_options.step_tokens = number(args, "--step-tokens", batch_options.step_tokens);
    batch_options.kv_budget_bytes = detail::product(
        {number(args, "--kv-budget-mib", batch_options.kv_budget_bytes / detail::mib), detail::mib},
        "batch KV budget");
    batch_options.session = session_options(args);
    const cli::BatchIoOptions io{args.flag("--stats"), args.flag("--ready"),
                                 number(args, "--max-result-values", std::size_t{0})};
    const auto tokenizer = args.option("--tokenizer").value_or("");
    args.finish();
    cli::run_batch(Model::load(path, options), path, tokenizer, batch_options, io);
}

void launch_frontend(std::string_view action, std::span<char*> options, bool help) {
    const auto executable = std::filesystem::read_symlink("/proc/self/exe");
    const auto bin = executable.parent_path();
    const auto script = bin / "nathr-serve.py";
    if (!std::filesystem::is_regular_file(script))
        throw Error(ErrorCode::io, "cannot find the installed nathr-serve.py beside nathr");
    auto data = bin / "nathr-data";
    if (!std::filesystem::is_directory(data))
        data = bin / NATHR_BIN_TO_DATA;
    std::vector<std::string> arguments{
        "python3", "-I", "-B", script.string(), executable.string(), data.string(), std::string(action)};
    if (help)
        arguments.emplace_back("--help");
    else
        arguments.insert(arguments.end(), options.begin(), options.end());
    std::vector<char*> command;
    for (auto& argument : arguments)
        command.push_back(argument.data());
    command.push_back(nullptr);
    execvp(command.front(), command.data());
    throw Error(ErrorCode::io, "cannot start Python 3 for nathr " + std::string(action));
}

long process_peak_rss_kib() {
    rusage info{};
    if (getrusage(RUSAGE_SELF, &info) != 0)
        return -1;
    return info.ru_maxrss;
}

Json distribution(std::vector<double> values) {
    std::ranges::sort(values);
    const auto quantile = [](const std::vector<double>& sorted, double fraction) {
        const auto offset = fraction * static_cast<double>(sorted.size() - 1);
        const auto low = static_cast<std::size_t>(offset);
        return std::lerp(sorted[low], sorted[std::min(low + 1, sorted.size() - 1)],
                         offset - static_cast<double>(low));
    };
    const double median = quantile(values, 0.5);
    const double p95 = quantile(values, 0.95);
    const double mean =
        std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
    const double squared_error =
        std::accumulate(values.begin(), values.end(), 0.0,
                        [&](double sum, double value) { return sum + (value - mean) * (value - mean); });
    for (auto& value : values)
        value = std::abs(value - median);
    std::ranges::sort(values);
    return {{"median", median},
            {"p95", p95},
            {"median_absolute_deviation", quantile(values, 0.5)},
            {"coefficient_of_variation",
             values.size() > 1 ? std::sqrt(squared_error / (values.size() - 1)) / mean : 0.0}};
}

void bench(Arguments& args, const std::filesystem::path& path, BackendOptions options) {
    using Clock = std::chrono::steady_clock;
    const auto milliseconds = [](auto delta) {
        return std::chrono::duration<double, std::milli>(delta).count();
    };
    const auto prompt_length = number(args, "--prompt-length", std::size_t{128});
    const auto decode_length = number(args, "--decode-length", std::size_t{32});
    const auto context = number(args, "--context", std::size_t{512});
    const auto repetitions = number(args, "--repetitions", std::size_t{7});
    const auto warmups = number(args, "--warmups", std::size_t{2});
    const auto settings = session_options(args);
    (void)args.flag("--json");
    args.finish();
    if (!prompt_length || !decode_length || prompt_length > context ||
        decode_length > context - prompt_length || !repetitions || repetitions > 100 || warmups > 100)
        invalid("invalid benchmark lengths or repetitions");

    const auto start = Clock::now();
    const auto model = Model::load(path, options);
    const auto loaded = Clock::now();
    auto session = Session::create(model, context, 0, settings);
    const auto session_created = Clock::now();
    auto ids = std::vector<Token>(prompt_length + decode_length);
    const auto vocabulary = model.config().vocabulary;
    const auto base = vocabulary > 17 ? 17 : 0;
    for (std::size_t i = 0; i < ids.size(); ++i)
        ids[i] = static_cast<Token>(base + (i * 7919 + 12345) % (vocabulary - base));
    const auto all = std::span(ids);
    std::vector<double> prefill, decode, inter_token;
    prefill.reserve(repetitions);
    decode.reserve(repetitions);
    inter_token.reserve(repetitions * decode_length);
    for (std::size_t i = 0; i < warmups + repetitions; ++i) {
        session.reset();
        const auto beginning = Clock::now();
        session.prefill(all.first(prompt_length));
        const auto prefetched = Clock::now();
        auto previous = prefetched;
        for (const auto token : all.subspan(prompt_length)) {
            session.append(std::span(&token, 1));
            const auto ready = Clock::now();
            if (i >= warmups)
                inter_token.push_back(milliseconds(ready - previous));
            previous = ready;
        }
        if (i >= warmups) {
            prefill.push_back(milliseconds(prefetched - beginning));
            decode.push_back(milliseconds(previous - prefetched));
        }
    }
    const auto p = distribution(prefill);
    const auto d = distribution(decode);
    const auto memory = session.memory();
    const auto disk = model.memory();
    std::cout << Json({{"backend", model.backend()},
                       {"threads", options.threads},
                       {"cpu_policy", "x86-64 FP32 activations, FP64 reductions, FMA contraction off"},
                       {"compiler", __VERSION__},
                       {"warmups", warmups},
                       {"context", context},
                       {"prompt_length", prompt_length},
                       {"decode_length", decode_length},
                       {"model_load_ms", milliseconds(loaded - start)},
                       {"session_create_ms", milliseconds(session_created - loaded)},
                       {"prefill_samples_ms", prefill},
                       {"decode_samples_ms", decode},
                       {"prefill_ms", p},
                       {"ttft_ms", p},
                       {"ttft_scope", "prefill to first logits, excluding model load, session creation, "
                                      "tokenization and sampling"},
                       {"decode_ms", d},
                       {"inter_token_ms", distribution(std::move(inter_token))},
                       {"inter_token_samples", repetitions * decode_length},
                       {"prefill_tokens_s", 1000.0 * prompt_length / p.at("median").get<double>()},
                       {"decode_tokens_s", 1000.0 * decode_length / d.at("median").get<double>()},
                       {"peak_rss_kib", process_peak_rss_kib()},
                       {"encoded_weight_bytes", disk.encoded_weight_bytes},
                       {"owned_file_bytes", disk.file_bytes - disk.mapped_file_bytes},
                       {"mapped_file_bytes", disk.mapped_file_bytes},
                       {"mapped_weight_bytes", disk.mapped_file_bytes ? disk.encoded_weight_bytes : 0},
                       {"linear_kernels", kernel_info(model)},
                       {"cpu_features", capabilities()},
                       {"kv_cache_bytes", memory.cache_bytes},
                       {"kv_cache_dtype", settings.kv_precision == KvPrecision::f16 ? "F16" : "F32"},
                       {"prefill_chunk", std::min(context, settings.prefill_chunk)},
                       {"workspace_bytes", memory.workspace_bytes},
                       {"history_bytes", memory.history_bytes}})
                     .dump(2)
              << '\n';
}

struct CommandHelp {
    std::string_view name, description, synopsis, options;
    bool accepts_prompt = false;
};

constexpr CommandHelp commands[] = {
    {"run", "Start a terminal session or run one prompt", "run [MODEL] [PROMPT]", ""},
    {"models", "List pinned checkpoint names and cache status", "models", ""},
    {"pull", "Download and verify a named checkpoint", "pull MODEL", ""},
    {"generate", "Continue a prompt or run chat messages",
     "generate MODEL (PROMPT | --stdin | --tokens IDS | --messages FILE)",
     "  --stdin                       Read one plain-text prompt through EOF.\n"
     "  --messages FILE               Read a JSON array of chat messages. Use - for stdin.\n"
     "  --max-new-tokens N            Set the token budget (checkpoint default or 32).\n"
     "  --greedy | --sample           Choose the highest score or sample.\n"
     "  --temperature T --top-k K --top-p P --seed N\n"
     "  --stop TEXT                   Stop on text. Repeat to add more strings.\n"
     "  --stop-ids IDS                Add comma-separated stop token IDs.\n"
     "  --bos                         Prepend the start token to decoder-only input.\n"
     "  --context N                   Limit session context (decoder context for T5).\n"
     "  --kv-cache f32|f16 --prefill-chunk N\n"
     "  --tokenizer FILE              Use a matching GGUF tokenizer sidecar.\n"
     "  --json                        Include token IDs and the finish reason.\n",
     true},
    {"infer", "Return a named tensor as JSON", "infer MODEL (PROMPT | --stdin | --tokens IDS) --output NAME",
     "  --output NAME                 Select an output from nathr inspect MODEL --summary.\n"
     "  --stdin                       Read one plain-text prompt through EOF.\n"
     "  --text-pair TEXT               Add a second segment to BERT text input.\n"
     "  --token-type-ids IDS           Supply BERT segment IDs with --tokens.\n"
     "  --decoder-tokens IDS          Supply a T5 prefix beginning with its decoder start ID.\n"
     "  --tokenizer FILE               Use a matching GGUF tokenizer sidecar.\n",
     true},
    {"inspect", "Show checkpoint capabilities and outputs", "inspect MODEL",
     "  --summary                     Print a readable list of capabilities and outputs.\n"
     "  --json                        Print JSON (default).\n"},
    {"tokenize", "Convert text to token IDs and back", "tokenize MODEL (PROMPT | --tokens IDS | --stdin)",
     "  --stdin                       Read one JSON string per line.\n"
     "  --stages                      Show pretokenization for a text input.\n"
     "  --text-pair TEXT               Return BERT token and segment IDs for a text pair.\n"
     "  --skip-special                 Omit special tokens when decoding --tokens.\n"
     "  --tokenizer FILE               Use a matching GGUF tokenizer sidecar.\n",
     true},
    {"batch", "Run JSONL requests with one loaded model", "batch MODEL",
     "  --max-active N                Set the number of active requests (default 4).\n"
     "  --max-queued N                Set the request queue limit (default 32).\n"
     "  --max-queued-tokens N         Limit queued prompt tokens (default 16384).\n"
     "  --max-queue-bypasses N        Let up to N fitting requests pass a blocked queue head (default 0).\n"
     "  --step-tokens N               Set the total tokens per scheduler step (default 32).\n"
     "  --speculative-tokens N        Verify up to 8 prompt lookup drafts in greedy decoding (default 0).\n"
     "  --kv-budget-mib N             Limit KV memory (default 512).\n"
     "  --context N --kv-cache f32|f16 --prefill-chunk N\n"
     "  --no-prefix-cache             Disable sharing of computed decoder KV pages.\n"
     "  --tokenizer FILE              Use a matching GGUF tokenizer sidecar.\n"
     "  --stats                       Emit timings and KV stats after the batch drains.\n"},
    {"serve", "Start the local OpenAI-compatible API", "serve [MODEL]", ""},
    {"bench", "Measure decoder inference or HTTP serving", "bench MODEL | nathr bench --serve",
     "  --serve                       Benchmark a running server (see bench --serve --help).\n"
     "  --prompt-length N             Set the synthetic prompt length (default 128).\n"
     "  --decode-length N             Set the decode length (default 32).\n"
     "  --context N                   Set session capacity (default 512).\n"
     "  --warmups N                    Set unmeasured runs (default 2).\n"
     "  --repetitions N                Set measured runs (default 7).\n"
     "  --kv-cache f32|f16 --prefill-chunk N\n"
     "  --json                        Print JSON (default).\n"},
    {"encode", "Return BERT hidden-state rows as JSON", "encode MODEL (PROMPT | --tokens IDS)", "", true},
    {"forward", "Inspect decoder-only logits and activations", "forward MODEL --tokens IDS",
     "  --cached                      Run with a KV cache.\n"
     "  --context N --kv-cache f32|f16 --prefill-chunk N\n"
     "  --json                        Include the last logit row in JSON.\n"
     "  --trace FILE                  Save intermediate activations.\n"
     "  --logits-f32                  Stream little-endian FP32 logit rows.\n"},
};

const CommandHelp* find_command(std::string_view name) {
    const auto command = std::ranges::find(commands, name, &CommandHelp::name);
    return command == std::end(commands) ? nullptr : command;
}

void usage(const CommandHelp* command = nullptr) {
    if (!command) {
        std::cout << "Usage: nathr COMMAND [OPTIONS]\n"
                     "       nathr help [COMMAND] | --version\n\n"
                     "Commands\n";
        for (const auto& item : commands)
            std::cout << "  " << item.name << std::string(11 - item.name.size(), ' ') << item.description
                      << '\n';
        std::cout << "\nRun nathr COMMAND --help for options and input formats.\n";
        return;
    }
    std::cout << "Usage: nathr " << command->synopsis << "\n\n"
              << command->description << ".\n\n"
              << "MODEL is a GGUF file or a Safetensors checkpoint directory. Use --model PATH as a named "
                 "option.\n";
    if (command->accepts_prompt)
        std::cout << "Use --prompt TEXT for a named prompt, or -- before a prompt that starts with --.\n";
    std::cout << "\n" << command->options;
    if (command->name != "tokenize")
        std::cout << "  --backend scalar|auto|avx2    Select CPU kernels (default scalar).\n"
                     "  --threads N                   Set CPU threads (default 1).\n";
    std::cout << "  -h, --help                    Show this help.\n";
}

void print_summary(const std::filesystem::path& path, const Json& info) {
    const auto& arch = info.at("architecture");
    const auto architecture =
        arch.is_string() ? arch.get<std::string>() : arch.value("general.architecture", "unknown");
    const auto yes_no = [](bool available) { return available ? "yes" : "no"; };
    std::cout << path.string() << "\n"
              << "  Format          " << info.at("format").get<std::string>() << '\n'
              << "  Architecture    " << architecture << '\n'
              << "  Backend         " << info.at("backend").get<std::string>() << '\n';
    if (info.contains("context"))
        std::cout << "  Context         " << info.at("context") << " tokens\n"
                  << "  Vocabulary      " << info.at("vocabulary") << " tokens\n";
    if (info.contains("decoder_start_id"))
        std::cout << "  Decoder start   " << info.at("decoder_start_id") << '\n';
    std::cout << "  Generation      " << yes_no(info.at("generation").get<bool>()) << '\n'
              << "  Chat template   " << yes_no(info.at("chat").get<bool>()) << '\n';
    if (info.contains("load_error")) {
        std::cout << "  Load error      " << info.at("load_error").get<std::string>() << '\n';
        return;
    }
    std::cout << "\nNamed outputs\n";
    for (const auto& output : info.at("outputs"))
        std::cout << "  " << output.at("name").get<std::string>() << " (width " << output.at("width")
                  << ", up to " << output.at("input_limit") << " input tokens, "
                  << (output.at("sequence") == true ? "sequence" : "vector") << ")\n";
}

} // namespace

int main(int argc, char** argv) try {
    if (argc == 1 || std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h") {
        usage();
        return 0;
    }
    std::string_view command(argv[1]);
    if (command == "--version" && argc == 2) {
        std::cout << "nathr " << NATHR_VERSION << '\n';
        return 0;
    }
    const auto arguments = std::span(argv + 2, static_cast<std::size_t>(argc - 2));
    if (command == "bench" && !arguments.empty() && std::string_view(arguments.front()) == "--serve")
        launch_frontend("bench-serve", arguments.subspan(1), false);
    bool help = command == "help";
    if (help) {
        if (arguments.empty()) {
            usage();
            return 0;
        }
        if (arguments.size() != 1)
            invalid("help takes one command name");
        command = arguments.front();
    } else {
        for (const std::string_view argument : arguments) {
            if (argument == "--")
                break;
            if (argument == "-h" || argument == "--help") {
                help = true;
                break;
            }
        }
    }
    const auto* info = find_command(command);
    if (!info)
        invalid("unknown command: " + std::string(command));
    if (command == "serve" || command == "run" || command == "models" || command == "pull")
        launch_frontend(command, arguments, help);
    if (help) {
        usage(info);
        return 0;
    }
    Arguments args(arguments, info->accepts_prompt);
    const auto path = std::filesystem::path(args.required("--model"));
    if (command == "tokenize") {
        tokenize(args, path);
        return 0;
    }
    const auto options = backend_options(args);
    if (command == "inspect") {
        const auto summary = args.flag("--summary");
        if (args.flag("--json") && summary)
            invalid("choose --summary or --json");
        args.finish();
        auto info = std::filesystem::is_directory(path) ? model_info(Model::load(path, options))
                                                        : gguf_info(path, options);
        if (info.at("format") == "gguf") {
            try {
                cli::add_capabilities(info, Model::load(path, options));
            } catch (const Error& error) {
                info["outputs"] = Json::array();
                info["generation"] = false;
                info["fused_decoder_batch"] = false;
                info["load_error"] = error.what();
            }
        }
        info["chat"] = info.value("generation", false) && cli::chat_system(path).has_value();
        if (summary)
            print_summary(path, info);
        else
            std::cout << info.dump(2) << '\n';
    } else if (command == "forward") {
        forward(args, path, options);
    } else if (command == "encode") {
        encode(args, path, options);
    } else if (command == "infer") {
        infer(args, path, options);
    } else if (command == "generate") {
        generate_text(args, path, options);
    } else if (command == "batch") {
        batch(args, path, options);
    } else if (command == "bench") {
        bench(args, path, options);
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr << "nathr: " << error.what() << '\n';
    if (const auto* typed = dynamic_cast<const Error*>(&error);
        typed && typed->code() == ErrorCode::invalid_argument) {
        const auto command = argc > 1 ? find_command(argv[1]) : nullptr;
        if (command)
            std::cerr << "See nathr " << command->name << " --help.\n";
        else
            std::cerr << "See nathr --help.\n";
    }
    return 1;
}
