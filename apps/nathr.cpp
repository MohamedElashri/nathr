#include <nathr/generation.hpp>
#include <nathr/tokenizer.hpp>

#include "backend/cpu/kernels.hpp"
#include "backend/cpu/thread_pool.hpp"
#include "core/input.hpp"
#include "formats/gguf.hpp"

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

namespace {
using namespace nathr;
using nathr::detail::Json;

[[noreturn]] void invalid(const std::string& message) {
    throw Error(ErrorCode::invalid_argument, message);
}

class Arguments {
  public:
    Arguments(int argc, char** argv, bool accepts_prompt) {
        int i = 2;
        if (i < argc && !std::string_view(argv[i]).starts_with("--")) {
            values_.emplace("--model", argv[i++]);
            if (accepts_prompt && i < argc && !std::string_view(argv[i]).starts_with("--"))
                values_.emplace("--prompt", argv[i++]);
        }
        for (; i < argc; ++i) {
            std::string key(argv[i]);
            if (!key.starts_with("--"))
                invalid("expected an option: " + key);
            const auto equals = key.find('=');
            auto [entry, added] = values_.try_emplace(key.substr(0, equals));
            if (!added)
                invalid("duplicate option: " + key);
            if (equals != key.npos) {
                entry->second = key.substr(equals + 1);
            } else if (i + 1 < argc && !std::string_view(argv[i + 1]).starts_with("--")) {
                entry->second = argv[++i];
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

    void finish() const {
        if (!values_.empty())
            invalid("unknown option: " + values_.begin()->first);
    }

  private:
    std::map<std::string, std::optional<std::string>> values_;
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
        tokens.push_back(parse_number<Token>(text.substr(0, comma), "--tokens"));
        if (comma == text.npos)
            break;
        text.remove_prefix(comma + 1);
    }
    return tokens;
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
        {"context", c.context},
        {"vocabulary", c.vocabulary},
        {"tied_embeddings", c.tied_embeddings},
        {"bos_id", c.bos},
        {"eos_id", c.eos},
        {"owned_file_bytes", m.file_bytes},
        {"mapped_weight_bytes", 0},
        {"encoded_weight_bytes", m.encoded_weight_bytes},
        {"stored_parameters", m.stored_parameters},
        {"tensors", m.tensors},
        {"dtype_histogram", {{"F32", m.f32_tensors}, {"F16", m.f16_tensors}, {"BF16", m.bf16_tensors}}}};
    if (model.architecture() == "gpt2") {
        info["architecture"] = model.architecture();
        info["layer_norm_epsilon"] = c.rms_epsilon;
    } else {
        info["rope_theta"] = c.rope_theta;
        info["rms_epsilon"] = c.rms_epsilon;
    }
    return info;
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
    const auto tokens = args.option("--tokens");
    const auto stdin_mode = args.flag("--stdin");
    const auto skip_special = args.flag("--skip-special");
    const auto stages = args.flag("--stages");
    if (static_cast<int>(prompt.has_value()) + tokens.has_value() + stdin_mode != 1 || (stages && tokens) ||
        (skip_special && !tokens)) {
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

void generate_text(Arguments& args, const std::filesystem::path& path, BackendOptions options) {
    const auto prompt = args.option("--prompt");
    const auto tokens = args.option("--tokens");
    const auto bos = args.flag("--bos");
    const auto as_json = args.flag("--json");
    const auto max_tokens = number(args, "--max-new-tokens", std::size_t{32});
    const auto context = number(args, "--context", std::size_t{0});
    const auto seed = number(args, "--seed", std::uint64_t{0});
    const auto settings = session_options(args);
    const SamplingOptions sampling{args.flag("--greedy"), number(args, "--temperature", 1.0),
                                   number(args, "--top-k", std::size_t{0}), number(args, "--top-p", 1.0)};
    if (prompt.has_value() == tokens.has_value())
        invalid("generate needs exactly one of --prompt or --tokens");
    const auto sidecar = args.option("--tokenizer").value_or("");
    args.finish();
    const auto model = Model::load(path, options);
    std::optional<Tokenizer> tokenizer;
    if (prompt || !as_json || !sidecar.empty())
        tokenizer = Tokenizer::load_model(path, sidecar);
    if (tokenizer && tokenizer->vocabulary() != model.config().vocabulary)
        invalid("model and tokenizer vocabulary differ");
    auto ids = prompt ? tokenizer->encode(*prompt) : parse_tokens(*tokens);
    if (bos)
        ids.insert(ids.begin(), model.config().bos);
    auto session = Session::create(model, context, seed, settings);
    if (ids.empty() || ids.size() > session.capacity() ||
        (max_tokens && max_tokens - 1 > session.capacity() - ids.size())) {
        invalid("prompt and generation budget exceed context, or prompt is empty");
    }
    Sampler sampler(sampling, model.config().vocabulary);
    session.prefill(ids);
    Utf8Stream decoder;
    std::vector<Token> emitted;
    std::string text;
    if (as_json)
        emitted.reserve(max_tokens);
    const auto append_text = [&](std::string_view piece) {
        if (as_json)
            text += piece;
        else
            std::cout << piece << std::flush;
    };
    const auto result = nathr::generate(session, sampler, model.config().eos, max_tokens, [&](Token token) {
        if (as_json)
            emitted.push_back(token);
        if (tokenizer)
            append_text(decoder.push(tokenizer->token_bytes(token, true)));
    });
    if (tokenizer)
        append_text(decoder.finish());
    if (as_json) {
        Json output = {{"prompt_ids", ids},
                       {"generated_ids", emitted},
                       {"emitted", result.emitted},
                       {"consumed", result.consumed},
                       {"stopped_at_eos", result.stopped_at_eos},
                       {"pending_id", result.pending ? Json(*result.pending) : Json(nullptr)}};
        if (tokenizer)
            output["text"] = text;
        std::cout << output.dump() << '\n';
    } else {
        std::cout << '\n';
    }
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

void usage() {
    std::cout << "nathr --help | --version\n"
                 "nathr inspect MODEL\n"
                 "nathr forward MODEL --tokens 1,2,3 [--cached] [--context N] [--trace FILE] [--json]\n"
                 "  --logits-f32 streams little-endian FP32 vocabulary rows to stdout\n"
                 "nathr tokenize MODEL (PROMPT | --tokens IDS | --stdin) [--tokenizer FILE]\n"
                 "  --stdin reads one JSON string per line. --stages shows pretokenization.\n"
                 "  --skip-special omits special tokens when decoding --tokens.\n"
                 "nathr generate MODEL (PROMPT | --tokens IDS) [--bos] [--max-new-tokens N] "
                 "[--context N]\n"
                 "  [--greedy | --temperature T --top-k K --top-p P] [--seed N] [--tokenizer FILE] [--json]\n"
                 "nathr bench MODEL [--prompt-length N] [--decode-length N] [--context N] "
                 "[--warmups N] [--repetitions N] [--json]\n"
                 "Use --model PATH and --prompt TEXT for named arguments.\n"
                 "Execution options: --backend scalar|auto|avx2 --threads 1..64\n"
                 "Cached execution: --kv-cache f32|f16 --prefill-chunk 1..32\n";
}

} // namespace

int main(int argc, char** argv) try {
    if (argc == 1 || std::string_view(argv[1]) == "--help") {
        usage();
        return 0;
    }
    const auto command = std::string_view(argv[1]);
    if (command == "--version" && argc == 2) {
        std::cout << "nathr " << NATHR_VERSION << '\n';
        return 0;
    }
    if (command != "inspect" && command != "forward" && command != "tokenize" && command != "generate" &&
        command != "bench")
        invalid("unknown command: " + std::string(command));
    if (argc == 3 && std::string_view(argv[2]) == "--help") {
        usage();
        return 0;
    }
    Arguments args(argc, argv, command == "generate" || command == "tokenize");
    const auto path = std::filesystem::path(args.required("--model"));
    if (command == "tokenize") {
        tokenize(args, path);
        return 0;
    }
    const auto options = backend_options(args);
    if (command == "inspect") {
        (void)args.flag("--json");
        args.finish();
        const auto info = std::filesystem::is_directory(path) ? model_info(Model::load(path, options))
                                                              : gguf_info(path, options);
        std::cout << info.dump(2) << '\n';
    } else if (command == "forward") {
        forward(args, path, options);
    } else if (command == "generate") {
        generate_text(args, path, options);
    } else if (command == "bench") {
        bench(args, path, options);
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr << "nathr: " << error.what() << '\n';
    return 1;
}
