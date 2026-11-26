#include "formats/safetensors.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>

namespace nathr::detail {
namespace {
DType read_dtype(const Json& value, std::string_view context) {
    if (value == "F32")
        return DType::f32;
    if (value == "F16")
        return DType::f16;
    if (value == "BF16")
        return DType::bf16;
    throw Error(ErrorCode::unsupported, std::string(context) + ": expected F32, F16 or BF16");
}
} // namespace

SafeTensors::SafeTensors(const std::filesystem::path& path) : path_(path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    const auto size = stream.tellg();
    if (size < 0)
        throw Error(ErrorCode::io, path.string() + ": cannot open file");
    if (static_cast<std::uint64_t>(size) > 1024 * mib)
        throw Error(ErrorCode::resource_limit, path.string() + ": file exceeds 1 GiB");
    std::array<std::byte, 8> prefix{};
    stream.seekg(0);
    if (!stream.read(reinterpret_cast<char*>(prefix.data()), prefix.size())) {
        throw Error(ErrorCode::malformed, path.string() + ": missing header size");
    }
    const auto header_size = little_integer(prefix);
    if (header_size > mib)
        throw Error(ErrorCode::resource_limit, path.string() + ": header exceeds 1 MiB");
    if (!header_size || header_size > static_cast<std::size_t>(size) - prefix.size()) {
        throw Error(ErrorCode::malformed, path.string() + ": invalid Safetensors header");
    }
    std::vector<std::byte> raw_header(header_size);
    if (!stream.read(reinterpret_cast<char*>(raw_header.data()), static_cast<std::streamsize>(header_size)) ||
        raw_header.front() != std::byte{'{'}) {
        throw Error(ErrorCode::malformed, path.string() + ": invalid Safetensors header");
    }
    const auto header = parse_json(raw_header, path);
    const auto data_offset = prefix.size() + header_size;
    const auto data_size = static_cast<std::size_t>(size) - data_offset;
    struct Entry {
        std::size_t begin, end;
        std::string name;
        TensorDesc desc;
    };
    std::vector<Entry> extents;
    for (const auto& [name, info] : header.items()) {
        const auto context = path.string() + ": " + name;
        if (name == "__metadata__") {
            if (!info.is_object() || info.size() > 2048 ||
                !std::all_of(info.begin(), info.end(), [](const Json& v) { return v.is_string(); })) {
                throw Error(ErrorCode::malformed, context + ": metadata must contain string values");
            }
            continue;
        }
        if (extents.size() == 512)
            throw Error(ErrorCode::resource_limit, path.string() + ": too many tensors");
        try {
            const auto& shape = info.at("shape");
            const auto& offsets = info.at("data_offsets");
            if (name.empty() || !shape.is_array() || shape.size() > 4 || !offsets.is_array() ||
                offsets.size() != 2) {
                throw Error(ErrorCode::malformed, context + ": invalid shape or data offsets");
            }
            TensorDesc desc{read_dtype(info.at("dtype"), context), {}, shape.size(), 1};
            for (std::size_t axis = 0; axis < desc.rank; ++axis) {
                desc.shape[axis] = natural(shape[axis], context);
            }
            // A tensor with a zero extent occupies zero bytes even if an earlier dimension is large.
            const auto dims = std::span(desc.shape).first(desc.rank);
            if (std::ranges::find(dims, 0) != dims.end()) {
                desc.elements = 0;
            } else {
                for (const auto dim : dims)
                    desc.elements = product({desc.elements, dim}, context);
            }
            const auto begin = natural(offsets[0], context);
            const auto end = natural(offsets[1], context);
            const auto size = product({desc.elements, encoding(desc.dtype)->bytes}, context);
            if (end < begin || end > data_size || end - begin != size) {
                throw Error(ErrorCode::malformed, context + ": extent does not match tensor storage");
            }
            extents.push_back({begin, end, name, desc});
            memory_.stored_parameters = sum(memory_.stored_parameters, desc.elements, context);
            ++memory_.tensors;
            switch (desc.dtype) {
            case DType::f32:
                ++memory_.f32_tensors;
                break;
            case DType::f16:
                ++memory_.f16_tensors;
                break;
            case DType::bf16:
                ++memory_.bf16_tensors;
                break;
            default:
                throw Error(ErrorCode::unsupported, context + ": unsupported Safetensors encoding");
            }
        } catch (const Json::exception& error) {
            throw Error(ErrorCode::malformed, context + ": " + error.what());
        }
    }
    std::ranges::sort(extents, [](const Entry& a, const Entry& b) {
        return std::pair(a.begin, a.end) < std::pair(b.begin, b.end);
    });
    std::size_t cursor = 0;
    for (const auto& entry : extents) {
        if (entry.begin != cursor)
            throw Error(ErrorCode::malformed, path.string() + ": tensor extents overlap or leave a gap");
        cursor = entry.end;
    }
    if (cursor != data_size)
        throw Error(ErrorCode::malformed, path.string() + ": trailing unclaimed bytes");
    bytes_.resize(static_cast<std::size_t>(size));
    std::ranges::copy(prefix, bytes_.begin());
    std::ranges::copy(raw_header, bytes_.begin() + prefix.size());
    if (!stream.read(reinterpret_cast<char*>(bytes_.data() + data_offset),
                     static_cast<std::streamsize>(data_size)) ||
        stream.peek() != std::char_traits<char>::eof() || stream.bad()) {
        throw Error(ErrorCode::io, path.string() + ": file changed or could not be read");
    }
    const auto data = std::span<const std::byte>(bytes_).subspan(data_offset);
    for (auto& entry : extents) {
        tensors_.emplace(std::move(entry.name),
                         TensorView{entry.desc, data.subspan(entry.begin, entry.end - entry.begin)});
    }
    memory_.file_bytes = bytes_.size();
    memory_.encoded_weight_bytes = data_size;
}

bool SafeTensors::contains(std::string_view name) const {
    return tensors_.contains(name);
}

TensorView SafeTensors::take(std::string_view name, std::initializer_list<std::size_t> shape) {
    const auto item = tensors_.find(name);
    if (item == tensors_.end()) {
        throw Error(ErrorCode::malformed, path_.string() + ": missing tensor " + std::string(name));
    }
    const auto tensor = item->second;
    if (!std::ranges::equal(shape, std::span(tensor.desc.shape).first(tensor.desc.rank))) {
        throw Error(ErrorCode::malformed, path_.string() + ": wrong shape for " + std::string(name));
    }
    tensors_.erase(item);
    return tensor;
}

ModelConfig read_config(const std::filesystem::path& path) {
    const auto json = read_json(path, mib);
    try {
        const auto field = [&](std::string_view name) {
            return natural(json.at(name), path.string() + ": " + std::string(name));
        };
        if (json.at("model_type") != "llama" || json.at("hidden_act") != "silu" ||
            (json.contains("architectures") &&
             json.at("architectures") != Json::array({"LlamaForCausalLM"})) ||
            json.value("attention_bias", false) || json.value("mlp_bias", false) ||
            json.value("rope_interleaved", false) ||
            (json.contains("rope_scaling") && !json.at("rope_scaling").is_null()) ||
            (json.contains("pretraining_tp") && field("pretraining_tp") != 1)) {
            throw Error(ErrorCode::unsupported, path.string() + ": unsupported Llama configuration");
        }
        auto config = ModelConfig{};
        config.layers = field("num_hidden_layers");
        config.hidden = field("hidden_size");
        config.intermediate = field("intermediate_size");
        config.query_heads = field("num_attention_heads");
        config.kv_heads = field("num_key_value_heads");
        config.vocabulary = field("vocab_size");
        config.context = field("max_position_embeddings");
        if (!config.layers || config.layers > 56 || !config.hidden || !config.intermediate ||
            !config.query_heads || !config.kv_heads || config.query_heads % config.kv_heads ||
            config.hidden % config.query_heads || !config.context || config.context > 8192 ||
            !config.vocabulary || config.vocabulary > 65536) {
            throw Error(ErrorCode::unsupported, path.string() + ": unsupported dimensions or head geometry");
        }
        config.head_dim = config.hidden / config.query_heads;
        if (config.head_dim % 2 || (json.contains("head_dim") && field("head_dim") != config.head_dim)) {
            throw Error(ErrorCode::unsupported, path.string() + ": unsupported rotary head width");
        }
        const auto bos = field("bos_token_id");
        const auto eos = field("eos_token_id");
        if (bos >= config.vocabulary || eos >= config.vocabulary) {
            throw Error(ErrorCode::malformed, path.string() + ": special token ID exceeds vocabulary");
        }
        config.bos = static_cast<Token>(bos);
        config.eos = static_cast<Token>(eos);
        config.rope_theta = json.at("rope_theta").get<float>();
        config.rms_epsilon = json.at("rms_norm_eps").get<float>();
        config.tied_embeddings = json.at("tie_word_embeddings").get<bool>();
        if (!(config.rope_theta > 0) || !(config.rms_epsilon > 0) || !std::isfinite(config.rope_theta) ||
            !std::isfinite(config.rms_epsilon)) {
            throw Error(ErrorCode::malformed, path.string() + ": invalid RMSNorm epsilon or RoPE theta");
        }
        return config;
    } catch (const Json::exception& error) {
        throw Error(ErrorCode::malformed, path.string() + ": " + error.what());
    }
}

} // namespace nathr::detail
