#include "formats/safetensors.hpp"

#include <algorithm>
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

} // namespace nathr::detail
