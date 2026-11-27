#include "formats/gguf.hpp"

#include "core/sha256.hpp"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <system_error>
#include <unistd.h>
#include <unordered_set>

namespace nathr::detail {
namespace {

class Reader {
  public:
    Reader(std::span<const std::byte> file, const std::filesystem::path& path)
        : header_(file.first(std::min(file.size(), 8 * mib))), path_(path) {}

    std::span<const std::byte> take(std::size_t count) {
        if (count > header_.size() - position_)
            fail("truncated or oversized GGUF header");
        const auto bytes = header_.subspan(position_, count);
        position_ += count;
        return bytes;
    }

    std::uint64_t uint(std::size_t width) { return little_integer(take(width)); }

    std::string text() { return std::string(text_view()); }

    Json value(std::uint32_t type, bool keep, std::size_t array_limit = 65536) {
        if (type == 9) {
            const auto element = static_cast<std::uint32_t>(uint(4));
            const auto count = uint(8);
            if (element == 9 || element > 12)
                fail("unsupported GGUF array element type");
            if (count > array_limit)
                fail("GGUF array exceeds element limit");
            Json array;
            if (keep) {
                if (count > 3 * max_vocabulary - kept_elements_)
                    fail("GGUF retained metadata exceeds aggregate element limit", ErrorCode::resource_limit);
                kept_elements_ += count;
                array = Json::array();
                array.get_ref<Json::array_t&>().reserve(count);
            }
            for (std::uint64_t i = 0; i < count; ++i) {
                auto item = scalar(element, keep);
                if (keep)
                    array.push_back(std::move(item));
            }
            return array;
        }
        return scalar(type, keep);
    }

    [[noreturn]] void fail(std::string_view what, ErrorCode code = ErrorCode::malformed) const {
        throw Error(code, path_.string() + ": byte " + std::to_string(position_) + ": " + std::string(what));
    }

    [[nodiscard]] std::size_t position() const noexcept { return position_; }
    void seek(std::size_t position) noexcept { position_ = position; }

  private:
    Json scalar(std::uint32_t type, bool keep) {
        switch (type) {
        case 0:
            return uint(1);
        case 1:
            return std::bit_cast<std::int8_t>(static_cast<std::uint8_t>(uint(1)));
        case 2:
            return uint(2);
        case 3:
            return std::bit_cast<std::int16_t>(static_cast<std::uint16_t>(uint(2)));
        case 4:
            return uint(4);
        case 5:
            return std::bit_cast<std::int32_t>(static_cast<std::uint32_t>(uint(4)));
        case 6:
            return std::bit_cast<float>(static_cast<std::uint32_t>(uint(4)));
        case 7: {
            const auto byte = uint(1);
            if (byte > 1)
                fail("invalid GGUF boolean");
            return byte != 0;
        }
        case 8: {
            const auto text = text_view();
            return keep ? Json(text) : Json{};
        }
        case 10:
            return uint(8);
        case 11:
            return std::bit_cast<std::int64_t>(uint(8));
        case 12:
            return std::bit_cast<double>(uint(8));
        default:
            fail("unknown GGUF metadata type");
        }
    }

    std::string_view text_view() {
        const auto count = uint(8);
        if (count > 65536)
            fail("GGUF string exceeds 64 KiB");
        const auto bytes = take(count);
        return {reinterpret_cast<const char*>(bytes.data()), bytes.size()};
    }

    std::span<const std::byte> header_;
    const std::filesystem::path& path_;
    std::size_t position_ = 0;
    std::size_t kept_elements_ = 0;
};

std::string system_message(int error) {
    return std::error_code(error, std::generic_category()).message();
}

} // namespace

MappedFile::MappedFile(const std::filesystem::path& path) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK);
    if (fd < 0)
        throw Error(ErrorCode::io, path.string() + ": " + system_message(errno));
    struct Descriptor {
        int value;
        ~Descriptor() { ::close(value); }
    } descriptor{fd};
    struct stat info{};
    if (::fstat(fd, &info) != 0)
        throw Error(ErrorCode::io, path.string() + ": " + system_message(errno));
    if (!S_ISREG(info.st_mode))
        throw Error(ErrorCode::unsupported, path.string() + ": expected a regular GGUF file");
    if (info.st_size < 24)
        throw Error(ErrorCode::malformed, path.string() + ": truncated GGUF prefix");
    if (info.st_size > static_cast<off_t>(max_checkpoint_bytes))
        throw Error(ErrorCode::resource_limit, path.string() + ": GGUF file exceeds 1 GiB");
    const auto size = static_cast<std::size_t>(info.st_size);
    void* base = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED)
        throw Error(ErrorCode::io, path.string() + ": " + system_message(errno));
    bytes_ = {static_cast<const std::byte*>(base), size};
}

MappedFile::~MappedFile() {
    ::munmap(const_cast<std::byte*>(bytes_.data()), bytes_.size());
}

Gguf::Gguf(const std::filesystem::path& path) : path_(path), file_(path) {
    Reader reader(file_.bytes(), path);
    if (reader.uint(4) != 0x46554747)
        reader.fail("invalid GGUF magic");
    if (reader.uint(4) != 3)
        reader.fail("only little-endian GGUF v3 is supported", ErrorCode::unsupported);
    const auto tensors = reader.uint(8);
    const auto keys = reader.uint(8);
    if (tensors > 512 || keys > 2048)
        reader.fail("GGUF tensor or metadata count exceeds limit", ErrorCode::resource_limit);
    struct MetadataEntry {
        std::string key;
        std::size_t offset;
        std::uint32_t type;
    };
    std::vector<MetadataEntry> deferred;
    std::unordered_set<std::string> seen;
    for (std::uint64_t i = 0; i < keys; ++i) {
        auto key = reader.text();
        if (key.empty() || key.find('\0') != key.npos || !seen.insert(key).second)
            reader.fail("invalid or duplicate GGUF metadata key");
        const auto type = static_cast<std::uint32_t>(reader.uint(4));
        if (key == "general.alignment") {
            if (type != 4)
                reader.fail("GGUF alignment must be UINT32");
            alignment_ = reader.uint(4);
            if (alignment_ < 8 || alignment_ > 4096 || !std::has_single_bit(alignment_))
                reader.fail("invalid GGUF alignment");
        } else if (key.starts_with("tokenizer.")) {
            const bool vocabulary = key == "tokenizer.ggml.tokens" || key == "tokenizer.ggml.token_type" ||
                                    key == "tokenizer.ggml.merges" || key == "tokenizer.ggml.scores";
            tokenizer_[std::move(key)] = reader.value(type, true, vocabulary ? max_vocabulary : 65536);
        } else if (key == "general.architecture") {
            architecture_[std::move(key)] = reader.value(type, true);
        } else if (key.starts_with("split.")) {
            split_[std::move(key)] = reader.value(type, true);
        } else {
            deferred.push_back({std::move(key), reader.position(), type});
            (void)reader.value(type, false);
        }
    }
    const auto metadata_end = reader.position();
    const auto arch = architecture_.find("general.architecture");
    if (arch != architecture_.end() && arch->is_string() && !arch->get_ref<const std::string&>().empty()) {
        // Metadata keys can appear before general.architecture in the file.
        const auto prefix = arch->get_ref<const std::string&>() + ".";
        for (auto& entry : deferred) {
            if (entry.key.starts_with(prefix)) {
                reader.seek(entry.offset);
                architecture_[std::move(entry.key)] = reader.value(entry.type, true);
            }
        }
        reader.seek(metadata_end);
    }

    std::vector<Record*> by_offset;
    by_offset.reserve(tensors);
    for (std::uint64_t i = 0; i < tensors; ++i) {
        auto name = reader.text();
        if (name.empty() || name.size() > 64 || name.find('\0') != name.npos)
            reader.fail("invalid GGUF tensor name");
        TensorDesc desc{};
        desc.rank = reader.uint(4);
        if (!desc.rank || desc.rank > desc.shape.size())
            reader.fail("invalid GGUF tensor rank");
        desc.elements = 1;
        for (std::size_t axis = 0; axis < desc.rank; ++axis) {
            const auto size = reader.uint(8);
            if (!size)
                reader.fail("zero GGUF tensor dimension");
            desc.shape[desc.rank - 1 - axis] = size; // Logical order puts the contiguous axis last.
            desc.elements = product({desc.elements, size}, path.string() + ": " + name);
        }
        desc.dtype = static_cast<DType>(reader.uint(4));
        const auto* format = encoding(desc.dtype);
        if (!format)
            reader.fail("unknown GGUF encoding ID", ErrorCode::unsupported);
        if (desc.shape[desc.rank - 1] % format->elements)
            reader.fail("GGUF tensor row does not fit encoded blocks");
        const auto offset = reader.uint(8);
        if (offset % alignment_)
            reader.fail("misaligned GGUF tensor offset");
        auto [entry, unique] = tensors_.emplace(std::move(name), Record{{desc, {}}, offset});
        if (!unique)
            reader.fail("duplicate GGUF tensor");
        by_offset.push_back(&entry->second);
    }
    data_offset_ =
        sum(reader.position(), (alignment_ - reader.position() % alignment_) % alignment_, path.string());
    if (data_offset_ > 8 * mib || data_offset_ > file_.bytes().size())
        reader.fail("GGUF data section begins outside file or header limit");
    const auto data = file_.bytes().subspan(data_offset_);
    std::ranges::sort(by_offset, {}, &Record::offset);
    std::size_t past_end = 0;
    for (auto* record : by_offset) {
        const auto& desc = record->view.desc;
        const auto& format = *encoding(desc.dtype);
        const auto size = product({desc.elements / format.elements, format.bytes}, path.string());
        if (record->offset < past_end || record->offset > data.size() || size > data.size() - record->offset)
            reader.fail("overlapping or truncated GGUF tensor");
        record->view.bytes = data.subspan(record->offset, size);
        past_end = record->offset + size;
        memory_.stored_parameters = sum(memory_.stored_parameters, desc.elements, path.string());
        memory_.encoded_weight_bytes = sum(memory_.encoded_weight_bytes, size, path.string());
        memory_.f32_tensors += desc.dtype == DType::f32;
        memory_.f16_tensors += desc.dtype == DType::f16;
        memory_.bf16_tensors += desc.dtype == DType::bf16;
    }
    memory_.file_bytes = file_.bytes().size();
    memory_.tensors = tensors_.size();
    memory_.mapped_file_bytes = memory_.file_bytes;
}

TensorView Gguf::take(std::string_view name, std::initializer_list<std::size_t> shape) {
    const auto it = tensors_.find(name);
    if (it == tensors_.end())
        throw Error(ErrorCode::malformed, path_.string() + ": missing GGUF tensor " + std::string(name));
    const auto view = it->second.view;
    if (!std::ranges::equal(shape, view.desc.dimensions()))
        throw Error(ErrorCode::malformed,
                    path_.string() + ": wrong shape for GGUF tensor " + std::string(name));
    tensors_.erase(it);
    return view;
}

std::string Gguf::tokenizer_fingerprint() const {
    try {
        return sha256(tokenizer_.dump());
    } catch (const Json::exception& error) {
        throw Error(ErrorCode::malformed, path_.string() + ": " + error.what());
    }
}

void Gguf::discard_metadata() {
    architecture_ = {};
    tokenizer_ = {};
    split_ = {};
}

} // namespace nathr::detail
