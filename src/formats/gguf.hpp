#pragma once

#include "core/input.hpp"
#include "formats/tensor.hpp"

#include <map>

namespace nathr::detail {

class MappedFile {
  public:
    explicit MappedFile(const std::filesystem::path& path);
    ~MappedFile();
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    [[nodiscard]] std::span<const std::byte> bytes() const noexcept { return bytes_; }

  private:
    std::span<const std::byte> bytes_;
};

class Gguf {
  public:
    struct Record {
        TensorView view;
        std::size_t offset;
    };

    explicit Gguf(const std::filesystem::path& path);
    [[nodiscard]] TensorView take(std::string_view name, std::initializer_list<std::size_t> shape);
    [[nodiscard]] std::string_view first_unbound() const noexcept {
        return tensors_.empty() ? std::string_view{} : tensors_.begin()->first;
    }
    [[nodiscard]] const auto& records() const noexcept { return tensors_; }
    [[nodiscard]] const Json& tokenizer() const noexcept { return tokenizer_; }
    [[nodiscard]] std::string tokenizer_fingerprint() const;
    [[nodiscard]] const Json& architecture() const noexcept { return architecture_; }
    [[nodiscard]] const Json& split() const noexcept { return split_; }
    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }
    [[nodiscard]] std::size_t alignment() const noexcept { return alignment_; }
    [[nodiscard]] std::size_t data_offset() const noexcept { return data_offset_; }
    [[nodiscard]] ModelMemory memory() const noexcept { return memory_; }
    void discard_metadata();

  private:
    std::filesystem::path path_;
    MappedFile file_;
    std::map<std::string, Record, std::less<>> tensors_;
    Json architecture_ = Json::object(), tokenizer_ = Json::object(), split_ = Json::object();
    std::size_t alignment_ = 32, data_offset_ = 0;
    ModelMemory memory_{};
};

} // namespace nathr::detail
