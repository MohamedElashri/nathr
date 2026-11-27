#pragma once

#include "core/input.hpp"
#include "formats/tensor.hpp"

#include <map>

namespace nathr::detail {

class SafeTensors {
  public:
    explicit SafeTensors(const std::filesystem::path& path, std::size_t byte_limit = max_checkpoint_bytes);
    SafeTensors(const SafeTensors&) = delete;
    SafeTensors& operator=(const SafeTensors&) = delete;
    [[nodiscard]] TensorView take(std::string_view name, std::initializer_list<std::size_t> shape);
    [[nodiscard]] TensorView take_transposed_matrix(std::string_view name, std::size_t rows,
                                                    std::size_t columns);
    [[nodiscard]] bool contains(std::string_view name) const;
    [[nodiscard]] std::string_view first_unbound() const noexcept {
        return tensors_.empty() ? std::string_view{} : tensors_.begin()->first;
    }
    [[nodiscard]] ModelMemory memory() const noexcept { return memory_; }

  private:
    std::filesystem::path path_;
    std::vector<std::byte> bytes_;
    std::map<std::string, TensorView, std::less<>> tensors_;
    ModelMemory memory_{};
};

} // namespace nathr::detail
