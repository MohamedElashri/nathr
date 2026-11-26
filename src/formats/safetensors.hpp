#pragma once

#include "core/input.hpp"
#include "formats/tensor.hpp"

#include <array>
#include <map>

namespace nathr::detail {

class SafeTensors {
  public:
    explicit SafeTensors(const std::filesystem::path& path);
    SafeTensors(const SafeTensors&) = delete;
    SafeTensors& operator=(const SafeTensors&) = delete;
    [[nodiscard]] TensorView take(std::string_view name, std::initializer_list<std::size_t> shape);
    [[nodiscard]] bool contains(std::string_view name) const;
    [[nodiscard]] bool bound_all() const noexcept { return tensors_.empty(); }
    [[nodiscard]] std::string_view first_unbound() const { return tensors_.begin()->first; }
    [[nodiscard]] ModelMemory memory() const noexcept { return memory_; }

  private:
    std::filesystem::path path_;
    std::vector<std::byte> bytes_;
    std::map<std::string, TensorView, std::less<>> tensors_;
    ModelMemory memory_{};
};

ModelConfig read_config(const std::filesystem::path& path);

} // namespace nathr::detail
