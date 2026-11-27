#pragma once

#include <nathr/batch.hpp>

namespace nathr::cli {

struct BatchIoOptions {
    bool stats = false, ready = false;
    std::size_t max_result_values = 0;
    bool metrics = false;
};

void run_batch(const Model& model, const std::filesystem::path& path, const std::filesystem::path& sidecar,
               BatchOptions options, BatchIoOptions io);

} // namespace nathr::cli
