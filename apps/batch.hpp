#pragma once

#include <nathr/batch.hpp>

namespace nathr::cli {

struct BatchIoOptions {
    bool stats = false, ready = false;
    std::size_t max_result_values = 0;
    std::vector<std::string> lora_adapters{};
    std::filesystem::path lora_root{};
    bool metrics = false;
    std::string chat_format = "auto";
    std::size_t max_output_tokens = 0;
};

void print_batch_memory_plan(const std::filesystem::path& path, BatchOptions options, BatchIoOptions io);
void run_batch(const Model& model, const std::filesystem::path& path, const std::filesystem::path& sidecar,
               BatchOptions options, BatchIoOptions io);

} // namespace nathr::cli
