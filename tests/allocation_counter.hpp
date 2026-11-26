#pragma once
#include <cstddef>

void start_allocation_count() noexcept;
std::size_t stop_allocation_count() noexcept;
