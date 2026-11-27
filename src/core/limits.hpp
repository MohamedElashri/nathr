#pragma once

#include <cstddef>

namespace nathr::detail {

inline constexpr std::size_t mib = 1024 * 1024;
inline constexpr std::size_t max_checkpoint_bytes = 1024 * mib;
inline constexpr std::size_t max_vocabulary = 196608;
inline constexpr std::size_t max_uncached_tokens = 128;
inline constexpr std::size_t max_workspace_bytes = 32 * mib;
inline constexpr std::size_t max_kv_cache_bytes = 512 * mib;

} // namespace nathr::detail
