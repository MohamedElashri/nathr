#pragma once

#include <cstddef>

namespace nathr::detail::x86 {

float f32_dot(const std::byte*, const float*, std::size_t);
float f16_dot(const std::byte*, const float*, std::size_t);
float bf16_dot(const std::byte*, const float*, std::size_t);
float q8_0_dot(const std::byte*, const float*, std::size_t);
float q4_0_dot(const std::byte*, const float*, std::size_t);
void f32_batch(const std::byte*, const float*, std::size_t, std::size_t, float*);
void f16_batch(const std::byte*, const float*, std::size_t, std::size_t, float*);
void bf16_batch(const std::byte*, const float*, std::size_t, std::size_t, float*);
void q8_0_batch(const std::byte*, const float*, std::size_t, std::size_t, float*);
void q4_0_batch(const std::byte*, const float*, std::size_t, std::size_t, float*);

} // namespace nathr::detail::x86
