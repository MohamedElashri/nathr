#pragma once

#include <cstddef>

namespace nathr::detail {

void decode_q4_1(const std::byte*, float*);
void decode_q5_0(const std::byte*, float*);
void decode_q5_1(const std::byte*, float*);
void decode_q2_k(const std::byte*, float*);
void decode_q3_k(const std::byte*, float*);
void decode_q4_k(const std::byte*, float*);
void decode_q5_k(const std::byte*, float*);
void decode_q6_k(const std::byte*, float*);
void decode_iq1_s(const std::byte*, float*);
void decode_iq1_m(const std::byte*, float*);
void decode_iq2_xxs(const std::byte*, float*);
void decode_iq2_xs(const std::byte*, float*);
void decode_iq2_s(const std::byte*, float*);
void decode_iq3_xxs(const std::byte*, float*);
void decode_iq3_s(const std::byte*, float*);
void decode_iq4_nl(const std::byte*, float*);
void decode_iq4_xs(const std::byte*, float*);

} // namespace nathr::detail
