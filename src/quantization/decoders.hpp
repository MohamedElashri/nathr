#pragma once

#include <cstddef>

namespace nathr::detail {

struct KScale {
    unsigned scale, minimum;
};

inline KScale k_scale(const std::byte* packed, unsigned group) {
    const unsigned i = group % 4;
    const auto lower_scale = std::to_integer<unsigned>(packed[i]);
    const auto lower_min = std::to_integer<unsigned>(packed[4 + i]);
    if (group < 4)
        return {lower_scale & 63, lower_min & 63};
    const auto upper = std::to_integer<unsigned>(packed[8 + i]);
    return {(upper & 15) | ((lower_scale >> 6) << 4), (upper >> 4) | ((lower_min >> 6) << 4)};
}

inline int q3_k_scale(const std::byte* packed, unsigned group) {
    const auto low = (std::to_integer<unsigned>(packed[group % 8]) >> (4 * (group / 8))) & 15;
    const auto high = (std::to_integer<unsigned>(packed[8 + group % 4]) >> (2 * (group / 4))) & 3;
    return static_cast<int>(low | (high << 4)) - 32;
}

void decode_q4_0(const std::byte*, float*);
void decode_q4_1(const std::byte*, float*);
void decode_q5_0(const std::byte*, float*);
void decode_q5_1(const std::byte*, float*);
void decode_q8_0(const std::byte*, float*);
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
