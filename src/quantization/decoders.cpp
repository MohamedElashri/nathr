#include "quantization/decoders.hpp"

#include "backend/cpu/scalar.hpp"
#include "quantization/iq_codebooks.hpp"

#include <bit>
#include <cstdint>

namespace nathr::detail {
namespace {

unsigned byte(const std::byte* p, std::size_t index) {
    return std::to_integer<unsigned>(p[index]);
}

int signed_byte(unsigned value) {
    return static_cast<int>(value) - (value >= 128 ? 256 : 0);
}

unsigned word(const std::byte* p, std::size_t offset) {
    return byte(p, offset) | (byte(p, offset + 1) << 8);
}

std::uint32_t dword(const std::byte* p, std::size_t offset) {
    return word(p, offset) | (std::uint32_t(word(p, offset + 2)) << 16);
}

float half(const std::byte* p, std::size_t offset) {
    return half_to_float(static_cast<std::uint16_t>(word(p, offset)));
}

unsigned bits(unsigned value, unsigned shift, unsigned mask) {
    return (value >> shift) & mask;
}

unsigned nibble(const std::byte* p, std::size_t index) {
    return bits(byte(p, index / 2), 4 * (index % 2), 15);
}

// The legacy nibbles occupy two planes of 16, not adjacent values.
unsigned legacy_nibble(const std::byte* p, std::size_t index) {
    return bits(byte(p, index % 16), 4 * (index / 16), 15);
}

template <bool affine> void legacy_q5(const std::byte* p, float* out) {
    constexpr auto high_offset = affine ? 4 : 2;
    const auto d = half(p, 0);
    const auto minimum = affine ? half(p, 2) : 0.0f;
    const auto high = dword(p, high_offset);
    for (unsigned i = 0; i < 32; ++i) {
        const int code = legacy_nibble(p + high_offset + 4, i) | (bits(high, i, 1) << 4);
        if constexpr (affine)
            out[i] = d * code + minimum;
        else
            out[i] = d * (code - 16);
    }
}

struct KScale {
    unsigned scale, minimum;
};
KScale k_scale(const std::byte* p, unsigned group) {
    const unsigned i = group % 4;
    const auto lower_scales = byte(p, i);
    const auto lower_mins = byte(p, 4 + i);
    if (group < 4)
        return {lower_scales & 63, lower_mins & 63};
    const auto packed = byte(p, 8 + i);
    return {(packed & 15) | ((lower_scales >> 6) << 4), (packed >> 4) | ((lower_mins >> 6) << 4)};
}

template <bool five> void affine_k(const std::byte* p, float* out) {
    const auto base_scale = half(p, 0);
    const auto base_min = half(p, 2);
    const auto* high = p + 16;
    const auto* quant = p + (five ? 48 : 16);
    for (unsigned group = 0; group < 8; ++group) {
        const auto [scale, minimum] = k_scale(p + 4, group);
        const auto d = base_scale * scale;
        const auto m = base_min * minimum;
        for (unsigned i = 0; i < 32; ++i) {
            auto code = bits(byte(quant, (group / 2) * 32 + i), 4 * (group % 2), 15);
            if constexpr (five)
                code |= bits(byte(high, i), group, 1) << 4;
            out[group * 32 + i] = d * code - m;
        }
    }
}

unsigned even_sign_mask(unsigned index) {
    return index | ((std::popcount(index) & 1) << 7);
}

template <class Grid>
void signed_grid(Grid values, float scale, unsigned sign_mask, unsigned sign_offset, float* out) {
    for (unsigned i = 0; i < sizeof(Grid); ++i) {
        const auto magnitude = static_cast<unsigned>((values >> (8 * i)) & 255);
        const auto sign = bits(sign_mask, sign_offset + i, 1) ? -1.f : 1.f;
        out[i] = (scale * magnitude) * sign;
    }
}

void ternary_grid(unsigned index, float scale, bool negative_delta, float* out) {
    const auto values = codebooks::iq1s_grid[index];
    const auto delta = negative_delta ? -0.125f : 0.125f;
    for (unsigned i = 0; i < 8; ++i) {
        const auto q = signed_byte((values >> (i * 8)) & 255);
        out[i] = scale * (q + delta);
    }
}

void nonlinear_nibbles(const std::byte* p, float scale, float* out) {
    for (unsigned i = 0; i < 32; ++i)
        out[i] = scale * codebooks::kvalues_iq4nl[legacy_nibble(p, i)];
}

} // namespace

void decode_q4_1(const std::byte* p, float* out) {
    const auto d = half(p, 0), m = half(p, 2);
    for (unsigned i = 0; i < 32; ++i)
        out[i] = d * legacy_nibble(p + 4, i) + m;
}

void decode_q5_0(const std::byte* p, float* out) {
    legacy_q5<false>(p, out);
}
void decode_q5_1(const std::byte* p, float* out) {
    legacy_q5<true>(p, out);
}

void decode_q2_k(const std::byte* p, float* out) {
    const auto base_scale = half(p, 80), base_min = half(p, 82);
    for (unsigned group = 0; group < 16; ++group) {
        const auto packed = byte(p, group);
        const auto d = base_scale * (packed & 15);
        const auto m = base_min * (packed >> 4);
        const auto offset = 16 + (group / 8) * 32 + (group % 2) * 16;
        const auto shift = 2 * ((group / 2) % 4);
        for (unsigned i = 0; i < 16; ++i)
            out[group * 16 + i] = d * bits(byte(p, offset + i), shift, 3) - m;
    }
}

void decode_q3_k(const std::byte* p, float* out) {
    const auto base = half(p, 108);
    const auto* scales = p + 96;
    for (unsigned group = 0; group < 16; ++group) {
        const auto low = bits(byte(scales, group % 8), 4 * (group / 8), 15);
        const auto high = bits(byte(scales, 8 + group % 4), 2 * (group / 4), 3);
        const int scale = static_cast<int>(low | (high << 4)) - 32;
        const auto d = base * scale;
        const auto mask_offset = (group % 2) * 16;
        const auto offset = 32 + (group / 8) * 32 + mask_offset;
        for (unsigned i = 0; i < 16; ++i) {
            const int low_bits = bits(byte(p, offset + i), 2 * ((group / 2) % 4), 3);
            const bool nonnegative = bits(byte(p, mask_offset + i), group / 2, 1);
            out[group * 16 + i] = d * (low_bits - (nonnegative ? 0 : 4));
        }
    }
}

void decode_q4_k(const std::byte* p, float* out) {
    affine_k<false>(p, out);
}
void decode_q5_k(const std::byte* p, float* out) {
    affine_k<true>(p, out);
}

void decode_q6_k(const std::byte* p, float* out) {
    const auto scale = half(p, 208);
    for (unsigned group = 0; group < 16; ++group) {
        const auto d = scale * signed_byte(byte(p, 192 + group));
        for (unsigned j = 0; j < 16; ++j) {
            const unsigned i = 16 * group + j;
            const auto low = bits(byte(p, i / 128 * 64 + i % 64), 4 * (i / 64 % 2), 15);
            const auto high = bits(byte(p, 128 + i / 128 * 32 + i % 32), 2 * (i / 32 % 4), 3);
            out[i] = d * (static_cast<int>(low | (high << 4)) - 32);
        }
    }
}

void decode_iq2_xxs(const std::byte* p, float* out) {
    const auto d = half(p, 0);
    for (unsigned group = 0; group < 8; ++group) {
        const unsigned offset = 2 + 8 * group;
        const auto meta = dword(p, offset + 4);
        const auto scale = (d * (0.5f + (meta >> 28))) * 0.25f;
        for (unsigned i = 0; i < 4; ++i) {
            const auto signs = even_sign_mask(bits(meta, 7 * i, 127));
            signed_grid(codebooks::iq2xxs_grid[byte(p, offset + i)], scale, signs, 0,
                        out + 32 * group + 8 * i);
        }
    }
}

void decode_iq2_xs(const std::byte* p, float* out) {
    const auto d = half(p, 0);
    for (unsigned i = 0; i < 32; ++i) {
        const auto packed = word(p, 2 + 2 * i);
        const auto scale = (d * (0.5f + nibble(p + 66, i / 2))) * 0.25f;
        signed_grid(codebooks::iq2xs_grid[packed & 511], scale, even_sign_mask(packed >> 9), 0, out + 8 * i);
    }
}

void decode_iq2_s(const std::byte* p, float* out) {
    const auto d = half(p, 0);
    for (unsigned i = 0; i < 32; ++i) {
        const auto high = bits(byte(p, 66 + i / 4), 2 * (i % 4), 3);
        const auto index = byte(p, 2 + i) | (high << 8);
        const auto scale = (d * (0.5f + nibble(p + 74, i / 2))) * 0.25f;
        signed_grid(codebooks::iq2s_grid[index], scale, byte(p, 34 + i), 0, out + 8 * i);
    }
}

void decode_iq3_xxs(const std::byte* p, float* out) {
    const auto d = half(p, 0);
    for (unsigned group = 0; group < 8; ++group) {
        const auto meta = dword(p, 66 + 4 * group);
        const auto scale = (d * (0.5f + (meta >> 28))) * 0.5f;
        for (unsigned i = 0; i < 8; ++i) {
            const auto signs = even_sign_mask(bits(meta, 7 * (i / 2), 127));
            const auto grid = codebooks::iq3xxs_grid[byte(p, 2 + 8 * group + i)];
            signed_grid(grid, scale, signs, 4 * (i % 2), out + 32 * group + 4 * i);
        }
    }
}

void decode_iq3_s(const std::byte* p, float* out) {
    const auto d = half(p, 0);
    for (unsigned i = 0; i < 64; ++i) {
        const auto high = bits(byte(p, 66 + i / 8), i % 8, 1);
        const auto index = byte(p, 2 + i) | (high << 8);
        const auto scale = d * (1 + 2 * nibble(p + 106, i / 8));
        signed_grid(codebooks::iq3s_grid[index], scale, byte(p, 74 + i / 2), 4 * (i % 2), out + 4 * i);
    }
}

void decode_iq1_s(const std::byte* p, float* out) {
    const auto d = half(p, 0);
    for (unsigned i = 0; i < 32; ++i) {
        const auto meta = word(p, 34 + 2 * (i / 4));
        const auto index = byte(p, 2 + i) | (bits(meta, 3 * (i % 4), 7) << 8);
        ternary_grid(index, d * (1 + 2 * bits(meta, 12, 7)), meta & 0x8000, out + 8 * i);
    }
}

void decode_iq1_m(const std::byte* p, float* out) {
    unsigned half_bits = 0;
    for (unsigned i = 0; i < 4; ++i)
        half_bits |= (word(p, 48 + 2 * i) >> 12) << (4 * i);
    const auto d = half_to_float(static_cast<std::uint16_t>(half_bits));
    for (unsigned i = 0; i < 32; ++i) {
        const auto local = word(p, 48 + 2 * (i / 8));
        const auto scale = d * (1 + 2 * bits(local, 3 * ((i / 2) % 4), 7));
        const auto meta = nibble(p + 32, i);
        const auto index = byte(p, i) | ((meta & 7) << 8);
        ternary_grid(index, scale, meta & 8, out + 8 * i);
    }
}

void decode_iq4_nl(const std::byte* p, float* out) {
    nonlinear_nibbles(p + 2, half(p, 0), out);
}

void decode_iq4_xs(const std::byte* p, float* out) {
    const auto d = half(p, 0);
    const auto high = word(p, 2);
    for (unsigned i = 0; i < 8; ++i) {
        const auto code = nibble(p + 4, i) | (bits(high, 2 * i, 3) << 4);
        nonlinear_nibbles(p + 8 + 16 * i, d * (static_cast<int>(code) - 32), out + 32 * i);
    }
}

} // namespace nathr::detail
