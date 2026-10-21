// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

#pragma once

#include "runtime/color-space.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace fsr4runtime {
using fsr4_411::ColorSpace;

inline float rcasSharpness(float sharpness) {
    if (!std::isfinite(sharpness)) {
        throw std::invalid_argument("non-finite RCAS sharpness");
    }
    return std::clamp(sharpness, 0.0f, 1.0f);
}

struct RcasConstants {
    std::array<uint32_t, 4> config{};
    float pre_exposure = 1.0f;
    float padding[3]{};
};

static_assert(sizeof(RcasConstants) == 32);

inline RcasConstants makeRcasConstants(float sharpness, float pre_exposure) {
    RcasConstants result{};
    const float factor = std::exp2(2.0f * rcasSharpness(sharpness) - 2.0f);
    result.config[0] = std::bit_cast<uint32_t>(factor);
    // Pack two FP16 copies as the SDK does. The factor is always positive and normal.
    const uint32_t bits = result.config[0];
    const uint32_t half = ((bits >> 13) & 0x3ffu) | (((bits >> 23) - 112u) << 10);
    result.config[1] = half | (half << 16);
    result.pre_exposure = pre_exposure;
    return result;
}
} // namespace fsr4runtime
