// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

#pragma once
#include <cstdint>
#include <filesystem>
#include <string>

namespace fsr4_411 {
enum class ColorSpace : uint8_t { Linear, Nonlinear, Srgb, Pq };

constexpr ColorSpace selectColorSpace(bool nonlinear, bool srgb, bool pq) {
    return pq          ? ColorSpace::Pq
           : srgb      ? ColorSpace::Srgb
           : nonlinear ? ColorSpace::Nonlinear
                       : ColorSpace::Linear;
}

constexpr const char* colorSpaceName(ColorSpace space) {
    switch (space) {
    case ColorSpace::Linear:
        return "linear";
    case ColorSpace::Nonlinear:
        return "nonlinear";
    case ColorSpace::Srgb:
        return "srgb";
    case ColorSpace::Pq:
        return "pq";
    }
    return "invalid";
}
} // namespace fsr4_411

namespace fsr4runtime {
struct ColorEndpointOptions {
    bool render_resolution_motion = false;
    bool inverted_depth = false;
    bool auto_exposure = false;
    bool motion_jitter_cancellation = false;
};

enum class PostpassFamily { Shared, Standard, UltraPerformance };

struct ColorEndpointShaders {
    std::filesystem::path prepass;
    std::filesystem::path postpass;
};

inline ColorEndpointShaders selectColorEndpointShaders(const std::filesystem::path& root,
                                                       fsr4_411::ColorSpace color,
                                                       const ColorEndpointOptions& options,
                                                       PostpassFamily family,
                                                       bool debug_view = false,
                                                       bool windows_fp8 = false) {
    const auto* motion = !options.render_resolution_motion ? "display-mv"
                         : options.inverted_depth          ? "render-mv-inverted-depth"
                                                           : "render-mv-normal-depth";
    const auto* exposure = options.auto_exposure ? "auto" : "external";
    const auto prepass = root / "prepass" / fsr4_411::colorSpaceName(color) / motion / exposure /
                         (options.motion_jitter_cancellation ? "jitter.spv" : "no-jitter.spv");
    const auto postpass_root =
        windows_fp8 && family == PostpassFamily::Shared
            ? root.parent_path().parent_path() / "windows/color-endpoints" / root.filename()
            : root;
    auto postpass = postpass_root / "postpass" / fsr4_411::colorSpaceName(color);
    if (family != PostpassFamily::Shared) {
        postpass /= family == PostpassFamily::UltraPerformance ? "ultraperf" : "standard";
    }
    postpass /= std::string(exposure) + (debug_view ? "-debug.spv" : ".spv");
    return {prepass, postpass};
}
} // namespace fsr4runtime
