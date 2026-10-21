// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

#include "runtime/context.hpp"

namespace fsr4runtime {
std::span<const VkPipeline> QualityContext::prepareColorSpacePipelines(ColorSpace requested,
                                                                       bool debug_view) {
    if (color_shader_root.empty()) {
        throw std::invalid_argument("missing color endpoint shader root");
    }
    const auto family = !use_int8_dispatch              ? PostpassFamily::Shared
                        : current_preset == "ultraperf" ? PostpassFamily::UltraPerformance
                                                        : PostpassFamily::Standard;
    const auto shaders =
        selectColorEndpointShaders(color_shader_root,
                                   requested,
                                   color_endpoints,
                                   family,
                                   debug_view,
                                   !use_int8_dispatch && model_pipeline_plan.windows_fp8_optimized);
    for (unsigned endpoint = 0; endpoint < 2; ++endpoint) {
        const auto index = endpoint ? pipelines.size() - 1 : 0;
        const auto& path = endpoint ? shaders.postpass : shaders.prepass;
        if (pipelines[index] && active_shader_paths[index] == path) {
            continue;
        }
        const auto code = readModelSpirv(path, color_api_version);
        const auto pipeline = createComputePipeline(code, color_wave32[endpoint]);
        retirePipeline(pipelines[index]);
        pipelines[index] = pipeline;
        active_shader_paths[index] = path;
        log("[color-space] shader=" + path.string());
    }
    return pipelines;
}
} // namespace fsr4runtime
