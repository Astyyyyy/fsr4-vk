// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

#include "assets/assets.hpp"
#include "models/models.hpp"
#include "runtime/context.hpp"

namespace fsr4fp8 {
using namespace fsr4runtime;

namespace {
bool hasModelAssets(const std::filesystem::path& root) {
    return fsr4assets::exists(root / "fp8") ||
           fsr4assets::exists(root / "fp8" / "1080" / "prepass.spv") ||
           fsr4assets::exists(root / "prepass.spv");
}

std::filesystem::path modelBundlePath(const std::filesystem::path& asset_root,
                                      uint32_t output_width,
                                      uint32_t output_height) {
    const char* bucket = fsr4::resolutionBucket(output_width, output_height);
    return fsr4assets::exists(asset_root / "fp8") ? asset_root / "fp8" / bucket : asset_root;
}

bool modelUsesWmma(const std::filesystem::path& bundle) {
    const char* disabled_value = std::getenv("FSR4_VK_DISABLE_WMMA");
    const bool disabled = disabled_value && std::strcmp(disabled_value, "0") != 0;
    return !disabled && fsr4assets::exists(bundle / "prepass_wmma.spv") &&
           fsr4assets::exists(bundle / "postpass_wmma.spv");
}
} // namespace

bool hasAssets(const fs::path& root) {
    return hasModelAssets(root);
}

fsr4_411::QualityContextPtr createContext(const fsr4_411::ContextCreateInfo& info) {
    const auto bundle =
        modelBundlePath(info.asset_root, info.max_output_width, info.max_output_height);
    const bool use_wmma = modelUsesWmma(bundle);

    const auto weights = info.asset_root / "fp8/models/4.1.1/weights_standard.bin";

    setDiagnosticRoot(info.diagnostic_root);

    return std::make_unique<fsr4runtime::QualityContext>(
        info.physical_device,
        info.device,
        bundle,
        weights,
        info.asset_root / "fp8" / "auto-exposure.spv",
        info.get_device_proc_addr,
        ModelRuntimeOptions{
            .rcas_shader_root = info.asset_root / "common/shaders/4.1.1/rcas",
            .color_shader_root =
                info.asset_root / "fp8/shaders/4.1.1/color-endpoints" /
                fsr4::resolutionBucket(info.max_output_width, info.max_output_height),
            .use_wmma = use_wmma,
            .use_int8_dispatch = false,
            .numbered_shader_files = false,
            .execute_padding_passes = use_wmma,
            .direct_display_motion_vectors = use_wmma && info.high_resolution_motion_vectors,
            .ultraperformance_model_data_path =
                info.asset_root / "fp8/models/4.1.1/weights_ultraperf.bin",
            .color_endpoints =
                {
                    .render_resolution_motion = !info.high_resolution_motion_vectors,
                    .inverted_depth = info.depth_inverted,
                    .auto_exposure = info.auto_exposure,
                    .motion_jitter_cancellation = info.motion_vector_jitter_cancellation,
                },
            .pass_policy =
                {
                    .dispatch_size = fsr4fp8::dispatchSize,
                    .dependency = fsr4fp8::passDependency,
                    .postpass_updates_exposure = !info.auto_exposure,
                }},
        info.auto_exposure,
        info.depth_inverted,
        info.extensive_logging,
        info.max_render_width,
        info.max_render_height,
        info.max_output_width,
        info.max_output_height,
        info.api_version);
}

using fsr4::divideUp;
using fsr4::roundUp;

PassDependency passDependency(uint32_t pass_index, uint32_t next_pass_index, bool use_wmma) {
    const auto current_type = kPasses[pass_index].type;
    const bool has_next_pass = next_pass_index < kPasses.size();

    if (!use_wmma) {
        return PassDependency::FullComputeMemory;
    }

    if (pass_index == kPrepassIndex) {
        // The next padding pass writes the same tensor border.
        return PassDependency::ScratchWrite;
    }

    if (pass_index == kFinalPaddingPassIndex) {
        return PassDependency::ScratchRead;
    }

    if (pass_index >= kFirstPaddingPassIndex && pass_index <= kLastModelPassIndex) {
        const bool model_to_padding = current_type == PassType::ModelPass && has_next_pass &&
                                      kPasses[next_pass_index].type == PassType::TensorPadding;

        return model_to_padding ? PassDependency::ScratchWrite : PassDependency::ScratchReadWrite;
    }

    return PassDependency::FullComputeMemory;
}

std::array<uint32_t, 2> dispatchSize(uint32_t pass_index,
                                     uint32_t output_width,
                                     uint32_t output_height,
                                     uint32_t model_bucket_width,
                                     uint32_t model_bucket_height) {
    const uint32_t padded_output_width = roundUp(output_width, 8);
    const uint32_t padded_output_height = roundUp(output_height, 8);
    const auto& pass = kPasses[pass_index];

    switch (pass.type) {
    case PassType::Prepass:
    case PassType::Postpass:
        return {divideUp(padded_output_width, 64), divideUp(padded_output_height, 2)};
    case PassType::ModelPass: {
        const uint32_t model_layer_index = model_binding::calculationLayer(pass_index) + 1;
        const uint32_t half_width = padded_output_width / 2, half_height = padded_output_height / 2;
        const uint32_t quarter_width = padded_output_width / 4,
                       quarter_height = padded_output_height / 4;
        const uint32_t eighth_width = padded_output_width / 8,
                       eighth_height = padded_output_height / 8;

        switch (model_layer_index) {
        case 1:
        case 2:
            return {divideUp(half_width, 16), divideUp(half_height, 8)};
        case 3:
            return {divideUp(quarter_width, 32), quarter_height};
        case 4:
        case 5:
            return {divideUp(quarter_width, 16), divideUp(quarter_height, 4)};
        case 6:
            return {divideUp(eighth_width, 32), eighth_height};
        case 7:
        case 8:
            return {divideUp(eighth_width, 16), divideUp(eighth_height, 2)};
        case 9:
            return {divideUp(quarter_width, 32), divideUp(quarter_height, 2)};
        case 10:
            return {divideUp(quarter_width, 16), divideUp(quarter_height, 4)};
        case 11:
            return {divideUp(half_width, 32), divideUp(half_height, 2)};
        case 12:
            return {divideUp(half_width, 16), divideUp(half_height, 8)};
        default:
            return {1, 1};
        }
    }
    case PassType::TensorPadding: {
        const uint32_t model_layer_index = model_binding::paddingLayer(pass_index);

        uint32_t resolution_divisor = 2;

        switch (model_layer_index) {
        case 0:
        case 1:
        case 2:
        case 11:
        case 12:
            resolution_divisor = 2;
            break;
        case 3:
        case 4:
        case 5:
        case 9:
        case 10:
            resolution_divisor = 4;
            break;
        case 7:
        case 8:
            resolution_divisor = 8;
            break;
        case 6:
            return {0, 0};
        }

        const uint32_t tensor_width = padded_output_width / resolution_divisor;
        const uint32_t tensor_height = padded_output_height / resolution_divisor;
        const uint32_t bucket_tensor_width = model_bucket_width / resolution_divisor;
        const uint32_t bucket_tensor_height = model_bucket_height / resolution_divisor;

        // Clear the unused tensor border.
        const uint32_t padding_columns =
            tensor_width <= bucket_tensor_width
                ? std::min(bucket_tensor_width + 1u - tensor_width, 33u)
                : 0u;
        const uint32_t padding_rows = tensor_height <= bucket_tensor_height
                                          ? std::min(bucket_tensor_height + 1u - tensor_height, 1u)
                                          : 0u;
        const uint64_t active_area = uint64_t{tensor_width} * tensor_height;
        const uint64_t padded_area =
            uint64_t{tensor_width + padding_columns} * (tensor_height + padding_rows);
        const uint64_t padding_element_count = padded_area - active_area;
        return {static_cast<uint32_t>((padding_element_count + 31u) / 32u), 1};
    }
    }
    return {1, 1};
}
} // namespace fsr4fp8

namespace fsr4runtime {

fs::path ModelShaderSelector::selectFp8Shader(const fs::path& directory,
                                              uint32_t pass_index,
                                              const fs::path& prepass_override,
                                              const fs::path& postpass_override) {
    auto shader_path = stockShaderPath(directory, pass_index, prepass_override, postpass_override);
    const bool is_endpoint = pass_index == kPrepassIndex || pass_index == kPostpassIndex;
    const bool has_endpoint_override =
        (pass_index == kPrepassIndex && !prepass_override.empty()) ||
        (pass_index == kPostpassIndex && options.numbered_shader_files &&
         !postpass_override.empty());
    if (options.use_wmma && is_endpoint && !has_endpoint_override) {
        shader_path = directory / (std::string(kPasses[pass_index].name) + "_wmma.spv");
    }

    if (options.windows_fp8_optimized && pass_index == kPostpassIndex &&
        !(options.numbered_shader_files && !postpass_override.empty())) {
        shader_path = directory / "postpass_wmma.win-opt.spv";
    }
    const bool is_model_pass = kPasses[pass_index].type == PassType::ModelPass;
    if (options.windows_fp8_optimized && is_model_pass) {
        shader_path = directory / (std::string(kPasses[pass_index].name) + ".win-opt.spv");
        if (options.extensive_logging) {
            fsr4runtime::log("[shader] model=fp8 variant=windows-optimized path='" +
                                 shader_path.string() + "'",
                             fsr4log::Level::Debug);
        }
    }
    return shader_path;
}

QualityContext::ModelPipelinePlan
QualityContext::prepareFp8Pipelines(const fs::path& shader_dir, const ShaderDriverPolicy& policy) {
    bool windows_fp8_optimized = use_wmma && policy.windows_amd_shaders_allowed;

    if (const char* value = std::getenv("FSR4_VK_FP8_WINDOWS_OPT")) {
        windows_fp8_optimized = windows_fp8_optimized && std::strcmp(value, "0") != 0;
    }

    log(std::string("[fp8] Windows optimized model shaders=") +
        (windows_fp8_optimized ? "enabled" : "disabled"));

    if (windows_fp8_optimized) {
        // Fall back to stock shaders unless every optimized pass is available.
        for (const auto& pass : kPasses) {
            if (pass.type != PassType::ModelPass) {
                continue;
            }

            const auto path = shader_dir / (std::string(pass.name) + ".win-opt.spv");

            if (!fsr4assets::exists(path)) {
                throw std::runtime_error("FP8 Windows optimized shader missing: " + path.string());
            }
        }
    }

    ModelPipelinePlan plan;
    plan.windows_fp8_optimized = windows_fp8_optimized;
    plan.require_wave32.fill(true);
    return plan;
}
} // namespace fsr4runtime
