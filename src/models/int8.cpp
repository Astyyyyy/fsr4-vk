// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

#include "models/models.hpp"
#include "runtime/context.hpp"

namespace fsr4int8 {
using namespace fsr4runtime;

const char* resolutionBucket(uint32_t output_width, uint32_t output_height) {
    return fsr4::resolutionBucket(output_width, output_height);
}

ShaderVariants::ShaderVariants(const std::filesystem::path& int8_root, bool enabled)
    : root(int8_root) {
    if (!enabled) {
        return;
    }

    const char* exact = std::getenv("FSR4_VK_INT8_EXACT");
    if (!exact || std::strcmp(exact, "0") != 0) {
        profile = "exact";
    }
}

std::filesystem::path ShaderVariants::candidate(const std::filesystem::path& original) const {
    if (profile.empty() || root.empty()) {
        return original;
    }

    const auto relative = original.lexically_relative(root);
    if (relative.empty() || relative.is_absolute() || *relative.begin() == "..") {
        throw std::invalid_argument("INT8 shader outside its variant asset root: " +
                                    original.string());
    }

    return root / "optimized" / profile / relative;
}

bool hasAssets(const fs::path& root) {
    return fsr4assets::exists(root / "int8" / "1080" / "native" / "pass-00.spv") &&
           fsr4assets::exists(root / "int8" / "auto-exposure.spv") &&
           fsr4assets::exists(root / "int8/models/4.1.1/weights_standard.bin");
}

fs::path modelRoot(const fs::path& asset_root) {
    return asset_root / "int8";
}

fsr4_411::QualityContextPtr createContext(const fsr4_411::ContextCreateInfo& info) {
    const fs::path root = modelRoot(info.asset_root);
    const std::string preset = fsr4::resolutionPreset(info.max_render_width, info.max_output_width);
    const fs::path tier_root =
        root / resolutionBucket(info.max_output_width, info.max_output_height);

    // Native through Performance share shaders and weights.
    // Ultra Performance uses its own model shaders, postpass and weights.
    const fs::path shader_bundle = tier_root / "native";
    const fs::path ultraperformance_shader_bundle = tier_root / "ultraperf";

    const bool render_resolution_motion_vectors = !info.high_resolution_motion_vectors;
    const ColorEndpointOptions color_endpoints{
        .render_resolution_motion = render_resolution_motion_vectors,
        .inverted_depth = info.depth_inverted,
        .auto_exposure = info.auto_exposure,
        .motion_jitter_cancellation = info.motion_vector_jitter_cancellation,
    };
    const auto endpoint_root = info.asset_root / "int8/shaders/4.1.1/color-endpoints" /
                               resolutionBucket(info.max_output_width, info.max_output_height);
    const auto standard = selectColorEndpointShaders(
        endpoint_root, ColorSpace::Linear, color_endpoints, PostpassFamily::Standard);
    const auto ultra = selectColorEndpointShaders(
        endpoint_root, ColorSpace::Linear, color_endpoints, PostpassFamily::UltraPerformance);

    const fs::path weights = root / "models/4.1.1/weights_standard.bin";
    const fs::path ultraperformance_weights = root / "models/4.1.1/weights_ultraperf.bin";

    for (const auto& path : {standard.prepass, standard.postpass, ultra.postpass}) {
        if (!fsr4assets::exists(path)) {
            throw std::runtime_error("missing FSR 4.1.1 INT8 endpoint shader: " + path.string());
        }
    }

    if (!fsr4assets::exists(weights) || !fsr4assets::exists(ultraperformance_weights)) {
        throw std::runtime_error("missing FSR 4.1.1 INT8 weights: " + weights.string());
    }

    fsr4runtime::setDiagnosticRoot(info.diagnostic_root);
    fsr4runtime::log(
        std::string("[int8] endpoint tier=") +
        resolutionBucket(info.max_output_width, info.max_output_height) + " initial_preset=" +
        preset + " motion_resolution=" + (render_resolution_motion_vectors ? "render" : "display") +
        " depth=" + (info.depth_inverted ? "inverted" : "standard") +
        " exposure=" + (info.auto_exposure ? "automatic" : "external") +
        " motion_jitter_cancellation=" + (info.motion_vector_jitter_cancellation ? "yes" : "no"));
    if (info.extensive_logging) {
        const auto log_endpoint = [](const char* role, const fs::path& asset) {
            fsr4runtime::log(std::string("[shader] model=int8 role=") + role + " asset='" +
                                 asset.string() + "'",
                             fsr4log::Level::Debug);
        };
        log_endpoint("standard-prepass", standard.prepass);
        log_endpoint("standard-postpass", standard.postpass);
        log_endpoint("ultraperf-prepass", ultra.prepass);
        log_endpoint("ultraperf-postpass", ultra.postpass);
    }

    // INT8 skips the interleaved tensor-padding passes used by FP8.
    const ShaderVariants policy(root, true);

    fsr4runtime::ShaderRecipe recipe{
        .profile = policy.profile,
        .select =
            [policy](const fs::path& path) {
                return policy.candidate(path);
            },
    };

    return std::make_unique<fsr4runtime::QualityContext>(
        info.physical_device,
        info.device,
        shader_bundle,
        weights,
        root / "auto-exposure.spv",
        info.get_device_proc_addr,
        fsr4runtime::ModelRuntimeOptions{
            .rcas_shader_root = info.asset_root / "common/shaders/4.1.1/rcas",
            .color_shader_root = endpoint_root,
            .use_wmma = false,
            .use_int8_dispatch = true,
            .numbered_shader_files = true,
            .execute_padding_passes = false,
            // These prepasses already sample motion vectors at output resolution.
            .direct_display_motion_vectors = info.high_resolution_motion_vectors,

            .ultraperformance_shader_dir = ultraperformance_shader_bundle,
            .ultraperformance_model_data_path = ultraperformance_weights,

            .color_endpoints = color_endpoints,
            .pass_policy =
                {
                    .dispatch_size = fsr4int8::dispatchSize,
                    .dependency =
                        [](uint32_t current, uint32_t next, bool) {
                            return fsr4int8::passDependency(current, next);
                        },
                    .postpass_updates_exposure = true,
                },
            .shader_recipe = std::move(recipe),
        },
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

PassDependency passDependency(uint32_t pass_index, uint32_t next_pass_index) {
    const bool final_model_to_postpass = kPasses[pass_index].type == PassType::ModelPass &&
                                         next_pass_index < kPasses.size() &&
                                         kPasses[next_pass_index].type == PassType::Postpass;
    return final_model_to_postpass ? PassDependency::ScratchRead : PassDependency::ScratchReadWrite;
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
        return {divideUp(padded_output_width, 16), divideUp(padded_output_height, 16)};
    case PassType::Postpass:
        // Each invocation writes 4x4 pixels; an 8x8 group covers 32x32 pixels.
        return {divideUp(padded_output_width, 32), divideUp(padded_output_height, 32)};
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
        case 12:
            return {divideUp(half_width, 64), half_height};
        case 3:
        case 4:
        case 5:
        case 10:
        case 11:
            return {divideUp(quarter_width, 64), quarter_height};
        case 6:
        case 7:
        case 8:
        case 9:
            return {divideUp(eighth_width, 64), eighth_height};
        default:
            return {1, 1};
        }
    }
    case PassType::TensorPadding: {
        const uint32_t model_layer_index = model_binding::paddingLayer(pass_index);

        if (model_layer_index == model_binding::unused_padding_layer) {
            return {0, 0};
        }
        // Count the one-pixel tensor border without duplicating corners.
        static constexpr uint32_t kPaddingResolutionDivisors[13] = {
            2, 2, 2, 4, 4, 4, 8, 8, 8, 4, 4, 2, 2};
        const uint32_t resolution_divisor = kPaddingResolutionDivisors[model_layer_index];
        const uint32_t tensor_width = padded_output_width / resolution_divisor;
        const uint32_t tensor_height = padded_output_height / resolution_divisor;
        const uint32_t padding_element_count =
            2 * tensor_width + (tensor_height > 2 ? 2 * (tensor_height - 2) : 0);
        return {divideUp(padding_element_count, 32), 1};
    }
    }
    return {1, 1};
}

} // namespace fsr4int8

namespace fsr4runtime {

fs::path ModelShaderSelector::windowsInt8ShaderPath(const fs::path& directory,
                                                    uint32_t pass_index,
                                                    const fs::path& stock_path) const {
    fs::path shader_path = stock_path;
    const bool is_model_pass = kPasses[pass_index].type == PassType::ModelPass;
    // The saturating-dot rewrite is only proved for the bundled model weights.
    if (options.windows_int8_optimized && is_model_pass && fsr4assets::isEmbedded(shader_path)) {
        const auto candidate = directory / (stock_path.stem().string() + ".win-dotacc.spv");

        if (fsr4assets::exists(candidate)) {
            shader_path = candidate;
        }
    }
    return shader_path;
}

fs::path ModelShaderSelector::selectInt8Variant(const fs::path& original) {
    fs::path shader_path = original;
    const auto candidate = recipe.candidate(shader_path);

    if (candidate != shader_path) {
        if (fsr4assets::exists(candidate)) {
            shader_path = candidate;
        } else if (!exact_fallback_logged) {
            fsr4runtime::log("[int8] exact variants unavailable in external assets; "
                             "using stock where missing (regenerate int8/optimized)",
                             fsr4log::Level::Warning);
            exact_fallback_logged = true;
        }
    }

    if (options.extensive_logging) {
        fsr4runtime::log("[shader] model=int8 profile=" +
                             (recipe.profile.empty() ? std::string("stock") : recipe.profile) +
                             " path=" + shader_path.string(),
                         fsr4log::Level::Debug);
    }

    return shader_path;
}

fs::path ModelShaderSelector::selectInt8Shader(const fs::path& directory,
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

    return selectInt8Variant(windowsInt8ShaderPath(directory, pass_index, shader_path));
}

QualityContext::ModelPipelinePlan
QualityContext::prepareInt8Pipelines(const ModelRuntimeOptions& runtime_options,
                                     const ShaderDriverPolicy& policy) {
    const bool rdna2_device = policy.rdna2_device;
    // Wave32 was slightly faster on RDNA2.
    bool force_int8_endpoint_wave32 = rdna2_device || runtime_options.use_wmma;
    bool force_int8_model_wave32 = rdna2_device || runtime_options.use_wmma;

    if (const char* value = std::getenv("FSR4_VK_INT8_ENDPOINT_WAVE32");
        value && !runtime_options.use_wmma) {
        force_int8_endpoint_wave32 = std::strcmp(value, "0") != 0;
    }

    if (const char* value = std::getenv("FSR4_VK_INT8_MODEL_WAVE32");
        value && !runtime_options.use_wmma) {
        force_int8_model_wave32 = std::strcmp(value, "0") != 0;
    }

    log(std::string("[int8] endpoint subgroup=") +
        (force_int8_endpoint_wave32 ? "forced-wave32" : "driver-selected") +
        " model subgroup=" + (force_int8_model_wave32 ? "forced-wave32" : "driver-selected"));

    ModelPipelinePlan plan;
    for (uint32_t index = 0; index < model_pass_policy.passes.size(); ++index) {
        const bool endpoint = model_pass_policy.passes[index].type == PassType::Prepass ||
                              model_pass_policy.passes[index].type == PassType::Postpass;
        const bool model_pass = model_pass_policy.passes[index].type == PassType::ModelPass;
        plan.require_wave32[index] = !((endpoint && !force_int8_endpoint_wave32) ||
                                       (model_pass && !force_int8_model_wave32));
    }
    // Windows vector/dot variants are separate from the shared exact INT8 patches.
    bool windows_int8_optimized = policy.windows_amd_shaders_allowed;

    if (const char* value = std::getenv("FSR4_VK_INT8_WINDOWS_OPT")) {
        windows_int8_optimized = windows_int8_optimized && std::strcmp(value, "0") != 0;
    }

    log(std::string("[int8] Windows combined vector/dot shaders=") +
        (windows_int8_optimized ? "enabled for embedded assets" : "disabled"));

    const auto& int8_variants = runtime_options.shader_recipe;

    log("[int8] shader patches=" +
        (int8_variants.profile.empty() ? std::string("stock") : int8_variants.profile));

    plan.windows_int8_optimized = windows_int8_optimized;
    return plan;
}

} // namespace fsr4runtime
