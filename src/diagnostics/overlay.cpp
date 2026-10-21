// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

#include "runtime/context.hpp"
#include "diagnostics/overlay_spv.hpp"
#include "diagnostics/overlay_font.hpp"

#include <array>
#include <string_view>

namespace fsr4runtime {
namespace {
constexpr uint32_t kOverlayOffset = 1024;
constexpr uint32_t kTextColumns = 96;
#if defined(_WIN32)
constexpr std::string_view kLibraryLabel = "Windows .dll";
#else
constexpr std::string_view kLibraryLabel = "Linux .so";
#endif

struct OverlayData {
    uint32_t render[2];
    uint32_t motion[2];
    float motion_scale[2];
    float motion_jitter[2];
    float pre_exposure;
    uint32_t flags;
    VkDeviceAddress font;
    uint32_t text[7][kTextColumns]{};
};

struct OverlayPush {
    VkDeviceAddress data;
    uint32_t width;
    uint32_t height;
    uint32_t font_pixels;
    uint32_t debug_view;
    uint32_t header_width;
    uint32_t header_height;
};

static_assert(sizeof(OverlayPush) == sizeof(RootConstants));
static_assert(offsetof(OverlayData, text) == 48);
static_assert(kOverlayOffset + sizeof(OverlayData) <= kConstantBufSize);

// Vulkan 1.3 and format_feature_flags2 expose read support per format.
// The caller may not have enabled shaderStorageImageReadWithoutFormat.
bool canReadOutput(VkPhysicalDevice physical_device, VkFormat format) {
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical_device, &properties);
    if (properties.apiVersion < VK_API_VERSION_1_3) {
        uint32_t count = 0;
        check(vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &count, nullptr),
              "overlay extension count");
        std::vector<VkExtensionProperties> extensions(count);
        check(vkEnumerateDeviceExtensionProperties(
                  physical_device, nullptr, &count, extensions.data()),
              "overlay extensions");
        if (std::none_of(extensions.begin(), extensions.end(), [](const auto& extension) {
                return std::string_view(extension.extensionName) ==
                       VK_KHR_FORMAT_FEATURE_FLAGS_2_EXTENSION_NAME;
            })) {
            return false;
        }
    }
    VkFormatProperties3 features{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3};
    VkFormatProperties2 properties2{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, &features};
    vkGetPhysicalDeviceFormatProperties2(physical_device, format, &properties2);
    return (features.optimalTilingFeatures & VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT) !=
           0;
}

uint32_t writeText(uint32_t* destination, std::string_view text) {
    // Sorted glyph positions let the shader search only the visible text.
    std::fill_n(destination, kTextColumns, 0xffffff00u);
    uint32_t position = 0;
    for (size_t i = 0; i < std::min<size_t>(text.size(), kTextColumns); ++i) {
        const auto character = static_cast<unsigned char>(text[i]);
        const uint32_t glyph = character >= 32 && character < 127 ? character - 32 : '?' - 32;
        destination[i] = (position << 8) | glyph;
        position += kOverlayFontAdvances[glyph];
    }
    return position;
}
} // namespace

void QualityContext::recordOverlay(VkCommandBuffer command_buffer,
                                   Frame& frame,
                                   const Image& output,
                                   const Image& motion,
                                   const OptimizedConstants& constants,
                                   const fsr4_411::OverlayOptions& options) {
    if (!overlay_font.buffer) {
        overlay_font = createBuffer(
            physical_device,
            device,
            sizeof(kOverlayFontPixels),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            true,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        std::memcpy(overlay_font.mapped, kOverlayFontPixels, sizeof(kOverlayFontPixels));
        overlay_font_address =
            deviceAddress(dispatch.get_buffer_device_address, device, overlay_font.buffer);
    }
    if (overlay_read_format != output.format) {
        overlay_read_supported = canReadOutput(physical_device, output.format);
        overlay_read_format = output.format;
    }
    // Debug view uses the write-only shader.
    const bool blend = !options.debug_view && overlay_read_supported;
    if (overlay_pipeline_blend != blend) {
        retirePipeline(overlay_pipeline);
        overlay_pipeline_blend = blend;
    }
    if (!overlay_pipeline) {
        const auto* data = blend ? kOverlayBlendSpirv : kOverlaySpirv;
        const auto size = blend ? std::size(kOverlayBlendSpirv) : std::size(kOverlaySpirv);
        overlay_pipeline = createComputePipeline(std::vector<uint32_t>(data, data + size), false);
        log(std::string("[overlay] pipeline ready; scene blending=") + (blend ? "yes" : "no") +
            " | " + std::string(kLibraryLabel) + " | RDNA2: " + (device_is_rdna2 ? "Yes" : "No") +
            " | " + model_wave_label);
    }

    const std::string title = "FSR4 | " + std::string(options.label);
    const std::string details = std::to_string(constants.width_lr) + "x" +
                                std::to_string(constants.height_lr) + " -> " +
                                std::to_string(output.width) + "x" + std::to_string(output.height);
    const std::string status = std::string(kLibraryLabel) +
                               " | RDNA2: " + (device_is_rdna2 ? "Yes" : "No") + " | " +
                               model_wave_label;
    OverlayData data{{constants.width_lr, constants.height_lr},
                     {motion.width, motion.height},
                     {constants.mv_scale[0], constants.mv_scale[1]},
                     {constants.motion_vector_jitter_cancellation[0],
                      constants.motion_vector_jitter_cancellation[1]},
                     constants.pre_exposure,
                     (options.hdr ? 1u : 0u) | (constants.rcas_enabled ? 4u : 0u) |
                         (static_cast<uint32_t>(color_space) << 3) |
                         (depth_inverted_enabled ? 32u : 0u),
                     overlay_font_address};
    const auto title_width = writeText(data.text[0], title);
    const auto details_width =
        writeText(data.text[1], options.debug_view ? details + " | " + status : details);
    const auto status_width = writeText(data.text[2], status);
    writeText(data.text[3], options.hdr ? "Input color | HDR" : "Input color | SDR");
    writeText(data.text[4], depth_inverted_enabled ? "Depth | inverted" : "Depth | normal");
    writeText(data.text[5], "Motion | jitter corrected");
    writeText(data.text[6], "History blend | purple = less history");
    std::memcpy(static_cast<char*>(frame.constants.mapped) + kOverlayOffset, &data, sizeof(data));

    const uint32_t font_pixels = std::clamp((output.height * 28 + 720) / 1440, 18u, 42u);
    const uint32_t detail_pixels = std::max(14u, (font_pixels * 6 + 3) / 7);
    const uint32_t details_y = fsr4::divideUp(36 * font_pixels, kOverlayFontSize) + 2;
    const uint32_t status_y = details_y + fsr4::divideUp(36 * detail_pixels, kOverlayFontSize) + 2;
    const uint32_t margin = options.debug_view ? 0 : std::min({12u, output.width, output.height});
    const uint32_t header_width = std::min(output.width - margin,
                                           fsr4::divideUp(std::max({title_width * font_pixels,
                                                                    details_width * detail_pixels,
                                                                    status_width * detail_pixels}) +
                                                              8 * font_pixels,
                                                          kOverlayFontSize));
    const uint32_t header_height = std::min(
        output.height - margin,
        (options.debug_view ? details_y : status_y) +
            fsr4::divideUp(36 * detail_pixels, kOverlayFontSize) + (options.debug_view ? 12u : 2u));
    const OverlayPush push{frame.constants_address + kOverlayOffset,
                           output.width,
                           output.height,
                           font_pixels,
                           options.debug_view ? 1u : 0u,
                           header_width,
                           header_height};

    // Wait for postpass writes before drawing the overlay.
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                            (blend ? VK_ACCESS_2_SHADER_STORAGE_READ_BIT : 0) |
                            (options.debug_view ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : 0);
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &barrier;
    dispatch.cmd_pipeline_barrier2(command_buffer, &dependency);
    trackPipelineUse(overlay_pipeline);
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, overlay_pipeline);
    vkCmdPushConstants(
        command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(command_buffer,
                  fsr4::divideUp(options.debug_view ? output.width : header_width, 8),
                  fsr4::divideUp(options.debug_view ? output.height : push.header_height, 8),
                  1);
}
} // namespace fsr4runtime
