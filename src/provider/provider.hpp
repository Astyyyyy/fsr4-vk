// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

#pragma once

#if !defined(_WIN32)
#define __declspec(x) __attribute__((visibility("default")))
#endif
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "diagnostics/diagnostics.hpp"
#include "ffx_upscale.h"
#include "models/models.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <new>
#include <optional>
#include <vector>
#include <sstream>
#include <vulkan/vulkan.h>

namespace fsr4provider {

constexpr ffxStructType_t kBackendVkDescType = 3u;

struct CreateBackendVkDesc {
    ffxCreateContextDescHeader header;
    VkDevice device;
    VkPhysicalDevice physicalDevice;

    PFN_vkGetDeviceProcAddr getDeviceProcAddr;
};

struct ProviderVersion {
    std::uint64_t id;
    const char* name;

    bool (*has_assets)(const std::filesystem::path&);
    fsr4_411::QualityContextPtr (*create_context)(const fsr4_411::ContextCreateInfo&);
};

inline const std::array<ProviderVersion, 2> kProviderVersions{{
    {fsr4int8::kVersionId, fsr4int8::kVersionName, fsr4int8::hasAssets, fsr4int8::createContext},
    {fsr4fp8::kVersionId, fsr4fp8::kVersionName, fsr4fp8::hasAssets, fsr4fp8::createContext},
}};

inline constexpr std::uint64_t kAutoVersionId =
    (0xF5A5CA1Dull << 32) | ((4ull << 22) | (1ull << 12) | 1ull);
inline constexpr const char* kAutoVersionName = "4.1.1 Vulkan Auto";

struct AdvertisedVersion {
    std::uint64_t id;
    const char* name;
};

// Keep the original list indices and IDs stable for existing loader settings.
inline const std::array<AdvertisedVersion, 3> kAdvertisedVersions{{
    {kAutoVersionId, kAutoVersionName},
    {fsr4int8::kVersionId, fsr4int8::kVersionName},
    {fsr4fp8::kVersionId, fsr4fp8::kVersionName},
}};

constexpr const char* kRequiredDeviceFunctionGroups[] = {
    "vkGetDescriptorEXT",
    "vkCmdBindDescriptorBuffersEXT",
    "vkCmdSetDescriptorBufferOffsetsEXT",
    "vkGetDescriptorSetLayoutSizeEXT",
    "vkGetDescriptorSetLayoutBindingOffsetEXT",
    "vkCmdPipelineBarrier2|vkCmdPipelineBarrier2KHR",
    "vkCmdWriteTimestamp2|vkCmdWriteTimestamp2KHR",
    "vkGetBufferDeviceAddress|vkGetBufferDeviceAddressKHR",
};

struct ColorCaptureState {
    bool enabled = false;
    bool pending = false;
    bool first_capture_armed = true;
    bool triggered_mode = false;
    bool coherent = false;

    std::filesystem::path base_path;
    std::filesystem::path trigger_path;

    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;

    VkDeviceSize buffer_capacity = 0;
    VkDeviceSize payload_size = 0;
    VkEvent event = VK_NULL_HANDLE;

    std::uint32_t width = 0;
    std::uint32_t height = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;

    std::uint64_t next_index = 0;
    std::uint64_t pending_index = 0;
    std::uint64_t pending_dispatch = 0;
};

struct CapturedImage {
    std::string name;
    fsr4_411::Image image;
    VkDeviceSize offset = 0;
    VkDeviceSize size = 0;
};

struct FrameCaptureSlot {
    ColorCaptureState storage;
    std::vector<CapturedImage> images;
    std::string metadata;
    uint32_t index = 0;
};

struct FrameCaptureState {
    bool enabled = false, active = false, started = false, failed = false;
    uint32_t count = 32, queued = 0, completed = 0;
    uint64_t last_dispatch = 0;
    std::filesystem::path base, trigger, directory;
    std::array<FrameCaptureSlot, 8> slots;
    std::vector<bool> written;
};

struct ProviderContext {
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    PFN_vkGetDeviceProcAddr getDeviceProcAddr = nullptr;

    std::uint32_t apiVersion = VK_API_VERSION_1_1;
    std::uint32_t flags = 0;

    FfxApiDimensions2D maxRenderSize{};
    FfxApiDimensions2D maxUpscaleSize{};

    ffxApiMessage message = nullptr;
    const ProviderVersion* version = nullptr;
    fsr4_411::QualityContextPtr model_context;

    bool first_dispatch = true;
    bool extensive_logging = false;
    bool use_staging_output = false;
    std::uint64_t dispatch_count = 0;

    float previous_jitter[2] = {0.0f, 0.0f};
    bool has_previous_jitter = false;

    ColorCaptureState color_capture;
    FrameCaptureState frame_capture;
};

bool environmentFlag(const char* name) noexcept;

std::filesystem::path providerModuleDirectory();
std::filesystem::path providerLogPath();

void diagnosticLine(const std::string& text, fsr4log::Level level = fsr4log::Level::Info) noexcept;
void diagnostic(const ProviderContext& context,
                const std::string& text,
                fsr4log::Level level = fsr4log::Level::Info) noexcept;
void providerMessage(const ProviderContext& context,
                     uint32_t type,
                     const std::string& text) noexcept;

void configureFrameCapture(ProviderContext& provider);
void pollFrameCapture(ProviderContext& provider) noexcept;
FrameCaptureSlot* beginFrameCapture(ProviderContext& provider,
                                    const ffxDispatchDescUpscale& desc,
                                    const std::array<fsr4_411::Image, 6>& images,
                                    uint64_t dispatch_index) noexcept;
void endFrameCapture(ProviderContext& provider,
                     FrameCaptureSlot* slot,
                     VkCommandBuffer command_buffer) noexcept;
void finishFrameCapture(ProviderContext& provider) noexcept;

void pollColorCapture(ProviderContext& provider) noexcept;
void recordColorCapture(ProviderContext& provider,
                        VkCommandBuffer command_buffer,
                        const fsr4_411::Image& color,
                        std::uint64_t dispatch_index) noexcept;
void finishColorCapture(ProviderContext& provider) noexcept;

std::uint64_t requestedVersionId(const ffxApiHeader* header);
const ProviderVersion* selectVersion(std::uint64_t id, VkPhysicalDevice physical_device);

ffxReturnCode_t createContext(ffxContext* context,
                              ffxCreateContextDescHeader* header,
                              const ffxAllocationCallbacks* callbacks);
ffxReturnCode_t destroyContext(ffxContext* context, const ffxAllocationCallbacks* callbacks);
ffxReturnCode_t configure(ffxContext* context, const ffxConfigureDescHeader* header);
ffxReturnCode_t query(ffxContext* context, ffxQueryDescHeader* header);
ffxReturnCode_t dispatch(ffxContext* context, const ffxDispatchDescHeader* header);
} // namespace fsr4provider

namespace fsr4provider {
inline const char* apiSurfaceFormatName(uint32_t value) {
    switch (value) {
    case FFX_API_SURFACE_FORMAT_UNKNOWN:
        return "UNKNOWN";
    case FFX_API_SURFACE_FORMAT_R32G32B32A32_TYPELESS:
        return "R32G32B32A32_TYPELESS";
    case FFX_API_SURFACE_FORMAT_R32G32B32A32_UINT:
        return "R32G32B32A32_UINT";
    case FFX_API_SURFACE_FORMAT_R32G32B32A32_FLOAT:
        return "R32G32B32A32_FLOAT";
    case FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT:
        return "R16G16B16A16_FLOAT";
    case FFX_API_SURFACE_FORMAT_R32G32B32_FLOAT:
        return "R32G32B32_FLOAT";
    case FFX_API_SURFACE_FORMAT_R32G32_FLOAT:
        return "R32G32_FLOAT";
    case FFX_API_SURFACE_FORMAT_R8_UINT:
        return "R8_UINT";
    case FFX_API_SURFACE_FORMAT_R32_UINT:
        return "R32_UINT";
    case FFX_API_SURFACE_FORMAT_R8G8B8A8_TYPELESS:
        return "R8G8B8A8_TYPELESS";
    case FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM:
        return "R8G8B8A8_UNORM";
    case FFX_API_SURFACE_FORMAT_R8G8B8A8_SNORM:
        return "R8G8B8A8_SNORM";
    case FFX_API_SURFACE_FORMAT_R8G8B8A8_SRGB:
        return "R8G8B8A8_SRGB";
    case FFX_API_SURFACE_FORMAT_B8G8R8A8_TYPELESS:
        return "B8G8R8A8_TYPELESS";
    case FFX_API_SURFACE_FORMAT_B8G8R8A8_UNORM:
        return "B8G8R8A8_UNORM";
    case FFX_API_SURFACE_FORMAT_B8G8R8A8_SRGB:
        return "B8G8R8A8_SRGB";
    case FFX_API_SURFACE_FORMAT_R11G11B10_FLOAT:
        return "R11G11B10_FLOAT";
    case FFX_API_SURFACE_FORMAT_R10G10B10A2_UNORM:
        return "R10G10B10A2_UNORM";
    case FFX_API_SURFACE_FORMAT_R16G16_FLOAT:
        return "R16G16_FLOAT";
    case FFX_API_SURFACE_FORMAT_R16G16_UINT:
        return "R16G16_UINT";
    case FFX_API_SURFACE_FORMAT_R16G16_SINT:
        return "R16G16_SINT";
    case FFX_API_SURFACE_FORMAT_R16_FLOAT:
        return "R16_FLOAT";
    case FFX_API_SURFACE_FORMAT_R16_UINT:
        return "R16_UINT";
    case FFX_API_SURFACE_FORMAT_R16_UNORM:
        return "R16_UNORM";
    case FFX_API_SURFACE_FORMAT_R16_SNORM:
        return "R16_SNORM";
    case FFX_API_SURFACE_FORMAT_R8_UNORM:
        return "R8_UNORM";
    case FFX_API_SURFACE_FORMAT_R8G8_UNORM:
        return "R8G8_UNORM";
    case FFX_API_SURFACE_FORMAT_R8G8_UINT:
        return "R8G8_UINT";
    case FFX_API_SURFACE_FORMAT_R32_FLOAT:
        return "R32_FLOAT";
    case FFX_API_SURFACE_FORMAT_R9G9B9E5_SHAREDEXP:
        return "R9G9B9E5_SHAREDEXP";
    case FFX_API_SURFACE_FORMAT_R16G16B16A16_TYPELESS:
        return "R16G16B16A16_TYPELESS";
    case FFX_API_SURFACE_FORMAT_R32G32_TYPELESS:
        return "R32G32_TYPELESS";
    case FFX_API_SURFACE_FORMAT_R10G10B10A2_TYPELESS:
        return "R10G10B10A2_TYPELESS";
    case FFX_API_SURFACE_FORMAT_R16G16_TYPELESS:
        return "R16G16_TYPELESS";
    case FFX_API_SURFACE_FORMAT_R16_TYPELESS:
        return "R16_TYPELESS";
    case FFX_API_SURFACE_FORMAT_R8_TYPELESS:
        return "R8_TYPELESS";
    case FFX_API_SURFACE_FORMAT_R8G8_TYPELESS:
        return "R8G8_TYPELESS";
    case FFX_API_SURFACE_FORMAT_R32_TYPELESS:
        return "R32_TYPELESS";
    default:
        return "UNKNOWN";
    }
}

inline std::string apiSurfaceFormat(uint32_t value) {
    return std::string(apiSurfaceFormatName(value)) + "(" + std::to_string(value) + ")";
}

inline const char* apiResourceTypeName(uint32_t value) {
    switch (value) {
    case FFX_API_RESOURCE_TYPE_BUFFER:
        return "BUFFER";
    case FFX_API_RESOURCE_TYPE_TEXTURE1D:
        return "TEXTURE1D";
    case FFX_API_RESOURCE_TYPE_TEXTURE2D:
        return "TEXTURE2D";
    case FFX_API_RESOURCE_TYPE_TEXTURE_CUBE:
        return "TEXTURE_CUBE";
    case FFX_API_RESOURCE_TYPE_TEXTURE3D:
        return "TEXTURE3D";
    default:
        return "UNKNOWN";
    }
}

inline std::string apiResourceType(uint32_t value) {
    return std::string(apiResourceTypeName(value)) + "(" + std::to_string(value) + ")";
}

inline const char* apiResourceStateName(uint32_t value) {
    switch (value) {
    case FFX_API_RESOURCE_STATE_COMMON:
        return "COMMON";
    case FFX_API_RESOURCE_STATE_UNORDERED_ACCESS:
        return "UNORDERED_ACCESS";
    case FFX_API_RESOURCE_STATE_COMPUTE_READ:
        return "COMPUTE_READ";
    case FFX_API_RESOURCE_STATE_PIXEL_READ:
        return "PIXEL_READ";
    case FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ:
        return "PIXEL_COMPUTE_READ";
    case FFX_API_RESOURCE_STATE_COPY_SRC:
        return "COPY_SRC";
    case FFX_API_RESOURCE_STATE_COPY_DEST:
        return "COPY_DEST";
    case FFX_API_RESOURCE_STATE_GENERIC_READ:
        return "GENERIC_READ";
    case FFX_API_RESOURCE_STATE_INDIRECT_ARGUMENT:
        return "INDIRECT_ARGUMENT";
    case FFX_API_RESOURCE_STATE_PRESENT:
        return "PRESENT";
    case FFX_API_RESOURCE_STATE_RENDER_TARGET:
        return "RENDER_TARGET";
    case FFX_API_RESOURCE_STATE_DEPTH_ATTACHMENT:
        return "DEPTH_ATTACHMENT";
    default:
        return "UNKNOWN";
    }
}

inline std::string apiResourceState(uint32_t value) {
    return std::string(apiResourceStateName(value)) + "(" + fsr4log::hexValue(value) + ")";
}
} // namespace fsr4provider
