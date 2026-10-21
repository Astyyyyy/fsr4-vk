// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

#pragma once

#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vulkan/vulkan.h>

namespace fsr4vk {

template <typename Function>
inline Function loadDeviceFunction(VkDevice device,
                                   PFN_vkGetDeviceProcAddr get_device_proc_addr,
                                   std::initializer_list<const char*> names) {
    if (!device || !get_device_proc_addr) {
        throw std::invalid_argument("missing Vulkan device dispatch");
    }
    for (const char* name : names) {
        if (auto function = get_device_proc_addr(device, name)) {
            return reinterpret_cast<Function>(function);
        }
    }
    std::string message = "missing Vulkan device entry point: ";
    bool first = true;
    for (const char* name : names) {
        if (!first) {
            message += " or ";
        }
        message += name;
        first = false;
    }
    throw std::runtime_error(message);
}

template <typename Function>
inline Function loadPromotedDeviceFunction(VkDevice device,
                                           PFN_vkGetDeviceProcAddr get_device_proc_addr,
                                           uint32_t api_version,
                                           const char* core_name,
                                           const char* khr_name) {
    const bool prefer_core =
        VK_API_VERSION_MAJOR(api_version) > 1 ||
        (VK_API_VERSION_MAJOR(api_version) == 1 && VK_API_VERSION_MINOR(api_version) >= 3);
    if (prefer_core) {
        return loadDeviceFunction<Function>(device, get_device_proc_addr, {core_name, khr_name});
    }
    return loadDeviceFunction<Function>(device, get_device_proc_addr, {khr_name, core_name});
}

// Prefer KHR names on Vulkan 1.1 and 1.2 to avoid unusable core trampolines.
struct DeviceDispatch {
    PFN_vkCmdPipelineBarrier2 cmd_pipeline_barrier2 = nullptr;

    PFN_vkCmdWriteTimestamp2 cmd_write_timestamp2 = nullptr;
    PFN_vkGetBufferDeviceAddress get_buffer_device_address = nullptr;

    PFN_vkSetDebugUtilsObjectNameEXT set_debug_utils_object_name = nullptr;

    DeviceDispatch() = default;

    DeviceDispatch(VkDevice device,
                   PFN_vkGetDeviceProcAddr get_device_proc_addr,
                   uint32_t api_version = VK_API_VERSION_1_3) {
        load(device, get_device_proc_addr, api_version);
    }

    void load(VkDevice device,
              PFN_vkGetDeviceProcAddr get_device_proc_addr,
              uint32_t api_version = VK_API_VERSION_1_3) {
        cmd_pipeline_barrier2 =
            loadPromotedDeviceFunction<PFN_vkCmdPipelineBarrier2>(device,
                                                                  get_device_proc_addr,
                                                                  api_version,
                                                                  "vkCmdPipelineBarrier2",
                                                                  "vkCmdPipelineBarrier2KHR");
        cmd_write_timestamp2 =
            loadPromotedDeviceFunction<PFN_vkCmdWriteTimestamp2>(device,
                                                                 get_device_proc_addr,
                                                                 api_version,
                                                                 "vkCmdWriteTimestamp2",
                                                                 "vkCmdWriteTimestamp2KHR");
        get_buffer_device_address =
            loadPromotedDeviceFunction<PFN_vkGetBufferDeviceAddress>(device,
                                                                     get_device_proc_addr,
                                                                     api_version,
                                                                     "vkGetBufferDeviceAddress",
                                                                     "vkGetBufferDeviceAddressKHR");
        if (get_device_proc_addr) {
            set_debug_utils_object_name = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
                get_device_proc_addr(device, "vkSetDebugUtilsObjectNameEXT"));
        }
    }
};

inline void setDebugName(const DeviceDispatch& dispatch,
                         VkDevice device,
                         VkObjectType type,
                         uint64_t handle,
                         const char* name) {
    if (dispatch.set_debug_utils_object_name && device && handle && name) {
        VkDebugUtilsObjectNameInfoEXT info{VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
        info.objectType = type;
        info.objectHandle = handle;
        info.pObjectName = name;
        dispatch.set_debug_utils_object_name(device, &info);
    }
}

} // namespace fsr4vk
