// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

#pragma once

#include "runtime/resources.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vulkan/vulkan.h>

namespace fsr4runtime {
namespace image_slot {
inline constexpr std::size_t history = 0;
inline constexpr std::size_t reprojected = 1;
inline constexpr std::size_t recurrent = 2;
inline constexpr std::size_t wmma_scratch0 = 3;
inline constexpr std::size_t wmma_scratch1 = 4;
inline constexpr std::size_t exposure = 5;
inline constexpr std::size_t fallback_mask = 6;
inline constexpr std::size_t exposure_counter = 7;
inline constexpr std::size_t exposure_mips = 8;
} // namespace image_slot

namespace descriptor_set {
inline constexpr uint32_t sampled_images = 0;
inline constexpr uint32_t model_reads = 1;
inline constexpr uint32_t samplers = 2;
inline constexpr uint32_t storage_images = 3;
inline constexpr uint32_t model_writes = 4;
inline constexpr uint32_t uniform_buffers = 5;
} // namespace descriptor_set

namespace exposure_binding {
inline constexpr uint32_t counter = 90;
inline constexpr uint32_t mips = 91;
inline constexpr uint32_t output = 92;
inline constexpr uint32_t color = 93;
} // namespace exposure_binding

namespace constant_offset {
inline constexpr uint32_t auto_exposure = 512;
} // namespace constant_offset

namespace model_binding {
inline constexpr uint32_t layer_count = 12;
inline constexpr uint32_t weight_binding_base = 18;
inline constexpr uint32_t scratch_offset_base = 19;
inline constexpr uint32_t descriptors_per_layer = 2;
inline constexpr uint32_t weight_offset_word = 1;
inline constexpr uint32_t scratch_offset_word = 4;
inline constexpr uint32_t unused_padding_layer = 6;

constexpr uint32_t scratchOffset(uint32_t zero_based_layer) {
    return scratch_offset_base + zero_based_layer * descriptors_per_layer;
}

constexpr uint32_t weightOffset(uint32_t zero_based_layer) {
    return scratchOffset(zero_based_layer) + 1;
}

constexpr uint32_t weightBinding(uint32_t zero_based_layer) {
    return weight_binding_base + weightOffset(zero_based_layer);
}

// Calculation passes use even indices from 2 to 24; padding uses odd indices from 1 to 25.
constexpr uint32_t calculationLayer(uint32_t pass_index) {
    return (pass_index - 2) / 2;
}

constexpr uint32_t paddingLayer(uint32_t pass_index) {
    return pass_index <= 1 ? 0 : (pass_index - 1) / 2;
}
} // namespace model_binding
} // namespace fsr4runtime

namespace fsr4runtime {

struct ModelConstants {
    struct LayerDimensions {
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t padding[2]{};
    };

    LayerDimensions layers[17]{};
};

static_assert(sizeof(ModelConstants) == 272);

inline ModelConstants makeModelConstants(uint32_t width, uint32_t height) {
    static constexpr std::array<uint32_t, 17> divisors{
        2,
        1,
        2,
        2,
        4,
        4,
        4,
        8,
        8,
        8,
        4,
        4,
        2,
        2,
        1,
        1,
        1,
    };

    width = fsr4::roundUp(width, 8);
    height = fsr4::roundUp(height, 8);

    ModelConstants constants{};
    for (uint32_t index = 0; index < divisors.size(); ++index) {
        constants.layers[index].width = width / divisors[index];
        constants.layers[index].height = height / divisors[index];
    }
    return constants;
}

} // namespace fsr4runtime

namespace fsr4runtime {

// Make scratch writes visible to the next model pass.
void scratchBufferBarrier(PFN_vkCmdPipelineBarrier2 pipeline_barrier2,
                          VkCommandBuffer command_buffer,
                          VkBuffer scratch_buffer,
                          VkAccessFlags2 dst_access);

enum class PassType : uint8_t {
    Prepass,
    ModelPass,
    TensorPadding,
    Postpass,
};

enum class CapturePoint : uint8_t {
    BeforePrepass,
    BeforePostpass,
    AfterPostpass,
};

using CaptureHook = void (*)(void*, CapturePoint, VkCommandBuffer);

struct Pass {
    const char* name;
    PassType type;
};

// The recorder skips disabled and unused padding passes.
constexpr std::array<Pass, 27> kPasses{{
    {"prepass", PassType::Prepass},   {"pass0_post", PassType::TensorPadding},
    {"pass1", PassType::ModelPass},   {"pass1_post", PassType::TensorPadding},
    {"pass2", PassType::ModelPass},   {"pass2_post", PassType::TensorPadding},
    {"pass3", PassType::ModelPass},   {"pass3_post", PassType::TensorPadding},
    {"pass4", PassType::ModelPass},   {"pass4_post", PassType::TensorPadding},
    {"pass5", PassType::ModelPass},   {"pass5_post", PassType::TensorPadding},
    {"pass6", PassType::ModelPass},   {"pass6_post", PassType::TensorPadding},
    {"pass7", PassType::ModelPass},   {"pass7_post", PassType::TensorPadding},
    {"pass8", PassType::ModelPass},   {"pass8_post", PassType::TensorPadding},
    {"pass9", PassType::ModelPass},   {"pass9_post", PassType::TensorPadding},
    {"pass10", PassType::ModelPass},  {"pass10_post", PassType::TensorPadding},
    {"pass11", PassType::ModelPass},  {"pass11_post", PassType::TensorPadding},
    {"pass12", PassType::ModelPass},  {"pass12_post", PassType::TensorPadding},
    {"postpass", PassType::Postpass},
}};

inline constexpr uint32_t kPrepassIndex = 0;
inline constexpr uint32_t kFirstPaddingPassIndex = 1;
inline constexpr uint32_t kPostpassIndex = kPasses.size() - 1;
inline constexpr uint32_t kFinalPaddingPassIndex = kPostpassIndex - 1;
inline constexpr uint32_t kLastModelPassIndex = kFinalPaddingPassIndex - 1;

enum class PassDependency {
    FullComputeMemory,
    ScratchRead,
    ScratchWrite,
    ScratchReadWrite,
};

struct RootConstants {
    uint32_t data[8] = {};
};

static_assert(sizeof(RootConstants) == 32);

struct ModelPassPolicy {
    using Dispatch = std::array<uint32_t, 2> (*)(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
    using Dependency = PassDependency (*)(uint32_t, uint32_t, bool);
    Dispatch dispatch_size = nullptr;
    Dependency dependency = nullptr;
    bool postpass_updates_exposure = false;
    std::span<const Pass> passes = kPasses;
    ModelConstants (*make_constants)(uint32_t, uint32_t) = makeModelConstants;
    const char* model_name = nullptr;
};

struct FrameRecordingInfo {
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;

    VkDeviceAddress descriptor_buffer_address = 0;
    VkDeviceSize descriptor_set_capacity = 0;
    PFN_vkCmdBindDescriptorBuffersEXT bind_descriptor_buffers = nullptr;
    PFN_vkCmdSetDescriptorBufferOffsetsEXT set_descriptor_buffer_offsets = nullptr;
    bool descriptor_buffers_bound = false;

    bool use_wmma = false;
    const ModelPassPolicy* model_policy = nullptr;
    bool execute_padding_passes = false;
    std::span<const VkPipeline> pipelines;
    VkDeviceAddress constants_address = 0;
    uint32_t output_width = 0;
    uint32_t output_height = 0;
    uint32_t model_bucket_width = 0;
    uint32_t model_bucket_height = 0;

    VkImage history = VK_NULL_HANDLE;
    VkImage recurrent = VK_NULL_HANDLE;
    VkImage reprojected = VK_NULL_HANDLE;
    VkImage auto_exposure_texture = VK_NULL_HANDLE;
    VkBuffer calculation_scratch_buffer = VK_NULL_HANDLE;
    bool transition_temporal_images = false;
    PFN_vkCmdPipelineBarrier2 pipeline_barrier2 = nullptr;

    VkQueryPool timestamp_queries = VK_NULL_HANDLE;
    uint32_t timestamp_base = 0;
    PFN_vkCmdWriteTimestamp2 write_timestamp2 = nullptr;
    bool verbose = false;
    bool timestamp_barriers = false;

    CaptureHook capture_hook = nullptr;
    void* capture_user = nullptr;
};

void recordModelPasses(const FrameRecordingInfo& recording);

} // namespace fsr4runtime
