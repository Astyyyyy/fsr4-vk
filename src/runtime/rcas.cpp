// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

#include "runtime/context.hpp"

namespace fsr4runtime {
namespace {
constexpr VkDeviceSize kRcasConstantOffset = 768;
constexpr uint32_t kRcasInput = 94;
constexpr uint32_t kRcasExposure = 95;
constexpr uint32_t kRcasOutput = 11;
constexpr uint32_t kRcasConstantDescriptor = 2;
static_assert(kRcasConstantOffset + sizeof(RcasConstants) <= 1024);

const char* rcasShaderName(ColorSpace transfer, bool external_exposure) {
    switch (transfer) {
    case ColorSpace::Linear:
        return external_exposure ? "linear.spv" : "linear-auto.spv";
    case ColorSpace::Nonlinear:
        return "nonlinear.spv";
    case ColorSpace::Srgb:
        return "srgb.spv";
    case ColorSpace::Pq:
        return "pq.spv";
    }
    throw std::logic_error("invalid RCAS transfer");
}
} // namespace

void QualityContext::prepareRcas(VkCommandBuffer command_buffer,
                                 Frame& frame,
                                 const Image& output,
                                 const Image& exposure,
                                 bool use_staging_output) {
    const auto variant = color_space == ColorSpace::Linear
                             ? (exposure.view ? 0u : 1u)
                             : static_cast<unsigned>(color_space) + 1u;
    auto& cached = rcas_pipeline;
    if (rcas_pipeline_variant != variant) {
        retirePipeline(cached.handle);
        rcas_pipeline_variant = variant;
    }
    auto& pipeline = cached.handle;
    if (!pipeline) {
        const auto path =
            rcas_shader_root / rcasShaderName(color_space, exposure.view != VK_NULL_HANDLE);
        pipeline = createComputePipeline(readSpirv(path), true);
        log("[rcas] SDK4.1.1 shader=" + path.string() + " tile=4x64 wave=32");
    }
    frame.rcas_pipeline = pipeline;
    frame.rcas_dispatch = {fsr4::divideUp(output.width, cached.output_tile.width),
                           fsr4::divideUp(output.height, cached.output_tile.height)};
    if (!rcas_input.image) {
        rcas_input = createImage(physical_device,
                                 device,
                                 fsr4::roundUp(max_output_width, 16),
                                 fsr4::roundUp(max_output_height, 16),
                                 VK_FORMAT_R16G16B16A16_SFLOAT,
                                 VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
    }
    imageBarrier(dispatch.cmd_pipeline_barrier2,
                 command_buffer,
                 rcas_input.image,
                 rcas_input_initialized ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED,
                 VK_IMAGE_LAYOUT_GENERAL,
                 rcas_input_initialized ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                                        : VK_PIPELINE_STAGE_2_NONE,
                 rcas_input_initialized ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : VK_ACCESS_2_NONE,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    rcas_input_initialized = true;
    writeStorageImageDescriptor(frame, descriptor_set::storage_images, 2, rcas_input.view);
    writeSampledImageDescriptor(frame,
                                descriptor_set::sampled_images,
                                kRcasInput,
                                rcas_input.view,
                                VK_IMAGE_LAYOUT_GENERAL);
    const bool direct_exposure = exposure.view != VK_NULL_HANDLE;
    writeSampledImageDescriptor(frame,
                                descriptor_set::sampled_images,
                                kRcasExposure,
                                direct_exposure ? exposure.view : images[image_slot::exposure].view,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    writeStorageImageDescriptor(frame,
                                descriptor_set::storage_images,
                                kRcasOutput,
                                use_staging_output ? staging_output.view : output.view);
    VkDescriptorAddressInfoEXT address{VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT};
    address.address = frame.constants_address + kRcasConstantOffset;
    address.range = sizeof(RcasConstants);
    VkDescriptorGetInfoEXT descriptor{VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT};
    descriptor.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptor.data.pUniformBuffer = &address;
    vkGetDescriptorEXT(
        device,
        &descriptor,
        descriptor_size.uniform_buffer,
        descriptorAddress(frame, descriptor_set::uniform_buffers, kRcasConstantDescriptor));
}

void QualityContext::recordRcas(VkCommandBuffer command_buffer,
                                Frame& frame,
                                const Image& output,
                                const OptimizedConstants& constants) {
    const auto rcas = makeRcasConstants(constants.rcas_sharpness, constants.pre_exposure);
    std::memcpy(
        static_cast<char*>(frame.constants.mapped) + kRcasConstantOffset, &rcas, sizeof(rcas));
    imageBarrier(dispatch.cmd_pipeline_barrier2,
                 command_buffer,
                 rcas_input.image,
                 VK_IMAGE_LAYOUT_GENERAL,
                 VK_IMAGE_LAYOUT_GENERAL,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    RootConstants root{};
    root.data[5] = kRcasConstantDescriptor;
    root.data[6] = output.width;
    root.data[7] = output.height;
    trackPipelineUse(frame.rcas_pipeline);
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, frame.rcas_pipeline);
    vkCmdPushConstants(
        command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(root), &root);
    vkCmdDispatch(command_buffer, frame.rcas_dispatch.width, frame.rcas_dispatch.height, 1);
}
} // namespace fsr4runtime
