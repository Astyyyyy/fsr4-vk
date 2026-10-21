// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

#include "runtime/resources.hpp"
#include "runtime/context.hpp"

namespace fsr4_411 {
void check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed with VkResult " +
                                 std::to_string(result));
    }
}

std::vector<uint8_t> readBytes(const fs::path& path) {
    return fsr4assets::read(path);
}

std::vector<uint32_t> readSpirv(const fs::path& path) {
    const auto bytes = readBytes(path);

    if (bytes.size() % 4) {
        throw std::runtime_error("invalid SPIR-V size");
    }

    std::vector<uint32_t> words(bytes.size() / 4);
    std::memcpy(words.data(), bytes.data(), bytes.size());

    return words;
}

std::vector<uint32_t> readModelSpirv(const fs::path& path, uint32_t api_version) {
    auto words = readSpirv(path);

    constexpr uint32_t spirv_magic = 0x07230203u;
    constexpr uint32_t spirv_1_5 = 0x00010500u;
    constexpr uint32_t spirv_1_6 = 0x00010600u;

    if (words.size() < 5 || words[0] != spirv_magic) {
        throw std::runtime_error("invalid SPIR-V header in " + path.string());
    }

    if (api_version >= VK_API_VERSION_1_2 && api_version < VK_API_VERSION_1_3) {
        // These shaders support a SPIR-V 1.5 header; replacements must support it too.
        if (words[1] == spirv_1_6) {
            words[1] = spirv_1_5;
        } else if (words[1] > spirv_1_5) {
            throw std::runtime_error("unsupported SPIR-V version in " + path.string());
        }
    }

    return words;
}

uint32_t findMemoryType(VkPhysicalDevice physical_device,
                        uint32_t allowed_type_bits,
                        VkMemoryPropertyFlags required_properties,
                        VkMemoryPropertyFlags preferred_properties) {
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(physical_device, &properties);

    uint32_t fallback_index = std::numeric_limits<uint32_t>::max();

    for (uint32_t index = 0; index < properties.memoryTypeCount; ++index) {
        const bool type_is_allowed = (allowed_type_bits & (1u << index)) != 0;
        const VkMemoryPropertyFlags available_properties =
            properties.memoryTypes[index].propertyFlags;
        const bool has_required_properties =
            (available_properties & required_properties) == required_properties;

        if (!type_is_allowed || !has_required_properties) {
            continue;
        }

        const bool has_preferred_properties =
            (available_properties & preferred_properties) == preferred_properties;

        if (has_preferred_properties) {
            return index;
        }

        if (fallback_index == std::numeric_limits<uint32_t>::max()) {
            fallback_index = index;
        }
    }

    if (fallback_index != std::numeric_limits<uint32_t>::max()) {
        return fallback_index;
    }

    throw std::runtime_error("no compatible Vulkan memory type");
}

Buffer createBuffer(VkPhysicalDevice physical_device,
                    VkDevice device,
                    VkDeviceSize size,
                    VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags properties,
                    bool enable_device_address,
                    VkMemoryPropertyFlags preferred_properties) {
    Buffer result{};
    result.size = size;

    try {
        VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        buffer_info.size = size;
        buffer_info.usage = usage;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        check(vkCreateBuffer(device, &buffer_info, nullptr, &result.buffer), "vkCreateBuffer");

        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device, result.buffer, &requirements);

        VkMemoryAllocateFlagsInfo allocation_flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        allocation_flags.flags = enable_device_address ? VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT : 0;

        VkMemoryAllocateInfo allocation_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation_info.pNext = enable_device_address ? &allocation_flags : nullptr;
        allocation_info.allocationSize = requirements.size;
        allocation_info.memoryTypeIndex = findMemoryType(
            physical_device, requirements.memoryTypeBits, properties, preferred_properties);

        VkPhysicalDeviceMemoryProperties memory_properties{};
        vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);

        result.memory_type_index = allocation_info.memoryTypeIndex;
        result.memory_property_flags =
            memory_properties.memoryTypes[allocation_info.memoryTypeIndex].propertyFlags;
        result.memory_heap_index =
            memory_properties.memoryTypes[allocation_info.memoryTypeIndex].heapIndex;

        check(vkAllocateMemory(device, &allocation_info, nullptr, &result.memory),
              "vkAllocateMemory(buffer)");

        check(vkBindBufferMemory(device, result.buffer, result.memory, 0), "vkBindBufferMemory");

        if (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
            check(vkMapMemory(device, result.memory, 0, size, 0, &result.mapped), "vkMapMemory");
        }

        return result;
    } catch (...) {
        if (result.mapped) {
            vkUnmapMemory(device, result.memory);
        }

        if (result.buffer) {
            vkDestroyBuffer(device, result.buffer, nullptr);
        }

        if (result.memory) {
            vkFreeMemory(device, result.memory, nullptr);
        }

        throw;
    }
}

Image createImage(VkPhysicalDevice physical_device,
                  VkDevice device,
                  uint32_t width,
                  uint32_t height,
                  VkFormat format,
                  VkImageUsageFlags usage,
                  VkImageAspectFlags aspect) {
    Image result{};
    result.format = format;
    result.width = width;
    result.height = height;

    try {
        VkImageCreateInfo image_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        image_info.imageType = VK_IMAGE_TYPE_2D;
        image_info.format = format;
        image_info.extent = {width, height, 1};
        image_info.mipLevels = 1;
        image_info.arrayLayers = 1;
        image_info.samples = VK_SAMPLE_COUNT_1_BIT;
        image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        image_info.usage = usage;
        image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        check(vkCreateImage(device, &image_info, nullptr, &result.image), "vkCreateImage");

        VkMemoryRequirements requirements{};
        vkGetImageMemoryRequirements(device, result.image, &requirements);

        VkMemoryAllocateInfo allocation_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation_info.allocationSize = requirements.size;
        allocation_info.memoryTypeIndex = findMemoryType(
            physical_device, requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        check(vkAllocateMemory(device, &allocation_info, nullptr, &result.memory),
              "vkAllocateMemory(image)");
        check(vkBindImageMemory(device, result.image, result.memory, 0), "vkBindImageMemory");

        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = result.image;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = format;
        vi.subresourceRange.aspectMask = aspect;
        vi.subresourceRange.levelCount = 1;
        vi.subresourceRange.layerCount = 1;

        check(vkCreateImageView(device, &vi, nullptr, &result.view), "vkCreateImageView");
        return result;
    } catch (...) {
        if (result.view) {
            vkDestroyImageView(device, result.view, nullptr);
        }

        if (result.image) {
            vkDestroyImage(device, result.image, nullptr);
        }

        if (result.memory) {
            vkFreeMemory(device, result.memory, nullptr);
        }

        throw;
    }
}

void destroyBuffer(VkDevice device, Buffer& buffer) {
    if (buffer.mapped) {
        vkUnmapMemory(device, buffer.memory);
    }

    if (buffer.buffer) {
        vkDestroyBuffer(device, buffer.buffer, nullptr);
    }

    if (buffer.memory) {
        vkFreeMemory(device, buffer.memory, nullptr);
    }

    buffer = {};
}

void destroyImage(VkDevice device, Image& image) {
    if (image.view) {
        vkDestroyImageView(device, image.view, nullptr);
    }

    if (image.image) {
        vkDestroyImage(device, image.image, nullptr);
    }

    if (image.memory) {
        vkFreeMemory(device, image.memory, nullptr);
    }

    image = {};
}

VkDeviceAddress deviceAddress(PFN_vkGetBufferDeviceAddress get_buffer_device_address,
                              VkDevice device,
                              VkBuffer buffer) {
    VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    info.buffer = buffer;
    return get_buffer_device_address(device, &info);
}

void imageBarrier(PFN_vkCmdPipelineBarrier2 pipeline_barrier2,
                  VkCommandBuffer command_buffer,
                  VkImage image,
                  VkImageLayout old_layout,
                  VkImageLayout new_layout,
                  VkPipelineStageFlags2 source_stage,
                  VkAccessFlags2 source_access,
                  VkPipelineStageFlags2 destination_stage,
                  VkAccessFlags2 destination_access,
                  VkImageAspectFlags aspect) {
    VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    barrier.srcStageMask = source_stage;
    barrier.srcAccessMask = source_access;
    barrier.dstStageMask = destination_stage;
    barrier.dstAccessMask = destination_access;
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = aspect;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;

    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.imageMemoryBarrierCount = 1;
    dependency.pImageMemoryBarriers = &barrier;
    pipeline_barrier2(command_buffer, &dependency);
}

void computeBarrier(PFN_vkCmdPipelineBarrier2 pipeline_barrier2, VkCommandBuffer command_buffer) {
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask =
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &barrier;
    pipeline_barrier2(command_buffer, &dependency);
}
} // namespace fsr4_411

namespace fsr4runtime {

void QualityContext::prepareExternalViews(Frame& frame,
                                          Image& color,
                                          Image& depth,
                                          Image& motion,
                                          Image& exposure,
                                          Image& optional_mask,
                                          Image& output) {
    for (auto view : frame.views) {
        vkDestroyImageView(device, view, nullptr);
    }

    frame.views.clear();
    const std::array<std::pair<const char*, Image*>, 6> external_images{{
        {"color", &color},
        {"depth", &depth},
        {"motion", &motion},
        {"external-exposure", &exposure},
        {"optional-mask", &optional_mask},
        {"output", &output},
    }};

    frame.views.reserve(external_images.size());
    for (std::size_t index = 0; index < external_images.size(); ++index) {
        const auto& [role, image] = external_images[index];
        if (!image->image || image->view) {
            continue;
        }

        // Reuse matching views for aliased inputs.
        for (std::size_t previous = 0; previous < index; ++previous) {
            const auto* candidate = external_images[previous].second;
            if (candidate->image == image->image && candidate->format == image->format &&
                candidate->aspect == image->aspect && candidate->view) {
                image->view = candidate->view;
                break;
            }
        }
        if (image->view) {
            continue;
        }

        VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view_info.image = image->image;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = image->format;
        view_info.subresourceRange = {image->aspect, 0, 1, 0, 1};

        check(vkCreateImageView(device, &view_info, nullptr, &image->view), "external image view");
        frame.views.push_back(image->view);

        if (extensive_logging && frame_count <= 3) {
            log("[resource] frame=" + std::to_string(frame_count) + " created view role=" + role +
                    " image=" + fsr4log::handle(image->image) +
                    " view=" + fsr4log::handle(image->view) +
                    " format=" + fsr4log::vulkanFormat(image->format) + " active_size=" +
                    std::to_string(image->width) + "x" + std::to_string(image->height),
                fsr4log::Level::Debug);
        }
    }
}

bool QualityContext::updateDynamicDescriptors(VkCommandBuffer command_buffer,
                                              Frame& frame,
                                              Image& color,
                                              Image& depth,
                                              Image& motion,
                                              Image& exposure_image,
                                              Image& optional_mask,
                                              Image& output,
                                              Image& reprojected,
                                              bool use_staging_output) {
    const bool run_auto_exposure = auto_exposure_enabled && !exposure_image.image;

    writeSampledImageDescriptor(
        frame, descriptor_set::sampled_images, 0, images[image_slot::history].view);
    // COMPUTE_READ inputs use the standard FidelityFX sampled-image layout.
    writeSampledImageDescriptor(frame,
                                descriptor_set::sampled_images,
                                1,
                                motion.view,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    writeSampledImageDescriptor(frame,
                                descriptor_set::sampled_images,
                                2,
                                depth.view,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    writeSampledImageDescriptor(frame,
                                descriptor_set::sampled_images,
                                3,
                                color.view,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    writeSampledImageDescriptor(
        frame, descriptor_set::sampled_images, 4, images[image_slot::recurrent].view);
    const VkImageView dummy_mask = images[image_slot::fallback_mask].view;
    writeExposureDescriptors(frame, exposure_image);

    writeSampledImageDescriptor(
        frame, descriptor_set::sampled_images, 9, images[image_slot::reprojected].view);
    writeSampledImageDescriptor(frame,
                                descriptor_set::sampled_images,
                                16,
                                optional_mask.view ? optional_mask.view : dummy_mask,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    writeStorageImageDescriptor(frame, descriptor_set::storage_images, 0, dummy_mask);
    prepareStagingOutput(command_buffer, output, use_staging_output);

    if (extensive_logging && frame_count <= 3) {
        log("[descriptor] frame=" + std::to_string(frame_count) +
                " binding=u2 role=upscaled-output staging_view=" +
                fsr4log::handle(staging_output.view) +
                " output_view=" + fsr4log::handle(output.view),
            fsr4log::Level::Debug);
    }

    writeStorageImageDescriptor(frame,
                                descriptor_set::storage_images,
                                2,
                                use_staging_output ? staging_output.view : output.view);
    if (extensive_logging && (frame_count <= 3 || frame_count % 120 == 0)) {
        log("[output] frame=" + std::to_string(frame_count) +
                " mode=" + std::string(use_staging_output ? "staging" : "direct") +
                " game_image=" + fsr4log::handle(output.image) +
                " staging_image=" + fsr4log::handle(staging_output.image),
            fsr4log::Level::Debug);
    }

    writeStorageImageDescriptor(frame, descriptor_set::storage_images, 3, reprojected.view);
    writeStorageImageDescriptor(frame, descriptor_set::storage_images, 4, dummy_mask);
    writeStorageImageDescriptor(frame, descriptor_set::storage_images, 5, dummy_mask);

    if (run_auto_exposure) {
        writeStorageImageDescriptor(frame,
                                    descriptor_set::storage_images,
                                    exposure_binding::counter,
                                    images[image_slot::exposure_counter].view);
        writeStorageImageDescriptor(frame,
                                    descriptor_set::storage_images,
                                    exposure_binding::mips,
                                    images[image_slot::exposure_mips].view);
        writeStorageImageDescriptor(frame,
                                    descriptor_set::storage_images,
                                    exposure_binding::output,
                                    images[image_slot::exposure].view);
        writeSampledImageDescriptor(frame,
                                    descriptor_set::sampled_images,
                                    exposure_binding::color,
                                    color.view,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    return run_auto_exposure;
}

void QualityContext::transitionExternalInputs(VkCommandBuffer command_buffer,
                                              const Image& color,
                                              const Image& depth,
                                              const Image& motion,
                                              const Image& exposure,
                                              const Image& optional_mask,
                                              const Image& output) {
    std::array<VkImageMemoryBarrier2, 6> barriers{};
    uint32_t barrier_count = 0;

    const auto add_barrier = [&](const Image& image) {
        if (!image.image) {
            return;
        }

        auto& barrier = barriers[barrier_count++];
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.image = image.image;
        barrier.subresourceRange = {image.aspect, 0, 1, 0, 1};
    };
    add_barrier(color);
    add_barrier(depth);
    add_barrier(motion);
    add_barrier(exposure);
    add_barrier(optional_mask);
    add_barrier(output);

    if (!barrier_count) {
        return;
    }

    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.imageMemoryBarrierCount = barrier_count;
    dependency.pImageMemoryBarriers = barriers.data();
    dispatch.cmd_pipeline_barrier2(command_buffer, &dependency);
}

uint8_t*
QualityContext::descriptorAddress(Frame& frame, uint32_t set_index, uint32_t binding_index) {
    auto* descriptor_bytes = static_cast<uint8_t*>(frame.descriptor_buffer.mapped);
    const VkDeviceSize set_offset = set_index * descriptor_capacity;
    const VkDeviceSize binding_offset =
        set_binding_offsets[set_index] + binding_index * set_strides[set_index];
    return descriptor_bytes + set_offset + binding_offset;
}

void QualityContext::writeImageDescriptorBuffer(Frame& frame,
                                                uint32_t set_index,
                                                uint32_t binding_index,
                                                VkDescriptorType type,
                                                VkImageView view,
                                                VkImageLayout image_layout) {
    if (!view) {
        return;
    }

    VkDescriptorImageInfo image_info{};
    image_info.imageView = view;
    image_info.imageLayout = image_layout;

    VkDescriptorGetInfoEXT descriptor_info{VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT};
    descriptor_info.type = type;
    size_t descriptor_byte_count;

    if (type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE) {
        descriptor_info.data.pSampledImage = &image_info;
        descriptor_byte_count = descriptor_size.sampled_image;
    } else {
        descriptor_info.data.pStorageImage = &image_info;
        descriptor_byte_count = descriptor_size.storage_image;
    }

    vkGetDescriptorEXT(device,
                       &descriptor_info,
                       descriptor_byte_count,
                       descriptorAddress(frame, set_index, binding_index));
}

void QualityContext::writeSampledImageDescriptor(Frame& frame,
                                                 uint32_t set_index,
                                                 uint32_t binding_index,
                                                 VkImageView view,
                                                 VkImageLayout image_layout) {
    writeImageDescriptorBuffer(
        frame, set_index, binding_index, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, view, image_layout);
}

void QualityContext::writeStorageImageDescriptor(Frame& frame,
                                                 uint32_t set_index,
                                                 uint32_t binding_index,
                                                 VkImageView view) {
    writeImageDescriptorBuffer(frame,
                               set_index,
                               binding_index,
                               VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                               view,
                               VK_IMAGE_LAYOUT_GENERAL);
}

void QualityContext::writeSamplerDescriptor(Frame& frame,
                                            uint32_t set_index,
                                            uint32_t binding_index,
                                            VkSampler sampler_handle) {
    VkDescriptorGetInfoEXT descriptor_info{VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT};
    descriptor_info.type = VK_DESCRIPTOR_TYPE_SAMPLER;
    descriptor_info.data.pSampler = &sampler_handle;

    vkGetDescriptorEXT(device,
                       &descriptor_info,
                       descriptor_size.sampler,
                       descriptorAddress(frame, set_index, binding_index));
}

void QualityContext::writeStorageBufferDescriptor(
    Frame& frame, uint32_t set_index, uint32_t binding_index, VkBuffer buffer, VkDeviceSize size) {
    VkDescriptorAddressInfoEXT buffer_address_info{VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT};
    buffer_address_info.address =
        fsr4_411::deviceAddress(dispatch.get_buffer_device_address, device, buffer);
    buffer_address_info.range = size;
    VkDescriptorGetInfoEXT descriptor_info{VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT};
    descriptor_info.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_info.data.pStorageBuffer = &buffer_address_info;

    vkGetDescriptorEXT(device,
                       &descriptor_info,
                       descriptor_size.storage_buffer,
                       descriptorAddress(frame, set_index, binding_index));
}

void QualityContext::fillStorageBufferDescriptors(Frame& frame,
                                                  uint32_t set_index,
                                                  VkBuffer buffer,
                                                  VkDeviceSize size) {
    writeStorageBufferDescriptor(frame, set_index, 0, buffer, size);
    auto* first_descriptor = descriptorAddress(frame, set_index, 0);

    for (uint32_t binding_index = 1; binding_index < kDescriptorCount; ++binding_index) {
        std::memcpy(first_descriptor + binding_index * set_strides[set_index],
                    first_descriptor,
                    descriptor_size.storage_buffer);
    }
}

void QualityContext::writeUniformBufferDescriptor(
    Frame& frame, uint32_t set_index, uint32_t binding_index, VkBuffer buffer, VkDeviceSize size) {
    VkDescriptorAddressInfoEXT buffer_address_info{VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT};
    buffer_address_info.address =
        fsr4_411::deviceAddress(dispatch.get_buffer_device_address, device, buffer);
    buffer_address_info.range = size;
    VkDescriptorGetInfoEXT descriptor_info{VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT};
    descriptor_info.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptor_info.data.pUniformBuffer = &buffer_address_info;

    vkGetDescriptorEXT(device,
                       &descriptor_info,
                       descriptor_size.uniform_buffer,
                       descriptorAddress(frame, set_index, binding_index));
}

void QualityContext::writeWeightDescriptors(Frame& frame, const Buffer& model_data) {
    if (frame.model_weights == model_data.buffer) {
        return;
    }
    writeStorageBufferDescriptor(frame,
                                 descriptor_set::model_reads,
                                 model_binding::weight_binding_base,
                                 model_data.buffer,
                                 model_data.size);

    for (uint32_t model_layer_index = 0; model_layer_index < model_binding::layer_count;
         ++model_layer_index) {
        const uint32_t weight_descriptor_index = model_binding::weightBinding(model_layer_index);
        writeStorageBufferDescriptor(frame,
                                     descriptor_set::model_reads,
                                     weight_descriptor_index,
                                     model_data.buffer,
                                     model_data.size);
    }
    frame.model_weights = model_data.buffer;
}

void QualityContext::writeStaticDescriptors(Frame& frame) {
    writeSampledImageDescriptor(
        frame, descriptor_set::sampled_images, 0, images[image_slot::history].view);
    writeSampledImageDescriptor(frame,
                                descriptor_set::sampled_images,
                                4,
                                images[image_slot::recurrent].view,
                                VK_IMAGE_LAYOUT_GENERAL);
    // Recurrent is sampled and written in GENERAL by postpass.
    writeStaticExposureDescriptor(frame);

    writeSampledImageDescriptor(
        frame, descriptor_set::sampled_images, 9, images[image_slot::reprojected].view);

    writeSamplerDescriptor(frame, descriptor_set::samplers, 0, sampler);

    writeStorageImageDescriptor(
        frame, descriptor_set::storage_images, 1, images[image_slot::history].view);
    writeStorageImageDescriptor(
        frame, descriptor_set::storage_images, 3, images[image_slot::reprojected].view);
    writeStorageImageDescriptor(
        frame, descriptor_set::storage_images, 6, images[image_slot::recurrent].view);
    writeStorageImageDescriptor(
        frame, descriptor_set::storage_images, 12, images[image_slot::wmma_scratch1].view);
    writeSampledImageDescriptor(frame,
                                descriptor_set::sampled_images,
                                20,
                                images[image_slot::wmma_scratch1].view,
                                VK_IMAGE_LAYOUT_GENERAL);

    writeUniformBufferDescriptor(frame,
                                 descriptor_set::uniform_buffers,
                                 0,
                                 frame.model_constants.buffer,
                                 frame.model_constants.size);
    writeUniformBufferDescriptor(
        frame, descriptor_set::uniform_buffers, 1, frame.constants.buffer, frame.constants.size);

    fillStorageBufferDescriptors(
        frame, descriptor_set::model_writes, scratch_buffer.buffer, scratch_buffer.size);
    fillStorageBufferDescriptors(
        frame, descriptor_set::model_reads, scratch_buffer.buffer, scratch_buffer.size);
    // Endpoint t18 reads the selected preset weight bank.
    writeWeightDescriptors(frame, weight_buffer);
}

void QualityContext::writeAllStaticDescriptors() {
    for (auto& frame : frames) {
        writeStaticDescriptors(frame);
    }
}

QualityContext::DescriptorGeometry
QualityContext::createDescriptorLayouts(PFN_vkGetDeviceProcAddr get_device_proc_addr_) {
    VkSamplerCreateInfo sampler_info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

    sampler_info.maxAnisotropy = 16.0f;
    sampler_info.compareOp = VK_COMPARE_OP_NEVER;
    sampler_info.maxLod = std::numeric_limits<float>::max();
    sampler_info.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;

    check(vkCreateSampler(device, &sampler_info, nullptr, &sampler), "vkCreateSampler");

    const auto make_set_layout = [&](uint32_t set_index, VkDescriptorType type, uint32_t count) {
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = 0;
        binding.descriptorType = type;
        binding.descriptorCount = count;
        binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo layout_info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layout_info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
        layout_info.bindingCount = 1;
        layout_info.pBindings = &binding;

        check(vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &set_layouts[set_index]),
              "vkCreateDescriptorSetLayout");
    };

    make_set_layout(
        descriptor_set::sampled_images, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kDescriptorCount);
    make_set_layout(
        descriptor_set::model_reads, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kDescriptorCount);
    make_set_layout(descriptor_set::samplers, VK_DESCRIPTOR_TYPE_SAMPLER, kDescriptorCount);
    make_set_layout(
        descriptor_set::storage_images, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kDescriptorCount);
    make_set_layout(
        descriptor_set::model_writes, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kDescriptorCount);
    make_set_layout(
        descriptor_set::uniform_buffers, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kDescriptorCount);

    VkPushConstantRange push_range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RootConstants)};

    VkPipelineLayoutCreateInfo pipeline_layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipeline_layout_info.setLayoutCount = kNumDescriptorSets;
    pipeline_layout_info.pSetLayouts = set_layouts.data();
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push_range;

    check(vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &pipeline_layout),
          "vkCreatePipelineLayout");

    vkGetDescriptorEXT = reinterpret_cast<PFN_vkGetDescriptorEXT>(
        get_device_proc_addr_(device, "vkGetDescriptorEXT"));
    vkCmdBindDescriptorBuffersEXT = reinterpret_cast<PFN_vkCmdBindDescriptorBuffersEXT>(
        get_device_proc_addr_(device, "vkCmdBindDescriptorBuffersEXT"));
    vkCmdSetDescriptorBufferOffsetsEXT = reinterpret_cast<PFN_vkCmdSetDescriptorBufferOffsetsEXT>(
        get_device_proc_addr_(device, "vkCmdSetDescriptorBufferOffsetsEXT"));
    vkGetDescriptorSetLayoutSizeEXT = reinterpret_cast<PFN_vkGetDescriptorSetLayoutSizeEXT>(
        get_device_proc_addr_(device, "vkGetDescriptorSetLayoutSizeEXT"));
    vkGetDescriptorSetLayoutBindingOffsetEXT =
        reinterpret_cast<PFN_vkGetDescriptorSetLayoutBindingOffsetEXT>(
            get_device_proc_addr_(device, "vkGetDescriptorSetLayoutBindingOffsetEXT"));

    if (!vkGetDescriptorEXT || !vkCmdBindDescriptorBuffersEXT ||
        !vkCmdSetDescriptorBufferOffsetsEXT || !vkGetDescriptorSetLayoutSizeEXT ||
        !vkGetDescriptorSetLayoutBindingOffsetEXT) {
        throw std::runtime_error("descriptor buffer entry points unavailable");
    }

    VkPhysicalDeviceDescriptorBufferPropertiesEXT buffer_properties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 properties2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    properties2.pNext = &buffer_properties;
    vkGetPhysicalDeviceProperties2(physical_device, &properties2);

    descriptor_size.sampled_image = buffer_properties.sampledImageDescriptorSize;
    descriptor_size.storage_image = buffer_properties.storageImageDescriptorSize;
    descriptor_size.uniform_buffer = buffer_properties.uniformBufferDescriptorSize;
    descriptor_size.storage_buffer = buffer_properties.storageBufferDescriptorSize;
    descriptor_size.sampler = buffer_properties.samplerDescriptorSize;

    const VkDeviceSize descriptor_offset_alignment =
        buffer_properties.descriptorBufferOffsetAlignment;

    const std::array<VkDeviceSize, kNumDescriptorSets> descriptor_strides{{
        descriptor_size.sampled_image,
        descriptor_size.storage_buffer,
        descriptor_size.sampler,
        descriptor_size.storage_image,
        descriptor_size.storage_buffer,
        descriptor_size.uniform_buffer,
    }};

    VkDeviceSize max_layout_size = 0;

    for (uint32_t set_index = 0; set_index < kNumDescriptorSets; set_index++) {
        VkDescriptorSetLayout layout = set_layouts[set_index];
        VkDeviceSize layout_size = 0;
        vkGetDescriptorSetLayoutSizeEXT(device, layout, &layout_size);
        set_strides[set_index] = descriptor_strides[set_index];

        if (layout_size > max_layout_size) {
            max_layout_size = layout_size;
        }

        VkDeviceSize binding_offset = 0;
        vkGetDescriptorSetLayoutBindingOffsetEXT(device, layout, 0, &binding_offset);
        set_binding_offsets[set_index] = binding_offset;

        const VkDeviceSize required_size =
            binding_offset + set_strides[set_index] * kDescriptorCount;

        if (!set_strides[set_index] || layout_size < required_size) {
            throw std::runtime_error("invalid descriptor-buffer layout size");
        }

        log("[descriptor] set=" + std::to_string(set_index) + " role=" +
            fsr4log::descriptorSetName(set_index) + " layout_bytes=" + std::to_string(layout_size) +
            " stride_bytes=" + std::to_string(set_strides[set_index]) +
            " binding_offset_bytes=" + std::to_string(binding_offset));
    }

    return {descriptor_offset_alignment, max_layout_size};
}

} // namespace fsr4runtime

namespace fsr4runtime {
namespace {
void remapAutoExposureDescriptors(std::vector<uint32_t>& words) {
    constexpr uint16_t op_decorate = 71;
    constexpr uint32_t decoration_binding = 33;
    constexpr uint32_t decoration_descriptor_set = 34;
    constexpr std::array<uint32_t, 4> resource_ids{{15, 19, 23, 26}};

    uint32_t set_edits = 0;
    uint32_t binding_edits = 0;

    for (std::size_t cursor = 5; cursor < words.size();) {
        const uint32_t instruction = words[cursor];
        const uint16_t word_count = static_cast<uint16_t>(instruction >> 16);
        const uint16_t opcode = static_cast<uint16_t>(instruction & 0xffffu);

        if (!word_count || cursor + word_count > words.size()) {
            throw std::runtime_error("invalid auto-exposure SPIR-V instruction stream");
        }

        if (opcode == op_decorate && word_count >= 4) {
            const uint32_t target = words[cursor + 1];
            const bool is_resource =
                std::find(resource_ids.begin(), resource_ids.end(), target) != resource_ids.end();

            if (is_resource && words[cursor + 2] == decoration_descriptor_set &&
                words[cursor + 3] == 1) {
                words[cursor + 3] = target == resource_ids[0] ? 0u : 3u;
                ++set_edits;
            } else if (is_resource && words[cursor + 2] == decoration_binding &&
                       words[cursor + 3] == 1) {
                words[cursor + 3] = 0;
                ++binding_edits;
            }
        }

        cursor += word_count;
    }

    if (set_edits != resource_ids.size() || binding_edits != resource_ids.size()) {
        throw std::runtime_error("unexpected auto-exposure descriptor ABI");
    }
}
} // namespace

Image QualityContext::prepareExactAutoExposureColor(VkCommandBuffer command_buffer,
                                                    const Image& source,
                                                    uint32_t active_width,
                                                    uint32_t active_height) {

    if (!source.image || !active_width || !active_height) {
        return {};
    }

    const uint32_t backing_width = source.resource_width ? source.resource_width : source.width;

    const uint32_t backing_height = source.resource_height ? source.resource_height : source.height;

    const int64_t padding_width =
        static_cast<int64_t>(backing_width) - static_cast<int64_t>(active_width);

    const int64_t padding_height =
        static_cast<int64_t>(backing_height) - static_cast<int64_t>(active_height);

    // Copy only small padding; a much larger image may be a fixed maximum-size allocation.
    const bool needs_exact_size_copy = padding_width >= 0 && padding_width <= 2 &&
                                       padding_height >= 0 && padding_height <= 2 &&
                                       (padding_width > 0 || padding_height > 0);

    if (!needs_exact_size_copy) {
        return {};
    }

    AutoExposureColorEntry* cached_color = nullptr;

    for (auto& entry : auto_exposure_color_cache) {
        if (entry.image.image && entry.image.width == active_width &&
            entry.image.height == active_height && entry.image.format == source.format) {
            cached_color = &entry;
            break;
        }
    }

    if (!cached_color) {
        // This age limit does not check GPU completion.
        if (auto_exposure_color_cache.size() >= kAutoExposureColorCacheSoftLimit) {
            for (auto cache_entry = auto_exposure_color_cache.begin();
                 cache_entry != auto_exposure_color_cache.end();) {

                const bool safe_to_retire =
                    frame_count >= cache_entry->last_used_frame &&
                    (frame_count - cache_entry->last_used_frame) >= frames.size();

                if (safe_to_retire) {
                    if (extensive_logging) {
                        log("[exposure-copy] retire " + std::to_string(cache_entry->image.width) +
                                "x" + std::to_string(cache_entry->image.height) + " age=" +
                                std::to_string(frame_count - cache_entry->last_used_frame),
                            fsr4log::Level::Debug);
                    }

                    if (cache_entry->image.image) {
                        destroyImage(device, cache_entry->image);
                    }

                    cache_entry = auto_exposure_color_cache.erase(cache_entry);

                    if (auto_exposure_color_cache.size() < kAutoExposureColorCacheSoftLimit) {
                        break;
                    }
                } else {
                    ++cache_entry;
                }
            }
        }

        AutoExposureColorEntry entry{};

        entry.image = createImage(physical_device,
                                  device,
                                  active_width,
                                  active_height,
                                  source.format,
                                  VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

        entry.initialized = false;
        entry.last_used_frame = frame_count;

        fsr4vk::setDebugName(dispatch,
                             device,
                             VK_OBJECT_TYPE_IMAGE,
                             (uint64_t)entry.image.image,
                             "FSR4_ExactAutoExposureColor");

        auto_exposure_color_cache.push_back(entry);
        cached_color = &auto_exposure_color_cache.back();

        if (extensive_logging) {
            log("[exposure-copy] created exact " + std::to_string(active_width) + "x" +
                    std::to_string(active_height) +
                    " format=" + fsr4log::vulkanFormat(source.format) +
                    " cache=" + std::to_string(auto_exposure_color_cache.size()),
                fsr4log::Level::Debug);
        }
    }

    cached_color->last_used_frame = frame_count;

    VkImageMemoryBarrier2 source_copy_barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};

    source_copy_barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    source_copy_barrier.srcAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    source_copy_barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    source_copy_barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    source_copy_barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    source_copy_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    source_copy_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    source_copy_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    source_copy_barrier.image = source.image;
    source_copy_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkImageMemoryBarrier2 destination_copy_barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};

    if (cached_color->initialized) {
        destination_copy_barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        destination_copy_barrier.srcAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        destination_copy_barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    } else {
        destination_copy_barrier.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
        destination_copy_barrier.srcAccessMask = VK_ACCESS_2_NONE;
        destination_copy_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    }

    destination_copy_barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    destination_copy_barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    destination_copy_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    destination_copy_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    destination_copy_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    destination_copy_barrier.image = cached_color->image.image;
    destination_copy_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    const VkImageMemoryBarrier2 pre_copy_barriers[] = {source_copy_barrier,
                                                       destination_copy_barrier};

    VkDependencyInfo pre_copy{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};

    pre_copy.imageMemoryBarrierCount = 2;
    pre_copy.pImageMemoryBarriers = pre_copy_barriers;

    dispatch.cmd_pipeline_barrier2(command_buffer, &pre_copy);

    VkImageCopy region{};

    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.srcOffset = {0, 0, 0};
    region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.dstOffset = {0, 0, 0};
    region.extent = {active_width, active_height, 1};

    vkCmdCopyImage(command_buffer,
                   source.image,
                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   cached_color->image.image,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   1,
                   &region);

    VkImageMemoryBarrier2 ready{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};

    ready.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    ready.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    ready.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    ready.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    ready.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    ready.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    ready.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ready.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ready.image = cached_color->image.image;
    ready.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkImageMemoryBarrier2 source_restore{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    source_restore.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    source_restore.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    source_restore.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    source_restore.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    source_restore.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    source_restore.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    source_restore.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    source_restore.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    source_restore.image = source.image;
    source_restore.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    const VkImageMemoryBarrier2 ready_barriers[] = {ready, source_restore};
    VkDependencyInfo ready_dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};

    ready_dep.imageMemoryBarrierCount = 2;
    ready_dep.pImageMemoryBarriers = ready_barriers;

    dispatch.cmd_pipeline_barrier2(command_buffer, &ready_dep);

    cached_color->initialized = true;

    if (extensive_logging && (frame_count <= 3 || frame_count % 300 == 0)) {
        log("[exposure-copy] backing=" + std::to_string(backing_width) + "x" +
                std::to_string(backing_height) + " active=" + std::to_string(active_width) + "x" +
                std::to_string(active_height) +
                " cache=" + std::to_string(auto_exposure_color_cache.size()),
            fsr4log::Level::Debug);
    }

    Image result = cached_color->image;

    result.width = active_width;
    result.height = active_height;

    return result;
}

void QualityContext::createAutoExposurePipeline(const fs::path& path) {
    auto code = readSpirv(path);
    remapAutoExposureDescriptors(code);
    auto_exposure_pipeline = createComputePipeline(code, false);
    log("[exposure] SPD auto-exposure pipeline ready");
}

void QualityContext::recordAutoExposure(VkCommandBuffer command_buffer,
                                        const Frame& frame,
                                        const OptimizedConstants& main_constants) {
    computeBarrier(dispatch.cmd_pipeline_barrier2, command_buffer);
    imageBarrier(dispatch.cmd_pipeline_barrier2,
                 command_buffer,
                 images[image_slot::exposure].image,
                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 VK_IMAGE_LAYOUT_GENERAL,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    VkDescriptorBufferBindingInfoEXT binding{VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT};
    binding.address = frame.descriptor_buffer_address;
    binding.usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT |
                    VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT;
    vkCmdBindDescriptorBuffersEXT(command_buffer, 1, &binding);
    constexpr std::array<uint32_t, kNumDescriptorSets> buffer_indices{};
    const std::array<VkDeviceSize, kNumDescriptorSets> offsets{{
        0 * descriptor_capacity,
        1 * descriptor_capacity,
        2 * descriptor_capacity,
        3 * descriptor_capacity,
        4 * descriptor_capacity,
        5 * descriptor_capacity,
    }};
    vkCmdSetDescriptorBufferOffsetsEXT(command_buffer,
                                       VK_PIPELINE_BIND_POINT_COMPUTE,
                                       pipeline_layout,
                                       0,
                                       kNumDescriptorSets,
                                       buffer_indices.data(),
                                       offsets.data());

    const VkDeviceAddress exposure_constants_address =
        frame.constants_address + constant_offset::auto_exposure;
    const std::array<uint32_t, 4> exposure_root_constants{{
        static_cast<uint32_t>(exposure_constants_address),
        static_cast<uint32_t>(exposure_constants_address >> 32),
        exposure_binding::counter,
        exposure_binding::color,
    }};
    trackPipelineUse(auto_exposure_pipeline);
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, auto_exposure_pipeline);
    vkCmdPushConstants(command_buffer,
                       pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0,
                       sizeof(exposure_root_constants),
                       exposure_root_constants.data());
    vkCmdDispatch(command_buffer,
                  fsr4::divideUp(main_constants.width_lr, 64),
                  fsr4::divideUp(main_constants.height_lr, 64),
                  1);

    imageBarrier(dispatch.cmd_pipeline_barrier2,
                 command_buffer,
                 images[image_slot::exposure].image,
                 VK_IMAGE_LAYOUT_GENERAL,
                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}

void QualityContext::writeAutoExposureConstants(Frame& frame,
                                                const OptimizedConstants& main_constants) {
    struct SpdConstants {
        uint32_t mip_count;
        uint32_t workgroup_count;
        uint32_t offset[2];
        float inverse_size[2];
        float pre_exposure;
        float padding;
    };

    static_assert(sizeof(SpdConstants) == 32);
    const uint32_t groups_x = fsr4::divideUp(main_constants.width_lr, 64);
    const uint32_t groups_y = fsr4::divideUp(main_constants.height_lr, 64);
    uint32_t mip_count = 0;

    for (uint32_t size = std::max(main_constants.width_lr, main_constants.height_lr); size > 1;
         size >>= 1) {
        ++mip_count;
    }

    mip_count = std::min(mip_count, 12u);
    const SpdConstants exposure_constants{
        mip_count,
        groups_x * groups_y,
        {0, 0},
        {1.0f / main_constants.width_lr, 1.0f / main_constants.height_lr},
        main_constants.pre_exposure,
        0.0f,
    };
    std::memcpy(static_cast<uint8_t*>(frame.constants.mapped) + constant_offset::auto_exposure,
                &exposure_constants,
                sizeof(exposure_constants));
}

} // namespace fsr4runtime

namespace fsr4runtime {

Image QualityContext::resampleDisplayMotionVectors(VkCommandBuffer command_buffer,
                                                   const Image& source,
                                                   uint32_t target_width,
                                                   uint32_t target_height) {
    if (!source.image || !source.width || !source.height ||
        (source.width == target_width && source.height == target_height)) {
        return source;
    }

    // Reserve the maximum render size so dynamic resolution only changes the blit area.
    const uint32_t capacity_width =
        max_render_width >= target_width ? max_render_width : target_width;
    const uint32_t capacity_height =
        max_render_height >= target_height ? max_render_height : target_height;

    const bool needs_motion_image_reallocation =
        !render_motion.image || render_motion.format != source.format ||
        render_motion.width < target_width || render_motion.height < target_height;

    if (needs_motion_image_reallocation) {
        // Reallocate only when the format changes or the required size exceeds capacity.
        if (render_motion.image) {
            vkDeviceWaitIdle(device);
            destroyImage(device, render_motion);
        }

        render_motion = createImage(physical_device,
                                    device,
                                    capacity_width,
                                    capacity_height,
                                    source.format,
                                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

        render_motion_initialized = false;

        fsr4vk::setDebugName(dispatch,
                             device,
                             VK_OBJECT_TYPE_IMAGE,
                             (uint64_t)render_motion.image,
                             "FSR4_RenderResolutionMotion");

        if (extensive_logging) {
            log("[motion] created render-resolution resample image size=" +
                std::to_string(capacity_width) + "x" + std::to_string(capacity_height) +
                " fmt=" + fsr4log::vulkanFormat(source.format));
        }
    }

    fsr4_411::recordMotionResample(dispatch.cmd_pipeline_barrier2,
                                   command_buffer,
                                   source,
                                   render_motion,
                                   target_width,
                                   target_height,
                                   render_motion_initialized);

    render_motion_initialized = true;

    // Keep the larger backing image but report only the active render size.
    Image result = render_motion;
    result.width = target_width;
    result.height = target_height;
    return result;
}

} // namespace fsr4runtime

namespace fsr4_411 {
void recordMotionResample(PFN_vkCmdPipelineBarrier2 pipeline_barrier2,
                          VkCommandBuffer command_buffer,
                          const Image& source,
                          const Image& destination,
                          uint32_t target_width,
                          uint32_t target_height,
                          bool destination_initialized) {
    // Restore the sampled layout after blitting the motion vectors.
    VkImageMemoryBarrier2 motion_copy_source_barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};

    motion_copy_source_barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    motion_copy_source_barrier.srcAccessMask =
        VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    motion_copy_source_barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    motion_copy_source_barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    motion_copy_source_barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    motion_copy_source_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    motion_copy_source_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    motion_copy_source_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    motion_copy_source_barrier.image = source.image;
    motion_copy_source_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkImageMemoryBarrier2 motion_copy_destination_barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};

    motion_copy_destination_barrier.srcStageMask =
        destination_initialized ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_2_NONE;

    motion_copy_destination_barrier.srcAccessMask =
        destination_initialized ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : VK_ACCESS_2_NONE;

    motion_copy_destination_barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    motion_copy_destination_barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    motion_copy_destination_barrier.oldLayout = destination_initialized
                                                    ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                                    : VK_IMAGE_LAYOUT_UNDEFINED;

    motion_copy_destination_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    motion_copy_destination_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    motion_copy_destination_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    motion_copy_destination_barrier.image = destination.image;
    motion_copy_destination_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    const VkImageMemoryBarrier2 pre_blit_barriers[] = {motion_copy_source_barrier,
                                                       motion_copy_destination_barrier};

    VkDependencyInfo copy_dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};

    copy_dependency.imageMemoryBarrierCount = 2;
    copy_dependency.pImageMemoryBarriers = pre_blit_barriers;

    pipeline_barrier2(command_buffer, &copy_dependency);

    // Prepass samples motion in render coordinates, so resample display-resolution vectors.
    VkImageBlit blit{};

    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};

    blit.srcOffsets[0] = {0, 0, 0};

    blit.srcOffsets[1] = {
        static_cast<int32_t>(source.width), static_cast<int32_t>(source.height), 1};

    blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};

    blit.dstOffsets[0] = {0, 0, 0};

    blit.dstOffsets[1] = {
        static_cast<int32_t>(target_width), static_cast<int32_t>(target_height), 1};

    vkCmdBlitImage(command_buffer,
                   source.image,
                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   destination.image,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   1,
                   &blit,
                   VK_FILTER_NEAREST);

    VkImageMemoryBarrier2 render_motion_ready{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};

    render_motion_ready.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    render_motion_ready.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    render_motion_ready.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    render_motion_ready.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    render_motion_ready.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    render_motion_ready.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    render_motion_ready.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    render_motion_ready.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    render_motion_ready.image = destination.image;
    render_motion_ready.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkImageMemoryBarrier2 source_ready = motion_copy_source_barrier;
    source_ready.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    source_ready.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    source_ready.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    source_ready.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    source_ready.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    source_ready.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    const VkImageMemoryBarrier2 post_blit_barriers[] = {source_ready, render_motion_ready};
    VkDependencyInfo sampling_dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};

    sampling_dependency.imageMemoryBarrierCount = 2;
    sampling_dependency.pImageMemoryBarriers = post_blit_barriers;

    pipeline_barrier2(command_buffer, &sampling_dependency);
}
} // namespace fsr4_411

namespace fsr4runtime {
void QualityContext::writeExposureDescriptors(Frame& frame, const Image& exposure_image) {
    // Keep temporal exposure between frames. Prepass reads game exposure at t6
    // and history at t17; postpass updates that history through u9.

    const VkImageView exposure_view =
        exposure_image.view ? exposure_image.view : images[image_slot::exposure].view;

    writeSampledImageDescriptor(frame,
                                descriptor_set::sampled_images,
                                6,
                                exposure_view,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    writeSampledImageDescriptor(
        frame, descriptor_set::sampled_images, 17, images[image_slot::exposure].view);
}

void QualityContext::writeStaticExposureDescriptor(Frame& frame) {
    writeStorageImageDescriptor(
        frame, descriptor_set::storage_images, 9, images[image_slot::exposure].view);
}

} // namespace fsr4runtime
