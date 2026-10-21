// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

#include "provider/provider.hpp"

namespace fsr4provider {
namespace {
void releaseCaptureStorage(ProviderContext& provider, ColorCaptureState& capture) noexcept {

    if (capture.mapped && capture.memory) {
        vkUnmapMemory(provider.device, capture.memory);
    }

    capture.mapped = nullptr;

    if (capture.buffer) {
        vkDestroyBuffer(provider.device, capture.buffer, nullptr);
    }

    if (capture.memory) {
        vkFreeMemory(provider.device, capture.memory, nullptr);
    }

    if (capture.event) {
        vkDestroyEvent(provider.device, capture.event, nullptr);
    }

    capture.buffer = VK_NULL_HANDLE;
    capture.memory = VK_NULL_HANDLE;
    capture.event = VK_NULL_HANDLE;
    capture.buffer_capacity = 0;
    capture.payload_size = 0;
    capture.pending = false;
}

std::uint32_t bytesPerPixel(VkFormat format) {
    switch (format) {
    case VK_FORMAT_R16G16B16A16_SFLOAT:
        return 8;
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
    case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32:
        return 4;
    default:
        return 0;
    }
}

const char* colorCaptureFormatName(VkFormat format) {
    switch (format) {
    case VK_FORMAT_R16G16B16A16_SFLOAT:
        return "VK_FORMAT_R16G16B16A16_SFLOAT";
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
        return "VK_FORMAT_B10G11R11_UFLOAT_PACK32";
    case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32:
        return "VK_FORMAT_E5B9G9R9_UFLOAT_PACK32";
    default:
        return "VK_FORMAT_UNDEFINED";
    }
}

std::filesystem::path colorCaptureOutputPath(const ColorCaptureState& capture) {
    std::filesystem::path path = capture.base_path;

    if (capture.triggered_mode) {
        path += "." + std::to_string(capture.pending_index);
    }

    switch (capture.format) {
    case VK_FORMAT_R16G16B16A16_SFLOAT:
        path += ".rgba16f";
        break;
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
        path += ".r11g11b10f";
        break;
    case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32:
        path += ".rgb9e5";
        break;
    default:
        path += ".raw";
        break;
    }

    return path;
}

bool colorCaptureMemoryType(ProviderContext& provider,
                            std::uint32_t type_bits,
                            std::uint32_t& index,
                            bool& coherent) {
    VkPhysicalDeviceMemoryProperties props{};
    vkGetPhysicalDeviceMemoryProperties(provider.physicalDevice, &props);

    for (std::uint32_t pass = 0; pass < 2; ++pass) {
        for (std::uint32_t i = 0; i < props.memoryTypeCount; ++i) {
            if (!(type_bits & (1u << i))) {
                continue;
            }

            const auto flags = props.memoryTypes[i].propertyFlags;

            if (!(flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
                continue;
            }

            const bool is_coherent = (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;

            if (pass == 0 && !is_coherent) {
                continue;
            }

            index = i;
            coherent = is_coherent;
            return true;
        }
    }

    return false;
}

bool ensureCaptureStorage(ProviderContext& provider,
                          ColorCaptureState& capture,
                          VkDeviceSize payload_size) {

    if (capture.buffer && capture.buffer_capacity >= payload_size && capture.event) {
        capture.payload_size = payload_size;
        return true;
    }

    if (capture.pending) {
        return false;
    }

    releaseCaptureStorage(provider, capture);

    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = payload_size;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(provider.device, &bci, nullptr, &capture.buffer) != VK_SUCCESS) {
        return false;
    }

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(provider.device, capture.buffer, &req);

    std::uint32_t memory_type = 0;

    if (!colorCaptureMemoryType(provider, req.memoryTypeBits, memory_type, capture.coherent)) {
        releaseCaptureStorage(provider, capture);
        return false;
    }

    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = memory_type;

    if (vkAllocateMemory(provider.device, &mai, nullptr, &capture.memory) != VK_SUCCESS) {
        releaseCaptureStorage(provider, capture);
        return false;
    }

    if (vkBindBufferMemory(provider.device, capture.buffer, capture.memory, 0) != VK_SUCCESS) {
        releaseCaptureStorage(provider, capture);
        return false;
    }

    if (vkMapMemory(provider.device, capture.memory, 0, VK_WHOLE_SIZE, 0, &capture.mapped) !=
        VK_SUCCESS) {
        releaseCaptureStorage(provider, capture);
        return false;
    }

    VkEventCreateInfo eci{VK_STRUCTURE_TYPE_EVENT_CREATE_INFO};

    if (vkCreateEvent(provider.device, &eci, nullptr, &capture.event) != VK_SUCCESS) {
        releaseCaptureStorage(provider, capture);
        return false;
    }

    capture.buffer_capacity = payload_size;
    capture.payload_size = payload_size;

    return true;
}

void writeCompletedColorCapture(ProviderContext& provider) noexcept {
    auto& capture = provider.color_capture;

    if (!capture.pending || !capture.mapped) {
        return;
    }

    try {
        if (!capture.coherent) {
            VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
            range.memory = capture.memory;
            range.offset = 0;
            range.size = VK_WHOLE_SIZE;

            if (vkInvalidateMappedMemoryRanges(provider.device, 1, &range) != VK_SUCCESS) {
                throw std::runtime_error("vkInvalidateMappedMemoryRanges failed");
            }
        }

        const auto output = colorCaptureOutputPath(capture);

        if (!output.parent_path().empty()) {
            std::error_code ec;
            std::filesystem::create_directories(output.parent_path(), ec);
        }

        std::ofstream raw(output, std::ios::binary | std::ios::trunc);

        if (!raw) {
            throw std::runtime_error("could not open color dump output");
        }

        raw.write(static_cast<const char*>(capture.mapped),
                  static_cast<std::streamsize>(capture.payload_size));

        if (!raw) {
            throw std::runtime_error("failed writing color dump");
        }

        raw.close();

        std::ofstream meta(output.string() + ".txt", std::ios::trunc);

        if (meta) {
            meta << "width=" << capture.width << "\n"
                 << "height=" << capture.height << "\n"
                 << "format=" << colorCaptureFormatName(capture.format) << "\n"
                 << "vk_format=" << static_cast<std::uint32_t>(capture.format) << "\n"
                 << "bytes_per_pixel=" << bytesPerPixel(capture.format) << "\n"
                 << "row_pitch=" << (std::uint64_t(capture.width) * bytesPerPixel(capture.format))
                 << "\n"
                 << "byte_size=" << capture.payload_size << "\n"
                 << "dispatch=" << capture.pending_dispatch << "\n";
        }

        diagnostic(provider,
                   "[FSR4-color-dump] wrote " + output.string() + " (" +
                       std::to_string(capture.width) + "x" + std::to_string(capture.height) + " " +
                       colorCaptureFormatName(capture.format) + ")");
    } catch (const std::exception& e) {
        diagnostic(provider,
                   std::string("[FSR4-color-dump] write failed: ") + e.what(),
                   fsr4log::Level::Error);
    }

    capture.pending = false;

    if (capture.event) {
        vkResetEvent(provider.device, capture.event);
    }
}

bool colorCaptureRequested(ProviderContext& provider) noexcept {
    auto& capture = provider.color_capture;

    if (!capture.enabled || capture.pending) {
        return false;
    }

    if (!capture.triggered_mode) {
        if (!capture.first_capture_armed) {
            return false;
        }

        capture.first_capture_armed = false;
        return true;
    }

    std::error_code ec;

    if (!std::filesystem::exists(capture.trigger_path, ec) || ec) {
        return false;
    }

    std::filesystem::remove(capture.trigger_path, ec);
    return true;
}

} // namespace

void pollColorCapture(ProviderContext& provider) noexcept {
    auto& capture = provider.color_capture;

    if (!capture.enabled || !capture.pending || !capture.event) {
        return;
    }

    const VkResult status = vkGetEventStatus(provider.device, capture.event);

    if (status == VK_EVENT_SET) {
        writeCompletedColorCapture(provider);
    }
}

void recordColorCapture(ProviderContext& provider,
                        VkCommandBuffer command_buffer,
                        const fsr4_411::Image& color,
                        std::uint64_t dispatch_index) noexcept {
    auto& capture = provider.color_capture;

    if (!colorCaptureRequested(provider)) {
        return;
    }

    try {
        const std::uint32_t bpp = bytesPerPixel(color.format);

        if (!bpp) {
            diagnostic(provider,
                       "[FSR4-color-dump] unsupported color format " +
                           std::to_string(static_cast<std::uint32_t>(color.format)));
            return;
        }

        const VkDeviceSize payload = VkDeviceSize(color.width) * color.height * bpp;

        if (!ensureCaptureStorage(provider, capture, payload)) {
            diagnostic(provider,
                       "[FSR4-color-dump] could not allocate readback buffer",
                       fsr4log::Level::Error);
            return;
        }

        capture.width = color.width;
        capture.height = color.height;
        capture.format = color.format;
        capture.payload_size = payload;
        capture.pending_index = capture.next_index++;
        capture.pending_dispatch = dispatch_index;

        if (vkResetEvent(provider.device, capture.event) != VK_SUCCESS) {
            diagnostic(provider, "[FSR4-color-dump] vkResetEvent failed", fsr4log::Level::Error);
            return;
        }

        VkImageMemoryBarrier image_barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        image_barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        image_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        // Restore the sampled layout expected by the caller.
        image_barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        image_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        image_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        image_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        image_barrier.image = color.image;
        image_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(command_buffer,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0,
                             0,
                             nullptr,
                             0,
                             nullptr,
                             1,
                             &image_barrier);

        VkBufferImageCopy copy{};
        copy.bufferOffset = 0;
        copy.bufferRowLength = 0;
        copy.bufferImageHeight = 0;
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageOffset = {0, 0, 0};
        copy.imageExtent = {color.width, color.height, 1};
        vkCmdCopyImageToBuffer(command_buffer,
                               color.image,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               capture.buffer,
                               1,
                               &copy);

        VkBufferMemoryBarrier buffer_barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        buffer_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        buffer_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        buffer_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        buffer_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        buffer_barrier.buffer = capture.buffer;
        buffer_barrier.offset = 0;
        buffer_barrier.size = payload;
        VkImageMemoryBarrier image_restore{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        image_restore.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        image_restore.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        image_restore.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        image_restore.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        image_restore.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        image_restore.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        image_restore.image = color.image;
        image_restore.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(command_buffer,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0,
                             0,
                             nullptr,
                             1,
                             &buffer_barrier,
                             1,
                             &image_restore);
        vkCmdSetEvent(command_buffer, capture.event, VK_PIPELINE_STAGE_TRANSFER_BIT);
        capture.pending = true;
        diagnostic(provider,
                   "[FSR4-color-dump] capture queued at dispatch " +
                       std::to_string(dispatch_index));
    } catch (const std::exception& e) {
        diagnostic(provider,
                   std::string("[FSR4-color-dump] capture failed: ") + e.what(),
                   fsr4log::Level::Error);
    }
}

void finishColorCapture(ProviderContext& provider) noexcept {
    auto& capture = provider.color_capture;

    if (capture.pending) {
        if (vkDeviceWaitIdle(provider.device) == VK_SUCCESS) {
            writeCompletedColorCapture(provider);
        }
    }

    releaseCaptureStorage(provider, capture);
}
} // namespace fsr4provider

namespace fsr4provider {
namespace {
uint32_t capturePixelBytes(VkFormat format) {
    switch (format) {
    case VK_FORMAT_R8_UNORM:
    case VK_FORMAT_R8_UINT:
        return 1;
    case VK_FORMAT_R8G8_UNORM:
    case VK_FORMAT_R8G8_UINT:
    case VK_FORMAT_R16_SFLOAT:
    case VK_FORMAT_D16_UNORM:
    case VK_FORMAT_R16_UNORM:
    case VK_FORMAT_R16_SNORM:
    case VK_FORMAT_R16_UINT:
        return 2;
    case VK_FORMAT_R16G16_SFLOAT:
    case VK_FORMAT_R16G16_SNORM:
    case VK_FORMAT_R16G16_UNORM:
    case VK_FORMAT_R32_SFLOAT:
    case VK_FORMAT_R32_UINT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_R8G8B8A8_SNORM:
    case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_B8G8R8A8_UNORM:
    case VK_FORMAT_B8G8R8A8_SRGB:
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
    case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32:
        return 4;
    case VK_FORMAT_R16G16B16A16_SFLOAT:
    case VK_FORMAT_R16G16B16A16_UNORM:
    case VK_FORMAT_R32G32_SFLOAT:
        return 8;
    case VK_FORMAT_R32G32B32A32_SFLOAT:
        return 16;
    default:
        throw std::runtime_error("unsupported frame capture format " + std::to_string(format));
    }
}

void publishSequence(ProviderContext& provider) {
    auto& c = provider.frame_capture;
    std::ofstream out(c.directory / "sequence.json", std::ios::trunc);
    if (!out) {
        throw std::runtime_error("cannot write sequence.json");
    }
    out << "{\n  \"schema\": \"fsr4vk-sequence-v1\",\n  \"complete\": "
        << (!c.failed && c.completed == c.count ? "true" : "false")
        << ",\n  \"context_flags\": " << provider.flags << ",\n  \"max_render_size\": ["
        << provider.maxRenderSize.width << ',' << provider.maxRenderSize.height
        << "],\n  \"max_output_size\": [" << provider.maxUpscaleSize.width << ','
        << provider.maxUpscaleSize.height << "],\n  \"frames\": [";
    bool comma = false;
    for (uint32_t i = 0; i < c.written.size(); ++i) {
        if (!c.written[i]) {
            continue;
        }
        if (comma) {
            out << ',';
        }
        out << "\n    \"frame-" << i << "/frame.json\"";
        comma = true;
    }
    out << "\n  ]\n}\n";
    if (!out) {
        throw std::runtime_error("failed writing sequence.json");
    }
}

void copyCapturedImage(ProviderContext& provider,
                       VkCommandBuffer cmd,
                       const CapturedImage& item,
                       VkBuffer buffer,
                       bool output) {
    const auto& image = item.image;
    const VkImageLayout layout =
        output ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.oldLayout = layout;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image.image;
    auto aspect = image.aspect;
    if ((image.format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
         image.format == VK_FORMAT_D24_UNORM_S8_UINT)) {
        aspect |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }
    barrier.subresourceRange = {aspect, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0,
                         0,
                         nullptr,
                         0,
                         nullptr,
                         1,
                         &barrier);
    VkBufferImageCopy copy{};
    copy.bufferOffset = item.offset;
    copy.imageSubresource = {image.aspect, 0, 0, 1};
    copy.imageExtent = {image.width, image.height, 1};
    vkCmdCopyImageToBuffer(
        cmd, image.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &copy);
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.newLayout = layout;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         0,
                         0,
                         nullptr,
                         0,
                         nullptr,
                         1,
                         &barrier);
}

void completeCapturedFrame(ProviderContext& provider, FrameCaptureSlot& slot) {
    auto& c = provider.frame_capture;
    auto& storage = slot.storage;
    if (!storage.coherent) {
        VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        range.memory = storage.memory;
        range.size = VK_WHOLE_SIZE;
        fsr4_411::check(vkInvalidateMappedMemoryRanges(provider.device, 1, &range),
                        "capture invalidate");
    }
    const auto dir = c.directory / ("frame-" + std::to_string(slot.index));
    std::filesystem::create_directories(dir);
    for (const auto& image : slot.images) {
        std::ofstream raw(dir / (image.name + ".bin"), std::ios::binary | std::ios::trunc);
        raw.write(static_cast<const char*>(storage.mapped) + image.offset, image.size);
        if (!raw) {
            throw std::runtime_error("failed writing captured " + image.name);
        }
    }
    std::ofstream meta(dir / "frame.json", std::ios::trunc);
    meta << slot.metadata;
    meta.close();
    if (!meta) {
        throw std::runtime_error("failed writing frame.json");
    }
    storage.pending = false;
    fsr4_411::check(vkResetEvent(provider.device, storage.event), "capture reset event");
    c.written[slot.index] = true;
    ++c.completed;
    publishSequence(provider);
    if (c.completed == c.count) {
        c.active = false;
        diagnostic(provider,
                   "[frame-capture] complete: " + c.directory.string() + "/sequence.json");
    }
}
}

void configureFrameCapture(ProviderContext& provider) {
    auto& c = provider.frame_capture;
    const char* base = std::getenv("FSR4_VK_DUMP_SEQUENCE");
    if (!base || !*base) {
        return;
    }
    try {
        c.base = base;
        if (const char* count = std::getenv("FSR4_VK_DUMP_SEQUENCE_FRAMES")) {
            char* end = nullptr;
            const auto n = std::strtoul(count, &end, 10);
            if (end == count || *end || n < 1 || n > 256) {
                diagnostic(
                    provider, "[frame-capture] frame count must be 1..256", fsr4log::Level::Error);
                return;
            }
            c.count = static_cast<uint32_t>(n);
        }
        if (const char* trigger = std::getenv("FSR4_VK_DUMP_SEQUENCE_TRIGGER")) {
            c.trigger = trigger;
        }
        c.enabled = true;
        diagnostic(provider,
                   "[frame-capture] enabled: " + c.base.string() +
                       " frames=" + std::to_string(c.count));
    } catch (const std::exception& e) {
        c.enabled = false;
        diagnostic(provider,
                   std::string("[frame-capture] configuration failed: ") + e.what(),
                   fsr4log::Level::Error);
    }
}

void pollFrameCapture(ProviderContext& provider) noexcept {
    auto& c = provider.frame_capture;
    if (!c.enabled || c.failed) {
        return;
    }
    try {
        for (auto& slot : c.slots) {
            if (slot.storage.pending &&
                vkGetEventStatus(provider.device, slot.storage.event) == VK_EVENT_SET) {
                completeCapturedFrame(provider, slot);
            }
        }
    } catch (const std::exception& e) {
        c.failed = true;
        c.active = false;
        diagnostic(provider,
                   std::string("[frame-capture] write failed: ") + e.what(),
                   fsr4log::Level::Error);
    }
}

FrameCaptureSlot* beginFrameCapture(ProviderContext& provider,
                                    const ffxDispatchDescUpscale& d,
                                    const std::array<fsr4_411::Image, 6>& images,
                                    uint64_t dispatch_index) noexcept {
    auto& c = provider.frame_capture;
    if (!c.enabled || c.failed) {
        return nullptr;
    }
    try {
        if (!c.active) {
            if (c.queued != c.completed) {
                return nullptr;
            }
            if (c.trigger.empty()) {
                if (c.started) {
                    return nullptr;
                }
            } else {
                if (!std::filesystem::exists(c.trigger)) {
                    return nullptr;
                }
                std::filesystem::remove(c.trigger);
            }
            std::filesystem::create_directories(c.base);
            bool created = false;
            for (uint32_t i = 0; i < 10000; ++i) {
                c.directory = c.base / ("capture-" + std::to_string(i));
                if (std::filesystem::create_directory(c.directory)) {
                    created = true;
                    break;
                }
            }
            if (!created) {
                throw std::runtime_error("no unused capture directory");
            }
            c.queued = c.completed = 0;
            c.written.assign(c.count, false);
            c.started = c.active = true;
            publishSequence(provider);
        }
        if (c.queued >= c.count) {
            return nullptr;
        }
        if (c.queued && dispatch_index != c.last_dispatch + 1) {
            throw std::runtime_error("nonconsecutive dispatches; sequence aborted");
        }
        FrameCaptureSlot* slot = nullptr;
        for (auto& candidate : c.slots) {
            if (!candidate.storage.pending) {
                slot = &candidate;
                break;
            }
        }
        if (!slot) {
            throw std::runtime_error(
                "readback ring full; sequence aborted instead of skipping frames");
        }
        slot->images.clear();
        constexpr const char* names[] = {"color", "depth", "motion", "exposure", "mask", "output"};
        VkDeviceSize total = 0;
        for (uint32_t i = 0; i < images.size(); ++i) {
            auto image = images[i];
            const FfxApiResource* raw[] = {&d.color,
                                           &d.depth,
                                           &d.motionVectors,
                                           &d.exposure,
                                           d.reactive.resource ? &d.reactive
                                                               : &d.transparencyAndComposition,
                                           &d.output};
            if (raw[i]->description.width) {
                image.resource_width = raw[i]->description.width;
            }
            if (raw[i]->description.height) {
                image.resource_height = raw[i]->description.height;
            }
            if (!image.image) {
                continue;
            }
            if (image.resource_width) {
                image.width = image.resource_width;
            }
            if (image.resource_height) {
                image.height = image.resource_height;
            }
            const VkDeviceSize size =
                VkDeviceSize(image.width) * image.height * capturePixelBytes(image.format);
            total = (total + 15) & ~VkDeviceSize(15);
            slot->images.push_back({names[i], image, total, size});
            total += size;
        }
        if (!ensureCaptureStorage(provider, slot->storage, total)) {
            throw std::runtime_error("readback allocation failed");
        }
        slot->index = c.queued;
        std::ostringstream j;
        j << std::setprecision(std::numeric_limits<float>::max_digits10)
          << "{\n\"schema\":\"fsr4vk-frame-v1\",\"dispatch\":" << dispatch_index
          << ",\"dispatch_flags\":" << d.flags << ",\"render_size\":[" << images[0].width << ','
          << images[0].height << "],\"output_size\":[" << images[5].width << ',' << images[5].height
          << "],\"jitter\":[" << d.jitterOffset.x << ',' << d.jitterOffset.y
          << "],\"motion_vector_scale\":[" << d.motionVectorScale.x << ',' << d.motionVectorScale.y
          << "],\"pre_exposure\":" << d.preExposure << ",\"frame_time_ms\":" << d.frameTimeDelta
          << ",\"reset\":" << (d.reset ? "true" : "false") << ",\"camera_near\":" << d.cameraNear
          << ",\"camera_far\":" << d.cameraFar << ",\"camera_fov\":" << d.cameraFovAngleVertical
          << ",\"view_space_to_meters\":" << d.viewSpaceToMetersFactor
          << ",\"sharpening\":" << (d.enableSharpening ? "true" : "false")
          << ",\"sharpness\":" << d.sharpness
          << ",\"reactive\":" << (d.reactive.resource ? "true" : "false")
          << ",\"transparency\":" << (d.transparencyAndComposition.resource ? "true" : "false")
          << ",\"reference_kind\":\"captured-output\",\"images\":{\n";
        for (size_t i = 0; i < slot->images.size(); ++i) {
            const auto& item = slot->images[i];
            if (i) {
                j << ",\n";
            }
            j << '"' << item.name << "\":{\"file\":\"" << item.name
              << ".bin\",\"vk_format\":" << static_cast<uint32_t>(item.image.format)
              << ",\"width\":" << item.image.width << ",\"height\":" << item.image.height
              << ",\"aspect\":" << item.image.aspect
              << ",\"row_pitch\":" << item.image.width * capturePixelBytes(item.image.format)
              << '}';
        }
        j << "\n}}\n";
        slot->metadata = j.str();
        for (const auto& item : slot->images) {
            if (item.name != "output") {
                copyCapturedImage(provider,
                                  static_cast<VkCommandBuffer>(d.commandList),
                                  item,
                                  slot->storage.buffer,
                                  false);
            }
        }
        c.last_dispatch = dispatch_index;
        return slot;
    } catch (const std::exception& e) {
        c.failed = true;
        c.active = false;
        diagnostic(
            provider, std::string("[frame-capture] aborted: ") + e.what(), fsr4log::Level::Error);
        return nullptr;
    }
}

void endFrameCapture(ProviderContext& provider,
                     FrameCaptureSlot* slot,
                     VkCommandBuffer cmd) noexcept {
    if (!slot) {
        return;
    }
    for (const auto& item : slot->images) {
        if (item.name == "output") {
            copyCapturedImage(provider, cmd, item, slot->storage.buffer, true);
        }
    }
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = slot->storage.buffer;
    barrier.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT,
                         0,
                         0,
                         nullptr,
                         1,
                         &barrier,
                         0,
                         nullptr);
    vkCmdSetEvent(cmd, slot->storage.event, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    slot->storage.pending = true;
    ++provider.frame_capture.queued;
}

void finishFrameCapture(ProviderContext& provider) noexcept {
    auto& c = provider.frame_capture;
    if (!c.enabled) {
        return;
    }
    if (vkDeviceWaitIdle(provider.device) == VK_SUCCESS) {
        pollFrameCapture(provider);
    }
    if (c.completed != c.count && c.started) {
        diagnostic(provider,
                   "[frame-capture] incomplete sequence: " + c.directory.string(),
                   fsr4log::Level::Warning);
    }
    for (auto& slot : c.slots) {
        releaseCaptureStorage(provider, slot.storage);
    }
}
}
