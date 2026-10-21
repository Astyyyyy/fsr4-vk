// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

#include "runtime/context.hpp"

namespace fsr4runtime {

uint32_t QualityContext::bytesPerPixel(VkFormat format) {
    switch (format) {
    case VK_FORMAT_R8_UNORM:
    case VK_FORMAT_R8_UINT:
        return 1;

    case VK_FORMAT_R8G8_UNORM:
    case VK_FORMAT_R16_UNORM:
    case VK_FORMAT_R16_SFLOAT:
        return 2;

    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_R8G8B8A8_SNORM:
    case VK_FORMAT_B8G8R8A8_UNORM:
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
    case VK_FORMAT_R16G16_SFLOAT:
    case VK_FORMAT_R32_SFLOAT:
    case VK_FORMAT_R32_UINT:
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
        return 4;

    case VK_FORMAT_R16G16B16A16_SFLOAT:
    case VK_FORMAT_R32G32_SFLOAT:
        return 8;

    case VK_FORMAT_R32G32B32A32_SFLOAT:
        return 16;

    default:
        return 0;
    }
}

const char* QualityContext::liveCapturePointName(CapturePoint point) {
    switch (point) {
    case CapturePoint::BeforePrepass:
        return "prepass_in";

    case CapturePoint::BeforePostpass:
        return "postpass_in";

    case CapturePoint::AfterPostpass:
        return "postpass_out";
    }

    return "unknown";
}

void QualityContext::liveCaptureWrite(const fs::path& path,
                                      const void* data,
                                      size_t bytes) noexcept {
    try {
        fs::create_directories(path.parent_path());

        std::ofstream out(path, std::ios::binary);

        if (!out) {
            log("[411-capture] cannot open " + path.string(), fsr4log::Level::Error);
            return;
        }

        out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));

        if (!out) {
            log("[411-capture] short write " + path.string(), fsr4log::Level::Error);
        }

    } catch (const std::exception& e) {
        log(std::string("[411-capture] write failed: ") + e.what(), fsr4log::Level::Error);
    }
}

void QualityContext::liveCaptureImageMeta(const fs::path& path,
                                          const char* role,
                                          const Image& image,
                                          VkImageLayout layout,
                                          uint32_t bpp,
                                          VkDeviceSize bytes) noexcept {
    try {
        fs::create_directories(path.parent_path());

        std::ofstream out(path);

        if (!out) {
            return;
        }

        out << "{\n"
            << "  \"role\": \"" << role << "\",\n"
            << "  \"format\": " << static_cast<uint32_t>(image.format) << ",\n"
            << "  \"width\": " << image.width << ",\n"
            << "  \"height\": " << image.height << ",\n"
            << "  \"aspect\": " << image.aspect << ",\n"
            << "  \"layout\": " << static_cast<uint32_t>(layout) << ",\n"
            << "  \"bytes_per_pixel\": " << bpp << ",\n"
            << "  \"bytes\": " << bytes << "\n"
            << "}\n";

    } catch (...) {
    }
}

void QualityContext::liveCaptureManifest(const Image& color,
                                         const Image& depth,
                                         const Image& motion,
                                         const Image& exposure,
                                         const Image& output,
                                         const OptimizedConstants& constants,
                                         bool run_auto_exposure,
                                         bool use_staging_output) noexcept {
    try {
        fs::create_directories(live_capture_frame_dir);

        std::ofstream out(live_capture_frame_dir / "manifest.json");

        if (!out) {
            return;
        }

        auto emit_image = [&](const char* name, const Image& image, bool comma) {
            out << "    \"" << name << "\": {"
                << "\"handle\": \"0x" << std::hex << static_cast<uint64_t>((uintptr_t)image.image)
                << std::dec << "\", \"format\": " << static_cast<uint32_t>(image.format)
                << ", \"width\": " << image.width << ", \"height\": " << image.height
                << ", \"aspect\": " << image.aspect << "}" << (comma ? "," : "") << "\n";
        };

        out << "{\n"
            << "  \"frame\": " << frame_count << ",\n"
            << "  \"use_wmma\": " << (use_wmma ? "true" : "false") << ",\n"
            << "  \"run_auto_exposure\": " << (run_auto_exposure ? "true" : "false") << ",\n"
            << "  \"use_staging_output\": " << (use_staging_output ? "true" : "false") << ",\n"
            << "  \"constants\": {"
            << "\"width\": " << constants.width << ", \"height\": " << constants.height
            << ", \"width_lr\": " << constants.width_lr
            << ", \"height_lr\": " << constants.height_lr << ", \"reset\": " << constants.reset
            << ", \"pre_exposure\": " << constants.pre_exposure
            << ", \"previous_pre_exposure\": " << constants.previous_pre_exposure << "},\n"
            << "  \"external_images\": {\n";

        emit_image("color", color, true);
        emit_image("depth", depth, true);
        emit_image("motion", motion, true);
        emit_image("exposure", exposure, true);
        emit_image("output", output, false);

        out << "  }\n"
            << "}\n";

    } catch (...) {
    }
}

void QualityContext::liveCaptureImage(VkCommandBuffer command_buffer,
                                      CapturePoint point,
                                      const char* role,
                                      const Image& image,
                                      VkImageLayout layout) {
    if (!image.image || !image.width || !image.height) {
        return;
    }

    const uint32_t bpp = bytesPerPixel(image.format);

    if (!bpp) {
        log("[411-capture] unsupported image format role=" + std::string(role) +
                " fmt=" + fsr4log::vulkanFormat(image.format),
            fsr4log::Level::Warning);

        return;
    }

    const VkDeviceSize bytes = VkDeviceSize(image.width) * image.height * bpp;

    LiveCaptureItem item{};

    item.relative_path = std::string(liveCapturePointName(point)) + "/" + role + ".bin";

    item.bytes = bytes;

    item.staging =
        createBuffer(physical_device,
                     device,
                     bytes,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    imageBarrier(dispatch.cmd_pipeline_barrier2,
                 command_buffer,
                 image.image,
                 layout,
                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                 VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_COPY_BIT,
                 VK_ACCESS_2_TRANSFER_READ_BIT,
                 image.aspect);

    VkBufferImageCopy copy{};

    copy.imageSubresource = {
        image.aspect,
        0,
        0,
        1,
    };

    copy.imageExtent = {
        image.width,
        image.height,
        1,
    };

    vkCmdCopyImageToBuffer(command_buffer,
                           image.image,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           item.staging.buffer,
                           1,
                           &copy);

    imageBarrier(dispatch.cmd_pipeline_barrier2,
                 command_buffer,
                 image.image,
                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 layout,
                 VK_PIPELINE_STAGE_2_COPY_BIT,
                 VK_ACCESS_2_TRANSFER_READ_BIT,
                 VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                 VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                 image.aspect);

    liveCaptureImageMeta(
        live_capture_frame_dir / (item.relative_path + ".json"), role, image, layout, bpp, bytes);

    live_capture_items.push_back(std::move(item));
}

void QualityContext::liveCaptureBuffer(VkCommandBuffer command_buffer,
                                       CapturePoint point,
                                       const char* role,
                                       const Buffer& buffer) {
    if (!buffer.buffer || !buffer.size) {
        return;
    }

    LiveCaptureItem item{};

    item.relative_path = std::string(liveCapturePointName(point)) + "/" + role + ".bin";

    item.bytes = buffer.size;

    item.staging =
        createBuffer(physical_device,
                     device,
                     buffer.size,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VkBufferMemoryBarrier2 before{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};

    before.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

    before.srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;

    before.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;

    before.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;

    before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

    before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

    before.buffer = buffer.buffer;
    before.offset = 0;
    before.size = VK_WHOLE_SIZE;

    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};

    dep.bufferMemoryBarrierCount = 1;
    dep.pBufferMemoryBarriers = &before;

    dispatch.cmd_pipeline_barrier2(command_buffer, &dep);

    VkBufferCopy copy{
        0,
        0,
        buffer.size,
    };

    vkCmdCopyBuffer(command_buffer, buffer.buffer, item.staging.buffer, 1, &copy);

    VkBufferMemoryBarrier2 after = before;

    after.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;

    after.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;

    after.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

    after.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;

    dep.pBufferMemoryBarriers = &after;

    dispatch.cmd_pipeline_barrier2(command_buffer, &dep);

    live_capture_items.push_back(std::move(item));
}

void QualityContext::liveCaptureFlush() noexcept {
    if (!live_capture_pending) {
        return;
    }

    try {
        // Wait for the recorded copies to execute, even if submission was delayed.
        const VkResult status = vkGetEventStatus(device, live_capture_completion);
        if (status != VK_EVENT_SET) {
            if (status != VK_EVENT_RESET) {
                log("[411-capture] completion check failed result=" + std::to_string(status),
                    fsr4log::Level::Error);
            }
            return;
        }

        for (auto& item : live_capture_items) {
            if (item.staging.mapped && item.bytes) {
                liveCaptureWrite(live_capture_frame_dir / item.relative_path,
                                 item.staging.mapped,
                                 static_cast<size_t>(item.bytes));
            }

            destroyBuffer(device, item.staging);
        }

        const size_t count = live_capture_items.size();

        live_capture_items.clear();
        live_capture_pending = false;

        log("[411-capture] COMPLETE frame=" + std::to_string(live_capture_target_frame) +
            " dir=" + live_capture_frame_dir.string() + " files=" + std::to_string(count));

    } catch (const std::exception& e) {
        log(std::string("[411-capture] flush failed: ") + e.what(), fsr4log::Level::Error);
    }
}

void QualityContext::captureAtStage(CapturePoint point, VkCommandBuffer command_buffer) {
    if (!live_capture_recording) {
        return;
    }

    switch (point) {
    case CapturePoint::BeforePrepass:
        liveCaptureImage(command_buffer,
                         point,
                         "history",
                         images[image_slot::history],
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        liveCaptureImage(command_buffer,
                         point,
                         "recurrent",
                         images[image_slot::recurrent],
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        liveCaptureImage(command_buffer,
                         point,
                         "reprojected",
                         images[image_slot::reprojected],
                         VK_IMAGE_LAYOUT_GENERAL);

        if (live_capture_run_auto_exposure) {
            liveCaptureImage(command_buffer,
                             point,
                             "exposure_internal",
                             images[image_slot::exposure],
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }

        liveCaptureBuffer(command_buffer, point, "scratch", scratch_buffer);

        break;

    case CapturePoint::BeforePostpass:
        liveCaptureImage(command_buffer,
                         point,
                         "reprojected",
                         images[image_slot::reprojected],
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        liveCaptureImage(command_buffer,
                         point,
                         "recurrent",
                         images[image_slot::recurrent],
                         VK_IMAGE_LAYOUT_GENERAL);

        liveCaptureImage(command_buffer,
                         point,
                         "history_before_write",
                         images[image_slot::history],
                         VK_IMAGE_LAYOUT_GENERAL);

        if (live_capture_run_auto_exposure) {
            liveCaptureImage(command_buffer,
                             point,
                             "exposure_internal",
                             images[image_slot::exposure],
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }

        liveCaptureBuffer(command_buffer, point, "scratch", scratch_buffer);

        break;

    case CapturePoint::AfterPostpass:
        liveCaptureImage(
            command_buffer, point, "history", images[image_slot::history], VK_IMAGE_LAYOUT_GENERAL);

        liveCaptureImage(command_buffer,
                         point,
                         "recurrent",
                         images[image_slot::recurrent],
                         VK_IMAGE_LAYOUT_GENERAL);

        liveCaptureImage(command_buffer,
                         point,
                         "scratch0",
                         images[image_slot::wmma_scratch0],
                         VK_IMAGE_LAYOUT_GENERAL);

        liveCaptureImage(command_buffer,
                         point,
                         "scratch1",
                         images[image_slot::wmma_scratch1],
                         VK_IMAGE_LAYOUT_GENERAL);

        if (live_capture_use_staging_output && staging_output.image) {
            liveCaptureImage(
                command_buffer, point, "output_staging", staging_output, VK_IMAGE_LAYOUT_GENERAL);
        }

        break;
    }
}

void QualityContext::liveCaptureHook(void* opaque,
                                     CapturePoint point,
                                     VkCommandBuffer command_buffer) {
    auto* self = static_cast<QualityContext*>(opaque);

    try {
        self->captureAtStage(point, command_buffer);

    } catch (const std::exception& e) {
        log(std::string("[411-capture] hook failed: ") + e.what(), fsr4log::Level::Error);

        self->live_capture_recording = false;
    }
}

void QualityContext::configureLiveCapture() {
    if (const char* dir = std::getenv("FSR4_VK_CAPTURE_DIR")) {
        live_capture_root = dir;
    } else {
        live_capture_root = "/tmp/fsr4vk-live-capture";
    }

    if (const char* value = std::getenv("FSR4_VK_CAPTURE_FRAME")) {
        char* end = nullptr;

        const unsigned long long parsed = std::strtoull(value, &end, 10);

        if (end != value && *end == '\0' && parsed > 0) {
            live_capture_enabled = true;

            live_capture_target_frame = static_cast<uint64_t>(parsed);

            log("[411-capture] armed frame=" + std::to_string(live_capture_target_frame) +
                " dir=" + live_capture_root.string());
        }
    }

    if (const char* value = std::getenv("FSR4_VK_CAPTURE_TRIGGER")) {
        if (*value) {
            live_capture_enabled = true;
            live_capture_trigger = value;

            log("[411-capture] armed trigger=" + live_capture_trigger.string() +
                " dir=" + live_capture_root.string());
        }
    }
}

} // namespace fsr4runtime
