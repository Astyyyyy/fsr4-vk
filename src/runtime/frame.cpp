// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

#include "runtime/frame.hpp"
#include "diagnostics/diagnostics.hpp"
#include "runtime/context.hpp"
#include "runtime/resources.hpp"

#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <stdexcept>

namespace fsr4runtime {

void QualityContext::recordFrame(VkCommandBuffer command_buffer,
                                 Image color,
                                 Image depth,
                                 Image motion,
                                 Image exposure_image,
                                 Image optional_mask,
                                 Image output,
                                 OptimizedConstants main_constants,
                                 bool use_staging_output,
                                 ColorSpace requested_color_space,
                                 fsr4_411::OverlayOptions overlay) {
    const auto record_start = extensive_logging ? std::chrono::high_resolution_clock::now()
                                                : std::chrono::high_resolution_clock::time_point{};

    if (live_capture_pending) {
        liveCaptureFlush();
    }

    collectFrameUses();
    beginFrameUse();

    ++frame_count;

    if (!depth.image) {
        depth = resolveDepthInput(command_buffer, depth, color.width, color.height);
    }

    bool live_capture_triggered = false;

    if (live_capture_enabled && !live_capture_done && !live_capture_trigger.empty()) {
        std::error_code ec;

        if (fs::exists(live_capture_trigger, ec) && !ec) {
            live_capture_triggered = true;
            live_capture_target_frame = frame_count;

            fs::remove(live_capture_trigger, ec);
        }
    }

    selectModelPreset(color.width, main_constants);

    auto& frame = frames[next_frame];
    prepareModelPipelines();
    std::span<const VkPipeline> active_pipelines = pipelines;
    prepareModelWeights(frame);

    active_pipelines = prepareColorSpacePipelines(requested_color_space, overlay.debug_view);
    if (frame_count > 1 && color_space != requested_color_space) {
        main_constants.reset = 1;
    }
    color_space = requested_color_space;

    const bool transition_temporal_images = initialized;

    const bool gpu_profile_active = beginGpuProfile(command_buffer, frame);

    if (extensive_logging && (frame_count <= 3 || frame_count % 300 == 0)) {
        std::ostringstream message;
        message << "[frame] frame=" << frame_count << " frame_slot=" << next_frame
                << " model=" << (use_int8_dispatch ? "int8" : "fp8")
                << " history_initialized=" << (initialized ? "yes" : "no")
                << " wmma=" << (use_wmma ? "yes" : "no");
        log(message.str(), fsr4log::Level::Debug);
        log(fsr4log::describeImage(frame_count, "color", color), fsr4log::Level::Debug);
        log(fsr4log::describeImage(frame_count, "depth", depth), fsr4log::Level::Debug);
        log(fsr4log::describeImage(frame_count, "motion", motion), fsr4log::Level::Debug);
        log(fsr4log::describeImage(frame_count, "external-exposure", exposure_image),
            fsr4log::Level::Debug);
        log(fsr4log::describeImage(frame_count, "output", output), fsr4log::Level::Debug);
    }

    prepareExternalViews(frame, color, depth, motion, exposure_image, optional_mask, output);
    auto& history = images[image_slot::history];
    auto& reprojected = images[image_slot::reprojected];
    auto& recurrent = images[image_slot::recurrent];
    const bool run_auto_exposure = updateDynamicDescriptors(command_buffer,
                                                            frame,
                                                            color,
                                                            depth,
                                                            motion,
                                                            exposure_image,
                                                            optional_mask,
                                                            output,
                                                            reprojected,
                                                            use_staging_output);

    if (main_constants.rcas_enabled) {
        prepareRcas(command_buffer, frame, output, exposure_image, use_staging_output);
    } else {
        retirePipeline(rcas_pipeline.handle);
        frame.rcas_pipeline = VK_NULL_HANDLE;
    }

    // SPD must see the active render extent rather than padded image dimensions.
    Image exact_auto_exposure_color{};

    if (run_auto_exposure) {
        if (!auto_exposure_pipeline) {
            createAutoExposurePipeline(auto_exposure_shader_path);
        }
        exact_auto_exposure_color = prepareExactAutoExposureColor(
            command_buffer, color, main_constants.width_lr, main_constants.height_lr);
    }
    if (!run_auto_exposure) {
        retirePipeline(auto_exposure_pipeline);
    }
    // Inputs arrive in SHADER_READ_ONLY_OPTIMAL and output in GENERAL.
    // Restore any temporary input transitions before returning.

    if (exact_auto_exposure_color.image) {
        // Only auto-exposure reads the exact-size copy; model passes use the original color.
        writeSampledImageDescriptor(frame,
                                    descriptor_set::sampled_images,
                                    93,
                                    exact_auto_exposure_color.view,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    if (!direct_display_motion_vectors && motion.image &&
        (motion.width != color.width || motion.height != color.height)) {
        motion = resampleDisplayMotionVectors(command_buffer, motion, color.width, color.height);

        // Use the resampled motion view for prepass.
        writeSampledImageDescriptor(frame,
                                    descriptor_set::sampled_images,
                                    1,
                                    motion.view,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    profileTimestamp(command_buffer, frame, gpu_profile_active, kProfileAfterExternalInputs);
    writeFrameConstants(frame, main_constants, run_auto_exposure);

    initializeFrameHistory(command_buffer, frame, main_constants);

    if (run_auto_exposure) {
        recordAutoExposure(command_buffer, frame, main_constants);
    }

    profileTimestamp(command_buffer, frame, gpu_profile_active, kProfileAfterExposure);
    const bool live_capture_this_frame =
        live_capture_enabled && !live_capture_done &&
        (live_capture_triggered ||
         (live_capture_target_frame != 0 && frame_count == live_capture_target_frame));

    if (live_capture_this_frame) {
        VkEventCreateInfo event_info{VK_STRUCTURE_TYPE_EVENT_CREATE_INFO};
        check(vkCreateEvent(device, &event_info, nullptr, &live_capture_completion),
              "live capture completion event");
        live_capture_target_frame = frame_count;

        live_capture_frame_dir = live_capture_root / ("frame-" + std::to_string(frame_count));

        fs::create_directories(live_capture_frame_dir);

        live_capture_run_auto_exposure = run_auto_exposure;

        live_capture_use_staging_output = use_staging_output;

        live_capture_recording = true;

        liveCaptureManifest(color,
                            depth,
                            motion,
                            exposure_image,
                            output,
                            main_constants,
                            run_auto_exposure,
                            use_staging_output);

        liveCaptureWrite(live_capture_frame_dir / "constants.bin",
                         frame.constants.mapped,
                         static_cast<size_t>(frame.constants.size));

        liveCaptureWrite(live_capture_frame_dir / "model_constants.bin",
                         frame.model_constants.mapped,
                         static_cast<size_t>(frame.model_constants.size));

        log("[411-capture] RECORDING frame=" + std::to_string(frame_count));
    }

    if (overlay.debug_view) {
        // Finish reading the previous debug preview before postpass overwrites it.
        imageBarrier(dispatch.cmd_pipeline_barrier2,
                     command_buffer,
                     images[image_slot::wmma_scratch1].image,
                     VK_IMAGE_LAYOUT_GENERAL,
                     VK_IMAGE_LAYOUT_GENERAL,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    }

    for (auto pipeline : active_pipelines) {
        trackPipelineUse(pipeline);
    }
    recordModelPasses(FrameRecordingInfo{
        .command_buffer = command_buffer,
        .pipeline_layout = pipeline_layout,

        .descriptor_buffer_address = frame.descriptor_buffer_address,
        .descriptor_set_capacity = descriptor_capacity,
        .bind_descriptor_buffers = vkCmdBindDescriptorBuffersEXT,
        .set_descriptor_buffer_offsets = vkCmdSetDescriptorBufferOffsetsEXT,
        .descriptor_buffers_bound = run_auto_exposure,

        .use_wmma = use_wmma,
        .model_policy = &model_pass_policy,
        .execute_padding_passes = execute_padding_passes,
        .pipelines = active_pipelines,
        .constants_address = frame.constants_address,
        .output_width = main_constants.width,
        .output_height = main_constants.height,
        .model_bucket_width = model_bucket_width,
        .model_bucket_height = model_bucket_height,

        .history = history.image,
        .recurrent = recurrent.image,
        .reprojected = reprojected.image,
        .auto_exposure_texture = images[image_slot::exposure].image,
        .calculation_scratch_buffer = scratch_buffer.buffer,
        .transition_temporal_images = transition_temporal_images,
        .pipeline_barrier2 = dispatch.cmd_pipeline_barrier2,

        .timestamp_queries = gpu_profile_active && !gpu_profile_summary_only ? frame.profile_queries
                                                                             : VK_NULL_HANDLE,
        .timestamp_base = kProfilePassBase,
        .write_timestamp2 = dispatch.cmd_write_timestamp2,
        .verbose = extensive_logging && frame_count <= 3,
        .timestamp_barriers = gpu_profile_active && !gpu_profile_summary_only,

        .capture_hook = live_capture_this_frame ? &QualityContext::liveCaptureHook : nullptr,
        .capture_user = live_capture_this_frame ? this : nullptr,
    });

    if (live_capture_this_frame) {
        live_capture_recording = false;
        VkMemoryBarrier readback{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        readback.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        readback.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(command_buffer,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT,
                             0,
                             1,
                             &readback,
                             0,
                             nullptr,
                             0,
                             nullptr);
        vkCmdSetEvent(command_buffer, live_capture_completion, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        live_capture_pending = true;
        live_capture_done = true;

        log("[411-capture] GPU copies queued; "
            "waiting for next provider frame to dump");
    }

    if (main_constants.rcas_enabled) {
        recordRcas(command_buffer, frame, output, main_constants);
    }

    profileTimestamp(command_buffer, frame, gpu_profile_active, kProfileAfterRecordFrame);

    if (overlay.watermark || overlay.debug_view) {
        recordOverlay(command_buffer, frame, output, motion, main_constants, overlay);
    } else {
        retirePipeline(overlay_pipeline);
    }

    copyStagingOutput(command_buffer, output, use_staging_output);

    profileTimestamp(command_buffer, frame, gpu_profile_active, kProfileAfterStagingCopy);

    // The caller's next output transition makes these writes visible to its consumer.
    profileTimestamp(command_buffer, frame, gpu_profile_active, kProfileFrameEnd);

    if (gpu_profile_active) {
        frame.profile_pending = true;
    }

    // Signal completion when this dispatch has executed on the GPU.
    vkCmdSetEvent(
        command_buffer, pending_frame_uses.back().completion, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    collectFrameUses();

    next_frame = (next_frame + 1) % frames.size();
    initialized = true;
    previous_exposure = main_constants.pre_exposure;

    if (extensive_logging) {
        const auto record_end = std::chrono::high_resolution_clock::now();
        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(record_end - record_start).count();
        total_record_ms += elapsed_ms;

        if (elapsed_ms < min_record_ms) {
            min_record_ms = elapsed_ms;
        }

        if (elapsed_ms > max_record_ms) {
            max_record_ms = elapsed_ms;
        }

        if (frame_count % 300 == 0) {
            const double avg_ms = total_record_ms / frame_count;
            std::ostringstream ps;
            ps << "[cpu-profile] frame=" << frame_count << " record_ms=" << std::fixed
               << std::setprecision(3) << elapsed_ms << " average_record_ms=" << avg_ms
               << " minimum_record_ms=" << min_record_ms << " maximum_record_ms=" << max_record_ms
               << " (CPU-side record() time, excludes GPU execution)";
            log(ps.str());
        }
    }
}

} // namespace fsr4runtime

namespace fsr4runtime {

void QualityContext::selectModelPreset(uint32_t render_width, OptimizedConstants& main_constants) {
    const std::string needed_preset = fsr4::resolutionPreset(render_width, main_constants.width);
    if ((current_preset == "ultraperf") != (needed_preset == "ultraperf")) {
        log("[model] switching family: " + current_preset + " -> " + needed_preset);
        if (frame_count > 1) {
            initialized = false;
            main_constants.reset = 1;
        }
    }
    current_preset = needed_preset;
}

void QualityContext::prepareModelWeights(Frame& frame) {
    const bool ultra = current_preset == "ultraperf";
    if (ultra && !ultraperformance_weight_buffer.buffer) {
        throw std::runtime_error("Ultra Performance model data is unavailable");
    }
    writeWeightDescriptors(frame, ultra ? ultraperformance_weight_buffer : weight_buffer);
}

void QualityContext::writeFrameConstants(Frame& frame,
                                         OptimizedConstants& main_constants,
                                         bool run_auto_exposure) {
    const uint32_t padded_output_width = fsr4::roundUp(main_constants.width, 8);
    const uint32_t padded_output_height = fsr4::roundUp(main_constants.height, 8);

    if (cached_model_width != main_constants.width ||
        cached_model_height != main_constants.height) {
        cached_model_constants =
            model_pass_policy.make_constants(main_constants.width, main_constants.height);
        cached_model_width = main_constants.width;
        cached_model_height = main_constants.height;
    }

    const ModelConstants& model_constants = cached_model_constants;
    std::memcpy(frame.model_constants.mapped, &model_constants, sizeof(model_constants));

    if (extensive_logging && frame_count <= 3) {
        log("[descriptor] frame=" + std::to_string(frame_count) +
                " set=5 role=uniform-buffers model_constants_buffer=" +
                fsr4log::handle(frame.model_constants.buffer) +
                " frame_constants_buffer=" + fsr4log::handle(frame.constants.buffer),
            fsr4log::Level::Debug);
        std::ostringstream model_message;
        model_message << "[model] frame=" << frame_count
                      << " padded_output_size=" << padded_output_width << 'x'
                      << padded_output_height;
        log(model_message.str(), fsr4log::Level::Debug);

        for (uint32_t layer = 0; layer <= model_binding::layer_count; ++layer) {
            const uint32_t slot = layer == 0 ? 0 : layer + 1;
            const auto& dimensions = model_constants.layers[slot];
            log("[model] frame=" + std::to_string(frame_count) + " layer=" + std::to_string(layer) +
                    " constants_slot=" + std::to_string(slot) + " tensor_size=" +
                    std::to_string(dimensions.width) + "x" + std::to_string(dimensions.height),
                fsr4log::Level::Debug);
        }
    }

    main_constants.previous_pre_exposure = previous_exposure;
    main_constants.reset = main_constants.reset || !initialized;
    std::memset(frame.constants.mapped, 0, kConstantBufSize);
    std::memcpy(frame.constants.mapped, &main_constants, sizeof(main_constants));

    if (run_auto_exposure) {
        writeAutoExposureConstants(frame, main_constants);
    }

    if (extensive_logging && frame_count <= 3) {
        auto* written = reinterpret_cast<const OptimizedConstants*>(frame.constants.mapped);
        std::ostringstream constants_readback_message;
        constants_readback_message
            << "[constants] frame=" << frame_count << " buffer_tex_size=[" << written->tex_size[0]
            << "," << written->tex_size[1] << "]"
            << " padded_output_width=" << written->width
            << " padded_output_height=" << written->height << " render_width=" << written->width_lr
            << " render_height=" << written->height_lr << " scale=[" << written->scale[0] << ","
            << written->scale[1] << "]"
            << " reset=" << written->reset;
        log(constants_readback_message.str(), fsr4log::Level::Debug);
    }
}

void QualityContext::initializeFrameHistory(VkCommandBuffer command_buffer,
                                            Frame& frame,
                                            const OptimizedConstants& main_constants) {
    const auto& history = images[image_slot::history];
    const auto& recurrent = images[image_slot::recurrent];

    const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    if (!initialized) {
        VkClearColorValue zero{};
        VkClearColorValue one{};
        one.float32[0] = 1.0f;

        for (auto& image : images) {
            const bool is_exposure = (&image == &images[image_slot::exposure]);
            imageBarrier(
                dispatch.cmd_pipeline_barrier2,
                command_buffer,
                image.image,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                frame_count > 1 ? VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT : VK_PIPELINE_STAGE_2_NONE,
                frame_count > 1 ? VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT : 0,
                VK_PIPELINE_STAGE_2_CLEAR_BIT,
                VK_ACCESS_2_TRANSFER_WRITE_BIT);
            vkCmdClearColorImage(command_buffer,
                                 image.image,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 is_exposure ? &one : &zero,
                                 1,
                                 &range);

            if (extensive_logging && (is_exposure || &image == &images[image_slot::history])) {
                log("[history] frame=" + std::to_string(frame_count) +
                        " initialized role=" + (is_exposure ? "temporal-exposure" : "history") +
                        " image=" + fsr4log::handle(image.image) +
                        " clear_value=" + (is_exposure ? "1.0" : "0.0"),
                    fsr4log::Level::Debug);
            }

            const bool is_sampled_only =
                image.image == history.image || image.image == recurrent.image || is_exposure;
            imageBarrier(dispatch.cmd_pipeline_barrier2,
                         command_buffer,
                         image.image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         is_sampled_only ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                         : VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_2_CLEAR_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
        }
        // RADV may drop the R32_SFLOAT clear, so upload exposure explicitly.
        {
            const float exposure_init[2] = {1.0f, 1.0f};
            vkCmdUpdateBuffer(command_buffer,
                              frame.exposure_upload.buffer,
                              0,
                              sizeof(exposure_init),
                              exposure_init);
            VkMemoryBarrier2 exposure_upload_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
            exposure_upload_barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            exposure_upload_barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            exposure_upload_barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            exposure_upload_barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dependency.memoryBarrierCount = 1;
            dependency.pMemoryBarriers = &exposure_upload_barrier;
            dispatch.cmd_pipeline_barrier2(command_buffer, &dependency);
            imageBarrier(dispatch.cmd_pipeline_barrier2,
                         command_buffer,
                         images[image_slot::exposure].image,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_PIPELINE_STAGE_2_NONE,
                         0,
                         VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT);
            VkBufferImageCopy region{};
            region.bufferOffset = 0;
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.imageExtent = {2, 1, 1};
            vkCmdCopyBufferToImage(command_buffer,
                                   frame.exposure_upload.buffer,
                                   images[image_slot::exposure].image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                   1,
                                   &region);
            imageBarrier(dispatch.cmd_pipeline_barrier2,
                         command_buffer,
                         images[image_slot::exposure].image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            log("[exposure] initialized temporal exposure: current=1.0 previous=1.0 via buffer upload");
        }
    }

    if (frame_count == 1 || main_constants.reset) {
        VkMemoryBarrier2 before{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        before.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        before.srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
        before.dstStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
        before.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.memoryBarrierCount = 1;
        dependency.pMemoryBarriers = &before;
        dispatch.cmd_pipeline_barrier2(command_buffer, &dependency);
        vkCmdFillBuffer(command_buffer, scratch_buffer.buffer, 0, VK_WHOLE_SIZE, 0);

        before.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
        before.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        before.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        before.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
        dispatch.cmd_pipeline_barrier2(command_buffer, &dependency);
    }
}

Image QualityContext::resolveDepthInput(VkCommandBuffer command_buffer,
                                        Image depth,
                                        uint32_t active_width,
                                        uint32_t active_height) {
    if (depth.image) {
        return depth;
    }

    const uint32_t capacity_width =
        max_render_width > active_width ? max_render_width : active_width;

    const uint32_t capacity_height =
        max_render_height > active_height ? max_render_height : active_height;

    const bool recreate = !fallback_depth.image || fallback_depth.format != VK_FORMAT_R32_SFLOAT ||
                          fallback_depth.width < active_width ||
                          fallback_depth.height < active_height;

    if (recreate) {
        // Allocate enough capacity for dynamic resolution without reallocating each frame.
        if (fallback_depth.image) {
            vkDeviceWaitIdle(device);
            destroyImage(device, fallback_depth);
        }

        fallback_depth = createImage(physical_device,
                                     device,
                                     capacity_width,
                                     capacity_height,
                                     VK_FORMAT_R32_SFLOAT,
                                     VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

        fallback_depth_initialized = false;

        fsr4vk::setDebugName(dispatch,
                             device,
                             VK_OBJECT_TYPE_IMAGE,
                             (uint64_t)fallback_depth.image,
                             "FSR4_NoDepthFallback");
    }

    if (!fallback_depth_initialized) {
        imageBarrier(dispatch.cmd_pipeline_barrier2,
                     command_buffer,
                     fallback_depth.image,
                     VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_PIPELINE_STAGE_2_NONE,
                     VK_ACCESS_2_NONE,
                     VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     VK_ACCESS_2_TRANSFER_WRITE_BIT);

        VkClearColorValue clear{};

        // Use neutral far-plane depth when the caller provides no depth image.
        const float far_depth = depth_inverted_enabled ? 0.0f : 1.0f;

        clear.float32[0] = far_depth;
        clear.float32[1] = far_depth;
        clear.float32[2] = far_depth;
        clear.float32[3] = far_depth;

        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        vkCmdClearColorImage(command_buffer,
                             fallback_depth.image,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             &clear,
                             1,
                             &range);

        imageBarrier(dispatch.cmd_pipeline_barrier2,
                     command_buffer,
                     fallback_depth.image,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     VK_ACCESS_2_TRANSFER_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

        fallback_depth_initialized = true;

        log(std::string("[depth] initialized neutral no-depth t2 value=") +
            (depth_inverted_enabled ? "0" : "1") + " capacity=" + std::to_string(capacity_width) +
            "x" + std::to_string(capacity_height));
    }

    Image result = fallback_depth;

    // Report the active size while retaining the larger backing image.
    result.width = active_width;
    result.height = active_height;

    return result;
}

void QualityContext::copyStagingOutput(VkCommandBuffer command_buffer,
                                       const Image& output,
                                       bool use_staging_output) {
    if (use_staging_output && staging_output.image) {
        VkImageMemoryBarrier2 staging_copy_barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        staging_copy_barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        staging_copy_barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        staging_copy_barrier.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        staging_copy_barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        staging_copy_barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        staging_copy_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        staging_copy_barrier.image = staging_output.image;
        staging_copy_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkImageMemoryBarrier2 output_copy_barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        output_copy_barrier.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
        output_copy_barrier.srcAccessMask = VK_ACCESS_2_NONE;
        output_copy_barrier.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        output_copy_barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        output_copy_barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        output_copy_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        output_copy_barrier.image = output.image;
        output_copy_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkImageMemoryBarrier2 pre_copy_barriers[] = {staging_copy_barrier, output_copy_barrier};
        VkDependencyInfo pre_copy{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        pre_copy.imageMemoryBarrierCount = 2;
        pre_copy.pImageMemoryBarriers = pre_copy_barriers;
        dispatch.cmd_pipeline_barrier2(command_buffer, &pre_copy);

        VkImageCopy region{};
        region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.extent = {output.width, output.height, 1};
        vkCmdCopyImage(command_buffer,
                       staging_output.image,
                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       output.image,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1,
                       &region);

        if (extensive_logging && frame_count <= 3) {
            log("[output] staging copy source_size=" + std::to_string(staging_output.width) + "x" +
                    std::to_string(staging_output.height) + " destination_size=" +
                    std::to_string(output.width) + "x" + std::to_string(output.height),
                fsr4log::Level::Debug);
        }

        VkImageMemoryBarrier2 staging_restore_barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        staging_restore_barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        staging_restore_barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        staging_restore_barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        staging_restore_barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        staging_restore_barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        staging_restore_barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        staging_restore_barrier.image = staging_output.image;
        staging_restore_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkImageMemoryBarrier2 output_ready_barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        output_ready_barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        output_ready_barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        output_ready_barrier.dstStageMask =
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        output_ready_barrier.dstAccessMask =
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        output_ready_barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        output_ready_barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        output_ready_barrier.image = output.image;
        output_ready_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkImageMemoryBarrier2 post_copy_barriers[] = {staging_restore_barrier,
                                                      output_ready_barrier};
        VkDependencyInfo post_copy{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        post_copy.imageMemoryBarrierCount = 2;
        post_copy.pImageMemoryBarriers = post_copy_barriers;
        dispatch.cmd_pipeline_barrier2(command_buffer, &post_copy);
    }
}

void QualityContext::prepareStagingOutput(VkCommandBuffer command_buffer,
                                          const Image& output,
                                          bool use_staging_output) {
    if (use_staging_output && (!staging_output.image || staging_output.format != output.format)) {
        auto replacement =
            createImage(physical_device,
                        device,
                        fsr4::roundUp(max_output_width, 8),
                        fsr4::roundUp(max_output_height, 8),
                        output.format,
                        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                            VK_IMAGE_USAGE_SAMPLED_BIT);
        if (staging_output.image) {
            try {
                retired_staging_outputs.push_back(staging_output);
            } catch (...) {
                destroyImage(device, replacement);
                throw;
            }
        }
        staging_output = replacement;
        log("[output] created staging image size=" + std::to_string(staging_output.width) + "x" +
            std::to_string(staging_output.height) +
            " format=" + fsr4log::vulkanFormat(output.format));

        VkImageMemoryBarrier2 staging_initialization_barrier{
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        staging_initialization_barrier.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
        staging_initialization_barrier.srcAccessMask = VK_ACCESS_2_NONE;
        staging_initialization_barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        staging_initialization_barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        staging_initialization_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        staging_initialization_barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        staging_initialization_barrier.image = staging_output.image;
        staging_initialization_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkDependencyInfo staging_initialization_dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        staging_initialization_dependency.imageMemoryBarrierCount = 1;
        staging_initialization_dependency.pImageMemoryBarriers = &staging_initialization_barrier;
        dispatch.cmd_pipeline_barrier2(command_buffer, &staging_initialization_dependency);
    }
    if (use_staging_output) {
        pending_frame_uses.back().staging_output = staging_output.image;
    }
}

} // namespace fsr4runtime

namespace fsr4runtime {
using fsr4_411::computeBarrier;
using fsr4_411::imageBarrier;

namespace {
void recordPassDependency(const FrameRecordingInfo& recording,
                          uint32_t pass_index,
                          uint32_t next_pass_index) {
    const auto dependency =
        recording.model_policy->dependency(pass_index, next_pass_index, recording.use_wmma);
    VkAccessFlags2 destination_access = 0;

    switch (dependency) {
    case PassDependency::FullComputeMemory:
        computeBarrier(recording.pipeline_barrier2, recording.command_buffer);
        return;
    case PassDependency::ScratchRead:
        destination_access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        break;
    case PassDependency::ScratchWrite:
        destination_access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        break;
    case PassDependency::ScratchReadWrite:
        destination_access =
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        break;
    }

    scratchBufferBarrier(recording.pipeline_barrier2,
                         recording.command_buffer,
                         recording.calculation_scratch_buffer,
                         destination_access);
}
} // namespace

void scratchBufferBarrier(PFN_vkCmdPipelineBarrier2 pipeline_barrier2,
                          VkCommandBuffer command_buffer,
                          VkBuffer scratch_buffer,
                          VkAccessFlags2 dst_access) {
    VkBufferMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};

    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask = dst_access;

    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

    barrier.buffer = scratch_buffer;
    barrier.offset = 0;
    barrier.size = VK_WHOLE_SIZE;

    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};

    dependency.bufferMemoryBarrierCount = 1;
    dependency.pBufferMemoryBarriers = &barrier;

    pipeline_barrier2(command_buffer, &dependency);
}

void recordModelPasses(const FrameRecordingInfo& recording) {
    const auto passes =
        recording.model_policy ? recording.model_policy->passes : std::span<const Pass>{};

    if (!recording.command_buffer || !recording.pipeline_layout ||
        !recording.descriptor_buffer_address || !recording.bind_descriptor_buffers ||
        !recording.set_descriptor_buffer_offsets || passes.empty() ||
        recording.pipelines.size() != passes.size() || !recording.constants_address ||
        !recording.history || !recording.recurrent || !recording.reprojected ||
        !recording.model_policy || !recording.model_policy->dispatch_size ||
        !recording.model_policy->dependency ||
        (recording.model_policy->postpass_updates_exposure && !recording.auto_exposure_texture)) {
        throw std::invalid_argument("incomplete FSR 4.1.1 frame recording arguments");
    }

    // Restore the layouts expected by prepass.
    if (recording.transition_temporal_images) {
        for (VkImage image : {recording.history, recording.recurrent}) {
            imageBarrier(recording.pipeline_barrier2,
                         recording.command_buffer,
                         image,
                         VK_IMAGE_LAYOUT_GENERAL,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }

        imageBarrier(recording.pipeline_barrier2,
                     recording.command_buffer,
                     recording.reprojected,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_IMAGE_LAYOUT_GENERAL,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    }

    if (recording.verbose) {
        fsr4runtime::log("[frame] recorder output_size=" + std::to_string(recording.output_width) +
                             "x" + std::to_string(recording.output_height) +
                             " wmma=" + std::to_string(recording.use_wmma) +
                             " planned_passes=" + std::to_string(passes.size()) +
                             " constants_addr=" + std::to_string(recording.constants_address) +
                             " history=" + fsr4log::handle(recording.history) +
                             " recurrent=" + fsr4log::handle(recording.recurrent) +
                             " reprojected=" + fsr4log::handle(recording.reprojected),
                         fsr4log::Level::Debug);
    }

    std::array<VkDescriptorBufferBindingInfoEXT, 1> binding_infos{};
    binding_infos[0].sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT;
    binding_infos[0].address = recording.descriptor_buffer_address;
    binding_infos[0].usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT |
                             VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT;
    if (!recording.descriptor_buffers_bound) {
        recording.bind_descriptor_buffers(recording.command_buffer, 1, binding_infos.data());
    }

    constexpr std::array<uint32_t, 6> buffer_indices{{0, 0, 0, 0, 0, 0}};
    const std::array<VkDeviceSize, 6> buffer_offsets{{
        0 * recording.descriptor_set_capacity,
        1 * recording.descriptor_set_capacity,
        2 * recording.descriptor_set_capacity,
        3 * recording.descriptor_set_capacity,
        4 * recording.descriptor_set_capacity,
        5 * recording.descriptor_set_capacity,
    }};
    recording.set_descriptor_buffer_offsets(recording.command_buffer,
                                            VK_PIPELINE_BIND_POINT_COMPUTE,
                                            recording.pipeline_layout,
                                            0,
                                            buffer_indices.size(),
                                            buffer_indices.data(),
                                            buffer_offsets.data());

    const auto should_skip_padding_pass = [&](uint32_t pass_index) {
        if (pass_index >= passes.size() || passes[pass_index].type != PassType::TensorPadding) {
            return false;
        }

        const bool padding_enabled = recording.execute_padding_passes;
        return !padding_enabled ||
               model_binding::paddingLayer(pass_index) == model_binding::unused_padding_layer;
    };

    for (uint32_t index = 0; index < passes.size(); ++index) {
        const auto& pass = passes[index];

        if (recording.capture_hook && pass.type == PassType::Prepass) {
            recording.capture_hook(
                recording.capture_user, CapturePoint::BeforePrepass, recording.command_buffer);
        }

        // Postpass reverses the temporal image roles.
        if (pass.type == PassType::Postpass) {
            for (VkImage image : {recording.history, recording.recurrent}) {
                imageBarrier(recording.pipeline_barrier2,
                             recording.command_buffer,
                             image,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             VK_IMAGE_LAYOUT_GENERAL,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                 VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            }

            imageBarrier(recording.pipeline_barrier2,
                         recording.command_buffer,
                         recording.reprojected,
                         VK_IMAGE_LAYOUT_GENERAL,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

            if (recording.model_policy->postpass_updates_exposure) {
                imageBarrier(recording.pipeline_barrier2,
                             recording.command_buffer,
                             recording.auto_exposure_texture,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             VK_IMAGE_LAYOUT_GENERAL,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                 VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            }
        }

        if (recording.capture_hook && pass.type == PassType::Postpass) {
            recording.capture_hook(
                recording.capture_user, CapturePoint::BeforePostpass, recording.command_buffer);
        }

        RootConstants root_constants{};

        switch (pass.type) {
        case PassType::Prepass:
            break;

        case PassType::ModelPass: {
            const uint32_t model_layer_index = model_binding::calculationLayer(index);
            root_constants.data[model_binding::scratch_offset_word] =
                model_binding::scratchOffset(model_layer_index);
            root_constants.data[model_binding::weight_offset_word] =
                model_binding::weightOffset(model_layer_index);
            break;
        }

        case PassType::TensorPadding: {
            const uint32_t model_layer_index = model_binding::paddingLayer(index);
            root_constants.data[model_binding::scratch_offset_word] =
                model_binding::scratchOffset(model_layer_index);
            break;
        }

        case PassType::Postpass:
            break;
        }

        const uint32_t timestamp_stride = recording.timestamp_barriers ? 3u : 2u;
        const uint32_t timestamp_index = recording.timestamp_base + index * timestamp_stride;

        // Fill timestamp slots for skipped passes so query reads can complete.
        if (pass.type == PassType::TensorPadding) {
            if (should_skip_padding_pass(index)) {
                if (recording.timestamp_queries && recording.write_timestamp2 &&
                    recording.timestamp_barriers) {
                    for (uint32_t q = 0; q < 3; ++q) {
                        recording.write_timestamp2(recording.command_buffer,
                                                   VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                                   recording.timestamp_queries,
                                                   timestamp_index + q);
                    }
                }

                continue;
            }
        }

        vkCmdBindPipeline(
            recording.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, recording.pipelines[index]);

        if (recording.timestamp_queries && recording.write_timestamp2) {
            recording.write_timestamp2(recording.command_buffer,
                                       VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                       recording.timestamp_queries,
                                       timestamp_index);
        }

        vkCmdPushConstants(recording.command_buffer,
                           recording.pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0,
                           sizeof(RootConstants),
                           &root_constants);

        auto groups = recording.model_policy->dispatch_size(index,
                                                            recording.output_width,
                                                            recording.output_height,
                                                            recording.model_bucket_width,
                                                            recording.model_bucket_height);

        if (recording.verbose) {
            const char* legacy_type_names[] = {"Prepass", "ModelPass", "ModelPost", "Postpass"};
            const char* role_names[] = {
                "prepass", "model-calculation", "tensor-padding", "postpass"};
            std::ostringstream message;
            message << "[pass] index=" << index << " '" << pass.name
                    << "' role=" << role_names[static_cast<unsigned>(pass.type)]
                    << " legacy_type=" << legacy_type_names[static_cast<unsigned>(pass.type)]
                    << " wmma=" << (recording.use_wmma ? "yes" : "no") << " push_constants=["
                    << root_constants.data[0] << "," << root_constants.data[1] << ","
                    << root_constants.data[2] << "," << root_constants.data[3] << ","
                    << root_constants.data[4] << "," << root_constants.data[5] << ","
                    << root_constants.data[6] << "," << root_constants.data[7] << "]"
                    << " dispatch_groups=[" << groups[0] << "," << groups[1] << ",1]"
                    << " pipeline=" << fsr4log::handle(recording.pipelines[index])
                    << " layout=" << fsr4log::handle(recording.pipeline_layout);
            fsr4runtime::log(message.str(), fsr4log::Level::Debug);
        }

        if (groups[0] > 0 && groups[1] > 0) {
            vkCmdDispatch(recording.command_buffer, groups[0], groups[1], 1);

            if (recording.capture_hook && pass.type == PassType::Postpass) {
                recording.capture_hook(
                    recording.capture_user, CapturePoint::AfterPostpass, recording.command_buffer);
            }

            if (recording.timestamp_queries && recording.write_timestamp2) {
                recording.write_timestamp2(recording.command_buffer,
                                           VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                           recording.timestamp_queries,
                                           timestamp_index + 1);
            }

            if (recording.model_policy->postpass_updates_exposure &&
                pass.type == PassType::Postpass) {
                imageBarrier(recording.pipeline_barrier2,
                             recording.command_buffer,
                             recording.auto_exposure_texture,
                             VK_IMAGE_LAYOUT_GENERAL,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            }

            if (index + 1 < passes.size()) {
                uint32_t next_active_pass_index = index + 1;

                while (next_active_pass_index < passes.size() &&
                       should_skip_padding_pass(next_active_pass_index)) {
                    ++next_active_pass_index;
                }

                recordPassDependency(recording, index, next_active_pass_index);

                if (recording.timestamp_queries && recording.write_timestamp2 &&
                    recording.timestamp_barriers) {
                    recording.write_timestamp2(recording.command_buffer,
                                               VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                               recording.timestamp_queries,
                                               timestamp_index + 2);
                }
            } else if (recording.timestamp_queries && recording.write_timestamp2 &&
                       recording.timestamp_barriers) {
                // Fill the final barrier timestamp slot even though no barrier follows.
                recording.write_timestamp2(recording.command_buffer,
                                           VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                           recording.timestamp_queries,
                                           timestamp_index + 2);
            }
        }
    }
}
} // namespace fsr4runtime
