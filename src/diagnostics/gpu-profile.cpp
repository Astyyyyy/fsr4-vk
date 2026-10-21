// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

#include "runtime/context.hpp"

namespace fsr4runtime {

double QualityContext::profileMs(uint64_t begin, uint64_t end) const {
    return static_cast<double>(end - begin) * gpu_timestamp_period_ns / 1.0e6;
}

bool QualityContext::collectGpuProfile(Frame& frame) {
    if (!gpu_profile_enabled || !frame.profile_pending || !frame.profile_queries) {
        return true;
    }

    const uint32_t query_count =
        gpu_profile_summary_only
            ? kProfilePassBase
            : kProfilePassBase + kProfilePassStride * model_pass_policy.passes.size();
    std::array<uint64_t, kProfileQueryCount * 2> values{};

    const VkResult result =
        vkGetQueryPoolResults(device,
                              frame.profile_queries,
                              0,
                              query_count,
                              sizeof(uint64_t) * 2 * query_count,
                              values.data(),
                              sizeof(uint64_t) * 2,
                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);

    if (result == VK_NOT_READY) {
        return false;
    }

    check(result, "vkGetQueryPoolResults(FSR4 GPU profile)");

    for (uint32_t query_index = 0; query_index < query_count; ++query_index) {
        if (!values[query_index * 2 + 1]) {
            return false;
        }
    }

    const auto timestamp_ticks = [&](uint32_t query_index) {
        return values[query_index * 2];
    };

    const auto csv_path = diagnosticPath("FSR4_VK_GPU_PROFILE_PATH", "fsr4-gpu-profile.csv");
    bool write_header = true;
    std::error_code file_error;

    if (std::filesystem::exists(csv_path, file_error) && !file_error) {
        const auto csv_size = std::filesystem::file_size(csv_path, file_error);

        if (!file_error && csv_size != 0) {
            write_header = false;
        }
    }

    std::ofstream stream(csv_path, std::ios::app);

    if (stream) {
        if (write_header) {
            stream << "frame,zone,ms\n";
        }

        stream << std::fixed << std::setprecision(6);
        const auto emit = [&](const std::string& zone, uint64_t begin, uint64_t end) {
            stream << frame.profile_frame_id << ',' << zone << ',' << profileMs(begin, end) << '\n';
        };

        emit("frame_total", timestamp_ticks(kProfileFrameStart), timestamp_ticks(kProfileFrameEnd));
        emit("external_inputs",
             timestamp_ticks(kProfileFrameStart),
             timestamp_ticks(kProfileAfterExternalInputs));
        emit("exposure_and_setup",
             timestamp_ticks(kProfileAfterExternalInputs),
             timestamp_ticks(kProfileAfterExposure));
        emit("record_frame",
             timestamp_ticks(kProfileAfterExposure),
             timestamp_ticks(kProfileAfterRecordFrame));
        emit("staging_copy",
             timestamp_ticks(kProfileAfterRecordFrame),
             timestamp_ticks(kProfileAfterStagingCopy));
        emit("final_barrier",
             timestamp_ticks(kProfileAfterStagingCopy),
             timestamp_ticks(kProfileFrameEnd));

        if (!gpu_profile_summary_only) {
            uint64_t previous = timestamp_ticks(kProfileAfterExposure);

            for (uint32_t index = 0; index < model_pass_policy.passes.size(); ++index) {
                const auto& pass = model_pass_policy.passes[index];

                if (pass.type == PassType::TensorPadding) {
                    const uint32_t padding_layer_index = index <= 1 ? 0 : (index - 1) / 2;

                    if (!execute_padding_passes || padding_layer_index == 6) {
                        continue;
                    }
                }

                const uint32_t base = kProfilePassBase + index * kProfilePassStride;
                const uint64_t start = timestamp_ticks(base);
                const uint64_t end = timestamp_ticks(base + 1);
                const uint64_t barrier_end = timestamp_ticks(base + 2);

                emit(std::string("gap_before/") + pass.name, previous, start);
                emit(std::string("dispatch/") + pass.name, start, end);

                if (index + 1 < model_pass_policy.passes.size()) {
                    emit(std::string("barrier_after/") + pass.name, end, barrier_end);
                }

                previous = barrier_end;
            }

            emit("record_tail", previous, timestamp_ticks(kProfileAfterRecordFrame));
        }
    }

    frame.profile_pending = false;

    return true;
}

bool QualityContext::beginGpuProfile(VkCommandBuffer command_buffer, Frame& frame) {
    if (!gpu_profile_enabled || !frame.profile_queries) {
        return false;
    }

    if (!collectGpuProfile(frame)) {
        return false;
    }

    if (frame_count % gpu_profile_interval != 0) {
        return false;
    }

    const uint32_t query_count =
        gpu_profile_summary_only
            ? kProfilePassBase
            : kProfilePassBase + kProfilePassStride * model_pass_policy.passes.size();
    vkCmdResetQueryPool(command_buffer, frame.profile_queries, 0, query_count);
    dispatch.cmd_write_timestamp2(command_buffer,
                                  VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                  frame.profile_queries,
                                  kProfileFrameStart);
    frame.profile_frame_id = frame_count;
    return true;
}

void QualityContext::profileTimestamp(VkCommandBuffer command_buffer,
                                      Frame& frame,
                                      bool active,
                                      uint32_t query) {
    if (active) {
        dispatch.cmd_write_timestamp2(
            command_buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, frame.profile_queries, query);
    }
}

void QualityContext::configureGpuProfile() {
    if (const char* value = std::getenv("FSR4_VK_GPU_PROFILE")) {
        gpu_profile_enabled = std::strcmp(value, "0") != 0;
    }

    if (gpu_profile_enabled) {
        if (const char* value = std::getenv("FSR4_VK_GPU_PROFILE_SUMMARY_ONLY")) {
            gpu_profile_summary_only = std::strcmp(value, "0") != 0;
        }

        if (const char* value = std::getenv("FSR4_VK_GPU_PROFILE_INTERVAL")) {
            char* end = nullptr;
            const unsigned long parsed = std::strtoul(value, &end, 10);

            if (end != value && *end == '\0' && parsed > 0 && parsed <= 10000) {
                gpu_profile_interval = static_cast<uint32_t>(parsed);
            }
        }

        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physical_device, &properties);
        gpu_timestamp_period_ns = properties.limits.timestampPeriod;

        if (!(gpu_timestamp_period_ns > 0.0)) {
            throw std::runtime_error("invalid Vulkan timestampPeriod for GPU profiler");
        }

        log("[gpu-profile] enabled interval_frames=" + std::to_string(gpu_profile_interval) +
            " mode=" + (gpu_profile_summary_only ? std::string("summary") : std::string("passes")) +
            " timestamp_period_ns=" + std::to_string(gpu_timestamp_period_ns));
    }
}

} // namespace fsr4runtime
