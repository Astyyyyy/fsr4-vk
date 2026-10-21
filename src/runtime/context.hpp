// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

#pragma once

#include "diagnostics/diagnostics.hpp"
#include "models/models.hpp"
#include "runtime/frame.hpp"
#include "runtime/resources.hpp"
#include "runtime/rcas.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace fsr4runtime {
// Models choose shader variants for the shared runtime.
struct ShaderRecipe {
    std::string profile;

    std::function<std::filesystem::path(const std::filesystem::path&)> select;

    std::filesystem::path candidate(const std::filesystem::path& path) const {
        return select ? select(path) : path;
    }
};
} // namespace fsr4runtime

namespace fsr4runtime {
struct ShaderSelectionOptions {
    bool use_wmma;
    bool use_int8_dispatch;
    bool numbered_shader_files;
    bool windows_fp8_optimized;
    bool windows_int8_optimized;
    bool extensive_logging;
};

// Share fallback logging state across presets.
class ModelShaderSelector {
    ShaderSelectionOptions options;
    const ShaderRecipe& recipe;
    bool exact_fallback_logged = false;

    std::filesystem::path stockShaderPath(const std::filesystem::path& directory,
                                          uint32_t pass_index,
                                          const std::filesystem::path& prepass_override,
                                          const std::filesystem::path& postpass_override) const;
    std::filesystem::path selectFp8Shader(const std::filesystem::path& directory,
                                          uint32_t pass_index,
                                          const std::filesystem::path& prepass_override,
                                          const std::filesystem::path& postpass_override);
    std::filesystem::path selectInt8Shader(const std::filesystem::path& directory,
                                           uint32_t pass_index,
                                           const std::filesystem::path& prepass_override,
                                           const std::filesystem::path& postpass_override);
    std::filesystem::path windowsInt8ShaderPath(const std::filesystem::path& directory,
                                                uint32_t pass_index,
                                                const std::filesystem::path& stock_path) const;
    std::filesystem::path selectInt8Variant(const std::filesystem::path& original);

public:
    ModelShaderSelector(ShaderSelectionOptions options, const ShaderRecipe& recipe)
        : options(options), recipe(recipe) {}

    std::filesystem::path select(const std::filesystem::path& directory,
                                 uint32_t pass_index,
                                 const std::filesystem::path& prepass_override,
                                 const std::filesystem::path& postpass_override);
};
} // namespace fsr4runtime

namespace fsr4runtime {
namespace fs = std::filesystem;

using fsr4_411::Buffer;
using fsr4_411::check;
using fsr4_411::computeBarrier;
using fsr4_411::createBuffer;
using fsr4_411::createImage;
using fsr4_411::destroyBuffer;
using fsr4_411::destroyImage;
using fsr4_411::deviceAddress;
using fsr4_411::Image;
using fsr4_411::imageBarrier;
using fsr4_411::kConstantBufSize;
using fsr4_411::kDescriptorCount;
using fsr4_411::kNumDescriptorSets;
using fsr4_411::OptimizedConstants;
using fsr4_411::readBytes;
using fsr4_411::readModelSpirv;
using fsr4_411::readSpirv;

struct ModelRuntimeOptions {
    fs::path rcas_shader_root;
    fs::path color_shader_root;
    bool use_wmma = false;
    bool use_int8_dispatch = false;
    bool numbered_shader_files = false;
    bool execute_padding_passes = false;
    bool direct_display_motion_vectors = false;

    fs::path ultraperformance_shader_dir;
    fs::path ultraperformance_model_data_path;

    ColorEndpointOptions color_endpoints;

    ModelPassPolicy pass_policy;
    ShaderRecipe shader_recipe;
};

class QualityContext final : public fsr4_411::QualityContext {
    VkPhysicalDevice physical_device;
    VkDevice device;
    fsr4vk::DeviceDispatch dispatch;

    PFN_vkGetDescriptorEXT vkGetDescriptorEXT = nullptr;
    PFN_vkCmdBindDescriptorBuffersEXT vkCmdBindDescriptorBuffersEXT = nullptr;
    PFN_vkCmdSetDescriptorBufferOffsetsEXT vkCmdSetDescriptorBufferOffsetsEXT = nullptr;
    PFN_vkGetDescriptorSetLayoutSizeEXT vkGetDescriptorSetLayoutSizeEXT = nullptr;
    PFN_vkGetDescriptorSetLayoutBindingOffsetEXT vkGetDescriptorSetLayoutBindingOffsetEXT = nullptr;

    VkDeviceSize descriptor_capacity = 0;
    std::array<VkDeviceSize, kNumDescriptorSets> set_strides{};
    std::array<VkDeviceSize, kNumDescriptorSets> set_binding_offsets{};

    VkSampler sampler{};
    VkPipelineLayout pipeline_layout{};
    std::array<VkDescriptorSetLayout, kNumDescriptorSets> set_layouts{};

    // Only the selected shaders own pipelines; inactive variants are retired.
    std::vector<VkPipeline> pipelines;
    std::vector<fs::path> active_shader_paths;
    std::array<std::vector<fs::path>, 2> model_shader_paths;
    unsigned model_pipeline_variant = 2;
    fs::path auto_exposure_shader_path;
    VkPipeline auto_exposure_pipeline{};
    VkPipeline overlay_pipeline{};
    bool overlay_pipeline_blend = false;

    struct PendingFrameUse {
        VkEvent completion{};
        std::vector<VkPipeline> pipelines;
        VkImage staging_output{};
    };

    std::vector<PendingFrameUse> pending_frame_uses;
    std::vector<VkEvent> available_frame_events;
    std::vector<VkPipeline> retired_pipelines;
    std::vector<Image> retired_staging_outputs;
    VkFormat overlay_read_format = VK_FORMAT_UNDEFINED;
    bool overlay_read_supported = false;
    bool device_is_rdna2 = false;
    uint32_t default_subgroup_size = 0;
    std::string automatic_wave_label;
    std::string model_wave_label;
    Buffer overlay_font{};
    VkDeviceAddress overlay_font_address{};

    struct RcasPipeline {
        VkPipeline handle{};
        VkExtent2D output_tile{4, 64};
    };

    RcasPipeline rcas_pipeline{};
    unsigned rcas_pipeline_variant = 10;
    ColorSpace color_space = ColorSpace::Linear;
    fs::path color_shader_root;
    ColorEndpointOptions color_endpoints;
    uint32_t color_api_version = VK_API_VERSION_1_3;
    std::array<bool, 2> color_wave32{};

    Image rcas_input{};
    bool rcas_input_initialized = false;
    uint32_t max_output_width = 0;
    uint32_t max_output_height = 0;
    fs::path rcas_shader_root;

    bool use_wmma = false;
    bool use_int8_dispatch = false;
    ModelPassPolicy model_pass_policy;
    bool numbered_shader_files = false;
    bool execute_padding_passes = false;
    bool direct_display_motion_vectors = false;
    bool auto_exposure_enabled = false;
    bool extensive_logging = false;
    bool depth_inverted_enabled = false;

    uint32_t max_render_width = 0;
    uint32_t max_render_height = 0;
    uint32_t model_bucket_width = 1920;
    uint32_t model_bucket_height = 1080;

    struct DescriptorSize {
        VkDeviceSize sampled_image;
        VkDeviceSize storage_image;
        VkDeviceSize sampler;
        VkDeviceSize storage_buffer;
        VkDeviceSize uniform_buffer;
    } descriptor_size{};

    Buffer scratch_buffer{};
    Buffer weight_buffer{};
    Buffer ultraperformance_weight_buffer{};

    std::string current_preset;

    std::vector<Image> images;
    Image staging_output{};

    // Cache exact-size color copies by active render size and format.
    struct AutoExposureColorEntry {
        Image image{};
        bool initialized = false;
        uint64_t last_used_frame = 0;
    };

    std::vector<AutoExposureColorEntry> auto_exposure_color_cache;

    // Recently used entries may exceed this cache target.
    static constexpr size_t kAutoExposureColorCacheSoftLimit = 8;

    Image fallback_depth{};
    bool fallback_depth_initialized = false;

    Image render_motion{};
    bool render_motion_initialized = false;

    bool initialized = false;
    float previous_exposure = 1.0f;

    ModelConstants cached_model_constants{};
    uint32_t cached_model_width = 0;
    uint32_t cached_model_height = 0;

    uint64_t frame_count = 0;
    double total_record_ms = 0;
    double min_record_ms = 1e9;
    double max_record_ms = 0;

    bool live_capture_enabled = false;
    bool live_capture_recording = false;
    bool live_capture_pending = false;
    VkEvent live_capture_completion{};
    bool live_capture_done = false;
    bool live_capture_run_auto_exposure = false;
    bool live_capture_use_staging_output = false;

    uint64_t live_capture_target_frame = 0;

    fs::path live_capture_trigger;
    fs::path live_capture_root;
    fs::path live_capture_frame_dir;

    static constexpr uint32_t kProfileFrameStart = 0;
    static constexpr uint32_t kProfileAfterExternalInputs = 1;
    static constexpr uint32_t kProfileAfterExposure = 2;
    static constexpr uint32_t kProfileAfterRecordFrame = 3;
    static constexpr uint32_t kProfileAfterStagingCopy = 4;
    static constexpr uint32_t kProfileFrameEnd = 5;

    static constexpr uint32_t kProfilePassBase = 6;
    static constexpr uint32_t kProfilePassStride = 3;
    static constexpr uint32_t kProfileQueryCount =
        kProfilePassBase + kProfilePassStride * kPasses.size();

    bool gpu_profile_enabled = false;
    bool gpu_profile_summary_only = false;
    uint32_t gpu_profile_interval = 60;
    double gpu_timestamp_period_ns = 0.0;

    struct Frame {
        VkPipeline rcas_pipeline{};
        VkExtent2D rcas_dispatch{};
        Buffer constants{};
        Buffer model_constants{};
        Buffer exposure_upload{};

        std::vector<VkImageView> views;
        Buffer descriptor_buffer{};
        VkBuffer model_weights{};

        VkDeviceAddress constants_address = 0;
        VkDeviceAddress descriptor_buffer_address = 0;

        VkQueryPool profile_queries = VK_NULL_HANDLE;
        uint64_t profile_frame_id = 0;
        bool profile_pending = false;
    };

    struct LiveCaptureItem {
        std::string relative_path;
        Buffer staging{};
        VkDeviceSize bytes = 0;
    };

    std::vector<LiveCaptureItem> live_capture_items;

    std::array<Frame, 64> frames{};
    size_t next_frame = 0;

    static void log(const std::string& text, fsr4log::Level level = fsr4log::Level::Info) noexcept;

    double profileMs(uint64_t begin, uint64_t end) const;
    bool collectGpuProfile(Frame& frame);
    bool beginGpuProfile(VkCommandBuffer command_buffer, Frame& frame);
    void
    profileTimestamp(VkCommandBuffer command_buffer, Frame& frame, bool active, uint32_t query);

    static uint32_t bytesPerPixel(VkFormat format);
    static const char* liveCapturePointName(CapturePoint point);
    void liveCaptureWrite(const fs::path& path, const void* data, size_t bytes) noexcept;
    void liveCaptureImageMeta(const fs::path& path,
                              const char* role,
                              const Image& image,
                              VkImageLayout layout,
                              uint32_t bpp,
                              VkDeviceSize bytes) noexcept;
    void liveCaptureManifest(const Image& color,
                             const Image& depth,
                             const Image& motion,
                             const Image& exposure,
                             const Image& output,
                             const OptimizedConstants& constants,
                             bool run_auto_exposure,
                             bool use_staging_output) noexcept;
    void liveCaptureImage(VkCommandBuffer command_buffer,
                          CapturePoint point,
                          const char* role,
                          const Image& image,
                          VkImageLayout layout);
    void liveCaptureBuffer(VkCommandBuffer command_buffer,
                           CapturePoint point,
                           const char* role,
                           const Buffer& buffer);
    void liveCaptureFlush() noexcept;
    void captureAtStage(CapturePoint point, VkCommandBuffer command_buffer);
    static void liveCaptureHook(void* opaque, CapturePoint point, VkCommandBuffer command_buffer);
    void release();

public:
    QualityContext(VkPhysicalDevice physical_device_,
                   VkDevice device_,
                   const fs::path& shader_dir,
                   const fs::path& weights_path,
                   const fs::path& auto_exposure_shader_path,
                   PFN_vkGetDeviceProcAddr get_device_proc_addr_,
                   ModelRuntimeOptions runtime_options,
                   bool auto_exposure_enabled_,
                   bool depth_inverted_enabled_,
                   bool extensive_logging_,
                   uint32_t max_input_width,
                   uint32_t max_input_height,
                   uint32_t max_output_width,
                   uint32_t max_output_height,
                   uint32_t api_version = VK_API_VERSION_1_3);
    ~QualityContext() override;

    void recordFrame(VkCommandBuffer command_buffer,
                     Image color,
                     Image depth,
                     Image motion,
                     Image exposure_image,
                     Image optional_mask,
                     Image output,
                     OptimizedConstants main_constants,
                     bool use_staging_output = false,
                     ColorSpace color_space = ColorSpace::Linear,
                     fsr4_411::OverlayOptions overlay = {}) override;

private:
    std::span<const VkPipeline> prepareColorSpacePipelines(ColorSpace requested,
                                                           bool debug_view = false);
    Image resampleDisplayMotionVectors(VkCommandBuffer command_buffer,
                                       const Image& source,
                                       uint32_t target_width,
                                       uint32_t target_height);
    Image resolveDepthInput(VkCommandBuffer command_buffer,
                            Image depth,
                            uint32_t active_width,
                            uint32_t active_height);
    Image prepareExactAutoExposureColor(VkCommandBuffer command_buffer,
                                        const Image& source,
                                        uint32_t active_width,
                                        uint32_t active_height);

    struct ShaderDriverPolicy {
        bool windows_amd_shaders_allowed;
        bool rdna2_device;
    };

    struct ModelPipelinePlan {
        bool windows_fp8_optimized = false;
        bool windows_int8_optimized = false;
        std::array<bool, kPasses.size()> require_wave32{};
    };

    ModelPipelinePlan model_pipeline_plan{};

    ModelPipelinePlan prepareFp8Pipelines(const fs::path& shader_dir,
                                          const ShaderDriverPolicy& policy);
    ModelPipelinePlan prepareInt8Pipelines(const ModelRuntimeOptions& options,
                                           const ShaderDriverPolicy& policy);
    void prepareModelWeights(Frame& frame);
    void writeExposureDescriptors(Frame& frame, const Image& exposure);
    void writeStaticExposureDescriptor(Frame& frame);

    struct DescriptorGeometry {
        VkDeviceSize alignment;
        VkDeviceSize max_layout_size;
    };

    ShaderDriverPolicy configureDeviceShaderPolicy();
    void configureLiveCapture();
    void configureGpuProfile();
    DescriptorGeometry createDescriptorLayouts(PFN_vkGetDeviceProcAddr get_device_proc_addr_);
    void configureModelPipelines(const fs::path& shader_dir,
                                 const fs::path& weights_path,
                                 const ModelRuntimeOptions& runtime_options,
                                 uint32_t api_version,
                                 const ShaderDriverPolicy& shader_policy);
    void prepareModelPipelines();
    VkPipeline createComputePipeline(const std::vector<uint32_t>& code, bool wave32);
    void retirePipeline(VkPipeline& pipeline);
    void collectFrameUses();
    void beginFrameUse();
    void trackPipelineUse(VkPipeline pipeline);
    void prepareRcas(VkCommandBuffer command_buffer,
                     Frame& frame,
                     const Image& output,
                     const Image& exposure,
                     bool use_staging_output);
    void recordRcas(VkCommandBuffer command_buffer,
                    Frame& frame,
                    const Image& output,
                    const OptimizedConstants& constants);
    void recordOverlay(VkCommandBuffer command_buffer,
                       Frame& frame,
                       const Image& output,
                       const Image& motion,
                       const OptimizedConstants& constants,
                       const fsr4_411::OverlayOptions& options);
    void createAutoExposurePipeline(const fs::path& auto_exposure_shader_path);
    void allocateModelBuffers(const fs::path& weights_path,
                              const ModelRuntimeOptions& runtime_options,
                              uint32_t max_input_width,
                              uint32_t max_output_width,
                              uint32_t max_output_height);
    void allocateFrameResources(const DescriptorGeometry& geometry);
    void allocateInternalImages(uint32_t max_input_width,
                                uint32_t max_input_height,
                                uint32_t max_output_width,
                                uint32_t max_output_height);

    void selectModelPreset(uint32_t render_width, OptimizedConstants& main_constants);
    void
    writeFrameConstants(Frame& frame, OptimizedConstants& main_constants, bool run_auto_exposure);
    void initializeFrameHistory(VkCommandBuffer command_buffer,
                                Frame& frame,
                                const OptimizedConstants& main_constants);
    void recordAutoExposure(VkCommandBuffer command_buffer,
                            const Frame& frame,
                            const OptimizedConstants& main_constants);
    void writeAutoExposureConstants(Frame& frame, const OptimizedConstants& main_constants);
    void prepareStagingOutput(VkCommandBuffer command_buffer,
                              const Image& output,
                              bool use_staging_output);
    void
    copyStagingOutput(VkCommandBuffer command_buffer, const Image& output, bool use_staging_output);

    void prepareExternalViews(Frame& frame,
                              Image& color,
                              Image& depth,
                              Image& motion,
                              Image& exposure,
                              Image& optional_mask,
                              Image& output);
    bool updateDynamicDescriptors(VkCommandBuffer command_buffer,
                                  Frame& frame,
                                  Image& color,
                                  Image& depth,
                                  Image& motion,
                                  Image& exposure_image,
                                  Image& optional_mask,
                                  Image& output,
                                  Image& reprojected,
                                  bool use_staging_output);
    void transitionExternalInputs(VkCommandBuffer command_buffer,
                                  const Image& color,
                                  const Image& depth,
                                  const Image& motion,
                                  const Image& exposure,
                                  const Image& optional_mask,
                                  const Image& output);

    void writeImageDescriptorBuffer(
        Frame& frame,
        uint32_t set_index,
        uint32_t binding_index,
        VkDescriptorType type,
        VkImageView view,
        VkImageLayout image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    uint8_t* descriptorAddress(Frame& frame, uint32_t set_index, uint32_t binding_index);
    void writeSampledImageDescriptor(
        Frame& frame,
        uint32_t set_index,
        uint32_t binding_index,
        VkImageView view,
        VkImageLayout image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    void writeStorageImageDescriptor(Frame& frame,
                                     uint32_t set_index,
                                     uint32_t binding_index,
                                     VkImageView view);
    void writeSamplerDescriptor(Frame& frame,
                                uint32_t set_index,
                                uint32_t binding_index,
                                VkSampler sampler_handle);
    void writeStorageBufferDescriptor(Frame& frame,
                                      uint32_t set_index,
                                      uint32_t binding_index,
                                      VkBuffer buffer,
                                      VkDeviceSize size);
    void fillStorageBufferDescriptors(Frame& frame,
                                      uint32_t set_index,
                                      VkBuffer buffer,
                                      VkDeviceSize size);
    void writeUniformBufferDescriptor(Frame& frame,
                                      uint32_t set_index,
                                      uint32_t binding_index,
                                      VkBuffer buffer,
                                      VkDeviceSize size);
    void writeWeightDescriptors(Frame& frame, const Buffer& model_data);

    void writeStaticDescriptors(Frame& frame);
    void writeAllStaticDescriptors();
};

} // namespace fsr4runtime
