// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

#include "runtime/context.hpp"
#include "assets/assets.hpp"
#include "diagnostics/diagnostics.hpp"

#include <cstdio>

namespace fsr4runtime {

void QualityContext::release() {
    if (!device) {
        return;
    }

    vkDeviceWaitIdle(device);

    if (live_capture_pending) {
        liveCaptureFlush();
    }

    for (auto& item : live_capture_items) {
        destroyBuffer(device, item.staging);
    }

    live_capture_items.clear();
    if (live_capture_completion) {
        vkDestroyEvent(device, live_capture_completion, nullptr);
        live_capture_completion = VK_NULL_HANDLE;
    }

    for (auto& frame : frames) {
        for (auto image_view : frame.views) {
            vkDestroyImageView(device, image_view, nullptr);
        }

        destroyBuffer(device, frame.constants);
        destroyBuffer(device, frame.model_constants);
        destroyBuffer(device, frame.exposure_upload);
        destroyBuffer(device, frame.descriptor_buffer);

        if (frame.profile_queries) {
            vkDestroyQueryPool(device, frame.profile_queries, nullptr);
        }
    }

    destroyImage(device, rcas_input);
    if (rcas_pipeline.handle) {
        vkDestroyPipeline(device, rcas_pipeline.handle, nullptr);
    }

    if (staging_output.image) {
        destroyImage(device, staging_output);
    }

    for (auto& image : retired_staging_outputs) {
        destroyImage(device, image);
    }

    for (auto& entry : auto_exposure_color_cache) {
        if (entry.image.image) {
            destroyImage(device, entry.image);
        }
    }

    auto_exposure_color_cache.clear();

    if (fallback_depth.image) {
        destroyImage(device, fallback_depth);
    }

    if (render_motion.image) {
        destroyImage(device, render_motion);
    }

    for (auto& image : images) {
        destroyImage(device, image);
    }

    destroyBuffer(device, scratch_buffer);
    destroyBuffer(device, weight_buffer);
    destroyBuffer(device, ultraperformance_weight_buffer);

    for (auto pipeline : pipelines) {
        if (pipeline) {
            vkDestroyPipeline(device, pipeline, nullptr);
        }
    }

    for (auto pipeline : retired_pipelines) {
        vkDestroyPipeline(device, pipeline, nullptr);
    }
    for (const auto& use : pending_frame_uses) {
        vkDestroyEvent(device, use.completion, nullptr);
    }
    for (auto event : available_frame_events) {
        vkDestroyEvent(device, event, nullptr);
    }
    if (overlay_pipeline) {
        vkDestroyPipeline(device, overlay_pipeline, nullptr);
    }
    destroyBuffer(device, overlay_font);

    if (auto_exposure_pipeline) {
        vkDestroyPipeline(device, auto_exposure_pipeline, nullptr);
    }

    if (pipeline_layout) {
        vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    }

    for (auto& layout : set_layouts) {
        if (layout) {
            vkDestroyDescriptorSetLayout(device, layout, nullptr);
        }
    }

    if (sampler) {
        vkDestroySampler(device, sampler, nullptr);
    }
}

QualityContext::QualityContext(VkPhysicalDevice physical_device_,
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
                               uint32_t api_version)
    : physical_device(physical_device_), device(device_), use_wmma(runtime_options.use_wmma),
      use_int8_dispatch(runtime_options.use_int8_dispatch),
      model_pass_policy(runtime_options.pass_policy),
      numbered_shader_files(runtime_options.numbered_shader_files),
      execute_padding_passes(runtime_options.execute_padding_passes),
      direct_display_motion_vectors(runtime_options.direct_display_motion_vectors),
      auto_exposure_enabled(auto_exposure_enabled_), extensive_logging(extensive_logging_),
      depth_inverted_enabled(depth_inverted_enabled_), max_render_width(max_input_width),
      max_render_height(max_input_height),
      model_bucket_width(fsr4::modelBucket(max_output_width, max_output_height)[0]),
      model_bucket_height(fsr4::modelBucket(max_output_width, max_output_height)[1]) {
    try {
        dispatch.load(device, get_device_proc_addr_, api_version);
        this->auto_exposure_shader_path = auto_exposure_shader_path;
        color_shader_root = runtime_options.color_shader_root;
        color_endpoints = runtime_options.color_endpoints;
        color_api_version = api_version;
        this->max_output_width = max_output_width;
        this->max_output_height = max_output_height;
        rcas_shader_root = runtime_options.rcas_shader_root;

        if (direct_display_motion_vectors) {
            log("[motion] direct display-resolution prepass sampling enabled");
        }

        if (model_pass_policy.passes.empty() || model_pass_policy.passes.size() > kPasses.size() ||
            !model_pass_policy.make_constants) {
            throw std::invalid_argument("unsupported model pass/constant layout");
        }
        const auto shader_policy = configureDeviceShaderPolicy();
        configureLiveCapture();
        configureGpuProfile();

        const auto descriptor_geometry = createDescriptorLayouts(get_device_proc_addr_);
        configureModelPipelines(
            shader_dir, weights_path, runtime_options, api_version, shader_policy);

        allocateModelBuffers(
            weights_path, runtime_options, max_input_width, max_output_width, max_output_height);
        allocateFrameResources(descriptor_geometry);
        allocateInternalImages(
            max_input_width, max_input_height, max_output_width, max_output_height);

        writeAllStaticDescriptors();

    } catch (...) {
        release();
        throw;
    }
}

QualityContext::~QualityContext() {
    release();
}

void QualityContext::allocateModelBuffers(const fs::path& weights_path,
                                          const ModelRuntimeOptions& runtime_options,
                                          uint32_t max_input_width,
                                          uint32_t max_output_width,
                                          uint32_t max_output_height) {
    scratch_buffer = createBuffer(
        physical_device,
        device,
        fsr4::scratchCapacity(max_output_width, max_output_height),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        true);

    const auto weight_bytes = readBytes(weights_path);
    weight_buffer = createBuffer(
        physical_device,
        device,
        std::max<VkDeviceSize>(weight_bytes.size(), 256),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        true,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    std::memcpy(weight_buffer.mapped, weight_bytes.data(), weight_bytes.size());

    if (!runtime_options.ultraperformance_model_data_path.empty()) {
        const auto ultraperformance_weight_bytes =
            readBytes(runtime_options.ultraperformance_model_data_path);
        ultraperformance_weight_buffer = createBuffer(
            physical_device,
            device,
            std::max<VkDeviceSize>(ultraperformance_weight_bytes.size(), 256),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            true,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        std::memcpy(ultraperformance_weight_buffer.mapped,
                    ultraperformance_weight_bytes.data(),
                    ultraperformance_weight_bytes.size());
    }

    const auto log_memory = [&](const char* role, const Buffer& buffer) {
        std::ostringstream message;
        message << "[memory] role=" << role << " size_bytes=" << buffer.size
                << " memory_type=" << buffer.memory_type_index
                << " heap=" << buffer.memory_heap_index
                << " properties=" << fsr4log::memoryProperties(buffer.memory_property_flags);
        log(message.str());
    };
    log_memory("scratch", scratch_buffer);
    log_memory("weights", weight_buffer);

    if (ultraperformance_weight_buffer.buffer) {
        log_memory("ultraperf-weights", ultraperformance_weight_buffer);
    }

    current_preset = fsr4::resolutionPreset(max_input_width, max_output_width);
    log("[model] initial_preset=" + current_preset + " weights_asset='" + weights_path.string() +
        "'");
}

void QualityContext::allocateFrameResources(const DescriptorGeometry& geometry) {
    const auto descriptor_offset_alignment = geometry.alignment;
    const auto max_layout_size = geometry.max_layout_size;

    if (!descriptor_offset_alignment) {
        throw std::runtime_error("invalid descriptor-buffer alignment");
    }

    descriptor_capacity =
        ((max_layout_size + descriptor_offset_alignment - 1) / descriptor_offset_alignment) *
        descriptor_offset_alignment;
    log("[descriptor] max_layout_bytes=" + std::to_string(max_layout_size) +
        " alignment_bytes=" + std::to_string(descriptor_offset_alignment) +
        " capacity_bytes_per_set=" + std::to_string(descriptor_capacity));

    for (auto& frame : frames) {
        frame.constants = createBuffer(
            physical_device,
            device,
            kConstantBufSize,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            true);
        frame.model_constants = createBuffer(
            physical_device,
            device,
            kConstantBufSize,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            true);
        frame.exposure_upload = createBuffer(
            physical_device,
            device,
            8,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        frame.descriptor_buffer =
            createBuffer(physical_device,
                         device,
                         descriptor_capacity * kNumDescriptorSets,
                         VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT |
                             VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT |
                             VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         true);
        std::memset(frame.descriptor_buffer.mapped,
                    0,
                    static_cast<std::size_t>(frame.descriptor_buffer.size));

        frame.constants_address =
            deviceAddress(dispatch.get_buffer_device_address, device, frame.constants.buffer);
        frame.descriptor_buffer_address = deviceAddress(
            dispatch.get_buffer_device_address, device, frame.descriptor_buffer.buffer);

        if (gpu_profile_enabled) {
            VkQueryPoolCreateInfo query_info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            query_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
            query_info.queryCount = kProfileQueryCount;
            check(vkCreateQueryPool(device, &query_info, nullptr, &frame.profile_queries),
                  "vkCreateQueryPool(FSR4 GPU profile)");
        }
    }

    log("[frame] resources allocated slots=" + std::to_string(frames.size()) +
        " constants_bytes_per_slot=" + std::to_string(frames.front().constants.size) +
        " descriptor_bytes_per_slot=" + std::to_string(frames.front().descriptor_buffer.size) +
        " profile_queries_per_slot=" +
        std::to_string(gpu_profile_enabled ? kProfileQueryCount : 0));

    if (extensive_logging) {
        const auto log_device_address_range = [&](const char* name, const Buffer& buffer) {
            if (!buffer.buffer) {
                return;
            }

            const VkDeviceAddress address =
                deviceAddress(dispatch.get_buffer_device_address, device, buffer.buffer);
            std::ostringstream message;
            message << "[address] " << name << "=[0x" << std::hex << address << ",0x"
                    << (address + buffer.size) << ")" << std::dec << " bytes=" << buffer.size;
            log(message.str(), fsr4log::Level::Debug);
        };
        log_device_address_range("scratch", scratch_buffer);
        log_device_address_range("weights", weight_buffer);
        log_device_address_range("ultraperf-weights", ultraperformance_weight_buffer);

        for (size_t frame_index = 0; frame_index < frames.size(); ++frame_index) {
            const auto& frame = frames[frame_index];
            std::ostringstream message;
            message << "[address] frame_slot=" << frame_index << " constants=[0x" << std::hex
                    << frame.constants_address << ",0x"
                    << (frame.constants_address + frame.constants.size) << ") descriptor=[0x"
                    << frame.descriptor_buffer_address << ",0x"
                    << (frame.descriptor_buffer_address + frame.descriptor_buffer.size) << ")"
                    << std::dec;
            log(message.str(), fsr4log::Level::Debug);
        }
    }
}

void QualityContext::allocateInternalImages(uint32_t max_input_width,
                                            uint32_t max_input_height,
                                            uint32_t max_output_width,
                                            uint32_t max_output_height) {
    const auto padded_output_width = fsr4::roundUp(max_output_width, 8);
    const auto padded_output_height = fsr4::roundUp(max_output_height, 8);
    const auto image_usage_flags = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                                   VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                   VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    images.resize(auto_exposure_enabled ? image_slot::exposure_mips + 1
                                        : image_slot::fallback_mask + 1);
    images[image_slot::history] = createImage(physical_device,
                                              device,
                                              padded_output_width,
                                              padded_output_height,
                                              VK_FORMAT_R16G16B16A16_SFLOAT,
                                              image_usage_flags);
    images[image_slot::reprojected] = createImage(physical_device,
                                                  device,
                                                  padded_output_width,
                                                  padded_output_height,
                                                  VK_FORMAT_R16G16B16A16_SFLOAT,
                                                  image_usage_flags);
    images[image_slot::recurrent] = createImage(physical_device,
                                                device,
                                                padded_output_width,
                                                padded_output_height,
                                                VK_FORMAT_R8G8B8A8_UNORM,
                                                image_usage_flags);
    images[image_slot::wmma_scratch0] = createImage(physical_device,
                                                    device,
                                                    padded_output_width,
                                                    padded_output_height,
                                                    VK_FORMAT_R16G16B16A16_SFLOAT,
                                                    image_usage_flags);
    images[image_slot::wmma_scratch1] = createImage(physical_device,
                                                    device,
                                                    padded_output_width,
                                                    padded_output_height,
                                                    VK_FORMAT_R16G16B16A16_SFLOAT,
                                                    image_usage_flags);
    // Prepass reads current and previous exposure from this 2x1 image.
    images[image_slot::exposure] =
        createImage(physical_device, device, 2, 1, VK_FORMAT_R32_SFLOAT, image_usage_flags);
    images[image_slot::fallback_mask] =
        createImage(physical_device, device, 1, 1, VK_FORMAT_R8_UNORM, image_usage_flags);

    if (auto_exposure_enabled) {
        images[image_slot::exposure_counter] =
            createImage(physical_device, device, 1, 1, VK_FORMAT_R32_UINT, image_usage_flags);
        images[image_slot::exposure_mips] = createImage(physical_device,
                                                        device,
                                                        fsr4::divideUp(max_input_width, 16),
                                                        fsr4::divideUp(max_input_height, 16),
                                                        VK_FORMAT_R32_SFLOAT,
                                                        image_usage_flags);
    }

    fsr4vk::setDebugName(dispatch,
                         device,
                         VK_OBJECT_TYPE_BUFFER,
                         (uint64_t)scratch_buffer.buffer,
                         "FSR4_ScratchBuffer");
    fsr4vk::setDebugName(dispatch,
                         device,
                         VK_OBJECT_TYPE_BUFFER,
                         (uint64_t)weight_buffer.buffer,
                         "FSR4_WeightBuffer");
    fsr4vk::setDebugName(dispatch,
                         device,
                         VK_OBJECT_TYPE_IMAGE,
                         (uint64_t)images[image_slot::history].image,
                         "FSR4_HistoryImage");
    fsr4vk::setDebugName(dispatch,
                         device,
                         VK_OBJECT_TYPE_IMAGE,
                         (uint64_t)images[image_slot::reprojected].image,
                         "FSR4_ReprojectedImage");
    fsr4vk::setDebugName(dispatch,
                         device,
                         VK_OBJECT_TYPE_IMAGE,
                         (uint64_t)images[image_slot::recurrent].image,
                         "FSR4_RecurrentImage");

    if (auto_exposure_enabled) {
        fsr4vk::setDebugName(dispatch,
                             device,
                             VK_OBJECT_TYPE_IMAGE,
                             (uint64_t)images[image_slot::exposure].image,
                             "FSR4_AutoExposure");
        fsr4vk::setDebugName(dispatch,
                             device,
                             VK_OBJECT_TYPE_IMAGE,
                             (uint64_t)images[image_slot::exposure_counter].image,
                             "FSR4_AutoExposureAtomic");
        fsr4vk::setDebugName(dispatch,
                             device,
                             VK_OBJECT_TYPE_IMAGE,
                             (uint64_t)images[image_slot::exposure_mips].image,
                             "FSR4_AutoExposureMips");
    }
}

void QualityContext::log(const std::string& text, fsr4log::Level level) noexcept {
    fsr4runtime::log(text, level);
}

} // namespace fsr4runtime

namespace fsr4runtime {
namespace fs = std::filesystem;

fs::path ModelShaderSelector::stockShaderPath(const fs::path& directory,
                                              uint32_t pass_index,
                                              const fs::path& prepass_override,
                                              const fs::path& postpass_override) const {
    const auto pass_name = kPasses[pass_index].name;
    char numbered_name[32]{};
    std::snprintf(numbered_name, sizeof(numbered_name), "pass-%02u.spv", pass_index);

    std::string filename =
        options.numbered_shader_files ? numbered_name : std::string(pass_name) + ".spv";

    fs::path shader_path = directory / filename;

    if (pass_index == kPrepassIndex && !prepass_override.empty()) {
        shader_path = prepass_override;
    }

    if (options.numbered_shader_files && pass_index == kPostpassIndex &&
        !postpass_override.empty()) {
        shader_path = postpass_override;
    }

    return shader_path;
}

fs::path ModelShaderSelector::select(const fs::path& directory,
                                     uint32_t pass_index,
                                     const fs::path& prepass_override,
                                     const fs::path& postpass_override) {
    return options.use_int8_dispatch
               ? selectInt8Shader(directory, pass_index, prepass_override, postpass_override)
               : selectFp8Shader(directory, pass_index, prepass_override, postpass_override);
}
} // namespace fsr4runtime

namespace fsr4runtime {

QualityContext::ShaderDriverPolicy QualityContext::configureDeviceShaderPolicy() {
    VkPhysicalDeviceDriverProperties driver_properties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceSubgroupProperties subgroup_properties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    driver_properties.pNext = &subgroup_properties;
    VkPhysicalDeviceProperties2 device_properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    device_properties.pNext = &driver_properties;
    vkGetPhysicalDeviceProperties2(physical_device, &device_properties);

    const bool amdvlk_or_proprietary_driver = fsr4vk::isAmdvlkOrProprietaryDriver(
        device_properties.properties.vendorID, driver_properties.driverID);

    bool force_windows_amd_shaders = false;

    if (const char* value = std::getenv("FSR4_VK_FORCE_WINDOWS_AMD_SHADERS")) {
        force_windows_amd_shaders = std::strcmp(value, "0") != 0;
    }
#if defined(FSR4_NATIVE_ASSETS_ONLY)
    constexpr bool windows_shader_assets_available = false;
#else
    constexpr bool windows_shader_assets_available = true;
#endif
    const bool windows_amd_shaders_allowed =
        fsr4vk::allowWindowsAmdShaders(device_properties.properties.vendorID,
                                       driver_properties.driverID,
                                       force_windows_amd_shaders,
                                       windows_shader_assets_available);
    const bool rdna2_device = fsr4vk::isAmdRdna2Device(device_properties.properties.vendorID,
                                                       device_properties.properties.deviceID);
    device_is_rdna2 = rdna2_device;
    default_subgroup_size = subgroup_properties.subgroupSize;
    uint32_t minimum = default_subgroup_size, maximum = default_subgroup_size;
    if (device_properties.properties.apiVersion >= VK_API_VERSION_1_3) {
        VkPhysicalDeviceSubgroupSizeControlProperties sizes{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
        VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                               &sizes};
        vkGetPhysicalDeviceProperties2(physical_device, &properties);
        minimum = sizes.minSubgroupSize;
        maximum = sizes.maxSubgroupSize;
    }
    automatic_wave_label = "Wave" + std::to_string(minimum);
    if (minimum != maximum) {
        automatic_wave_label += (maximum == minimum * 2 ? "/" : "-") + std::to_string(maximum);
    }
    automatic_wave_label += " (auto)";

    std::ostringstream policy_message;
    policy_message << "[device-policy] device="
                   << std::quoted(device_properties.properties.deviceName)
                   << " vendor_id=" << fsr4log::hexValue(device_properties.properties.vendorID)
                   << " device_id=" << fsr4log::hexValue(device_properties.properties.deviceID)
                   << " driver_id=" << driver_properties.driverID
                   << " driver_name=" << std::quoted(driver_properties.driverName)
                   << " driver_info=" << std::quoted(driver_properties.driverInfo)
                   << " amd_proprietary=" << (amdvlk_or_proprietary_driver ? "yes" : "no")
                   << " rdna2=" << (rdna2_device ? "yes" : "no") << " windows_shaders_available="
                   << (windows_shader_assets_available ? "yes" : "no")
                   << " windows_shaders_allowed=" << (windows_amd_shaders_allowed ? "yes" : "no")
                   << " force_windows_shaders=" << (force_windows_amd_shaders ? "yes" : "no");
    log(policy_message.str());

    return {windows_amd_shaders_allowed, rdna2_device};
}

void QualityContext::configureModelPipelines(const fs::path& shader_dir,
                                             const fs::path& weights_path,
                                             const ModelRuntimeOptions& runtime_options,
                                             uint32_t api_version,
                                             const ShaderDriverPolicy& shader_policy) {
    const auto passes = model_pass_policy.passes;
    ModelPipelinePlan plan;
    if (use_int8_dispatch) {
        plan = prepareInt8Pipelines(runtime_options, shader_policy);
    } else {
        plan = prepareFp8Pipelines(shader_dir, shader_policy);
    }

    model_pipeline_plan = plan;
    color_wave32 = {plan.require_wave32.front(), plan.require_wave32[passes.size() - 1]};
    log(std::string("[shader-source] source=") + "original version=4.1.1 model=" +
        (model_pass_policy.model_name ? model_pass_policy.model_name
         : use_int8_dispatch          ? "int8"
                                      : "fp8"));

    ModelShaderSelector shader_selector(
        ShaderSelectionOptions{
            .use_wmma = use_wmma,
            .use_int8_dispatch = use_int8_dispatch,
            .numbered_shader_files = numbered_shader_files,
            .windows_fp8_optimized = plan.windows_fp8_optimized,
            .windows_int8_optimized = plan.windows_int8_optimized,
            .extensive_logging = extensive_logging,
        },
        runtime_options.shader_recipe);

    pipelines.resize(passes.size());
    active_shader_paths.resize(passes.size());
    model_shader_paths[0].resize(passes.size());
    if (!runtime_options.ultraperformance_shader_dir.empty()) {
        model_shader_paths[1].resize(passes.size());
    }
    for (uint32_t index = 0; index < passes.size(); ++index) {
        const auto type = passes[index].type;
        if (type == PassType::Prepass || type == PassType::Postpass ||
            (type == PassType::TensorPadding &&
             (!execute_padding_passes ||
              model_binding::paddingLayer(index) == model_binding::unused_padding_layer))) {
            continue;
        }
        const auto standard = shader_selector.select(shader_dir, index, {}, {});
        model_shader_paths[0][index] = standard;
        if (!model_shader_paths[1].empty()) {
            auto ultra =
                shader_selector.select(runtime_options.ultraperformance_shader_dir, index, {}, {});
            // Identical programs share their active handle across preset changes.
            if (ultra != standard &&
                readModelSpirv(ultra, api_version) == readModelSpirv(standard, api_version)) {
                ultra = standard;
            }
            model_shader_paths[1][index] = std::move(ultra);
        }
    }
}

} // namespace fsr4runtime

namespace fsr4runtime {
VkPipeline QualityContext::createComputePipeline(const std::vector<uint32_t>& code, bool wave32) {
    VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    module_info.codeSize = code.size() * sizeof(uint32_t);
    module_info.pCode = code.data();
    VkShaderModule module{};
    check(vkCreateShaderModule(device, &module_info, nullptr, &module), "compute shader module");
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    subgroup.requiredSubgroupSize = 32;
    VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.flags = VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
    info.layout = pipeline_layout;
    info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = module;
    info.stage.pName = "main";
    if (wave32) {
        info.stage.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
        info.stage.pNext = &subgroup;
    }
    VkPipeline pipeline{};
    const auto result =
        vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline);
    vkDestroyShaderModule(device, module, nullptr);
    if (result != VK_SUCCESS && pipeline) {
        vkDestroyPipeline(device, pipeline, nullptr);
    }
    check(result, "compute pipeline");
    return pipeline;
}

void QualityContext::prepareModelPipelines() {
    const unsigned variant = use_int8_dispatch && current_preset == "ultraperf" ? 1u : 0u;
    if (model_pipeline_variant == variant) {
        return;
    }
    const auto& paths = model_shader_paths[variant];
    if (paths.size() != pipelines.size()) {
        throw std::runtime_error("requested model pipeline set is unavailable");
    }
    const auto& plan = model_pipeline_plan;
    unsigned created = 0;
    for (uint32_t index = 0; index < paths.size(); ++index) {
        const auto type = model_pass_policy.passes[index].type;
        if (type == PassType::Prepass || type == PassType::Postpass ||
            (type == PassType::TensorPadding &&
             (!execute_padding_passes ||
              model_binding::paddingLayer(index) == model_binding::unused_padding_layer))) {
            continue;
        }
        if (pipelines[index] && active_shader_paths[index] == paths[index]) {
            continue;
        }
        const auto code = readModelSpirv(paths[index], color_api_version);
        const auto pipeline = createComputePipeline(code, plan.require_wave32[index]);
        retirePipeline(pipelines[index]);
        pipelines[index] = pipeline;
        active_shader_paths[index] = paths[index];
        if (type == PassType::ModelPass) {
            const auto wave = plan.require_wave32[index] ? std::string("Wave32")
                              : code[1] >= 0x00010600u
                                  ? automatic_wave_label
                                  : "Wave" + std::to_string(default_subgroup_size);
            if (model_wave_label.empty()) {
                model_wave_label = wave;
            } else if (model_wave_label != wave) {
                model_wave_label = "Wave mixed";
            }
        }
        ++created;
    }
    model_pipeline_variant = variant;
    log("[pipelines] model=" + std::string(variant ? "ultraperf" : "standard") +
        " created=" + std::to_string(created));
}

void QualityContext::retirePipeline(VkPipeline& pipeline) {
    if (pipeline) {
        retired_pipelines.push_back(pipeline);
        pipeline = VK_NULL_HANDLE;
    }
}

void QualityContext::collectFrameUses() {
    std::erase_if(pending_frame_uses, [&](const PendingFrameUse& use) {
        const auto status = vkGetEventStatus(device, use.completion);
        if (status == VK_EVENT_RESET) {
            return false;
        }
        if (status != VK_EVENT_SET) {
            check(status, "frame completion event");
        }
        available_frame_events.push_back(use.completion);
        return true;
    });
    std::erase_if(retired_pipelines, [&](VkPipeline pipeline) {
        for (const auto& use : pending_frame_uses) {
            if (std::find(use.pipelines.begin(), use.pipelines.end(), pipeline) !=
                use.pipelines.end()) {
                return false;
            }
        }
        vkDestroyPipeline(device, pipeline, nullptr);
        return true;
    });
    std::erase_if(retired_staging_outputs, [&](Image& image) {
        for (const auto& use : pending_frame_uses) {
            if (use.staging_output == image.image) {
                return false;
            }
        }
        destroyImage(device, image);
        return true;
    });
}

void QualityContext::beginFrameUse() {
    VkEvent event{};
    if (available_frame_events.empty()) {
        VkEventCreateInfo info{VK_STRUCTURE_TYPE_EVENT_CREATE_INFO};
        check(vkCreateEvent(device, &info, nullptr, &event), "frame completion event");
    } else {
        event = available_frame_events.back();
        check(vkResetEvent(device, event), "reset frame completion event");
        available_frame_events.pop_back();
    }
    pending_frame_uses.push_back({event, {}});
    pending_frame_uses.back().pipelines.reserve(pipelines.size() + 3);
}

void QualityContext::trackPipelineUse(VkPipeline pipeline) {
    if (pipeline) {
        pending_frame_uses.back().pipelines.push_back(pipeline);
    }
}
} // namespace fsr4runtime
