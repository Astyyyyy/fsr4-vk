// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

#include "provider/provider.hpp"
#include "runtime/rcas.hpp"

#if defined(__linux__)
#include "assets/assets.hpp"
#include <dlfcn.h>
#endif

namespace fsr4provider {
bool environmentFlag(const char* name) noexcept {
    const char* value = std::getenv(name);

    return value && std::strcmp(value, "0") != 0;
}

namespace {
std::optional<bool> overlayEnvironmentValue(const char* name) {
#if defined(_WIN32)
    char value[32]{};
    SetLastError(ERROR_SUCCESS);
    const DWORD length = GetEnvironmentVariableA(name, value, sizeof(value));
    if (length == 0 && GetLastError() == ERROR_ENVVAR_NOT_FOUND) {
        return std::nullopt;
    }
    return length != 0 && (length != 1 || value[0] != '0');
#else
    const char* value = std::getenv(name);
    if (!value) {
        return std::nullopt;
    }
    return *value && std::strcmp(value, "0") != 0;
#endif
}

bool overlayEnvironmentFlag(const char* name, bool fallback = false) {
    return overlayEnvironmentValue(name).value_or(fallback);
}

std::filesystem::path assetRoot() {
    if (const char* root = std::getenv("FSR4_VK_ASSET_ROOT")) {
        return root;
    }

#if defined(FSR4_EMBEDDED_ASSETS)
    return fsr4assets::root;
#endif
#if defined(_WIN32)
    HMODULE module{};
    wchar_t path[32768]{};

    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(&ffxCreateContext),
                           &module)) {
        const auto length = GetModuleFileNameW(module, path, 32768);

        if (length && length < 32768) {
            return std::filesystem::path(path).parent_path() / "fsr4-vulkan-assets";
        }
    }
#endif
#if defined(__linux__)
    Dl_info module{};
    // Use an internal symbol to locate this module, not another provider's export.
    if (dladdr(reinterpret_cast<const void*>(&assetRoot), &module) && module.dli_fname) {
        if (auto root = fsr4assets::adjacentRoot(std::filesystem::absolute(module.dli_fname))) {
            return *root;
        }
    }
#endif
    throw std::invalid_argument(
        "Cannot locate shader/model sidecars: place assets beside the provider "
        "or set FSR4_VK_ASSET_ROOT");
}

bool validAllocationCallbacks(const ffxAllocationCallbacks* callbacks) {
    return callbacks == nullptr || (callbacks->alloc != nullptr && callbacks->dealloc != nullptr);
}

void* allocate(const ffxAllocationCallbacks* callbacks, std::size_t size) {
    return callbacks ? callbacks->alloc(callbacks->pUserData, size) : std::malloc(size);
}

void deallocate(const ffxAllocationCallbacks* callbacks, void* memory) {
    if (callbacks) {
        callbacks->dealloc(callbacks->pUserData, memory);
    } else {
        std::free(memory);
    }
}

const CreateBackendVkDesc* findVulkanBackend(const ffxApiHeader* header) {
    for (auto* current = header; current; current = current->pNext) {
        if (current->type == kBackendVkDescType) {
            return reinterpret_cast<const CreateBackendVkDesc*>(current);
        }
    }

    return nullptr;
}

void validateDeviceEntryPoints(const ProviderContext& provider) {
    std::vector<std::string> missing;

    for (const char* groupValue : kRequiredDeviceFunctionGroups) {
        const std::string group(groupValue);
        bool found = false;
        std::size_t begin = 0;

        while (begin <= group.size()) {
            const std::size_t end = group.find('|', begin);
            const std::string name =
                group.substr(begin, end == std::string::npos ? std::string::npos : end - begin);

            if (!name.empty() && provider.getDeviceProcAddr(provider.device, name.c_str())) {
                found = true;
                break;
            }

            if (end == std::string::npos) {
                break;
            }

            begin = end + 1;
        }

        if (!found) {
            missing.push_back(group);
        }
    }

    if (!missing.empty()) {
        std::string message = "missing Vulkan device entry point groups: ";

        for (std::size_t i = 0; i < missing.size(); ++i) {
            if (i) {
                message += ", ";
            }

            message += missing[i];
        }

        throw std::runtime_error(message);
    }
}

} // namespace

ffxReturnCode_t createContext(ffxContext* context,
                              ffxCreateContextDescHeader* header,
                              const ffxAllocationCallbacks* callbacks) {
    if (!context || !header || !validAllocationCallbacks(callbacks)) {
        return FFX_API_RETURN_ERROR_PARAMETER;
    }

    *context = nullptr;

    if (header->type != FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE) {
        return FFX_API_RETURN_NO_PROVIDER;
    }

    const auto* backend = findVulkanBackend(header);

    if (!backend || !backend->device || !backend->physicalDevice || !backend->getDeviceProcAddr) {
        return FFX_API_RETURN_ERROR_PARAMETER;
    }

    const std::uint64_t requested_version = requestedVersionId(header);
    const auto* version = selectVersion(requested_version, backend->physicalDevice);

    if (!version) {
        return FFX_API_RETURN_NO_PROVIDER;
    }

    const auto* create = reinterpret_cast<const ffxCreateContextDescUpscale*>(header);
    void* storage = allocate(callbacks, sizeof(ProviderContext));

    if (!storage) {
        return FFX_API_RETURN_ERROR_MEMORY;
    }

    auto* provider = new (storage) ProviderContext{};
    provider->device = backend->device;
    provider->physicalDevice = backend->physicalDevice;
    provider->getDeviceProcAddr = backend->getDeviceProcAddr;

    provider->flags = create->flags;
    provider->maxRenderSize = create->maxRenderSize;
    provider->maxUpscaleSize = create->maxUpscaleSize;

    provider->message = create->fpMessage;
    provider->version = version;

    provider->extensive_logging = environmentFlag("FSR4_VK_EXTENSIVE_LOG");

    // Allow a staging copy when the caller cannot expose its output for storage writes.
    provider->use_staging_output = environmentFlag("FSR4_VK_USE_STAGING_OUTPUT") &&
                                   !environmentFlag("FSR4_VK_USE_DIRECT_OUTPUT");

    {
        const char* environment_model = std::getenv("FSR4_VK_MODEL");
        const char* source =
            requested_version != 0
                ? "loader"
                : (environment_model && *environment_model ? "environment" : "automatic");
        std::ostringstream selection;
        selection << "[model] selection source=" << source << " requested_id=0x" << std::hex
                  << requested_version << " selected_id=0x" << version->id << std::dec
                  << " selected='" << version->name << "'";
        diagnostic(*provider, selection.str());
    }

    configureFrameCapture(*provider);

    if (const char* dump_path = std::getenv("FSR4_VK_DUMP_COLOR"); dump_path && *dump_path) {
        provider->color_capture.enabled = true;
        provider->color_capture.base_path = dump_path;

        if (const char* trigger_path = std::getenv("FSR4_VK_DUMP_COLOR_TRIGGER");
            trigger_path && *trigger_path) {
            provider->color_capture.triggered_mode = true;
            provider->color_capture.trigger_path = trigger_path;
        }

        diagnostic(*provider,
                   "[FSR4-color-dump] enabled base=" + provider->color_capture.base_path.string() +
                       (provider->color_capture.triggered_mode
                            ? " trigger=" + provider->color_capture.trigger_path.string()
                            : " first-dispatch mode"));
    }

    try {
        const auto assets = assetRoot();

        if (!version->has_assets(assets)) {
            throw std::invalid_argument(std::string("missing assets for ") + version->name);
        }

        // The caller creates the device; validate only the entry points used here.
        validateDeviceEntryPoints(*provider);

        if (!backend->getDeviceProcAddr(backend->device, "vkGetDescriptorEXT")) {
            throw std::invalid_argument(
                "descriptor buffer extension must be enabled on caller device");
        }

        fsr4::validateDimensions(create->maxRenderSize.width,
                                 create->maxRenderSize.height,
                                 create->maxUpscaleSize.width,
                                 create->maxUpscaleSize.height);

        const std::string preset =
            fsr4::resolutionPreset(create->maxRenderSize.width, create->maxUpscaleSize.width);

        provider->model_context = version->create_context({
            .physical_device = provider->physicalDevice,
            .device = provider->device,
            .asset_root = assets,
            .diagnostic_root = providerModuleDirectory(),
            .get_device_proc_addr = provider->getDeviceProcAddr,
            .auto_exposure = (create->flags & FFX_UPSCALE_ENABLE_AUTO_EXPOSURE) != 0,
            .depth_inverted = (create->flags & FFX_UPSCALE_ENABLE_DEPTH_INVERTED) != 0,
            .high_resolution_motion_vectors =
                (create->flags & FFX_UPSCALE_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS) != 0,
            .motion_vector_jitter_cancellation =
                (create->flags & FFX_UPSCALE_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION) != 0,
            .extensive_logging = provider->extensive_logging,
            .max_render_width = create->maxRenderSize.width,
            .max_render_height = create->maxRenderSize.height,
            .max_output_width = create->maxUpscaleSize.width,
            .max_output_height = create->maxUpscaleSize.height,
            .api_version = provider->apiVersion,
        });
        std::ostringstream summary;
        summary << "[context] created model=" << std::quoted(version->name)
                << " entrypoint_api=" << VK_API_VERSION_MAJOR(provider->apiVersion) << '.'
                << VK_API_VERSION_MINOR(provider->apiVersion)
                << " max_render_size=" << create->maxRenderSize.width << 'x'
                << create->maxRenderSize.height
                << " max_output_size=" << create->maxUpscaleSize.width << 'x'
                << create->maxUpscaleSize.height << " initial_preset=" << preset
                << " flags=" << fsr4log::hexValue(create->flags) << " exposure="
                << ((create->flags & FFX_UPSCALE_ENABLE_AUTO_EXPOSURE) ? "automatic" : "external")
                << " depth="
                << ((create->flags & FFX_UPSCALE_ENABLE_DEPTH_INVERTED) ? "inverted" : "standard")
                << " motion_resolution="
                << ((create->flags & FFX_UPSCALE_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS)
                        ? "display"
                        : "render")
                << " output_mode=" << (provider->use_staging_output ? "staging-copy" : "direct")
                << " assets=" << (fsr4assets::isEmbedded(assets) ? "embedded" : "external")
                << " asset_root=" << std::quoted(assets.string())
                << " detailed_logging=" << (provider->extensive_logging ? "enabled" : "disabled");
        diagnostic(*provider, summary.str());
    } catch (const std::exception& error) {
        diagnostic(*provider,
                   std::string("[context] context rejected: ") + error.what(),
                   fsr4log::Level::Error);
        finishFrameCapture(*provider);
        finishColorCapture(*provider);
        provider->~ProviderContext();
        deallocate(callbacks, provider);
        return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
    }

    *context = provider;
    return FFX_API_RETURN_OK;
}

ffxReturnCode_t destroyContext(ffxContext* context, const ffxAllocationCallbacks* callbacks) {
    if (!context || !*context || !validAllocationCallbacks(callbacks)) {
        return FFX_API_RETURN_ERROR_PARAMETER;
    }

    auto* provider = reinterpret_cast<ProviderContext*>(*context);
    finishFrameCapture(*provider);
    finishColorCapture(*provider);
    provider->~ProviderContext();
    deallocate(callbacks, provider);
    *context = nullptr;
    return FFX_API_RETURN_OK;
}

ffxReturnCode_t configure(ffxContext* context, const ffxConfigureDescHeader* header) {
    if (!context || !*context || !header) {
        return FFX_API_RETURN_ERROR_PARAMETER;
    }

    if (header->type == FFX_API_CONFIGURE_DESC_TYPE_UPSCALE_KEYVALUE) {
        return FFX_API_RETURN_OK;
    }

    return FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE;
}

} // namespace fsr4provider

namespace fsr4provider {
std::uint64_t requestedVersionId(const ffxApiHeader* header) {
    for (auto* current = header; current; current = current->pNext) {
        if (current->type == FFX_API_DESC_TYPE_OVERRIDE_VERSION) {
            return reinterpret_cast<const ffxOverrideVersion*>(current)->versionId;
        }
    }

    return 0;
}

namespace {
bool physicalDeviceHasExtension(VkPhysicalDevice physical_device, const char* wanted) {
    std::uint32_t count = 0;

    if (vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &count, nullptr) !=
        VK_SUCCESS) {
        return false;
    }

    std::vector<VkExtensionProperties> extensions(count);

    if (vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &count, extensions.data()) !=
        VK_SUCCESS) {
        return false;
    }

    return std::any_of(
        extensions.begin(), extensions.end(), [wanted](const VkExtensionProperties& extension) {
            return std::strcmp(extension.extensionName, wanted) == 0;
        });
}

bool physicalDeviceSupportsFp8Matrices(VkPhysicalDevice physical_device) {
    VkPhysicalDeviceIDProperties target_id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2 target{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    target.pNext = &target_id;
    vkGetPhysicalDeviceProperties2(physical_device, &target);

    // The backend supplies no VkInstance for resolving physical-device extension commands.
    struct ProbeInstance {
        VkInstance handle{};

        ~ProbeInstance() {
            if (handle) {
                vkDestroyInstance(handle, nullptr);
            }
        }
    } probe;

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    info.pApplicationInfo = &app;
    if (vkCreateInstance(&info, nullptr, &probe.handle) != VK_SUCCESS) {
        return false;
    }
    const auto get_matrices =
        reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
            vkGetInstanceProcAddr(probe.handle,
                                  "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
    if (!get_matrices) {
        return false;
    }
    uint32_t count = 0;
    if (vkEnumeratePhysicalDevices(probe.handle, &count, nullptr) != VK_SUCCESS) {
        return false;
    }
    std::vector<VkPhysicalDevice> devices(count);
    if (vkEnumeratePhysicalDevices(probe.handle, &count, devices.data()) != VK_SUCCESS) {
        return false;
    }
    devices.resize(count);
    for (auto candidate : devices) {
        VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceCooperativeMatrixPropertiesKHR cooperative{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR};
        VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        properties.pNext = &id;
        id.pNext = &cooperative;
        vkGetPhysicalDeviceProperties2(candidate, &properties);
        if (std::memcmp(id.deviceUUID, target_id.deviceUUID, VK_UUID_SIZE) != 0 ||
            std::memcmp(id.driverUUID, target_id.driverUUID, VK_UUID_SIZE) != 0) {
            continue;
        }
        if (!(cooperative.cooperativeMatrixSupportedStages & VK_SHADER_STAGE_COMPUTE_BIT)) {
            return false;
        }
        count = 0;
        if (get_matrices(candidate, &count, nullptr) != VK_SUCCESS) {
            return false;
        }
        std::vector<VkCooperativeMatrixPropertiesKHR> matrices(
            count, {VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
        if (get_matrices(candidate, &count, matrices.data()) != VK_SUCCESS) {
            return false;
        }
        matrices.resize(count);
        const auto supports = [&](VkComponentTypeKHR input_type) {
            return std::any_of(matrices.begin(), matrices.end(), [&](const auto& matrix) {
                return matrix.scope == VK_SCOPE_SUBGROUP_KHR && matrix.MSize == 16 &&
                       matrix.NSize == 16 && matrix.KSize == 16 && matrix.AType == input_type &&
                       matrix.BType == input_type &&
                       matrix.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                       matrix.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                       !matrix.saturatingAccumulation;
            });
        };
        return supports(VK_COMPONENT_TYPE_FLOAT16_KHR) &&
               supports(VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT);
    }
    return false;
}

bool physicalDeviceSupportsFp8(VkPhysicalDevice physical_device) {
    if (!physicalDeviceHasExtension(physical_device, VK_EXT_SHADER_FLOAT8_EXTENSION_NAME) ||
        !physicalDeviceHasExtension(physical_device, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) ||
        !physicalDeviceHasExtension(physical_device, VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME)) {
        return false;
    }

    VkPhysicalDeviceShaderFloat8FeaturesEXT float8{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT};
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperative{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkPhysicalDeviceCooperativeMatrix2FeaturesNV cooperative2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV};
    VkPhysicalDeviceShaderFloat16Int8Features float16{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};

    features.pNext = &cooperative;
    cooperative.pNext = &float8;
    float8.pNext = &cooperative2;
    cooperative2.pNext = &float16;
    vkGetPhysicalDeviceFeatures2(physical_device, &features);

    return cooperative.cooperativeMatrix && float8.shaderFloat8 &&
           float8.shaderFloat8CooperativeMatrix && cooperative2.cooperativeMatrixConversions &&
           float16.shaderFloat16 && physicalDeviceSupportsFp8Matrices(physical_device);
}

} // namespace

const ProviderVersion* selectVersion(std::uint64_t id, VkPhysicalDevice physical_device) {
    // An explicit loader version takes precedence over the environment.
    if (id == kAutoVersionId) {
        return &kProviderVersions[physicalDeviceSupportsFp8(physical_device) ? 1u : 0u];
    }

    if (id != 0) {
        for (const auto& version : kProviderVersions) {
            if (version.id == id) {
                return &version;
            }
        }

        return nullptr;
    }

    const char* model = std::getenv("FSR4_VK_MODEL");

    if (model && *model) {
        if (std::strcmp(model, "int8") == 0) {
            return &kProviderVersions[0];
        }

        if (std::strcmp(model, "fp8") == 0) {
            return &kProviderVersions[1];
        }

        return nullptr;
    }

    return physicalDeviceSupportsFp8(physical_device) ? &kProviderVersions[1]
                                                      : &kProviderVersions[0];
}
} // namespace fsr4provider

namespace fsr4provider {
namespace {
float upscaleRatio(std::uint32_t qualityMode) {
    constexpr float ratios[] = {1.0f, 1.5f, 1.7f, 2.0f, 3.0f};
    return qualityMode < std::size(ratios) ? ratios[qualityMode] : 0.0f;
}

float radicalInverse(std::uint32_t index, std::uint32_t base) {
    float value = 0.0f;
    const float inverseBase = 1.0f / static_cast<float>(base);
    float fraction = inverseBase;

    while (index) {
        value += static_cast<float>(index % base) * fraction;
        index /= base;
        fraction *= inverseBase;
    }

    return value;
}

ffxReturnCode_t queryUpscale(ffxContext* context, ffxQueryDescHeader* header) {
    switch (header->type) {
    case FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION: {
        auto* query = reinterpret_cast<ffxQueryGetProviderVersion*>(header);

        if (!context || !*context) {
            return FFX_API_RETURN_ERROR_PARAMETER;
        }

        const auto* provider = static_cast<const ProviderContext*>(*context);
        query->versionId = provider->version->id;
        query->versionName = provider->version->name;
        return FFX_API_RETURN_OK;
    }
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETUPSCALERATIOFROMQUALITYMODE: {
        auto* query = reinterpret_cast<ffxQueryDescUpscaleGetUpscaleRatioFromQualityMode*>(header);
        const float ratio = upscaleRatio(query->qualityMode);

        if (!query->pOutUpscaleRatio || ratio == 0.0f) {
            return FFX_API_RETURN_ERROR_PARAMETER;
        }

        *query->pOutUpscaleRatio = ratio;
        return FFX_API_RETURN_OK;
    }
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETRENDERRESOLUTIONFROMQUALITYMODE: {
        auto* query =
            reinterpret_cast<ffxQueryDescUpscaleGetRenderResolutionFromQualityMode*>(header);
        const float ratio = upscaleRatio(query->qualityMode);

        if (!query->pOutRenderWidth || !query->pOutRenderHeight || ratio == 0.0f) {
            return FFX_API_RETURN_ERROR_PARAMETER;
        }

        *query->pOutRenderWidth =
            static_cast<std::uint32_t>(std::ceil(static_cast<float>(query->displayWidth) / ratio));
        *query->pOutRenderHeight =
            static_cast<std::uint32_t>(std::ceil(static_cast<float>(query->displayHeight) / ratio));
        return FFX_API_RETURN_OK;
    }
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT: {
        auto* query = reinterpret_cast<ffxQueryDescUpscaleGetJitterPhaseCount*>(header);

        if (!query->pOutPhaseCount || !query->renderWidth) {
            return FFX_API_RETURN_ERROR_PARAMETER;
        }

        const float ratio =
            static_cast<float>(query->displayWidth) / static_cast<float>(query->renderWidth);
        *query->pOutPhaseCount = static_cast<std::int32_t>(8.0f * ratio * ratio);
        return FFX_API_RETURN_OK;
    }
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET: {
        auto* query = reinterpret_cast<ffxQueryDescUpscaleGetJitterOffset*>(header);

        if (!query->pOutX || !query->pOutY || query->phaseCount <= 0 || query->index < 0) {
            return FFX_API_RETURN_ERROR_PARAMETER;
        }

        const std::uint32_t sample =
            static_cast<std::uint32_t>(query->index % query->phaseCount) + 1;
        *query->pOutX = radicalInverse(sample, 2) - 0.5f;
        *query->pOutY = radicalInverse(sample, 3) - 0.5f;
        return FFX_API_RETURN_OK;
    }
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GET_RESOURCE_REQUIREMENTS: {
        auto* query = reinterpret_cast<ffxQueryDescUpscaleGetResourceRequirements*>(header);
        query->required_resources =
            FFX_API_QUERY_RESOURCE_INPUT_COLOR | FFX_API_QUERY_RESOURCE_INPUT_DEPTH |
            FFX_API_QUERY_RESOURCE_INPUT_MV | FFX_API_QUERY_RESOURCE_INPUT_EXPOSURE;
        query->optional_resources = 0;

        if (context && *context &&
            (static_cast<ProviderContext*>(*context)->flags & FFX_UPSCALE_ENABLE_AUTO_EXPOSURE)) {
            query->required_resources &= ~FFX_API_QUERY_RESOURCE_INPUT_EXPOSURE;
        }

        return FFX_API_RETURN_OK;
    }
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GPU_MEMORY_USAGE: {
        auto* query = reinterpret_cast<ffxQueryDescUpscaleGetGPUMemoryUsage*>(header);

        if (!context || !*context || !query->gpuMemoryUsageUpscaler) {
            return FFX_API_RETURN_ERROR_PARAMETER;
        }

        return FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE;
    }
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GPU_MEMORY_USAGE_V2: {
        auto* query = reinterpret_cast<ffxQueryDescUpscaleGetGPUMemoryUsageV2*>(header);

        if (!query->gpuMemoryUsageUpscaler) {
            return FFX_API_RETURN_ERROR_PARAMETER;
        }

        return FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE;
    }
    default:
        return FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE;
    }
}
} // namespace

ffxReturnCode_t query(ffxContext* context, ffxQueryDescHeader* header) {
    if (!header) {
        return FFX_API_RETURN_ERROR_PARAMETER;
    }

    if (!context && header->type == FFX_API_QUERY_DESC_TYPE_GET_VERSIONS) {
        auto* query = reinterpret_cast<ffxQueryDescGetVersions*>(header);

        if (!query->outputCount) {
            return FFX_API_RETURN_ERROR_PARAMETER;
        }

        if (query->createDescType != FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE) {
            *query->outputCount = 0;
            return FFX_API_RETURN_OK;
        }

        const std::uint64_t capacity = *query->outputCount;
        const std::uint64_t write_count =
            std::min<std::uint64_t>(capacity, kAdvertisedVersions.size());

        for (std::uint64_t index = 0; index < write_count; ++index) {
            if (query->versionIds) {
                query->versionIds[index] = kAdvertisedVersions[index].id;
            }

            if (query->versionNames) {
                query->versionNames[index] = kAdvertisedVersions[index].name;
            }
        }

        *query->outputCount = kAdvertisedVersions.size();
        return FFX_API_RETURN_OK;
    }

    return queryUpscale(context, header);
}

} // namespace fsr4provider

namespace fsr4provider {
namespace {
// Use the RGB9E5 ABI value because the bundled SDK headers lack its enum name.
constexpr uint32_t kFfxSurfaceFormatR9G9B9E5SharedExp = 29u;

struct PreparedModelDispatch {
    fsr4_411::Image color;
    fsr4_411::Image depth;
    fsr4_411::Image motion;
    fsr4_411::Image exposure;
    fsr4_411::Image output;

    fsr4_411::OptimizedConstants constants{};
    bool motion_vector_jitter_cancellation = false;
};

VkFormat motionVectorVkFormat(uint32_t ffxFormat) {
    switch (ffxFormat) {
    case FFX_API_SURFACE_FORMAT_R16G16_FLOAT:
        return VK_FORMAT_R16G16_SFLOAT;
    case FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT:
        // Preserve RGBA16F storage; the shaders consume the XY motion components.
        return VK_FORMAT_R16G16B16A16_SFLOAT;
    case FFX_API_SURFACE_FORMAT_R8G8B8A8_SNORM:
        // The RGBA8 SNORM API tag represents an RG16 SNORM Vulkan image.
        return VK_FORMAT_R16G16_SNORM;
    case FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM:
        return VK_FORMAT_R16G16_UNORM;
    case FFX_API_SURFACE_FORMAT_R32G32_FLOAT:
        return VK_FORMAT_R32G32_SFLOAT;
    default:
        throw std::invalid_argument("unsupported motion-vector surface format " +
                                    apiSurfaceFormat(ffxFormat));
    }
}

// Optional masks are sampled only; preserve the underlying Vulkan format.
VkFormat optionalMaskVkFormat(uint32_t ffxFormat) {
    switch (ffxFormat) {
    case FFX_API_SURFACE_FORMAT_R8_UNORM:
        return VK_FORMAT_R8_UNORM;
    case FFX_API_SURFACE_FORMAT_R8_UINT:
        return VK_FORMAT_R8_UINT;
    case FFX_API_SURFACE_FORMAT_R8G8_UNORM:
        return VK_FORMAT_R8G8_UNORM;
    case FFX_API_SURFACE_FORMAT_R8G8_UINT:
        return VK_FORMAT_R8G8_UINT;
    case FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM:
        return VK_FORMAT_R8G8B8A8_UNORM;
    case FFX_API_SURFACE_FORMAT_R8G8B8A8_SNORM:
        return VK_FORMAT_R8G8B8A8_SNORM;
    case FFX_API_SURFACE_FORMAT_R8G8B8A8_SRGB:
        return VK_FORMAT_R8G8B8A8_SRGB;
    case FFX_API_SURFACE_FORMAT_B8G8R8A8_UNORM:
        return VK_FORMAT_B8G8R8A8_UNORM;
    case FFX_API_SURFACE_FORMAT_B8G8R8A8_SRGB:
        return VK_FORMAT_B8G8R8A8_SRGB;
    case FFX_API_SURFACE_FORMAT_R16_FLOAT:
        return VK_FORMAT_R16_SFLOAT;
    case FFX_API_SURFACE_FORMAT_R16_UNORM:
        return VK_FORMAT_R16_UNORM;
    case FFX_API_SURFACE_FORMAT_R16_SNORM:
        return VK_FORMAT_R16_SNORM;
    case FFX_API_SURFACE_FORMAT_R16_UINT:
        return VK_FORMAT_R16_UINT;
    case FFX_API_SURFACE_FORMAT_R32_FLOAT:
        return VK_FORMAT_R32_SFLOAT;
    default:
        throw std::invalid_argument("unsupported optional-mask surface format " +
                                    apiSurfaceFormat(ffxFormat));
    }
}

VkFormat inputColorVkFormat(uint32_t ffxFormat) {
    switch (ffxFormat) {
    case FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT:
        return VK_FORMAT_R16G16B16A16_SFLOAT;
    case FFX_API_SURFACE_FORMAT_R11G11B10_FLOAT:
        return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    case kFfxSurfaceFormatR9G9B9E5SharedExp:
        return VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;
    default:
        throw std::invalid_argument("unsupported color surface format " +
                                    apiSurfaceFormat(ffxFormat));
    }
}

VkFormat outputColorVkFormat(uint32_t format) {
    switch (format) {
    case FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT:
        return VK_FORMAT_R16G16B16A16_SFLOAT;
    case FFX_API_SURFACE_FORMAT_R11G11B10_FLOAT:
        return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    default:
        throw std::invalid_argument("unsupported output surface format " +
                                    apiSurfaceFormat(format));
    }
}

PreparedModelDispatch prepareModelDispatch(ProviderContext& provider,
                                           const ffxDispatchDescUpscale& dispatch,
                                           const fsr4_411::Image& color,
                                           const fsr4_411::Image& depth,
                                           const fsr4_411::Image& motion,
                                           const fsr4_411::Image& output,
                                           uint32_t render_width,
                                           uint32_t render_height,
                                           uint32_t output_width,
                                           uint32_t output_height) {
    const bool auto_exposure = (provider.flags & FFX_UPSCALE_ENABLE_AUTO_EXPOSURE) != 0;

    const bool exposure_is_r16 =
        dispatch.exposure.description.format == FFX_API_SURFACE_FORMAT_R16_FLOAT;
    const bool exposure_is_r32 =
        dispatch.exposure.description.format == FFX_API_SURFACE_FORMAT_R32_FLOAT;
    const bool exposure_is_rgba32 =
        dispatch.exposure.description.format == FFX_API_SURFACE_FORMAT_R32G32B32A32_FLOAT;

    if (!auto_exposure && (!dispatch.exposure.resource ||
                           (!exposure_is_r16 && !exposure_is_r32 && !exposure_is_rgba32))) {
        throw std::invalid_argument(
            "unsupported exposure resource: present=" +
            std::string(dispatch.exposure.resource ? "yes" : "no") +
            " api_format=" + apiSurfaceFormat(dispatch.exposure.description.format) +
            "; expected R16_FLOAT, R32_FLOAT or R32G32B32A32_FLOAT");
    }

    PreparedModelDispatch prepared{};
    prepared.color = {
        color.image, VK_NULL_HANDLE, color.view, color.format, render_width, render_height};
    // Keep backing size separate from active size; some inputs have extra padding.
    prepared.color.resource_width =
        dispatch.color.description.width ? dispatch.color.description.width : render_width;
    prepared.color.resource_height =
        dispatch.color.description.height ? dispatch.color.description.height : render_height;

    prepared.depth = {depth.image,
                      VK_NULL_HANDLE,
                      depth.view,
                      depth.format,
                      render_width,
                      render_height,
                      depth.aspect};

    const bool display_resolution_motion_vectors =
        (provider.flags & FFX_UPSCALE_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS) != 0;
    const uint32_t motion_resource_width =
        display_resolution_motion_vectors
            ? (dispatch.motionVectors.description.width ? dispatch.motionVectors.description.width
                                                        : output_width)
            : render_width;
    const uint32_t motion_resource_height =
        display_resolution_motion_vectors
            ? (dispatch.motionVectors.description.height ? dispatch.motionVectors.description.height
                                                         : output_height)
            : render_height;
    prepared.motion = {motion.image,
                       VK_NULL_HANDLE,
                       motion.view,
                       motion.format,
                       motion_resource_width,
                       motion_resource_height};

    if (!auto_exposure) {
        prepared.exposure = {reinterpret_cast<VkImage>(dispatch.exposure.resource),
                             VK_NULL_HANDLE,
                             VK_NULL_HANDLE,
                             exposure_is_r16 ? VK_FORMAT_R16_SFLOAT
                                             : (exposure_is_rgba32 ? VK_FORMAT_R32G32B32A32_SFLOAT
                                                                   : VK_FORMAT_R32_SFLOAT),
                             dispatch.exposure.description.width,
                             dispatch.exposure.description.height};
    }

    prepared.output = output;

    auto& constants = prepared.constants;
    const uint32_t padded_width = fsr4::roundUp(output_width, 8);
    const uint32_t padded_height = fsr4::roundUp(output_height, 8);

    constants.inv_size[0] = 1.0f / padded_width;
    constants.inv_size[1] = 1.0f / padded_height;

    constants.scale[0] = float(padded_width) / render_width;
    constants.scale[1] = float(padded_height) / render_height;
    constants.inv_scale[0] = 1.0f / constants.scale[0];
    constants.inv_scale[1] = 1.0f / constants.scale[1];

    constants.jitter[0] = dispatch.jitterOffset.x;
    constants.jitter[1] = dispatch.jitterOffset.y;

    prepared.motion_vector_jitter_cancellation =
        (provider.flags & FFX_UPSCALE_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION) != 0;

    const float motion_width = display_resolution_motion_vectors ? output_width : render_width;
    const float motion_height = display_resolution_motion_vectors ? output_height : render_height;
    constants.mv_scale[0] = dispatch.motionVectorScale.x / motion_width;
    constants.mv_scale[1] = dispatch.motionVectorScale.y / motion_height;

    if (prepared.motion_vector_jitter_cancellation) {
        if (dispatch.reset || !provider.has_previous_jitter) {
            provider.previous_jitter[0] = dispatch.jitterOffset.x;
            provider.previous_jitter[1] = dispatch.jitterOffset.y;
            provider.has_previous_jitter = true;
        }

        constants.motion_vector_jitter_cancellation[0] =
            (provider.previous_jitter[0] - dispatch.jitterOffset.x) / motion_width;
        constants.motion_vector_jitter_cancellation[1] =
            (provider.previous_jitter[1] - dispatch.jitterOffset.y) / motion_height;

        provider.previous_jitter[0] = dispatch.jitterOffset.x;
        provider.previous_jitter[1] = dispatch.jitterOffset.y;
    }

    constants.tex_size[0] = fsr4::roundUp(provider.maxUpscaleSize.width, 8);
    constants.tex_size[1] = fsr4::roundUp(provider.maxUpscaleSize.height, 8);

    constants.max_render_size[0] = provider.maxRenderSize.width;
    constants.max_render_size[1] = provider.maxRenderSize.height;

    constants.width = padded_width;
    constants.height = padded_height;
    constants.width_lr = render_width;
    constants.height_lr = render_height;

    constants.reset = dispatch.reset;
    constants.pre_exposure = fsr4::sanitizePreExposure(dispatch.preExposure);

    constants.rcas_enabled = dispatch.enableSharpening;
    constants.rcas_sharpness =
        dispatch.enableSharpening ? fsr4runtime::rcasSharpness(dispatch.sharpness) : 0.0f;

    return prepared;
}

} // namespace

ffxReturnCode_t dispatch(ffxContext* context, const ffxDispatchDescHeader* header) {
    if (!context || !*context || !header) {
        return FFX_API_RETURN_ERROR_PARAMETER;
    }

    if (header->type != FFX_API_DISPATCH_DESC_TYPE_UPSCALE) {
        return FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE;
    }

    auto* provider = static_cast<ProviderContext*>(*context);
    pollFrameCapture(*provider);
    pollColorCapture(*provider);

    const auto& dispatch_description = *reinterpret_cast<const ffxDispatchDescUpscale*>(header);
    const auto render_width = dispatch_description.renderSize.width;
    const auto render_height = dispatch_description.renderSize.height;
    const auto output_width = dispatch_description.upscaleSize.width
                                  ? dispatch_description.upscaleSize.width
                                  : provider->maxUpscaleSize.width;
    const auto output_height = dispatch_description.upscaleSize.height
                                   ? dispatch_description.upscaleSize.height
                                   : provider->maxUpscaleSize.height;

    const std::uint64_t dispatch_log_index = provider->dispatch_count++;
    const bool log_this_dispatch =
        provider->extensive_logging && (dispatch_log_index < 3 || dispatch_log_index % 120 == 0);

    const auto reject = [provider](const char* reason) {
        providerMessage(*provider,
                        FFX_API_MESSAGE_TYPE_ERROR,
                        std::string("[dispatch] FSR4 Vulkan dispatch rejected: ") + reason);
        return FFX_API_RETURN_ERROR_PARAMETER;
    };

    if (!dispatch_description.commandList) {
        return reject("null command list");
    }

    // The active render region may be smaller than the backing texture.
    const auto validate_resource = [](const char* name,
                                      const FfxApiResource& api_resource,
                                      uint32_t width,
                                      uint32_t height,
                                      uint32_t format,
                                      VkFormat vulkan_format,
                                      uint32_t state,
                                      uint32_t depth_usage = 0,
                                      VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT) {
        const uint32_t required_usage =
            state == FFX_API_RESOURCE_STATE_UNORDERED_ACCESS ? FFX_API_RESOURCE_USAGE_UAV : 0u;
        const uint32_t actual_depth_usage =
            api_resource.description.usage &
            (FFX_API_RESOURCE_USAGE_DEPTHTARGET | FFX_API_RESOURCE_USAGE_STENCILTARGET);

        if (!api_resource.resource ||
            api_resource.description.type != FFX_API_RESOURCE_TYPE_TEXTURE2D ||
            actual_depth_usage != depth_usage || api_resource.description.width < width ||
            api_resource.description.height < height || api_resource.description.format != format ||
            api_resource.state != state ||
            (api_resource.description.usage & required_usage) != required_usage) {
            std::ostringstream reason;
            reason << "unsupported " << name << " resource: handle=0x" << std::hex
                   << reinterpret_cast<uintptr_t>(api_resource.resource) << std::dec
                   << " type=" << apiResourceType(api_resource.description.type)
                   << " api_format=" << apiSurfaceFormat(api_resource.description.format)
                   << " backing_size=" << api_resource.description.width << "x"
                   << api_resource.description.height << " usage=0x" << std::hex
                   << api_resource.description.usage
                   << " state=" << apiResourceState(api_resource.state) << std::dec
                   << "; expected handle!=0 type="
                   << apiResourceType(FFX_API_RESOURCE_TYPE_TEXTURE2D)
                   << " api_format=" << apiSurfaceFormat(format) << " minimum_size=" << width << "x"
                   << height << " depth_usage=0x" << std::hex << depth_usage << " required_usage=0x"
                   << required_usage << " state=" << apiResourceState(state) << std::dec;
            throw std::invalid_argument(reason.str());
        }

        return fsr4_411::Image{reinterpret_cast<VkImage>(api_resource.resource),
                               VK_NULL_HANDLE,
                               VK_NULL_HANDLE,
                               vulkan_format,
                               width,
                               height,
                               aspect};
    };

    try {
        fsr4::validateDimensions(render_width, render_height, output_width, output_height);

        if (render_width > provider->maxRenderSize.width ||
            render_height > provider->maxRenderSize.height ||
            output_width > provider->maxUpscaleSize.width ||
            output_height > provider->maxUpscaleSize.height) {
            throw std::invalid_argument("dispatch dimensions exceed context capacity");
        }

        const VkFormat color_vk_format =
            inputColorVkFormat(dispatch_description.color.description.format);
        const auto output =
            validate_resource("output",
                              dispatch_description.output,
                              output_width,
                              output_height,
                              dispatch_description.output.description.format,
                              outputColorVkFormat(dispatch_description.output.description.format),
                              FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);

        if (log_this_dispatch) {
            const auto log_resource = [&](const char* name, const FfxApiResource& logged_resource) {
                std::ostringstream line;
                line << "[resource] dispatch=" << dispatch_log_index << " role=" << name
                     << " present=" << (logged_resource.resource ? "yes" : "no");
                if (!logged_resource.resource) {
                    diagnostic(*provider, line.str(), fsr4log::Level::Debug);
                    return;
                }
                line << " handle=0x" << std::hex
                     << reinterpret_cast<uintptr_t>(logged_resource.resource) << std::dec
                     << " type=" << apiResourceType(logged_resource.description.type)
                     << " api_format=" << apiSurfaceFormat(logged_resource.description.format)
                     << " backing_size=" << logged_resource.description.width << "x"
                     << logged_resource.description.height << " usage=0x" << std::hex
                     << logged_resource.description.usage
                     << " state=" << apiResourceState(logged_resource.state) << std::dec;
                diagnostic(*provider, line.str(), fsr4log::Level::Debug);
            };
            log_resource("color", dispatch_description.color);
            log_resource("depth", dispatch_description.depth);
            log_resource("motion", dispatch_description.motionVectors);
            log_resource("exposure", dispatch_description.exposure);
            log_resource("output", dispatch_description.output);
        }

        const uint32_t depth_stencil =
            FFX_API_RESOURCE_USAGE_DEPTHTARGET | FFX_API_RESOURCE_USAGE_STENCILTARGET;
        const uint32_t depth_usage = dispatch_description.depth.description.usage & depth_stencil;
        const bool is_depth_target = (depth_usage & FFX_API_RESOURCE_USAGE_DEPTHTARGET) != 0;

        const auto color = validate_resource("color",
                                             dispatch_description.color,
                                             render_width,
                                             render_height,
                                             dispatch_description.color.description.format,
                                             color_vk_format,
                                             FFX_API_RESOURCE_STATE_COMPUTE_READ);

        const bool missing_depth = dispatch_description.depth.resource == nullptr;

        const bool allow_missing_depth =
            missing_depth &&
            (provider->flags & FFX_UPSCALE_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS) != 0;

        if (missing_depth && !allow_missing_depth) {
            throw std::invalid_argument("missing depth is only supported with "
                                        "display-resolution motion vectors");
        }

        fsr4_411::Image depth{};

        if (!missing_depth) {
            VkFormat depth_format = VK_FORMAT_R32_SFLOAT;

            if (is_depth_target) {
                const bool has_stencil = (depth_usage & FFX_API_RESOURCE_USAGE_STENCILTARGET) != 0;
                depth_format = has_stencil ? VK_FORMAT_D32_SFLOAT_S8_UINT : VK_FORMAT_D32_SFLOAT;
            }

            depth = validate_resource("depth",
                                      dispatch_description.depth,
                                      render_width,
                                      render_height,
                                      FFX_API_SURFACE_FORMAT_R32_FLOAT,
                                      depth_format,
                                      FFX_API_RESOURCE_STATE_COMPUTE_READ,
                                      depth_usage,
                                      is_depth_target ? VK_IMAGE_ASPECT_DEPTH_BIT
                                                      : VK_IMAGE_ASPECT_COLOR_BIT);
        } else {
            // The model supplies neutral depth when no depth image is provided.
            depth = {VK_NULL_HANDLE,
                     VK_NULL_HANDLE,
                     VK_NULL_HANDLE,
                     VK_FORMAT_R32_SFLOAT,
                     render_width,
                     render_height,
                     VK_IMAGE_ASPECT_COLOR_BIT};

            if (log_this_dispatch) {
                diagnostic(*provider,
                           "[depth] external depth absent; "
                           "using no-depth compatibility path",
                           fsr4log::Level::Debug);
            }
        }

        const VkFormat motion_vk_format =
            motionVectorVkFormat(dispatch_description.motionVectors.description.format);
        const auto motion = validate_resource("motion",
                                              dispatch_description.motionVectors,
                                              render_width,
                                              render_height,
                                              dispatch_description.motionVectors.description.format,
                                              motion_vk_format,
                                              FFX_API_RESOURCE_STATE_COMPUTE_READ);

        fsr4_411::Image optional_mask{};
        const FfxApiResource* optional_mask_resource = nullptr;

        if (dispatch_description.reactive.resource &&
            dispatch_description.transparencyAndComposition.resource &&
            dispatch_description.reactive.resource !=
                dispatch_description.transparencyAndComposition.resource) {
            throw std::invalid_argument(
                "distinct reactive and transparency masks are not yet supported");
        }

        if (dispatch_description.reactive.resource) {
            optional_mask_resource = &dispatch_description.reactive;
        } else if (dispatch_description.transparencyAndComposition.resource) {
            optional_mask_resource = &dispatch_description.transparencyAndComposition;
        }

        if (optional_mask_resource) {
            optional_mask =
                validate_resource("optional mask",
                                  *optional_mask_resource,
                                  render_width,
                                  render_height,
                                  optional_mask_resource->description.format,
                                  optionalMaskVkFormat(optional_mask_resource->description.format),
                                  FFX_API_RESOURCE_STATE_COMPUTE_READ);

            if (log_this_dispatch) {
                std::ostringstream message;
                message << "[resource] dispatch=" << dispatch_log_index
                        << " role=optional-mask handle=0x" << std::hex
                        << reinterpret_cast<uintptr_t>(optional_mask_resource->resource) << std::dec
                        << " api_format="
                        << apiSurfaceFormat(optional_mask_resource->description.format)
                        << " backing_size=" << optional_mask_resource->description.width << "x"
                        << optional_mask_resource->description.height
                        << " reactive=" << (dispatch_description.reactive.resource ? 1 : 0)
                        << " transparency="
                        << (dispatch_description.transparencyAndComposition.resource ? 1 : 0);
                diagnostic(*provider, message.str(), fsr4log::Level::Debug);
            }
        }

        {
            auto prepared = prepareModelDispatch(*provider,
                                                 dispatch_description,
                                                 color,
                                                 depth,
                                                 motion,
                                                 output,
                                                 render_width,
                                                 render_height,
                                                 output_width,
                                                 output_height);

            if (log_this_dispatch) {
                std::ostringstream message;
                message << "[dispatch] index=" << dispatch_log_index << " parameters "
                        << "render=" << render_width << "x" << render_height
                        << " output_size=" << output_width << "x" << output_height
                        << " frame_time_ms=" << dispatch_description.frameTimeDelta
                        << " pre_exposure=" << dispatch_description.preExposure
                        << " reset=" << dispatch_description.reset
                        << " jitter=" << dispatch_description.jitterOffset.x << ","
                        << dispatch_description.jitterOffset.y
                        << " mv_scale=" << dispatch_description.motionVectorScale.x << ","
                        << dispatch_description.motionVectorScale.y
                        << " vertical_fov_radians=" << dispatch_description.cameraFovAngleVertical;
                diagnostic(*provider, message.str(), fsr4log::Level::Debug);
            }

            const auto& model_constants = prepared.constants;

            if (log_this_dispatch) {
                std::ostringstream message;
                message << "[constants] dispatch=" << dispatch_log_index
                        << " model=" << std::quoted(provider->version->name)
                        << " padded_output_size=" << model_constants.width << 'x'
                        << model_constants.height
                        << " buffer_tex_size=" << model_constants.tex_size[0] << 'x'
                        << model_constants.tex_size[1]
                        << " pre_exposure=" << model_constants.pre_exposure << " jitter=["
                        << model_constants.jitter[0] << ',' << model_constants.jitter[1] << ']'
                        << " normalized_motion_scale=[" << model_constants.mv_scale[0] << ','
                        << model_constants.mv_scale[1] << ']' << " motion_jitter_cancellation="
                        << (prepared.motion_vector_jitter_cancellation ? "yes" : "no")
                        << " provider_rcas="
                        << (model_constants.rcas_enabled ? "enabled" : "disabled")
                        << " rcas_sharpness=" << model_constants.rcas_sharpness;
                diagnostic(*provider, message.str(), fsr4log::Level::Debug);
            }

            if (model_constants.pre_exposure != dispatch_description.preExposure &&
                provider->first_dispatch) {
                providerMessage(*provider,
                                FFX_API_MESSAGE_TYPE_WARNING,
                                "[exposure] invalid pre-exposure replaced with neutral value=1.0");
            }

            auto* captured_frame = beginFrameCapture(*provider,
                                                     dispatch_description,
                                                     {prepared.color,
                                                      prepared.depth,
                                                      prepared.motion,
                                                      prepared.exposure,
                                                      optional_mask,
                                                      prepared.output},
                                                     dispatch_log_index);
            provider->model_context->recordFrame(
                static_cast<VkCommandBuffer>(dispatch_description.commandList),
                prepared.color,
                prepared.depth,
                prepared.motion,
                prepared.exposure,
                fsr4_411::Image{optional_mask.image,
                                VK_NULL_HANDLE,
                                optional_mask.view,
                                optional_mask.format,
                                optional_mask.width,
                                optional_mask.height,
                                optional_mask.aspect},
                prepared.output,
                model_constants,
                provider->use_staging_output,
                fsr4_411::selectColorSpace(
                    (provider->flags & FFX_UPSCALE_ENABLE_NON_LINEAR_COLORSPACE) != 0,
                    (dispatch_description.flags & FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_SRGB) != 0,
                    (dispatch_description.flags & FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_PQ) != 0),
                fsr4_411::OverlayOptions{
                    .label = provider->version->name,
                    .watermark = overlayEnvironmentFlag("FSR4_VK_WATERMARK",
                                                        overlayEnvironmentFlag("MLSR-WATERMARK")),
                    .debug_view =
                        (dispatch_description.flags & FFX_UPSCALE_FLAG_DRAW_DEBUG_VIEW) != 0 ||
                        overlayEnvironmentFlag("FSR4_VK_DEBUG_VIEW"),
                    .hdr = (provider->flags & FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE) != 0,
                });
            endFrameCapture(*provider,
                            captured_frame,
                            static_cast<VkCommandBuffer>(dispatch_description.commandList));
            recordColorCapture(*provider,
                               static_cast<VkCommandBuffer>(dispatch_description.commandList),
                               color,
                               dispatch_log_index);

            if (provider->first_dispatch) {
                diagnostic(*provider,
                           "[dispatch] first dispatch recorded render_size=" +
                               std::to_string(render_width) + "x" + std::to_string(render_height) +
                               " output_size=" + std::to_string(output_width) + "x" +
                               std::to_string(output_height) + " model=\"" +
                               provider->version->name + "\"");
                provider->first_dispatch = false;
            }

            return FFX_API_RETURN_OK;
        }
    } catch (const std::invalid_argument& error) {
        providerMessage(*provider,
                        FFX_API_MESSAGE_TYPE_ERROR,
                        std::string("FSR4 Vulkan dispatch rejected: ") + error.what());
        return FFX_API_RETURN_ERROR_PARAMETER;
    } catch (const std::exception& error) {
        providerMessage(*provider,
                        FFX_API_MESSAGE_TYPE_ERROR,
                        std::string("[dispatch] FSR4 Vulkan dispatch failed: ") + error.what());
        return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
    } catch (...) {
        providerMessage(*provider,
                        FFX_API_MESSAGE_TYPE_ERROR,
                        "[dispatch] FSR4 Vulkan dispatch failed: unknown exception");
        return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
    }
}
} // namespace fsr4provider

extern "C" FFX_API_ENTRY ffxReturnCode_t ffxCreateContext(ffxContext* context,
                                                          ffxCreateContextDescHeader* header,
                                                          const ffxAllocationCallbacks* callbacks) {
    return fsr4provider::createContext(context, header, callbacks);
}

extern "C" FFX_API_ENTRY ffxReturnCode_t
ffxDestroyContext(ffxContext* context, const ffxAllocationCallbacks* callbacks) {
    return fsr4provider::destroyContext(context, callbacks);
}

extern "C" FFX_API_ENTRY ffxReturnCode_t ffxConfigure(ffxContext* context,
                                                      const ffxConfigureDescHeader* header) {
    return fsr4provider::configure(context, header);
}

extern "C" FFX_API_ENTRY ffxReturnCode_t ffxQuery(ffxContext* context, ffxQueryDescHeader* header) {
    return fsr4provider::query(context, header);
}

extern "C" FFX_API_ENTRY ffxReturnCode_t ffxDispatch(ffxContext* context,
                                                     const ffxDispatchDescHeader* header) {
    return fsr4provider::dispatch(context, header);
}
