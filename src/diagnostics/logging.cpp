// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

#include "diagnostics/diagnostics.hpp"
#include "provider/provider.hpp"
#include "runtime/frame.hpp"
#include "runtime/resources.hpp"

#include <ctime>
#include <fstream>
#include <iomanip>
#include <locale>
#include <mutex>
#include <sstream>

namespace fsr4log {
namespace {
const char* levelName(Level level) {
    switch (level) {
    case Level::Debug:
        return "DEBUG";
    case Level::Info:
        return "INFO";
    case Level::Warning:
        return "WARN";
    case Level::Error:
        return "ERROR";
    }
    return "UNKNOWN";
}

std::string utcTimestamp(std::chrono::system_clock::time_point timestamp) {
    const auto seconds = std::chrono::floor<std::chrono::seconds>(timestamp);
    const auto milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(timestamp - seconds).count();
    const auto calendar_time = std::chrono::system_clock::to_time_t(seconds);
    std::tm utc{};
#if defined(_WIN32)
    const bool converted = gmtime_s(&utc, &calendar_time) == 0;
#else
    const bool converted = gmtime_r(&calendar_time, &utc) != nullptr;
#endif
    std::ostringstream result;
    result.imbue(std::locale::classic());

    if (!converted) {
        result << "unix_ms="
               << std::chrono::duration_cast<std::chrono::milliseconds>(
                      timestamp.time_since_epoch())
                      .count();
        return result.str();
    }

    result << std::put_time(&utc, "%Y-%m-%d %H:%M:%S") << '.' << std::setfill('0') << std::setw(3)
           << milliseconds << " UTC";
    return result.str();
}
} // namespace

std::string formatRecord(std::chrono::system_clock::time_point timestamp,
                         Level level,
                         std::string_view message) {
    const auto prefix = "[" + utcTimestamp(timestamp) + "] [" + levelName(level) + "] ";
    std::string record;
    std::size_t cursor = 0;

    // Prefix each line of a multiline message.
    do {
        const auto newline = message.find('\n', cursor);
        const auto end = newline == std::string_view::npos ? message.size() : newline;
        auto line = message.substr(cursor, end - cursor);

        if (!line.empty() && line.back() == '\r') {
            line.remove_suffix(1);
        }

        record += prefix;
        record.append(line);
        record += '\n';

        if (newline == std::string_view::npos) {
            break;
        }
        cursor = newline + 1;
    } while (cursor < message.size());

    return record;
}

void writeRecord(const std::filesystem::path& path,
                 Level level,
                 std::string_view message,
                 bool truncate) noexcept {
    try {
        // Keep messages from concurrent contexts together.
        static std::mutex writer_mutex;
        const std::lock_guard lock(writer_mutex);
        std::ofstream stream(path, truncate ? std::ios::trunc : std::ios::app);

        if (stream) {
            stream << formatRecord(std::chrono::system_clock::now(), level, message);
        }
    } catch (...) {
    }
}

std::string hexValue(std::uint64_t value) {
    std::ostringstream result;
    result.imbue(std::locale::classic());
    result << "0x" << std::hex << value;
    return result.str();
}

std::string vulkanFormat(VkFormat format) {
    switch (format) {
    case VK_FORMAT_UNDEFINED:
        return "VK_FORMAT_UNDEFINED";
    case VK_FORMAT_R8_UNORM:
        return "VK_FORMAT_R8_UNORM";
    case VK_FORMAT_R8_UINT:
        return "VK_FORMAT_R8_UINT";
    case VK_FORMAT_R8G8_UNORM:
        return "VK_FORMAT_R8G8_UNORM";
    case VK_FORMAT_R8G8B8A8_UNORM:
        return "VK_FORMAT_R8G8B8A8_UNORM";
    case VK_FORMAT_R8G8B8A8_SNORM:
        return "VK_FORMAT_R8G8B8A8_SNORM";
    case VK_FORMAT_B8G8R8A8_UNORM:
        return "VK_FORMAT_B8G8R8A8_UNORM";
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
        return "VK_FORMAT_A2B10G10R10_UNORM_PACK32";
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
        return "VK_FORMAT_B10G11R11_UFLOAT_PACK32";
    case VK_FORMAT_R16_SFLOAT:
        return "VK_FORMAT_R16_SFLOAT";
    case VK_FORMAT_R16G16_SFLOAT:
        return "VK_FORMAT_R16G16_SFLOAT";
    case VK_FORMAT_R16G16B16A16_SFLOAT:
        return "VK_FORMAT_R16G16B16A16_SFLOAT";
    case VK_FORMAT_R32_SFLOAT:
        return "VK_FORMAT_R32_SFLOAT";
    case VK_FORMAT_R32_UINT:
        return "VK_FORMAT_R32_UINT";
    case VK_FORMAT_R32G32_SFLOAT:
        return "VK_FORMAT_R32G32_SFLOAT";
    case VK_FORMAT_D32_SFLOAT:
        return "VK_FORMAT_D32_SFLOAT";
    case VK_FORMAT_D24_UNORM_S8_UINT:
        return "VK_FORMAT_D24_UNORM_S8_UINT";
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
        return "VK_FORMAT_D32_SFLOAT_S8_UINT";
    case VK_FORMAT_R16G16_SNORM:
        return "VK_FORMAT_R16G16_SNORM";
    case VK_FORMAT_R16G16_UNORM:
        return "VK_FORMAT_R16G16_UNORM";
    case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32:
        return "VK_FORMAT_E5B9G9R9_UFLOAT_PACK32";
    case VK_FORMAT_R32G32B32A32_SFLOAT:
        return "VK_FORMAT_R32G32B32A32_SFLOAT";
    case VK_FORMAT_R16_UNORM:
        return "VK_FORMAT_R16_UNORM";
    case VK_FORMAT_R16_SNORM:
        return "VK_FORMAT_R16_SNORM";
    case VK_FORMAT_R16_UINT:
        return "VK_FORMAT_R16_UINT";
    case VK_FORMAT_R8G8_UINT:
        return "VK_FORMAT_R8G8_UINT";
    case VK_FORMAT_R8G8B8A8_SRGB:
        return "VK_FORMAT_R8G8B8A8_SRGB";
    case VK_FORMAT_B8G8R8A8_SRGB:
        return "VK_FORMAT_B8G8R8A8_SRGB";

    default:
        return "VkFormat(" + std::to_string(static_cast<int>(format)) + ")";
    }
}

std::string memoryProperties(VkMemoryPropertyFlags flags) {
    std::string result;
    const auto add = [&](VkMemoryPropertyFlags bit, const char* name) {
        if ((flags & bit) != 0) {
            if (!result.empty()) {
                result += '|';
            }
            result += name;
        }
    };
    add(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "device-local");
    add(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, "host-visible");
    add(VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, "host-coherent");
    add(VK_MEMORY_PROPERTY_HOST_CACHED_BIT, "host-cached");
    add(VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT, "lazily-allocated");
    add(VK_MEMORY_PROPERTY_PROTECTED_BIT, "protected");

    // Include the numeric mask so unknown flags remain visible.
    return (result.empty() ? (flags == 0 ? "none" : "unknown") : result) + " (" + hexValue(flags) +
           ")";
}

const char* descriptorSetName(uint32_t set_index) {
    using namespace fsr4runtime;
    switch (set_index) {
    case descriptor_set::sampled_images:
        return "sampled-images";
    case descriptor_set::model_reads:
        return "model-reads";
    case descriptor_set::samplers:
        return "samplers";
    case descriptor_set::storage_images:
        return "storage-images";
    case descriptor_set::model_writes:
        return "model-writes";
    case descriptor_set::uniform_buffers:
        return "uniform-buffers";
    default:
        return "unknown";
    }
}

std::string describeImage(uint64_t frame, std::string_view role, const fsr4_411::Image& image) {
    std::ostringstream message;
    message << "[resource] frame=" << frame << " role=" << role
            << " present=" << (image.image ? "yes" : "no");

    if (image.image) {
        message << " image=" << handle(image.image) << " view=" << handle(image.view)
                << " format=" << vulkanFormat(image.format) << " active_size=" << image.width << 'x'
                << image.height
                << " backing_size=" << (image.resource_width ? image.resource_width : image.width)
                << 'x' << (image.resource_height ? image.resource_height : image.height)
                << " aspect_flags=" << hexValue(image.aspect);
    }

    return message.str();
}
} // namespace fsr4log

namespace fsr4runtime {
namespace {
std::filesystem::path diagnostic_root;
std::mutex diagnostic_root_mutex;
} // namespace

void setDiagnosticRoot(const std::filesystem::path& root) {
    const std::lock_guard lock(diagnostic_root_mutex);
    diagnostic_root = root;
}

std::filesystem::path diagnosticPath(const char* env_name, const char* filename) {
    if (const char* override_path = std::getenv(env_name); override_path && *override_path) {
        return std::filesystem::path(override_path);
    }

    const std::lock_guard lock(diagnostic_root_mutex);
    if (!diagnostic_root.empty()) {
        return diagnostic_root / filename;
    }

    return std::filesystem::path(filename);
}

void log(const std::string& text, fsr4log::Level level) noexcept {
    try {
        fsr4log::writeRecord(diagnosticPath("FSR4_VK_LOG_PATH", "fsr4.log"), level, text);
    } catch (...) {
    }
}
} // namespace fsr4runtime

namespace fsr4provider {
std::filesystem::path providerModuleDirectory() {
#if defined(_WIN32)
    HMODULE module{};
    wchar_t path[32768]{};

    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(&ffxCreateContext),
                           &module)) {
        const auto length = GetModuleFileNameW(module, path, 32768);

        if (length && length < 32768) {
            return std::filesystem::path(path).parent_path();
        }
    }
#endif
    return {};
}

std::filesystem::path providerLogPath() {
    if (const char* override_path = std::getenv("FSR4_VK_LOG_PATH");
        override_path && *override_path) {
        return std::filesystem::path(override_path);
    }

    const auto directory = providerModuleDirectory();
    return directory.empty() ? std::filesystem::path("fsr4.log") : directory / "fsr4.log";
}

void diagnosticLine(const std::string& text, fsr4log::Level level) noexcept {
    try {
        const auto path = providerLogPath();
        static std::once_flag started;
        bool wrote_first_record = false;
        std::call_once(started, [&] {
            fsr4log::writeRecord(path, level, text, true);
            wrote_first_record = true;
        });
        if (!wrote_first_record) {
            fsr4log::writeRecord(path, level, text);
        }
    } catch (...) {
    }
}

void diagnostic(const ProviderContext& context,
                const std::string& text,
                fsr4log::Level level) noexcept {
    (void)context;
    diagnosticLine(text, level);
}

void providerMessage(const ProviderContext& context,
                     uint32_t type,
                     const std::string& text) noexcept {
    const auto level = type == FFX_API_MESSAGE_TYPE_ERROR     ? fsr4log::Level::Error
                       : type == FFX_API_MESSAGE_TYPE_WARNING ? fsr4log::Level::Warning
                                                              : fsr4log::Level::Info;
    diagnostic(context, text, level);

    if (!context.message) {
        return;
    }

    try {
        const std::wstring wide(text.begin(), text.end());
        context.message(type, wide.c_str());
    } catch (...) {
    }
}
} // namespace fsr4provider
