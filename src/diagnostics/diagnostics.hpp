// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <type_traits>
#include <vulkan/vulkan.h>

namespace fsr4_411 {
struct Image;
}

namespace fsr4log {
enum class Level { Debug, Info, Warning, Error };

std::string formatRecord(std::chrono::system_clock::time_point timestamp,
                         Level level,
                         std::string_view message);
void writeRecord(const std::filesystem::path& path,
                 Level level,
                 std::string_view message,
                 bool truncate = false) noexcept;

std::string hexValue(std::uint64_t value);

template <typename Handle> std::string handle(Handle value) {
    if constexpr (std::is_pointer_v<Handle>) {
        return hexValue(reinterpret_cast<std::uintptr_t>(value));
    } else {
        return hexValue(static_cast<std::uint64_t>(value));
    }
}

std::string vulkanFormat(VkFormat format);
std::string memoryProperties(VkMemoryPropertyFlags flags);
const char* descriptorSetName(uint32_t set_index);
std::string describeImage(uint64_t frame, std::string_view role, const fsr4_411::Image& image);
} // namespace fsr4log

namespace fsr4runtime {
void setDiagnosticRoot(const std::filesystem::path& root);

std::filesystem::path diagnosticPath(const char* env_name, const char* filename);

void log(const std::string& text, fsr4log::Level level = fsr4log::Level::Info) noexcept;

} // namespace fsr4runtime
