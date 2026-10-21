// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace fsr4assets {
inline constexpr const char* root = "__fsr4_embedded__";

bool isEmbedded(const std::filesystem::path& path);
bool exists(const std::filesystem::path& path);
std::vector<std::uint8_t> read(const std::filesystem::path& path);
std::vector<std::uint8_t> decompressPayload(std::span<const std::uint8_t> packed,
                                            std::size_t expected_size);
} // namespace fsr4assets

namespace fsr4assets {
// Find sidecar assets beside the provider DLL or shared library.
inline std::optional<std::filesystem::path> adjacentRoot(const std::filesystem::path& module) {
    for (const char* name : {"assets", "fsr4-vulkan-assets"}) {
        const auto root = module.parent_path() / name;
        if (std::filesystem::is_regular_file(root / "manifest.json")) {
            return root;
        }
    }
    return std::nullopt;
}
} // namespace fsr4assets

namespace fsr4assets {
std::vector<std::uint8_t> applyDelta(std::span<const std::uint8_t> base,
                                     std::span<const std::uint8_t> delta);
}

namespace fsr4assets {
std::string sha256(std::span<const std::uint8_t> bytes);
void verifySha256(std::span<const std::uint8_t> bytes, const char* expected);
} // namespace fsr4assets
