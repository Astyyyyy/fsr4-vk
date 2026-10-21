// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

#include "assets/assets.hpp"
#include "asset-index.hpp"
#include "zstd.h"

#include <algorithm>
#include <array>
#include <bit>
#include <fstream>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>

#if defined(FSR4_EMBEDDED_ASSETS) && defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#if defined(FSR4_ELF_ASSETS)
#include "embedded-assets.hpp"
extern "C" const std::uint8_t fsr4_embedded_assets_begin[];
#endif

namespace fsr4assets {
namespace {
struct AssetLocation {
    std::filesystem::path asset_root;
    std::string catalog_key;
};

AssetLocation resolveAssetLocation(const std::filesystem::path& path) {
    if (isEmbedded(path)) {
        return {root, path.lexically_relative(root).generic_string()};
    }

    // Use the nearest catalog, regardless of its parent directory names.
    for (auto directory = path.parent_path(); !directory.empty();) {
        std::error_code error;
        if (std::filesystem::is_regular_file(directory / "manifest.json", error)) {
            return {directory, path.lexically_relative(directory).generic_string()};
        }
        const auto parent = directory.parent_path();
        if (parent == directory) {
            break;
        }
        directory = parent;
    }
    return {{}, path.generic_string()};
}

std::span<const Payload> payloadCatalog(bool embedded) {
    return embedded ? std::span<const Payload>(payload_index)
                    : std::span<const Payload>(sidecar_payload_index);
}

std::span<const Alias> aliasCatalog(bool embedded) {
    return embedded ? std::span<const Alias>(asset_aliases)
                    : std::span<const Alias>(sidecar_asset_aliases);
}

const Alias* findAssetAlias(const std::string& catalog_key, bool embedded) {
    const auto index = aliasCatalog(embedded);

    const auto found = std::lower_bound(
        index.begin(), index.end(), catalog_key, [](const Alias& entry, const std::string& value) {
            return entry.key < value;
        });

    return found != index.end() && found->key == catalog_key ? &*found : nullptr;
}

std::vector<std::uint8_t> readFileBytes(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);

    if (!input) {
        throw std::runtime_error("cannot open asset: " + path.string());
    }

    const auto file_size = input.tellg();

    if (file_size < 0 || file_size > 64 * 1024 * 1024) {
        throw std::runtime_error("invalid asset size");
    }

    std::vector<std::uint8_t> file_contents(static_cast<std::size_t>(file_size));
    input.seekg(0);

    if (!input.read(reinterpret_cast<char*>(file_contents.data()), file_size)) {
        throw std::runtime_error("cannot read asset");
    }

    return file_contents;
}

std::vector<std::uint8_t>
readStoredPayload(const Payload& entry, const AssetLocation& location, bool embedded) {
#if defined(FSR4_ELF_ASSETS)
    if (embedded) {
        const auto embedded_index = entry.resource - 100u;

        if (entry.resource < 100u || embedded_index >= std::size(embedded_payload_offsets)) {
            throw std::runtime_error("invalid embedded ELF asset ID");
        }

        const auto offset = embedded_payload_offsets[embedded_index];

        if (offset > embedded_asset_bytes || entry.packed_size > embedded_asset_bytes - offset) {
            throw std::runtime_error("invalid embedded ELF asset range");
        }

        const auto data = fsr4_embedded_assets_begin + offset;
        const std::span<const std::uint8_t> packed(data, entry.packed_size);
        return entry.compressed ? decompressPayload(packed, entry.stored_size)
                                : std::vector<std::uint8_t>(packed.begin(), packed.end());
    }
#elif defined(FSR4_EMBEDDED_ASSETS) && defined(_WIN32)
    if (embedded) {
        HMODULE module{};

        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                    GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                reinterpret_cast<LPCWSTR>(&read),
                                &module)) {
            throw std::runtime_error("cannot locate asset module");
        }

        const auto resource =
            FindResourceW(module, MAKEINTRESOURCEW(entry.resource), MAKEINTRESOURCEW(10));

        if (!resource || SizeofResource(module, resource) != entry.packed_size) {
            throw std::runtime_error("missing or incorrectly sized embedded asset");
        }

        const auto loaded = LoadResource(module, resource);
        const auto data = static_cast<const std::uint8_t*>(LockResource(loaded));

        if (!data) {
            throw std::runtime_error("cannot load embedded asset");
        }

        const std::span<const std::uint8_t> packed(data, entry.packed_size);
        return entry.compressed ? decompressPayload(packed, entry.stored_size)
                                : std::vector<std::uint8_t>(packed.begin(), packed.end());
    }
#else
    (void)embedded;
#endif
    return readFileBytes(location.asset_root / entry.file);
}

std::vector<std::uint8_t> reconstructPayload(unsigned payload_id,
                                             const AssetLocation& location,
                                             bool embedded,
                                             unsigned delta_depth = 0) {
    const auto catalog = payloadCatalog(embedded);

    if (payload_id >= catalog.size()) {
        throw std::runtime_error("invalid asset ID");
    }

    const auto& payload_entry = catalog[payload_id];
    auto reconstructed_bytes = readStoredPayload(payload_entry, location, embedded);

    if (reconstructed_bytes.size() != payload_entry.stored_size) {
        throw std::runtime_error("asset stored size mismatch");
    }

    verifySha256(reconstructed_bytes, payload_entry.stored_sha256);

    if (payload_entry.base >= 0) {
        if (unsigned(payload_entry.base) >= payload_id ||
            delta_depth >= (embedded ? max_delta_depth : sidecar_max_delta_depth)) {
            throw std::runtime_error("invalid/excessive-depth shader delta base");
        }

        auto base_bytes =
            reconstructPayload(unsigned(payload_entry.base), location, embedded, delta_depth + 1);
        reconstructed_bytes = applyDelta(base_bytes, reconstructed_bytes);
    }

    if (reconstructed_bytes.size() != payload_entry.size) {
        throw std::runtime_error("asset reconstructed size mismatch");
    }

    verifySha256(reconstructed_bytes, payload_entry.sha256);

    return reconstructed_bytes;
}
} // namespace

std::vector<std::uint8_t> decompressPayload(std::span<const std::uint8_t> packed,
                                            std::size_t expected_size) {
    constexpr std::size_t max_size = 64 * 1024 * 1024;
    if (expected_size > max_size || packed.empty() || packed.size() > max_size) {
        throw std::runtime_error("invalid compressed payload size");
    }
    const auto content_size = ZSTD_getFrameContentSize(packed.data(), packed.size());
    const auto frame_size = ZSTD_findFrameCompressedSize(packed.data(), packed.size());
    if (content_size != expected_size || ZSTD_isError(frame_size) || frame_size != packed.size()) {
        throw std::runtime_error("invalid compressed payload frame");
    }
    std::vector<std::uint8_t> decoded(expected_size);
    const auto result =
        ZSTD_decompress(decoded.data(), decoded.size(), packed.data(), packed.size());
    if (ZSTD_isError(result) || result != expected_size) {
        throw std::runtime_error("cannot decompress payload");
    }
    return decoded;
}

bool isEmbedded(const std::filesystem::path& path) {
#if defined(FSR4_EMBEDDED_ASSETS)
    return !path.empty() && *path.begin() == root;
#else
    (void)path;
    return false;
#endif
}

bool exists(const std::filesystem::path& path) {
    if (!isEmbedded(path) && std::filesystem::exists(path)) {
        return true;
    }

    const auto location = resolveAssetLocation(path);

    if (location.catalog_key.empty()) {
        return false;
    }

    const bool embedded = isEmbedded(path);
    const auto index = payloadCatalog(embedded);

    if (const auto alias = findAssetAlias(location.catalog_key, embedded)) {
        return embedded ||
               std::filesystem::is_regular_file(location.asset_root / index[alias->payload].file);
    }

    const auto prefix = location.catalog_key + '/';
    const auto names = aliasCatalog(embedded);
    const auto first = std::lower_bound(
        names.begin(), names.end(), prefix, [](const Alias& entry, const std::string& catalog_key) {
            return entry.key < catalog_key;
        });

    return first != names.end() && std::string(first->key).starts_with(prefix) &&
           (embedded ||
            std::filesystem::is_regular_file(location.asset_root / index[first->payload].file));
}

std::vector<std::uint8_t> read(const std::filesystem::path& path) {
    const auto location = resolveAssetLocation(path);
    // Read catalog entries through the decoder; external full modules remain overrides.
    const bool embedded = isEmbedded(path);
    const auto alias = findAssetAlias(location.catalog_key, embedded);
    const auto index = payloadCatalog(embedded);

    if (!isEmbedded(path) && std::filesystem::is_regular_file(path) &&
        (!alias || location.asset_root / index[alias->payload].file != path)) {
        return readFileBytes(path);
    }

    if (alias) {
        return reconstructPayload(alias->payload, location, isEmbedded(path));
    }

    if (!isEmbedded(path)) {
        return readFileBytes(path);
    }

    throw std::runtime_error("unknown embedded asset: " + path.string());
}
} // namespace fsr4assets

namespace fsr4assets {
std::vector<std::uint8_t> applyDelta(std::span<const std::uint8_t> base,
                                     std::span<const std::uint8_t> delta) {
    constexpr std::size_t max_payload_bytes = 64 * 1024 * 1024;
    constexpr std::uint8_t magic[] = {'F', '4', 'D', 'E', 'L', 'T', 'A'};

    if (delta.size() < 16 || delta.size() > max_payload_bytes || base.size() > max_payload_bytes ||
        !std::equal(std::begin(magic), std::end(magic), delta.begin()) ||
        (delta[7] != '1' && delta[7] != '2')) {
        throw std::runtime_error("invalid shader delta header");
    }

    std::size_t cursor = 8;
    const auto read_u32_le = [&]() -> std::uint32_t {
        if (delta.size() - cursor < 4) {
            throw std::runtime_error("truncated shader delta");
        }

        std::uint32_t word_value = 0;

        for (unsigned i = 0; i < 4; ++i) {
            word_value |= std::uint32_t(delta[cursor++]) << (8 * i);
        }

        return word_value;
    };

    const auto output_size = read_u32_le();
    const auto base_size = read_u32_le();

    if (output_size > max_payload_bytes || base_size != base.size()) {
        throw std::runtime_error("shader delta base/output size mismatch");
    }

    std::vector<std::uint8_t> relocated_base;

    if (delta[7] == '2') {
        if (base.size() % 4) {
            throw std::runtime_error("unaligned relocation base");
        }

        const auto relocation_count = read_u32_le();

        if (relocation_count > (delta.size() - cursor) / 12) {
            throw std::runtime_error("truncated relocation table");
        }

        struct RelocationRange {
            std::uint32_t first;
            std::uint32_t last;
            std::uint32_t replacement;
        };

        std::vector<RelocationRange> relocation_ranges;
        relocation_ranges.reserve(relocation_count);

        for (std::uint32_t i = 0; i < relocation_count; ++i) {
            const auto first = read_u32_le();
            const auto last = read_u32_le();
            const auto replacement = read_u32_le();

            if (last < first || std::uint64_t(replacement) + last - first > UINT32_MAX ||
                (!relocation_ranges.empty() && first <= relocation_ranges.back().last)) {
                throw std::runtime_error("invalid relocation range");
            }

            relocation_ranges.push_back({first, last, replacement});
        }

        relocated_base.assign(base.begin(), base.end());

        for (std::size_t i = 0; i < relocated_base.size(); i += 4) {
            std::uint32_t word_value = 0;

            for (unsigned byte_index = 0; byte_index < 4; ++byte_index) {
                word_value |= std::uint32_t(relocated_base[i + byte_index]) << (8 * byte_index);
            }

            auto matching_range =
                std::upper_bound(relocation_ranges.begin(),
                                 relocation_ranges.end(),
                                 word_value,
                                 [](std::uint32_t value, const RelocationRange& range) {
                                     return value < range.first;
                                 });

            if (matching_range != relocation_ranges.begin()) {
                --matching_range;

                if (word_value <= matching_range->last) {
                    word_value = matching_range->replacement + word_value - matching_range->first;

                    for (unsigned byte_index = 0; byte_index < 4; ++byte_index) {
                        relocated_base[i + byte_index] =
                            std::uint8_t(word_value >> (8 * byte_index));
                    }
                }
            }
        }

        base = relocated_base;
    }

    std::vector<std::uint8_t> output;
    output.reserve(output_size);

    constexpr std::uint8_t copy_from_base = 0;
    constexpr std::uint8_t append_literal = 1;

    while (cursor < delta.size()) {
        const auto operation = delta[cursor++];
        std::span<const std::uint8_t> piece;

        if (operation == copy_from_base) {
            const auto offset = read_u32_le();
            const auto length = read_u32_le();

            if (offset > base.size() || length > base.size() - offset) {
                throw std::runtime_error("shader delta copy outside base");
            }

            piece = base.subspan(offset, length);
        } else if (operation == append_literal) {
            const auto length = read_u32_le();

            if (length > delta.size() - cursor) {
                throw std::runtime_error("truncated shader delta literal");
            }

            piece = delta.subspan(cursor, length);
            cursor += length;
        } else {
            throw std::runtime_error("unknown shader delta operation");
        }

        if (piece.empty() || piece.size() > output_size - output.size()) {
            throw std::runtime_error("shader delta output overflow");
        }

        output.insert(output.end(), piece.begin(), piece.end());
    }

    if (output.size() != output_size) {
        throw std::runtime_error("incomplete shader delta output");
    }

    return output;
}
} // namespace fsr4assets

namespace fsr4assets {
namespace {
constexpr std::size_t block_bytes = 64;
constexpr std::size_t word_bytes = 4;
constexpr std::size_t length_field_bytes = 8;
constexpr std::size_t words_per_block = block_bytes / word_bytes;
constexpr std::size_t round_count = 64;

using HashState = std::array<std::uint32_t, 8>;
using MessageSchedule = std::array<std::uint32_t, round_count>;

constexpr std::array<std::uint32_t, round_count> round_constants{
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

constexpr HashState initial_hash_state{
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

std::uint32_t scheduleSigma0(std::uint32_t word) {
    return std::rotr(word, 7) ^ std::rotr(word, 18) ^ (word >> 3);
}

std::uint32_t scheduleSigma1(std::uint32_t word) {
    return std::rotr(word, 17) ^ std::rotr(word, 19) ^ (word >> 10);
}

std::uint32_t compressionSigma0(std::uint32_t word) {
    return std::rotr(word, 2) ^ std::rotr(word, 13) ^ std::rotr(word, 22);
}

std::uint32_t compressionSigma1(std::uint32_t word) {
    return std::rotr(word, 6) ^ std::rotr(word, 11) ^ std::rotr(word, 25);
}

std::uint32_t chooseBits(std::uint32_t selector, std::uint32_t when_set, std::uint32_t when_clear) {
    return (selector & when_set) ^ (~selector & when_clear);
}

std::uint32_t majorityBits(std::uint32_t first, std::uint32_t second, std::uint32_t third) {
    return (first & second) ^ (first & third) ^ (second & third);
}

std::uint8_t paddedInputByte(std::span<const std::uint8_t> input,
                             std::size_t position,
                             std::size_t padded_size,
                             std::uint64_t input_bit_length) {
    if (position < input.size()) {
        return input[position];
    }

    if (position == input.size()) {
        return 0x80;
    }

    if (position >= padded_size - length_field_bytes) {
        const auto bytes_from_end = padded_size - 1 - position;
        const auto bit_shift = bytes_from_end * 8;

        return static_cast<std::uint8_t>(input_bit_length >> bit_shift);
    }

    return 0;
}

MessageSchedule buildMessageSchedule(std::span<const std::uint8_t> input,
                                     std::size_t block_index,
                                     std::size_t padded_size,
                                     std::uint64_t input_bit_length) {
    MessageSchedule schedule{};
    const auto block_start = block_index * block_bytes;

    for (std::size_t byte_index = 0; byte_index < block_bytes; ++byte_index) {
        const auto byte =
            paddedInputByte(input, block_start + byte_index, padded_size, input_bit_length);
        const auto word_index = byte_index / word_bytes;
        const auto byte_in_word = byte_index % word_bytes;
        const auto bit_shift = (word_bytes - 1 - byte_in_word) * 8;

        schedule[word_index] |= static_cast<std::uint32_t>(byte) << bit_shift;
    }

    for (std::size_t word_index = words_per_block; word_index < round_count; ++word_index) {
        const auto earlier_mix = scheduleSigma0(schedule[word_index - 15]);
        const auto recent_mix = scheduleSigma1(schedule[word_index - 2]);

        schedule[word_index] =
            schedule[word_index - 16] + earlier_mix + schedule[word_index - 7] + recent_mix;
    }

    return schedule;
}

void compressBlock(HashState& hash_state, const MessageSchedule& schedule) {
    auto [a, b, c, d, e, f, g, h] = hash_state;

    for (std::size_t round = 0; round < round_count; ++round) {
        const auto mixed_e = compressionSigma1(e);
        const auto chosen_bits = chooseBits(e, f, g);
        const auto message_sum =
            h + mixed_e + chosen_bits + round_constants[round] + schedule[round];

        const auto mixed_a = compressionSigma0(a);
        const auto majority = majorityBits(a, b, c);
        const auto state_sum = mixed_a + majority;

        h = g;
        g = f;
        f = e;
        e = d + message_sum;

        d = c;
        c = b;
        b = a;
        a = message_sum + state_sum;
    }

    hash_state[0] += a;
    hash_state[1] += b;
    hash_state[2] += c;
    hash_state[3] += d;
    hash_state[4] += e;
    hash_state[5] += f;
    hash_state[6] += g;
    hash_state[7] += h;
}

std::string digestToHex(const HashState& hash_state) {
    constexpr char hex_digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(hash_state.size() * 8);

    for (const auto word : hash_state) {
        for (int bit_shift = 28; bit_shift >= 0; bit_shift -= 4) {
            result += hex_digits[(word >> bit_shift) & 0xf];
        }
    }

    return result;
}
} // namespace

std::string sha256(std::span<const std::uint8_t> bytes) {
    auto hash_state = initial_hash_state;

    // Reserve room for the 0x80 marker and length field, adding a block if needed.
    const auto final_block_has_room = bytes.size() % block_bytes < block_bytes - length_field_bytes;
    const auto padding_blocks = final_block_has_room ? 1 : 2;
    const auto block_count = bytes.size() / block_bytes + padding_blocks;
    const auto padded_size = block_count * block_bytes;
    const auto input_bit_length = static_cast<std::uint64_t>(bytes.size()) * 8;

    for (std::size_t block_index = 0; block_index < block_count; ++block_index) {
        const auto schedule =
            buildMessageSchedule(bytes, block_index, padded_size, input_bit_length);

        compressBlock(hash_state, schedule);
    }

    return digestToHex(hash_state);
}

void verifySha256(std::span<const std::uint8_t> bytes, const char* expected) {
    if (sha256(bytes) != expected) {
        throw std::runtime_error("asset SHA256 mismatch");
    }
}
} // namespace fsr4assets
