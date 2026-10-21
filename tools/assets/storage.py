# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Encode, compress and verify asset payloads."""

import bisect, difflib, hashlib, json, struct
from collections import defaultdict, Counter

MAGIC = b"F4DELTA1"

LIMIT = 64 * 1024 * 1024


def copy_insert(base, target, header=None):
    if max(len(base), len(target)) > LIMIT:
        raise ValueError("Shader exceeds delta format limit")
    # Limit candidate matches for repeated SPIR-V words.
    index = defaultdict(list)
    for pos in range(0, len(base) - 15, 4):
        token = base[pos : pos + 16]
        if len(index[token]) < 8:
            index[token].append(pos)
    out = bytearray(
        header
        if header is not None
        else MAGIC + struct.pack("<II", len(target), len(base))
    )
    literal = bytearray()
    pos = 0

    def flush():
        if literal:
            out.extend(b"\x01" + struct.pack("<I", len(literal)) + literal)
            literal.clear()

    while pos < len(target):
        best, source = 0, 0
        for candidate in index.get(target[pos : pos + 16], ()):
            length = 16
            while (
                candidate + length + 4 <= len(base)
                and pos + length + 4 <= len(target)
                and base[candidate + length : candidate + length + 4]
                == target[pos + length : pos + length + 4]
            ):
                length += 4
            while (
                candidate + length < len(base)
                and pos + length < len(target)
                and base[candidate + length] == target[pos + length]
            ):
                length += 1
            if length > best:
                best, source = length, candidate
        if best >= 16:
            flush()
            out.extend(b"\x00" + struct.pack("<II", source, best))
            pos += best
        else:
            take = min(4, len(target) - pos)
            literal.extend(target[pos : pos + take])
            pos += take
    flush()
    return bytes(out)


def word_ranges(base, target):
    """Encode word substitutions that reduce differences after shader IDs are renumbered.

    Reconstruct and hash-check every candidate. The runtime only applies byte edits.
    """

    def words(blob):
        if len(blob) < 20 or len(blob) % 4:
            return None
        values = struct.unpack("<" + "I" * (len(blob) // 4), blob)
        if values[0] != 0x07230203:
            return None
        rows = []
        pos = 5
        while pos < len(values):
            count = values[pos] >> 16
            if not count or pos + count > len(values):
                return None
            rows.append(values[pos : pos + count])
            pos += count
        return values, rows

    b, t = words(base), words(target)
    if b is None or t is None:
        return []
    bw, bi = b
    tw, ti = t
    matches = difflib.SequenceMatcher(
        None, [r[0] for r in bi], [r[0] for r in ti], autojunk=True
    )
    pairs = defaultdict(Counter)
    for match in matches.get_matching_blocks():
        for x, y in zip(
            bi[match.a : match.a + match.size], ti[match.b : match.b + match.size]
        ):
            for a, b in zip(x[1:], y[1:]):
                if 0 < a < bw[3] and 0 < b < tw[3]:
                    pairs[a][b] += 1
    mapping = {
        a: c.most_common(1)[0][0]
        for a, c in pairs.items()
        if a != c.most_common(1)[0][0]
        and c.most_common(1)[0][1] / sum(c.values()) >= 0.8
    }
    ranges = []
    for a, b in sorted(mapping.items()):
        if ranges and a == ranges[-1][1] + 1 and b == ranges[-1][2] + a - ranges[-1][0]:
            ranges[-1][1] = a
        else:
            ranges.append([a, a, b])
    return ranges


def relocate(base, ranges):
    starts = [r[0] for r in ranges]
    values = list(struct.unpack("<" + "I" * (len(base) // 4), base))
    for i, v in enumerate(values):
        j = bisect.bisect_right(starts, v) - 1
        if j >= 0 and v <= ranges[j][1]:
            values[i] = ranges[j][2] + v - ranges[j][0]
    return struct.pack("<" + "I" * len(values), *values)


def generate(base, target):
    plain = copy_insert(base, target)
    ranges = word_ranges(base, target)
    if not ranges:
        return plain
    header = b"F4DELTA2" + struct.pack("<III", len(target), len(base), len(ranges))
    header += b"".join(struct.pack("<III", *r) for r in ranges)
    mapped = copy_insert(relocate(base, ranges), target, header)
    chosen = mapped if len(mapped) < len(plain) else plain
    if reconstruct(base, chosen) != target:
        raise RuntimeError("Delta reconstruction mismatch")
    return chosen


def reconstruct(base, delta):
    if len(delta) < 16 or delta[:8] not in (MAGIC, b"F4DELTA2"):
        raise ValueError("Invalid delta header")
    size, base_size = struct.unpack_from("<II", delta, 8)
    if max(size, base_size, len(delta)) > LIMIT or base_size != len(base):
        raise ValueError("Invalid delta sizes")
    out = bytearray()
    pos = 16
    if delta[:8] == b"F4DELTA2":
        if len(delta) < 20 or len(base) % 4:
            raise ValueError("Invalid relocation header")
        (count,) = struct.unpack_from("<I", delta, 16)
        pos = 20
        if count > (len(delta) - pos) // 12:
            raise ValueError("Truncated relocations")
        ranges = []
        for _ in range(count):
            first, last, new = struct.unpack_from("<III", delta, pos)
            pos += 12
            if (
                last < first
                or new + last - first > 0xFFFFFFFF
                or (ranges and first <= ranges[-1][1])
            ):
                raise ValueError("Invalid relocation range")
            ranges.append([first, last, new])
        base = relocate(base, ranges)
    while pos < len(delta):
        op = delta[pos]
        pos += 1
        count = 8 if op == 0 else 4 if op == 1 else 0
        if not count or pos + count > len(delta):
            raise ValueError("Invalid delta instruction")
        if op == 0:
            offset, length = struct.unpack_from("<II", delta, pos)
            pos += 8
            if offset + length > len(base):
                raise ValueError("Delta copy outside base")
            piece = base[offset : offset + length]
        else:
            (length,) = struct.unpack_from("<I", delta, pos)
            pos += 4
            if pos + length > len(delta):
                raise ValueError("Truncated delta insert")
            piece = delta[pos : pos + length]
            pos += length
        if not length or len(out) + length > size:
            raise ValueError("Invalid delta output extent")
        out.extend(piece)
    if len(out) != size:
        raise ValueError("Incomplete delta output")
    return bytes(out)


def compress_shader(blob):
    try:
        from compression import zstd
    except ImportError:
        try:
            import zstandard
        except ImportError as error:
            raise RuntimeError(
                "Shader compression needs Python 3.14+ or pip install zstandard; "
                "alternatively configure FSR4_COMPRESS_SHADERS=OFF"
            ) from error
        packed = zstandard.ZstdCompressor(level=6, write_checksum=True).compress(blob)
    else:
        packed = zstd.compress(
            blob,
            options={
                zstd.CompressionParameter.compression_level: 6,
                zstd.CompressionParameter.checksum_flag: 1,
            },
        )
    # Compression never makes a payload larger.
    return (packed, "zstd") if len(packed) < len(blob) else (blob, "none")


def storage_index(path, catalog):
    if path is None:
        return [
            dict(
                id=r["id"],
                packed_size=r["stored_size"],
                compression="none",
                packed_sha256=r["stored_sha256"],
            )
            for r in catalog["payloads"]
        ]
    index = json.loads(path.read_text())
    if index.get("schema") != 1 or len(index["payloads"]) != len(catalog["payloads"]):
        raise ValueError("Invalid embedded storage index")
    for i, row in enumerate(index["payloads"]):
        if row["id"] != i or row["compression"] not in ("none", "zstd"):
            raise ValueError("Invalid embedded payload encoding")
    return index["payloads"]


def unpack_payload(packed, storage, original):
    if (
        len(packed) != storage["packed_size"]
        or hashlib.sha256(packed).hexdigest() != storage["packed_sha256"]
    ):
        raise ValueError("Embedded packed payload mismatch")
    if storage["compression"] == "zstd":
        try:
            from compression import zstd
        except ImportError:
            import zstandard

            blob = zstandard.ZstdDecompressor().decompress(
                packed, max_output_size=original["stored_size"]
            )
        else:
            blob = zstd.decompress(packed)
    else:
        blob = packed
    if (
        len(blob) != original["stored_size"]
        or hashlib.sha256(blob).hexdigest() != original["stored_sha256"]
    ):
        raise ValueError("Embedded decoded payload mismatch")
    return blob
