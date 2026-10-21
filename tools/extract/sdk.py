# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Read pinned SDK PE tables without loading or executing the Windows DLL."""

from pathlib import Path
import hashlib, json, re, struct

SDK_HASH = "d0dcccc74a43c44ba435b7a369b456e0970d8a4464e4bd683119b374f2c9fb46"
TIERS = ("1080", "2160", "4320")
PRESETS = ("native", "quality", "balanced", "performance", "custom", "ultraperf")
COLORS = ("linear", "nonlinear", "srgb", "pq")
SELECTION_FIELDS = [
    (0, 8, 8),
    (8, 0, 4),
    (16, 16, 8),
    (24, 24, 4),
    (28, 64, 4),
    (32, 104, 4),
    (36, 144, 4),
    (40, 184, 4),
    (44, 224, 4),
    (48, 264, 4),
    (56, 32, 8),
    (64, 40, 8),
    (72, 48, 8),
    (80, 56, 8),
    (88, 72, 8),
    (96, 80, 8),
    (104, 88, 8),
    (112, 96, 8),
    (120, 112, 8),
    (128, 120, 8),
    (136, 128, 8),
    (144, 136, 8),
    (152, 152, 8),
    (160, 160, 8),
    (168, 168, 8),
    (176, 176, 8),
    (184, 192, 8),
    (192, 200, 8),
    (200, 208, 8),
    (208, 216, 8),
    (216, 232, 8),
    (224, 240, 8),
    (232, 248, 8),
    (240, 256, 8),
    (248, 272, 8),
    (256, 280, 8),
    (264, 288, 8),
    (272, 296, 8),
]


class SDK:
    def __init__(self, path):
        self.path = Path(path)
        self.data = self.path.read_bytes()
        if hashlib.sha256(self.data).hexdigest() != SDK_HASH:
            raise ValueError("Unsupported SDK DLL; expected FSR 4.1.1.2740")
        pe = struct.unpack_from("<I", self.data, 0x3C)[0]
        count = struct.unpack_from("<H", self.data, pe + 6)[0]
        opt = pe + 24
        size = struct.unpack_from("<H", self.data, pe + 20)[0]
        self.image_base = struct.unpack_from("<Q", self.data, opt + 24)[0]
        self.sections = []
        for i in range(count):
            virtual_size, rva, raw_size, raw = struct.unpack_from(
                "<4I", self.data, opt + size + i * 40 + 8
            )
            self.sections.append((rva, raw, raw_size))
        self._int8_tables = None

    def offset(self, rva, size=1):
        for start, raw, length in self.sections:
            if start <= rva and rva + size <= start + length:
                return raw + rva - start
        raise ValueError(f"Unmapped SDK RVA/size: {rva:#x}/{size}")

    def read(self, rva, size):
        offset = self.offset(rva, size)
        return self.data[offset : offset + size]

    def uint(self, rva):
        return struct.unpack("<I", self.read(rva, 4))[0]

    def tables(self):
        if self._int8_tables is None:
            targets = [self.uint(0x8FA68 + 4 * i) for i in range(27)] + [0x8FA33]
            tables = []
            for start, end in zip(targets, targets[1:]):
                code = self.read(start, end - start)
                refs = []
                for m in re.finditer(b"\x48\x8d\x0d(.{4})", code, re.S):
                    rva = start + m.start() + 7 + struct.unpack("<i", m[1])[0]
                    if rva not in refs:
                        refs.append(rva)
                if len(refs) != 12:
                    raise ValueError(
                        f"Unexpected INT8 selector table layout at {start:#x}"
                    )
                tables.append(list(zip(refs[::2], refs[1::2])))
            self._int8_tables = tables
        return self._int8_tables

    def table_record(self, index_rva, records_rva, key):
        index = self.uint(index_rva + key * 4)
        return self.read(records_rva + index * 0x130, 0x130)

    def select(self, family, pass_index, tier, preset=1, flags=0, color=0, debug=False):
        if (
            not 0 <= pass_index < 27
            or not 0 <= tier < 3
            or not 0 <= preset < 6
            or not 0 <= flags < 16
            or not 0 <= color < 4
        ):
            raise ValueError("Invalid SDK selector configuration")
        key = (flags & 7) | (color << 3) | ((flags & 8) << 2)
        if family == "int8":
            pair = self.tables()[pass_index][tier * 2 + (0 if preset == 5 else 1)]
            if pass_index == 26:
                key = color | ((flags & 4)) | (int(debug) << 3)
            elif pass_index != 0:
                key = 0
        elif family == "fp8" and pass_index in (0, 26):
            if pass_index == 0:
                pair = [
                    (0xE98360, 0x1B2AAE0),
                    (0x14EE480, 0x1B32180),
                    (0xEC0700, 0x1B34B80),
                ][tier]
            else:
                pair = [
                    (0x12D6C30, 0x1B2FED0),
                    (0x13C3850, 0x1B2D1C0),
                    (0x1210B40, 0x1B25FA0),
                ][tier]
                key = color | (flags & 4) | (int(debug) << 3) | (tier << 4)
        else:
            raise ValueError("Unsupported SDK table selection")
        return self.table_record(*pair, key)

    def shader(self, record):
        size = struct.unpack_from("<I", record)[0]
        pointer = struct.unpack_from("<Q", record, 8)[0]
        if not 28 <= size <= 64 * 1024 * 1024:
            raise ValueError("Invalid selected shader size")
        rva = pointer - self.image_base
        data = self.read(rva, size)
        if data[:4] != b"DXBC" or struct.unpack_from("<I", data, 24)[0] != size:
            raise ValueError("Invalid selected DXBC container")
        return rva, data

    def selection(self, record):
        result = bytearray(0x118)
        for target, source, size in SELECTION_FIELDS:
            result[target : target + size] = record[source : source + size]
        return bytes(result)


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def extract_endpoints(sdk, output, debug=None):
    output = Path(output)
    if output.exists():
        raise ValueError("Use a fresh extraction directory")
    output.mkdir(parents=True)
    rows = ["family\ttier\tcolor\tflags\tpreset\tpass\trva\tsize\tfile"]
    for family in ("fp8", "int8"):
        for tier in range(len(TIERS)):
            for color, name in enumerate(COLORS):
                for flags in range(16):
                    for preset in (1, 5):
                        for debug_mode in ((False, True) if debug is None else (debug,)):
                            for stage in ((26,) if debug_mode else (0, 26)):
                                record = sdk.select(
                                    family, stage, tier, preset, flags, color, debug_mode
                                )
                                rva, data = sdk.shader(record)
                                suffix = "-debug" if debug is None and debug_mode else ""
                                path = (
                                    Path("sdk-" + family)
                                    / TIERS[tier]
                                    / name
                                    / f"{flags:02x}"
                                    / ("ultraperf" if preset == 5 else "standard")
                                    / f"pass-{stage:02}{suffix}.dxbc"
                                )
                                write(output / path, data)
                                rows.append(
                                    f"sdk-{family}\t{TIERS[tier]}\t{name}\t{flags}\t{preset}\t{stage}\t{rva}\t{len(data)}\t{path.as_posix()}"
                                )
    (output / "selection.tsv").write_text("\n".join(rows) + "\n")


def extract_rcas(sdk, output):
    output = Path(output)
    if output.exists():
        raise ValueError("Use a fresh RCAS extraction directory")
    output.mkdir(parents=True)
    records = {}
    for name, offset in {
        "linear": 0xDAA40, "linear-auto": 0xCFE50, "nonlinear": 0xD2940,
        "srgb": 0xCBF40, "pq": 0xD4940,
    }.items():
        size = struct.unpack_from("<I", sdk.data, offset + 24)[0]
        data = sdk.data[offset:offset + size]
        if data[:4] != b"DXBC" or len(data) != size:
            raise ValueError("Invalid SDK RCAS shader: " + name)
        write(output / (name + ".dxbc"), data)
        records[name] = {"offset": offset, "sha256": hashlib.sha256(data).hexdigest()}
    (output / "manifest.json").write_text(json.dumps({
        "source_sha256": SDK_HASH, "shaders": records,
    }, indent=2) + "\n")
