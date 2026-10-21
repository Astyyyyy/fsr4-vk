#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Compile overlay shaders into a C++ header."""
import argparse
from pathlib import Path
import struct
import subprocess
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "convert"))
from toolchain import find_tool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--glslang", help="override the bundled host GLSL compiler")
    parser.add_argument("--spirv-val", help="override the bundled host validator")
    args = parser.parse_args()
    args.glslang = find_tool("glslangValidator", args.glslang)
    args.spirv_val = find_tool("spirv-val", args.spirv_val)

    header = "#pragma once\n#include <cstdint>\nnamespace fsr4runtime {\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temporary:
        for readable, name in [(0, "kOverlaySpirv"), (1, "kOverlayBlendSpirv")]:
            binary = Path(temporary) / "overlay.spv"
            subprocess.run([args.glslang, "-V", "--target-env", "vulkan1.1",
                            f"-DREAD_OUTPUT={readable}", "-o", str(binary),
                            str(args.source)], check=True)
            subprocess.run([args.spirv_val, "--target-env", "vulkan1.1", str(binary)], check=True)
            blob = binary.read_bytes()
            words = struct.unpack("<" + "I" * (len(blob) // 4), blob)
            lines = [", ".join(f"0x{w:08x}u" for w in words[i:i+8]) for i in range(0, len(words), 8)]
            header += f"inline constexpr uint32_t {name}[] = {{\n    "
            header += ",\n    ".join(lines) + "\n};\n"
        header += "} // namespace fsr4runtime\n"
        generated = Path(temporary) / args.output.name
        generated.write_text(header, encoding="utf-8", newline="\n")
        generated.replace(args.output)


if __name__ == "__main__":
    main()
