#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Convert the SDK 4.1.1 prepass and postpass variants.

Read DLL tables directly and validate each converted shader.
"""

from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import importlib.util
import subprocess

SDK_HASH = "d0dcccc74a43c44ba435b7a369b456e0970d8a4464e4bd683119b374f2c9fb46"


def endpoint_alias(family, tier, color, flags, preset, pass_index, debug=False):
    """Map SDK permutations to prepass and postpass asset names."""
    family = family.removeprefix("sdk-")
    if family not in ("fp8", "int8") or pass_index not in (0, 26):
        raise ValueError("Unknown endpoint family/pass")
    root = Path(family) / "shaders/4.1.1/color-endpoints" / str(tier)
    exposure = "auto" if flags & 4 else "external"
    if pass_index == 0:
        motion = (
            "display-mv"
            if not flags & 2
            else "render-mv-inverted-depth" if flags & 1 else "render-mv-normal-depth"
        )
        jitter = "jitter.spv" if flags & 8 else "no-jitter.spv"
        return (root / "prepass" / color / motion / exposure / jitter).as_posix()
    root = root / "postpass" / color
    if family == "int8":
        root /= "ultraperf" if preset == 5 else "standard"
    return (root / (exposure + ("-debug.spv" if debug else ".spv"))).as_posix()


def main():
    from toolchain import find_tool

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sdk", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument(
        "translator",
        type=Path,
        nargs="?",
        help="dxil-spirv executable (defaults to the bundled submodule)",
    )
    parser.add_argument("--spirv-val", help="SPIR-V validator (auto-detected by default)")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--debug-postpass", action="store_const", const=True, default=None,
        help="convert only debug postpasses (default: normal and debug)",
    )
    selection.add_argument(
        "--no-debug-postpass", dest="debug_postpass", action="store_const",
        const=False, help="convert only normal endpoints",
    )
    args = parser.parse_args()
    if args.sdk.is_file() and hashlib.sha256(args.sdk.read_bytes()).hexdigest() != SDK_HASH:
        parser.error("Unsupported reference DLL: " + str(args.sdk))
    if args.output.exists():
        parser.error("Use a fresh output directory")
    try:
        args.translator = find_tool("dxil-spirv", args.translator, "DXIL_SPIRV")
        validator = find_tool("spirv-val", args.spirv_val)
    except ValueError as error:
        parser.error(str(error))
    args.output.mkdir(parents=True)
    spec = importlib.util.spec_from_file_location(
        "endpoint_sdk", Path(__file__).resolve().parents[1] / "extract/sdk.py"
    )
    sdk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sdk)
    extracted = args.sdk
    if not extracted.is_dir():
        extracted = args.output / "extracted"
        sdk.extract_endpoints(sdk.SDK(args.sdk), extracted, args.debug_postpass)
    files = sorted(extracted.rglob("*.dxbc"))
    if not files:
        parser.error("No extracted endpoint shaders")
    aliases, unique = {}, {}
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        relative = path.relative_to(extracted)
        family, tier, color, flags, preset, filename = relative.parts
        key = endpoint_alias(
            family,
            tier,
            color,
            int(flags, 16),
            5 if preset == "ultraperf" else 1,
            int(Path(filename).stem.split("-")[1]),
            args.debug_postpass is True or filename.endswith("-debug.dxbc"),
        )
        if key in aliases and aliases[key] != digest:
            raise ValueError("Endpoint settings cannot be collapsed: " + key)
        aliases[key] = digest
        unique.setdefault(digest, path)
    converted = args.output / "converted"
    converted.mkdir()

    def convert(item):
        digest, path = item
        output = converted / (digest + ".spv")
        flags = (
            ["--full-wmma", "1", "1"]
            if "sdk-fp8" in path.parts
            else ["--enable-shader-i8-dot", "--assume-ssbo-32bit-wrapping"]
        )
        subprocess.run(
            [
                str(args.translator),
                str(path),
                "--output",
                str(output),
                "--bindless",
                "--ssbo-uav",
                "--ssbo-srv",
                *flags,
            ],
            check=True,
        )
        validation = subprocess.run(
            [validator, "--target-env", "vulkan1.3", str(output)],
            capture_output=True,
            text=True,
        )
        if validation.returncode:
            raise ValueError(str(path) + ": " + validation.stderr)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(convert, unique.items()))


if __name__ == "__main__":
    main()
