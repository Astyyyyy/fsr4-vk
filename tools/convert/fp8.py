# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Convert extracted FP8 DXBC into unpatched SPIR-V."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

from auto_exposure import convert as convert_exposure

HERE = Path(__file__).resolve().parent


def run(*arguments):
    subprocess.run([str(value) for value in arguments], check=True)


def main():
    from toolchain import find_library, find_tool

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, nargs="?", default=Path("assets/extracted/fp8")
    )
    parser.add_argument(
        "output", type=Path, nargs="?", default=Path("build/converted-fp8")
    )
    parser.add_argument(
        "--translator",
        type=Path,
        help="dxil-spirv executable (defaults to the bundled submodule)",
    )
    parser.add_argument(
        "--library", type=Path, help="dxil-spirv C shared library for SPD conversion"
    )
    parser.add_argument("--spirv-val", help="SPIR-V validator (auto-detected by default)")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output directory; catalog assets are immutable")
    for tier in ("1080", "2160", "4320"):
        if not (args.input / "sm66" / tier).is_dir():
            parser.error("Missing input tier: " + tier)
    if not (args.input / "auto-exposure.dxbc").is_file():
        parser.error("Missing shared auto-exposure.dxbc; rerun extraction")
    try:
        args.translator = find_tool("dxil-spirv", args.translator, "DXIL_SPIRV")
        args.spirv_val = find_tool("spirv-val", args.spirv_val)
        library = find_library(args.library, args.translator)
    except ValueError as error:
        parser.error(str(error))
    args.output.mkdir(parents=True)
    for tier in ("1080", "2160", "4320"):
        destination = args.output / tier
        destination.mkdir()
        for source in sorted((args.input / "sm66" / tier).glob("*.dxbc")):
            target = destination / source.with_suffix(".spv").name
            run(
                args.translator,
                source,
                "--output",
                target,
                "--bindless",
                "--ssbo-uav",
                "--ssbo-srv",
                "--full-wmma",
                "1",
                "1",
            )
            run(args.spirv_val, target)
        if (args.input / "initializers.bin").is_file():
            shutil.copy2(
                args.input / "initializers.bin", destination / "initializers.bin"
            )
        if (args.input / "weights").is_dir():
            for source in (args.input / "weights").glob("weights_*.bin"):
                shutil.copy2(source, destination / source.name)
            shutil.copy2(
                args.input / "weights/weights_balanced.bin", destination / "weights.bin"
            )
    convert_exposure(
        args.input / "auto-exposure.dxbc", args.output / "auto-exposure.spv", library
    )
    run(args.spirv_val, args.output / "auto-exposure.spv")


if __name__ == "__main__":
    main()
