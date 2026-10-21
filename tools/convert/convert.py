# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Convert extracted SDK shaders into the stock asset folders."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "assets"))
from workflow import ROOT, run, generated_index, prepare_assets, validate_extraction, baseline, install


def install_stock(raw, assets):
    manifest = json.loads((assets / "manifest.json").read_text())
    inputs = generated_index(raw)
    files = {}
    for row in manifest["payloads"]:
        if baseline(row):
            if row["sha256"] not in inputs:
                raise ValueError("Stock asset was not regenerated: " + row["file"])
            files[row["file"]] = inputs[row["sha256"]]
    for name, index in manifest["aliases"].items():
        if "/stock/" in name:
            row = manifest["payloads"][index]
            files[name] = inputs[row["sha256"]]
    install(files, assets)


def convert(args):
    from toolchain import find_library, find_tool, patching_tools

    args.input = args.input or args.assets / "extracted"
    validate_extraction(args.input)
    prepare_assets(args.assets)
    tools = patching_tools(args.spirv_tools)
    os.environ["FSR4_SPIRV_TOOLS"] = str(tools)
    os.environ["PATH"] = str(tools) + os.pathsep + os.environ.get("PATH", "")
    translator = find_tool("dxil-spirv", args.translator, "DXIL_SPIRV")
    library = find_library(args.library, translator)
    (ROOT / "build").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="convert-", dir=ROOT / "build") as directory:
        raw = Path(directory)
        run(HERE / "fp8.py", args.input / "fp8", raw / "fp8",
            "--translator", translator, "--library", library)
        run(HERE / "int8.py", args.input / "int8", raw / "int8",
            "--dxil-spirv", translator, "--auto-exposure", raw / "fp8/auto-exposure.spv")
        run(HERE / "endpoints.py", args.input / "endpoints", raw / "endpoints", translator)
        install_stock(raw, args.assets)
    print("Converted stock assets:", args.assets.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="extraction directory (defaults to ASSETS/extracted)")
    parser.add_argument("--assets", type=Path, default=ROOT / "assets")
    parser.add_argument("--translator", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--spirv-tools", type=Path)
    args = parser.parse_args()
    try:
        convert(args)
    except (ValueError, RuntimeError, OSError, KeyError, subprocess.CalledProcessError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
