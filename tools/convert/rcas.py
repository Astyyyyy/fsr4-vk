# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Translate extracted RCAS DXBC into unpatched SPIR-V."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "extract"))
from sdk import SDK_HASH
from toolchain import find_tool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--translator")
    args = parser.parse_args()
    metadata = json.loads((args.input / "manifest.json").read_text())
    if metadata.get("source_sha256") != SDK_HASH:
        parser.error("Unsupported RCAS extraction")
    if args.output.exists():
        parser.error("Use a fresh output directory")
    translator = find_tool("dxil-spirv", args.translator, "DXIL_SPIRV")
    validator = find_tool("spirv-val")
    args.output.mkdir(parents=True)
    for name, record in metadata["shaders"].items():
        source = args.input / (name + ".dxbc")
        if hashlib.sha256(source.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("RCAS extraction changed: " + name)
        destination = args.output / (name + ".spv")
        subprocess.run([translator, str(source), "--output", str(destination),
                        "--bindless", "--ssbo-uav", "--ssbo-srv"], check=True)
        subprocess.run([validator, "--target-env", "vulkan1.1", str(destination)], check=True)


if __name__ == "__main__":
    main()
