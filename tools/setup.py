# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Extract, convert and patch the complete FSR 4.1.1 asset catalog."""

import argparse
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "assets"))
import workflow
from workflow import ROOT


def setup(args):
    extraction = args.assets / "extracted"
    if args.dll:
        sdk = workflow.load("setup_sdk", HERE / "extract/sdk.py")
        sdk.SDK(args.dll)
    if extraction.exists():
        workflow.validate_extraction(extraction)
    elif args.dll:
        workflow.run(HERE / "extract/extract.py", "--dll", args.dll, "--output", extraction)
    else:
        raise ValueError("Provide --dll SDK_DLL for the first extraction")
    shared = ["--assets", args.assets]
    for name in ("translator", "spirv_tools"):
        value = getattr(args, name)
        if value:
            shared += ["--" + name.replace("_", "-"), value]
    conversion = [*shared]
    if args.library:
        conversion += ["--library", args.library]
    workflow.run(HERE / "convert/convert.py", *conversion)
    workflow.run(HERE / "patches/patch.py", *shared)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", type=Path, help="SDK DLL (required for the first extraction)")
    parser.add_argument("--assets", type=Path, default=ROOT / "assets")
    parser.add_argument("--translator", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--spirv-tools", type=Path)
    args = parser.parse_args()
    try:
        setup(args)
    except (ValueError, RuntimeError, OSError, KeyError, subprocess.CalledProcessError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
