# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Correct FP8 exposure-history coordinates."""

import argparse
from pathlib import Path
import re
import subprocess
import tempfile

def patch_exposure_assembly(text):
    lines = text.splitlines(keepends=True)
    output = []
    patched_fetches = 0

    for index, line in enumerate(lines):
        is_exposure_coordinate = "OpCompositeConstruct %v2uint %uint_1 %uint_0" in line
        next_is_fetch = index + 1 < len(lines) and "OpImageFetch" in lines[index + 1]

        if is_exposure_coordinate and next_is_fetch:
            line = line.replace("%uint_1 %uint_0", "%uint_0 %uint_0")
            patched_fetches += 1

        output.append(line)

    return "".join(output), patched_fetches


def patch_exposure(
    source, output, spirv_dis="spirv-dis", spirv_as="spirv-as", spirv_val="spirv-val"
):
    source, output = Path(source), Path(output)
    catalog_assets = Path(__file__).resolve().parents[4] / "assets"

    if output.exists():
        raise RuntimeError("Use a fresh exposure-patch output path")
    if output.resolve().is_relative_to(catalog_assets.resolve()):
        raise RuntimeError("Do not replace production assets")

    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="fsr4-exposure-", dir=output.parent
    ) as temporary:
        temporary = Path(temporary)
        assembly = temporary / "input.spvasm"
        candidate = temporary / "candidate.spv"
        subprocess.run([spirv_dis, str(source), "-o", str(assembly)], check=True)
        text, patched_fetches = patch_exposure_assembly(assembly.read_text())
        assembly.write_text(text)
        subprocess.run(
            [
                spirv_as,
                str(assembly),
                "-o",
                str(candidate),
                "--target-env",
                "vulkan1.2",
            ],
            check=True,
        )
        subprocess.run([spirv_val, str(candidate)], check=True)
        candidate.replace(output)

    return patched_fetches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    for name in ("spirv-dis", "spirv-as", "spirv-val"):
        parser.add_argument("--" + name, default=name)
    args = parser.parse_args()
    patch_exposure(args.source, args.output, args.spirv_dis, args.spirv_as, args.spirv_val)


if __name__ == "__main__":
    main()
