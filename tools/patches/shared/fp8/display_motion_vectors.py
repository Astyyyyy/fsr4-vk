# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Fetch display-resolution motion vectors directly."""

import argparse
from pathlib import Path
import re
import subprocess
import tempfile

FETCH = re.compile(
    r"^(?P<indent>\s*)(?P<result>%\S+) = "
    r"OpImageFetch %v4float %185 (?P<coord>%\S+) Lod %uint_0\s*$"
)


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def patch_motion_assembly(text: str, source: Path) -> str:
    output: list[str] = []
    patched = 0

    for line in text.splitlines():
        match = FETCH.match(line)
        if not match:
            output.append(line)
            continue

        index = patched
        patched += 1
        indent = match.group("indent")
        result = match.group("result")
        coord = match.group("coord")
        prefix = f"direct_mv_{index}"
        instructions = [
            f"%{prefix}_cb4_ptr = OpAccessChain %_ptr_Uniform_v4float %201 %uint_0 %uint_4",
            f"%{prefix}_cb4 = OpLoad %v4float %{prefix}_cb4_ptr",
            f"%{prefix}_cb4_u = OpBitcast %v4uint %{prefix}_cb4",
            f"%{prefix}_width = OpCompositeExtract %uint %{prefix}_cb4_u 3",
            f"%{prefix}_cb5_ptr = OpAccessChain %_ptr_Uniform_v4float %201 %uint_0 %uint_5",
            f"%{prefix}_cb5 = OpLoad %v4float %{prefix}_cb5_ptr",
            f"%{prefix}_cb5_u = OpBitcast %v4uint %{prefix}_cb5",
            f"%{prefix}_height = OpCompositeExtract %uint %{prefix}_cb5_u 0",
            f"%{prefix}_x = OpCompositeExtract %uint {coord} 0",
            f"%{prefix}_y = OpCompositeExtract %uint {coord} 1",
            f"%{prefix}_x_f = OpConvertUToF %float %{prefix}_x",
            f"%{prefix}_y_f = OpConvertUToF %float %{prefix}_y",
            f"%{prefix}_x_center = OpFAdd %float %{prefix}_x_f %float_0_5",
            f"%{prefix}_y_center = OpFAdd %float %{prefix}_y_f %float_0_5",
            f"%{prefix}_width_f = OpConvertUToF %float %{prefix}_width",
            f"%{prefix}_height_f = OpConvertUToF %float %{prefix}_height",
            f"%{prefix}_u = OpFDiv %float %{prefix}_x_center %{prefix}_width_f",
            f"%{prefix}_v = OpFDiv %float %{prefix}_y_center %{prefix}_height_f",
            f"%{prefix}_uv = OpCompositeConstruct %v2float %{prefix}_u %{prefix}_v",
            f"%{prefix}_sampled = OpSampledImage %130 %185 %197",
            f"{result} = OpImageSampleExplicitLod %v4float %{prefix}_sampled %{prefix}_uv Lod %float_0",
        ]
        output.extend(indent + instruction for instruction in instructions)

    if patched != 2:
        raise RuntimeError(
            f"{source}: expected two t1 motion fetches, patched {patched}"
        )
    return "\n".join(output) + "\n"


def patch_motion(
    source, output, spirv_dis="spirv-dis", spirv_as="spirv-as", spirv_val="spirv-val"
):
    source, output = Path(source), Path(output)
    if output.exists() or output.resolve().is_relative_to(
        (Path(__file__).resolve().parents[4] / "assets").resolve()
    ):
        raise RuntimeError("Use a fresh path outside production assets")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fsr4-motion-") as temporary:
        assembly = Path(temporary) / "input.spvasm"
        patched = Path(temporary) / "patched.spvasm"
        run(spirv_dis, str(source), "-o", str(assembly))
        patched.write_text(patch_motion_assembly(assembly.read_text(), source))
        run(spirv_as, "--target-env", "spv1.5", str(patched), "-o", str(output))
        run(spirv_val, "--target-env", "vulkan1.3", str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    for name in ("spirv-dis", "spirv-as", "spirv-val"):
        parser.add_argument("--" + name, default=name)
    args = parser.parse_args()
    patch_motion(args.source, args.output, args.spirv_dis, args.spirv_as, args.spirv_val)


if __name__ == "__main__":
    main()
