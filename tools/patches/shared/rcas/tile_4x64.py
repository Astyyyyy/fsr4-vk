# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Patch RCAS bindings and use 4x64 tiles without changing the filter."""

import argparse
from pathlib import Path
import re
import subprocess
import tempfile


def adapt(text):
    # Reserve unused sampled-image slots so RCAS leaves model bindings intact.
    source_add = re.search(r"(%\w+) = OpIAdd %uint %\w+ %uint_18\b", text)
    if not source_add:
        raise ValueError("Missing RCAS input descriptor")
    access = re.search(
        r"(%\w+) = OpAccessChain %\w+ %\w+ " + source_add[1] + r"\b", text
    )
    image = re.search(r"(%\w+) = OpLoad %\w+ " + access[1] + r"\b", text)[1]
    text, count = re.subn(
        r"(OpIAdd %uint %\w+) %uint_18\b", r"\1 %rcas_input_slot", text
    )
    if count != 1:
        raise ValueError("Unexpected input descriptor count")
    text = re.sub(
        r"(OpIAdd %uint %\w+) %uint_(?:6|17)\b", r"\1 %rcas_exposure_slot", text
    )
    extension = re.search(r'(%\w+) = OpExtInstImport "GLSL.std.450"', text)[1]
    boolean = re.search(r"(%\w+) = OpTypeBool", text)
    if boolean:
        text = re.sub(re.escape(boolean[1]) + r"\b", "%rcas_bool", text)
    declarations = """%rcas_input_slot = OpConstant %uint 94
%rcas_exposure_slot = OpConstant %uint 95
%rcas_width_slot = OpConstant %uint 6
%rcas_height_slot = OpConstant %uint 7
%rcas_one = OpConstant %uint 1
%rcas_zero = OpConstantNull %v4float
%rcas_bool4 = OpTypeVector %rcas_bool 4
"""
    if not boolean:
        declarations = declarations.replace(
            "%rcas_bool4 =", "%rcas_bool = OpTypeBool\n%rcas_bool4 ="
        )
    start = text.index("       %main = OpFunction")
    text = text[:start] + declarations + text[start:]
    label = re.search(r"%\w+ = OpLabel\n", text[start:])
    location = start + label.end()
    text = (
        text[:location]
        + """%rcas_wptr = OpAccessChain %_ptr_PushConstant_uint %registers %rcas_width_slot
%rcas_hptr = OpAccessChain %_ptr_PushConstant_uint %registers %rcas_height_slot
%rcas_width = OpLoad %uint %rcas_wptr
%rcas_height = OpLoad %uint %rcas_hptr
%rcas_maxx = OpISub %uint %rcas_width %rcas_one
%rcas_maxy = OpISub %uint %rcas_height %rcas_one
"""
        + text[location:]
    )
    result = []
    reads = writes = 0
    for line in text.splitlines():
        fetch = re.match(
            r"\s*(%\w+) = OpImageFetch (%\w+) " + re.escape(image) + r" (%\w+)(.*)",
            line,
        )
        write = re.match(r"\s*OpImageWrite (%\w+) (%\w+) (%\w+)(.*)", line)
        if not fetch and not write:
            result.append(line)
            continue
        tag = f'rcas_{"load" if fetch else "store"}_{reads if fetch else writes}'
        coords = fetch[3] if fetch else write[2]
        result.extend(
            [
                f"%{tag}_x = OpCompositeExtract %uint {coords} 0",
                f"%{tag}_y = OpCompositeExtract %uint {coords} 1",
                f"%{tag}_vx = OpULessThan %rcas_bool %{tag}_x %rcas_width",
                f"%{tag}_vy = OpULessThan %rcas_bool %{tag}_y %rcas_height",
                f"%{tag}_valid = OpLogicalAnd %rcas_bool %{tag}_vx %{tag}_vy",
            ]
        )
        if fetch:
            if fetch[2] != "%v4float":
                raise ValueError("Unexpected fetch type")
            result.extend(
                [
                    f"%{tag}_cx = OpExtInst %uint {extension} UMin %{tag}_x %rcas_maxx",
                    f"%{tag}_cy = OpExtInst %uint {extension} UMin %{tag}_y %rcas_maxy",
                    f"%{tag}_coords = OpCompositeConstruct %v2uint %{tag}_cx %{tag}_cy",
                    f"%{tag}_raw = OpImageFetch %v4float {image} %{tag}_coords{fetch[4]}",
                    f"%{tag}_v4 = OpCompositeConstruct %rcas_bool4 %{tag}_valid %{tag}_valid %{tag}_valid %{tag}_valid",
                    f"{fetch[1]} = OpSelect %v4float %{tag}_v4 %{tag}_raw %rcas_zero",
                ]
            )
            reads += 1
        else:
            result.extend(
                [
                    f"OpSelectionMerge %{tag}_end None",
                    f"OpBranchConditional %{tag}_valid %{tag}_write %{tag}_end",
                    f"%{tag}_write = OpLabel",
                    line,
                    f"OpBranch %{tag}_end",
                    f"%{tag}_end = OpLabel",
                ]
            )
            writes += 1
    if reads != 20 or writes != 4:
        raise ValueError(f"Unexpected kernel footprint: {reads} reads, {writes} writes")
    return "\n".join(result) + "\n"


def one_pixel(text):
    """Cover the SDK 16x16 tile with 256 threads and one copy of the filter.

    The original 64-thread layout repeats the filter for four quadrants.
    Invocation bits 6 and 7 select those quadrants in the new layout.
    """
    if "LocalSize 64 1 1" not in text:
        raise ValueError("Unexpected SDK RCAS workgroup size")
    text = text.replace("LocalSize 64 1 1", "LocalSize 256 1 1")
    store = re.search(r"OpImageWrite %\w+ (%\w+) ", text)
    coordinates = re.search(
        re.escape(store[1]) + r" = OpCompositeConstruct %v2uint (%\w+) (%\w+)", text
    )
    if coordinates is None:
        raise ValueError("Missing first output coordinate")
    local = re.search(r"OpDecorate (%\w+) BuiltIn LocalInvocationId", text)[1]
    declarations = (
        "%one_c3 = OpConstant %uint 3\n"
        "%one_c4 = OpConstant %uint 4\n"
        "%one_c8 = OpConstant %uint 8\n"
    )
    position = text.index("       %main = OpFunction")
    text = text[:position] + declarations + text[position:]
    pointer = re.search(
        r"(%\w+) = OpAccessChain %_ptr_Input_uint " + re.escape(local) + r" %uint_0",
        text,
    )[1]
    lane = re.search(r"(%\w+) = OpLoad %uint " + re.escape(pointer), text)[1]
    for coordinate, shift, axis in zip(coordinates.groups(), (3, 4), ("x", "y")):
        pattern = (
            r"^(\s*)" + re.escape(coordinate) + r" = (OpBitwiseOr %uint %\w+ %\w+)$"
        )
        match = re.search(pattern, text, re.M)
        if not match:
            raise ValueError("Unexpected output coordinate formula: " + coordinate)
        replacement = (
            f"%one_base_{axis} = {match[2]}\n"
            f"%one_shift_{axis} = OpShiftRightLogical %uint {lane} %one_c{shift}\n"
            f"%one_bit_{axis} = OpBitwiseAnd %uint %one_shift_{axis} %one_c8\n"
            f"{coordinate} = OpBitwiseOr %uint %one_base_{axis} %one_bit_{axis}"
        )
        text = text[: match.start()] + replacement + text[match.end() :]
    end = re.search(r"%rcas_store_0_end = OpLabel\n", text).end()
    return text[:end] + "OpReturn\nOpFunctionEnd\n"


def interior(text):
    """Skip border checks for workgroups fully inside the image.

    Use a uniform branch and keep zero-filled reads at image edges.
    Preserve the SDK filter arithmetic after merging the five samples.
    """
    start = re.search(r"^\s*%rcas_load_0_x = ", text, re.M).start()
    end_pattern = r"^\s*%\w+ = OpSelect %v4float %rcas_load_4_v4 .*$"
    end = re.search(end_pattern, text, re.M).end()
    blocks = []
    pattern = (
        r"\s*%(rcas_load_(\d+))_x = OpCompositeExtract %uint (%\w+) 0\n"
        r".*?^(\s*(%\w+) = OpSelect %v4float %\1_v4 .*?)$"
    )

    def extract(match):
        tag, number, coordinate, _, destination = match.groups()
        blocks.append((tag, int(number), coordinate, destination, match[0]))
        return "\n"

    remainder = re.sub(pattern, extract, text[start:end], flags=re.M | re.S)
    if len(blocks) != 5:
        raise ValueError("Expected five RCAS neighborhood loads")
    destinations = {row[3] for row in blocks}
    before_branch, after_branch = [], []
    for line in remainder.splitlines():
        match = re.match(r"\s*%\w+ = OpCompositeExtract %float (%\w+) \d", line)
        target = after_branch if match and match[1] in destinations else before_branch
        target.append(line)
    bases = re.findall(r"(%\w+) = OpShiftLeftLogical %uint %\w+ %uint_4", text[:start])
    if len(bases) != 2:
        raise ValueError("Unexpected RCAS workgroup coordinate expressions")
    base_x, base_y = bases
    position = text.index("       %main = OpFunction")
    text = text[:position] + "%interior_c16 = OpConstant %uint 16\n" + text[position:]
    start = re.search(r"^\s*%rcas_load_0_x = ", text, re.M).start()
    end = re.search(end_pattern, text, re.M).end()
    body = (
        "\n".join(before_branch)
        + "\n"
        + f"""%interior_x0 = OpUGreaterThan %rcas_bool {base_x} %uint_0
%interior_y0 = OpUGreaterThan %rcas_bool {base_y} %uint_0
%interior_right = OpIAdd %uint {base_x} %interior_c16
%interior_bottom = OpIAdd %uint {base_y} %interior_c16
%interior_x1 = OpULessThan %rcas_bool %interior_right %rcas_width
%interior_y1 = OpULessThan %rcas_bool %interior_bottom %rcas_height
%interior_x = OpLogicalAnd %rcas_bool %interior_x0 %interior_x1
%interior_y = OpLogicalAnd %rcas_bool %interior_y0 %interior_y1
%interior_valid = OpLogicalAnd %rcas_bool %interior_x %interior_y
OpSelectionMerge %interior_merge None
OpBranchConditional %interior_valid %interior_fast %interior_border
%interior_fast = OpLabel
"""
    )
    image = re.search(r"%rcas_load_0_raw = OpImageFetch %v4float (%\w+)", text)[1]
    for _, number, coordinate, _, _ in blocks:
        body += (
            f"%interior_fast_{number} = OpImageFetch %v4float "
            f"{image} {coordinate} Lod %uint_0\n"
        )
    body += "OpBranch %interior_merge\n%interior_border = OpLabel\n"
    for _, number, _, destination, block in blocks:
        body += (
            re.sub(
                re.escape(destination) + r"(?= = OpSelect)",
                f"%interior_border_{number}",
                block,
            )
            + "\n"
        )
    body += "OpBranch %interior_merge\n%interior_merge = OpLabel\n"
    for _, number, _, destination, _ in blocks:
        body += (
            f"{destination} = OpPhi %v4float "
            f"%interior_fast_{number} %interior_fast "
            f"%interior_border_{number} %interior_border\n"
        )
    body += "\n".join(after_branch) + "\n"
    return text[:start] + body + text[end:]


def vertical_tile(text):
    """Use a 4x64 tile to improve locality without changing the filter or edge reads.

    Update the workgroup origin and interior check together. The runtime must
    dispatch enough 4x64 groups to cover the output.
    """
    local = re.search(r"OpDecorate (%\w+) BuiltIn LocalInvocationId", text)[1]
    pointer = re.search(
        r"(%\w+) = OpAccessChain %_ptr_Input_uint " + re.escape(local) + r" %uint_0",
        text,
    )[1]
    lane = re.search(r"(%\w+) = OpLoad %uint " + re.escape(pointer), text)[1]
    bases = re.findall(r"(%\w+) = OpShiftLeftLogical %uint (%\w+) %uint_4", text)
    if len(bases) != 2:
        raise ValueError("Unexpected RCAS tile origins")
    store = re.search(r"OpImageWrite %\w+ (%\w+) ", text)
    coordinates = re.search(
        re.escape(store[1]) + r" = OpCompositeConstruct %v2uint (%\w+) (%\w+)", text
    )
    if coordinates is None:
        raise ValueError("Missing RCAS output coordinates")
    position = text.index("       %main = OpFunction")
    text = text[:position] + """%tile_x_shift = OpConstant %uint 2
%tile_y_shift = OpConstant %uint 6
%tile_x_mask = OpConstant %uint 3
%tile_width = OpConstant %uint 4
%tile_height = OpConstant %uint 64
""" + text[position:]
    for (base, group), shift in zip(bases, ("%tile_x_shift", "%tile_y_shift")):
        text = text.replace(
            f"{base} = OpShiftLeftLogical %uint {group} %uint_4",
            f"{base} = OpShiftLeftLogical %uint {group} {shift}",
        )
    expressions = (
        f"OpBitwiseAnd %uint {lane} %tile_x_mask",
        f"OpShiftRightLogical %uint {lane} %tile_x_shift",
    )
    for coordinate, (base, _), expression, axis in zip(
        coordinates.groups(), bases, expressions, ("x", "y")
    ):
        pattern = r"^\s*" + re.escape(coordinate) + r" = OpBitwiseOr %uint %\w+ %\w+$"
        replacement = (
            f"%tile_{axis} = {expression}\n"
            f"{coordinate} = OpIAdd %uint {base} %tile_{axis}"
        )
        text, count = re.subn(pattern, replacement, text, flags=re.M)
        if count != 1:
            raise ValueError("Unexpected RCAS pixel coordinate: " + coordinate)
    for edge, (base, _), extent in zip(
        ("right", "bottom"), bases, ("%tile_width", "%tile_height")
    ):
        old = f"%interior_{edge} = OpIAdd %uint {base} %interior_c16"
        if text.count(old) != 1:
            raise ValueError("Missing RCAS interior extent: " + edge)
        text = text.replace(old, f"%interior_{edge} = OpIAdd %uint {base} {extent}")
    return text


def patch(source, output, spirv_dis="spirv-dis", spirv_as="spirv-as", spirv_val="spirv-val"):
    source, output = Path(source), Path(output)
    if output.exists() or output.resolve().is_relative_to(
        (Path(__file__).resolve().parents[4] / "assets").resolve()
    ):
        raise RuntimeError("Use a fresh path outside production assets")
    text = subprocess.check_output([spirv_dis, str(source)], text=True)
    text = vertical_tile(interior(one_pixel(adapt(text))))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fsr4-rcas-") as directory:
        assembly = Path(directory) / "patched.spvasm"
        assembly.write_text(text)
        subprocess.run([spirv_as, "--target-env", "spv1.3", str(assembly), "-o", str(output)], check=True)
        subprocess.run([spirv_val, "--target-env", "vulkan1.1", str(output)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    for name in ("spirv-dis", "spirv-as", "spirv-val"):
        parser.add_argument("--" + name, default=name)
    args = parser.parse_args()
    patch(args.source, args.output, args.spirv_dis, args.spirv_as, args.spirv_val)


if __name__ == "__main__":
    main()
