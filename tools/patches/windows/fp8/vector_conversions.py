# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Lower Windows FP8 matrix conversions using ordinary vector casts."""

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys

spec = importlib.util.spec_from_file_location(
    "conversion_width", Path(__file__).resolve().parents[2] / "shared/fp8/packed_width.py"
)
width = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = width
spec.loader.exec_module(width)

OP_UNDEF = 1
OP_EXECUTION_MODE = 16
OP_CAPABILITY = 17
OP_TYPE_VOID = 19
OP_TYPE_BOOL = 20
OP_TYPE_FLOAT = 22
OP_FUNCTION_PARAMETER = 55
OP_FUNCTION_CALL = 57
OP_LOAD = 61
OP_COMPOSITE_CONSTRUCT = 80
OP_COMPOSITE_EXTRACT = 81
OP_COMPOSITE_INSERT = 82
OP_COPY_OBJECT = 83
OP_FCONVERT = 115
OP_PHI = 245
OP_TYPE_COOP_MATRIX_KHR = 4456
OP_COOP_MATRIX_MUL_ADD_KHR = 4459

EXEC_LOCAL_SIZE = 17
ENC_FLOAT8_E4M3_EXT = 4214
DEC_SATURATED_TO_FLOAT8_NORMAL_EXT = 4216
CAP_COOP_MATRIX_CONVERSIONS_NV = 5431

MATRIX_VALUE_OPS = {
    OP_UNDEF, OP_FUNCTION_PARAMETER, OP_FUNCTION_CALL, OP_LOAD,
    OP_COMPOSITE_CONSTRUCT, OP_COMPOSITE_EXTRACT, OP_COMPOSITE_INSERT,
    OP_COPY_OBJECT, OP_FCONVERT, width.OP_SELECT, OP_PHI,
    width.OP_COOP_MATRIX_LOAD_KHR, OP_COOP_MATRIX_MUL_ADD_KHR,
}


def patch(source, output):
    """Apply conversion lowering as the last step of a Windows shader recipe."""
    source, output = Path(source), Path(output)
    header, insts = width.read_module(source)
    components, vectors = [None, None], [None, None]
    matrices, constants, saturated = {}, {}, set()
    wave32 = False
    for inst in insts:
        a = inst.args
        if inst.op == OP_TYPE_FLOAT:
            if a[1:] == [32]:
                components[0] = a[0]
            elif a[1:] == [8, ENC_FLOAT8_E4M3_EXT]:
                components[1] = a[0]
        elif inst.op == OP_TYPE_COOP_MATRIX_KHR:
            matrices[a[0]] = a[1:]
        elif inst.op == width.OP_CONSTANT and len(a) == 3:
            constants[a[1]] = a[2]
        elif inst.op == width.OP_DECORATE and a[1:] == [DEC_SATURATED_TO_FLOAT8_NORMAL_EXT]:
            saturated.add(a[0])
        elif inst.op == OP_EXECUTION_MODE and len(a) == 5 and a[1] == EXEC_LOCAL_SIZE:
            wave32 = a[2:] == [32, 1, 1]

    def component(matrix):
        a = matrices[matrix]
        if not wave32 or [constants.get(ident) for ident in a[1:4]] != [3, 16, 16]:
            raise RuntimeError("FP8 conversions require subgroup 16x16 Wave32 matrices")
        if a[0] not in components:
            raise RuntimeError("Unsupported matrix conversion component")
        return components.index(a[0])

    selected = {}
    if None not in components:
        value_types = {inst.args[1]: inst.args[0] for inst in insts
                       if inst.op in MATRIX_VALUE_OPS and len(inst.args) >= 2}
        for inst in insts:
            a = inst.args
            if inst.op == width.OP_TYPE_VECTOR and a[2] == 4 and a[1] in components:
                vectors[components.index(a[1])] = a[0]
            if inst.op != OP_FCONVERT or a[0] not in matrices:
                continue
            source_type = value_types.get(a[2])
            if source_type not in matrices:
                raise RuntimeError("Unsupported cooperative matrix conversion source")
            src, dst = component(source_type), component(a[0])
            if src == dst:
                raise RuntimeError("Unsupported cooperative matrix conversion")
            selected[a[1]] = (src, dst)

    next_id = header[3]

    def fresh():
        nonlocal next_id
        result = next_id
        next_id += 1
        return result

    globals_, annotations, body = [], [], []

    def emit(target, opcode, *args):
        target.append(width.Inst(opcode, list(args)))

    if selected:
        for index in range(2):
            if vectors[index] is None:
                vectors[index] = fresh()
                emit(globals_, width.OP_TYPE_VECTOR, vectors[index], components[index], 4)

    for inst in insts:
        a = inst.args
        if selected and (
            (inst.op == OP_CAPABILITY and a[0] == CAP_COOP_MATRIX_CONVERSIONS_NV)
            or (inst.op == width.OP_DECORATE and a[0] in selected
                and a[1] == DEC_SATURATED_TO_FLOAT8_NORMAL_EXT)
        ):
            continue
        if inst.op != OP_FCONVERT or a[1] not in selected:
            body.append(inst)
            continue
        target_type, result, value = a
        src, dst = selected[result]
        previous = fresh()
        emit(globals_, OP_UNDEF, target_type, previous)
        for start in (0, 4):
            elements = [fresh() for _ in range(4)]
            for index, element in enumerate(elements):
                emit(body, OP_COMPOSITE_EXTRACT, components[src], element, value, start + index)
            packed, converted = fresh(), fresh()
            emit(body, OP_COMPOSITE_CONSTRUCT, vectors[src], packed, *elements)
            emit(body, OP_FCONVERT, vectors[dst], converted, packed)
            if result in saturated:
                emit(annotations, width.OP_DECORATE, converted, DEC_SATURATED_TO_FLOAT8_NORMAL_EXT)
            for index in range(4):
                scalar = fresh()
                inserted = result if start + index == 7 else fresh()
                emit(body, OP_COMPOSITE_EXTRACT, components[dst], scalar, converted, index)
                emit(body, OP_COMPOSITE_INSERT, target_type, inserted, scalar, previous, start + index)
                previous = inserted

    patched = []
    annotations_written = globals_written = False
    for inst in body:
        if not annotations_written and inst.op in (
            OP_TYPE_VOID, OP_TYPE_FLOAT, width.OP_TYPE_INT, OP_TYPE_BOOL
        ):
            patched.extend(annotations)
            annotations_written = True
        if not globals_written and inst.op == width.OP_FUNCTION:
            patched.extend(globals_)
            globals_written = True
        patched.append(inst)
    if selected and (not annotations_written or not globals_written):
        raise RuntimeError("Missing SPIR-V declaration sections")
    header[3] = next_id
    width.write_module(output, header, patched)
    result = subprocess.run(["spirv-val", str(output), "--target-env", "vulkan1.3"],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(result.stdout)
    return {"matrix_conversions_lowered": bool(selected)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(patch(args.input, args.output))


if __name__ == "__main__":
    main()
