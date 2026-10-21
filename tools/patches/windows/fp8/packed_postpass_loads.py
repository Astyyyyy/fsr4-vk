#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Use packed Wave32 loads for FP8 postpass matrices without changing the arithmetic.

Requires a 32x1x1 workgroup, a forced 32-lane subgroup and a dispatch depth of 1.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

spec=importlib.util.spec_from_file_location('postpass_width', Path(__file__).resolve().parents[2]/'shared/fp8/packed_width.py')
width=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=width
spec.loader.exec_module(width)

def run(cmd: list[str]) -> str:
    r = subprocess.run([str(x) for x in cmd], text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f"{cmd[0]} exited {r.returncode}:\n{r.stdout}")
    return r.stdout

def one(regex: str, text: str, what: str, flags=0) -> re.Match[str]:
    xs = list(re.finditer(regex, text, flags))
    if len(xs) != 1:
        raise RuntimeError(f"expected one {what}, found {len(xs)}")
    return xs[0]

def _widen(input_path: Path, output_path: Path, variant: str, *, discover_bindings: bool = False) -> dict[str, object]:
    if variant not in {"u32", "uvec2", "uvec4"}:
        raise RuntimeError(f"unsupported variant: {variant}")
    lanes = {"u32": 1, "uvec2": 2, "uvec4": 4}[variant]
    element_bytes = 4 * lanes
    shift_amount = {4: 2, 8: 3, 16: 4}[element_bytes]

    header, insts = width.read_module(input_path)
    original_bound = header[3]
    next_id = original_bound

    def alloc_id() -> int:
        nonlocal next_id
        out = next_id
        next_id += 1
        return out

    uint32_types: list[int] = []
    uint8_types: list[int] = []
    vector_types: dict[tuple[int, int], int] = {}
    pointer_types: dict[int, tuple[int, int]] = {}
    runtime_arrays: dict[int, int] = {}
    structs: dict[int, list[int]] = {}
    constants: dict[int, tuple[int, int]] = {}
    variable_types: dict[int, tuple[int, int]] = {}
    decorations: dict[int, dict[int, list[list[int]]]] = {}

    for inst in insts:
        a = inst.args
        if inst.op == width.OP_TYPE_INT and len(a) == 3:
            if a[1:] == [32, 0]: uint32_types.append(a[0])
            elif a[1:] == [8, 0]: uint8_types.append(a[0])
        elif inst.op == width.OP_TYPE_VECTOR and len(a) == 3:
            vector_types[(a[1], a[2])] = a[0]
        elif inst.op == width.OP_TYPE_POINTER and len(a) == 3:
            pointer_types[a[0]] = (a[1], a[2])
        elif inst.op == width.OP_TYPE_RUNTIME_ARRAY and len(a) == 2:
            runtime_arrays[a[0]] = a[1]
        elif inst.op == width.OP_TYPE_STRUCT and a:
            structs[a[0]] = a[1:]
        elif inst.op == width.OP_CONSTANT and len(a) == 3:
            constants[a[1]] = (a[0], a[2])
        elif inst.op == width.OP_VARIABLE and len(a) >= 3:
            variable_types[a[1]] = (a[0], a[2])
        elif inst.op == width.OP_DECORATE and len(a) >= 2:
            decorations.setdefault(a[0], {}).setdefault(a[1], []).append(a[2:])

    uint32 = width.unique_one(uint32_types, "uint32 type")
    uint8 = width.unique_one(uint8_types, "uint8 type")

    # Dispatch depth is 1, so WorkGroupID.z is zero. Making that explicit lets
    # LLPC prove the wider loads are aligned.
    workgroup_vars = []
    for var_id in variable_types:
        entries = decorations.get(var_id, {}).get(11, [])  # BuiltIn
        if any(entry == [26] for entry in entries):        # WorkgroupId
            workgroup_vars.append(var_id)
    workgroup_var = width.unique_one(workgroup_vars, "WorkgroupId variable")
    zero_u32 = width.unique_one(
        [cid for cid, (ty, value) in constants.items() if ty == uint32 and value == 0],
        "uint32 zero constant",
    )
    two_u32 = width.unique_one(
        [cid for cid, (ty, value) in constants.items() if ty == uint32 and value == 2],
        "uint32 constant 2",
    )
    z_ptrs = {
        inst.args[1] for inst in insts
        if inst.op == width.OP_ACCESS_CHAIN and len(inst.args) >= 4
        and inst.args[2] == workgroup_var and inst.args[-1] == two_u32
    }
    if len(z_ptrs) != 1:
        raise RuntimeError(f"expected one WorkgroupId.z access chain, found {sorted(z_ptrs)}")
    z_loads = [
        inst.args[1] for inst in insts
        if inst.op == 61 and len(inst.args) >= 3 and inst.args[2] in z_ptrs
    ]
    z_value = width.unique_one(z_loads, "WorkgroupId.z load")
    z_replacements = 0
    for inst in insts:
        # Only arithmetic uses are permitted. Literals are not inspected.
        if inst.op in (width.OP_IADD, width.OP_ISUB, width.OP_IMUL, width.OP_BITWISE_OR,
                       width.OP_BITWISE_XOR, width.OP_BITWISE_AND,
                       width.OP_SHIFT_LEFT_LOGICAL, width.OP_SHIFT_RIGHT_LOGICAL,
                       width.OP_SHIFT_RIGHT_ARITHMETIC):
            for pos in range(2, len(inst.args)):
                if inst.args[pos] == z_value:
                    inst.args[pos] = zero_u32
                    z_replacements += 1
    if z_replacements == 0:
        raise RuntimeError("WorkgroupId.z was not used by arithmetic")

    # Move existing vector types before use; SPIR-V forbids duplicate vector types.
    new_global_types: list[width.Inst] = []
    relocated_type: width.Inst | None = None
    if lanes == 1:
        wide_type = uint32
    elif (uint32, lanes) in vector_types:
        wide_type = vector_types[(uint32, lanes)]
        relocated_type = next(
            x for x in insts if x.op == width.OP_TYPE_VECTOR and x.args[0] == wide_type
        )
    else:
        wide_type = alloc_id()
        new_global_types.append(width.Inst(width.OP_TYPE_VECTOR, [wide_type, uint32, lanes]))

    def decoration_scalar(target: int, decoration: int) -> int | None:
        entries = decorations.get(target, {}).get(decoration, [])
        if not entries:
            return None
        if len(entries) != 1 or len(entries[0]) != 1:
            raise RuntimeError(f"unexpected decoration shape on %{target}: {decoration} -> {entries}")
        return entries[0][0]

    # Bindings can have typed aliases. Identify raw byte buffers by their full type chain.
    target_vars: list[int] = []
    target_bindings: list[tuple[int, int]] = []
    payload_arrays: set[int] = set()
    payload_accesses: list[width.Inst] = []
    payload_pointer_types: set[int] = set()

    for var_id, (var_type, storage) in variable_types.items():
        if storage != width.STORAGE_BUFFER:
            continue
        ptr_storage, descriptor_array_type = pointer_types.get(var_type, (None, None))
        if ptr_storage != width.STORAGE_BUFFER:
            continue
        descriptor_struct = runtime_arrays.get(descriptor_array_type)
        if descriptor_struct is None:
            continue
        members = structs.get(descriptor_struct)
        if not members or len(members) != 1:
            continue
        payload_array = members[0]
        if runtime_arrays.get(payload_array) != uint8:
            continue
        if decoration_scalar(payload_array, width.DEC_ARRAY_STRIDE) != 1:
            raise RuntimeError(f"raw-u8 payload %{payload_array} ArrayStride != 1")
        descriptor_set = decoration_scalar(var_id, width.DEC_DESCRIPTOR_SET)
        binding = decoration_scalar(var_id, width.DEC_BINDING)
        if descriptor_set is None or binding is None:
            raise RuntimeError(f"raw-u8 descriptor %{var_id} lacks set/binding")

        selected_descriptor_ptrs = {
            inst.args[1] for inst in insts
            if inst.op == width.OP_ACCESS_CHAIN and len(inst.args) >= 4 and inst.args[2] == var_id
        }
        if not selected_descriptor_ptrs:
            continue
        local_payload_accesses = [
            inst for inst in insts
            if inst.op == width.OP_ACCESS_CHAIN and len(inst.args) >= 5
            and inst.args[2] in selected_descriptor_ptrs
        ]
        if not local_payload_accesses:
            continue

        target_vars.append(var_id)
        target_bindings.append((descriptor_set, binding))
        payload_arrays.add(payload_array)
        payload_accesses.extend(local_payload_accesses)
        payload_pointer_types.update(inst.args[0] for inst in local_payload_accesses)

    if not target_vars:
        raise RuntimeError("no raw-u8 cooperative-matrix StorageBuffer descriptors discovered")

    # Only accept raw model buffers at binding 0 in descriptor sets 1 and 4.
    expected_bindings = {(1, 0), (4, 0)}
    if set(target_bindings) != expected_bindings:
        raise RuntimeError(
            f"unexpected raw-u8 descriptor bindings: {sorted(set(target_bindings))}; "
            f"expected {sorted(expected_bindings)}"
        )

    if not payload_accesses:
        raise RuntimeError("no raw payload accesses found")
    for ptr_type in payload_pointer_types:
        storage, pointee = pointer_types.get(ptr_type, (None, None))
        if storage != width.STORAGE_BUFFER or pointee != uint8:
            raise RuntimeError(f"payload pointer %{ptr_type} is not StorageBuffer u8")

    payload_ptr_results = {inst.args[1] for inst in payload_accesses}
    load_insts = [
        inst for inst in insts
        if inst.op == width.OP_COOP_MATRIX_LOAD_KHR and len(inst.args) >= 3 and inst.args[2] in payload_ptr_results
    ]
    store_insts = [
        inst for inst in insts
        if inst.op == width.OP_COOP_MATRIX_STORE_KHR and inst.args and inst.args[0] in payload_ptr_results
    ]
    matrix_ptrs = {inst.args[2] for inst in load_insts} | {inst.args[0] for inst in store_insts}
    if matrix_ptrs != payload_ptr_results:
        raise RuntimeError("some raw payload pointers are not used exclusively by cooperative matrix load/store")

    unexpected: list[tuple[int,int]] = []
    for inst in insts:
        if inst.op == 61 and len(inst.args) >= 3 and inst.args[2] in payload_ptr_results:
            unexpected.append((inst.op, inst.args[2]))
        elif inst.op == 62 and inst.args and inst.args[0] in payload_ptr_results:
            unexpected.append((inst.op, inst.args[0]))
        elif inst.op == width.OP_ACCESS_CHAIN and len(inst.args) >= 3 and inst.args[2] in payload_ptr_results:
            unexpected.append((inst.op, inst.args[2]))
    if unexpected:
        raise RuntimeError(f"raw payload pointers have unexpected uses: {unexpected[:8]}")

    value_to_const: dict[int,int] = {}
    for const_id, (type_id, value) in constants.items():
        if type_id == uint32:
            value_to_const.setdefault(value, const_id)
    shift_const = value_to_const.get(shift_amount)
    new_global_constants: list[width.Inst] = []
    if shift_const is None:
        shift_const = alloc_id()
        new_global_constants.append(width.Inst(width.OP_CONSTANT, [uint32, shift_const, shift_amount]))
        value_to_const[shift_amount] = shift_const

    result_def_index: dict[int,int] = {}
    result_type_ops = {
        61,65,66,124,126,127,128,129,130,131,132,133,134,
        194,195,196,197,198,199,200,201,202,203,204,205,
        width.OP_SELECT,245,4457,4459,4460,
    }
    for idx, inst in enumerate(insts):
        if inst.op in result_type_ops and len(inst.args) >= 2:
            result_def_index[inst.args[1]] = idx
        elif inst.op in (width.OP_CONSTANT, width.OP_VARIABLE) and len(inst.args) >= 2:
            result_def_index[inst.args[1]] = idx
    definition_by_result = {rid: insts[idx] for rid, idx in result_def_index.items()}

    alignment_cache: dict[int,int] = {}
    alignment_active: set[int] = set()
    def ctz(value: int) -> int:
        value &= 0xffffffff
        if value == 0: return 32
        return min(32, (value & -value).bit_length()-1)
    def guaranteed_tz(value_id: int) -> int:
        if value_id in alignment_cache: return alignment_cache[value_id]
        const = constants.get(value_id)
        if const is not None and const[0] == uint32:
            out = ctz(const[1]); alignment_cache[value_id] = out; return out
        if value_id in alignment_active: return 0
        inst = definition_by_result.get(value_id)
        if inst is None or len(inst.args) < 3: return 0
        alignment_active.add(value_id)
        operands = inst.args[2:]
        out = 0
        try:
            if inst.op == width.OP_BITCAST and operands:
                out = guaranteed_tz(operands[0])
            elif inst.op in (width.OP_IADD, width.OP_ISUB, width.OP_BITWISE_OR, width.OP_BITWISE_XOR) and len(operands) >= 2:
                out = min(guaranteed_tz(operands[0]), guaranteed_tz(operands[1]))
            elif inst.op == width.OP_IMUL and len(operands) >= 2:
                out = min(32, guaranteed_tz(operands[0]) + guaranteed_tz(operands[1]))
            elif inst.op == width.OP_BITWISE_AND and len(operands) >= 2:
                out = max(guaranteed_tz(operands[0]), guaranteed_tz(operands[1]))
            elif inst.op == width.OP_SHIFT_LEFT_LOGICAL and len(operands) >= 2:
                s = constants.get(operands[1])
                if s is not None and s[0] == uint32: out = min(32, guaranteed_tz(operands[0]) + s[1])
            elif inst.op in (width.OP_SHIFT_RIGHT_LOGICAL, width.OP_SHIFT_RIGHT_ARITHMETIC) and len(operands) >= 2:
                s = constants.get(operands[1])
                if s is not None and s[0] == uint32: out = max(0, guaranteed_tz(operands[0]) - s[1])
            elif inst.op == width.OP_SELECT and len(operands) >= 3:
                out = min(guaranteed_tz(operands[1]), guaranteed_tz(operands[2]))
        finally:
            alignment_active.remove(value_id)
        alignment_cache[value_id] = out
        return out

    index_ids = list(dict.fromkeys(inst.args[-1] for inst in payload_accesses))
    align_bits = {value_id: guaranteed_tz(value_id) for value_id in index_ids}
    if any(bits < shift_amount for bits in align_bits.values()):
        bad = [rid for rid,bits in align_bits.items() if bits < shift_amount]
        raise RuntimeError(f"cannot prove {element_bytes}-byte alignment for offsets: {bad[:8]}")

    replacement_index: dict[int,int] = {}
    dynamic_insert_after: dict[int,list[width.Inst]] = {}
    for old_index in index_ids:
        const = constants.get(old_index)
        if const is not None:
            type_id, value = const
            if type_id != uint32 or value % element_bytes:
                raise RuntimeError(f"offset %{old_index} is not aligned")
            scaled = value // element_bytes
            existing = value_to_const.get(scaled)
            if existing is None:
                existing = alloc_id()
                new_global_constants.append(width.Inst(width.OP_CONSTANT, [uint32, existing, scaled]))
                value_to_const[scaled] = existing
            replacement_index[old_index] = existing
        else:
            def_idx = result_def_index.get(old_index)
            if def_idx is None:
                raise RuntimeError(f"cannot find dynamic offset definition %{old_index}")
            new_id = alloc_id()
            replacement_index[old_index] = new_id
            dynamic_insert_after.setdefault(def_idx, []).append(
                width.Inst(width.OP_SHIFT_RIGHT_LOGICAL, [uint32, new_id, old_index, shift_const])
            )

    stride_replacements: dict[int,int] = {}
    stride_ids = [x.args[4] for x in load_insts if len(x.args) >= 5] + [x.args[3] for x in store_insts if len(x.args) >= 4]
    for stride_id in dict.fromkeys(stride_ids):
        c = constants.get(stride_id)
        if c is None or c[0] != uint32:
            raise RuntimeError(f"stride %{stride_id} is not uint32 constant")
        value = c[1]
        if value == 0:
            stride_replacements[stride_id] = stride_id
            continue
        if value % element_bytes:
            raise RuntimeError(f"matrix stride {value} is not divisible by {element_bytes}")
        scaled = value // element_bytes
        existing = value_to_const.get(scaled)
        if existing is None:
            existing = alloc_id()
            new_global_constants.append(width.Inst(width.OP_CONSTANT, [uint32, existing, scaled]))
            value_to_const[scaled] = existing
        stride_replacements[stride_id] = existing

    for inst in insts:
        if inst.op == width.OP_TYPE_RUNTIME_ARRAY and inst.args[0] in payload_arrays:
            inst.args[1] = wide_type
        elif inst.op == width.OP_TYPE_POINTER and inst.args[0] in payload_pointer_types:
            inst.args[2] = wide_type
        elif inst.op == width.OP_DECORATE and len(inst.args) == 3 and inst.args[0] in payload_arrays and inst.args[1] == width.DEC_ARRAY_STRIDE:
            inst.args[2] = element_bytes

    payload_access_ids = {id(x) for x in payload_accesses}
    load_ids = {id(x) for x in load_insts}
    store_ids = {id(x) for x in store_insts}
    for inst in insts:
        if id(inst) in payload_access_ids:
            inst.args[-1] = replacement_index[inst.args[-1]]
        elif id(inst) in load_ids and len(inst.args) >= 5:
            inst.args[4] = stride_replacements[inst.args[4]]
        elif id(inst) in store_ids and len(inst.args) >= 4:
            inst.args[3] = stride_replacements[inst.args[3]]

    first_function = next((i for i,x in enumerate(insts) if x.op == width.OP_FUNCTION), None)
    if first_function is None:
        raise RuntimeError("module has no function")

    payload_type_indices = [
        i for i, x in enumerate(insts)
        if x.op == width.OP_TYPE_RUNTIME_ARRAY and x.args[0] in payload_arrays
    ]
    if not payload_type_indices:
        raise RuntimeError("cannot locate payload runtime-array declarations")
    first_payload_type = min(payload_type_indices)

    # Define uint32 before adding its vector type.
    uint32_type_index = next(
        (i for i, x in enumerate(insts)
         if x.op == width.OP_TYPE_INT and x.args and x.args[0] == uint32),
        None,
    )
    if uint32_type_index is None or uint32_type_index >= first_payload_type:
        raise RuntimeError("uint32 type is not defined before the payload arrays")

    rebuilt: list[width.Inst] = []
    for idx, inst in enumerate(insts):
        if inst.op == width.OP_TYPE_INT and inst.args[0] == uint32 and relocated_type is not None:
            rebuilt.append(inst)
            rebuilt.append(relocated_type)
            continue
        if inst is relocated_type:
            continue
        if idx == first_payload_type:
            rebuilt.extend(new_global_types)
        if idx == first_function:
            rebuilt.extend(new_global_constants)
        rebuilt.append(inst)
        rebuilt.extend(dynamic_insert_after.get(idx, []))

    header[3] = next_id
    width.write_module(output_path, header, rebuilt)
    out_header, _ = width.read_module(output_path)
    if out_header[3] != next_id: raise RuntimeError("output bound mismatch")

    return {
        "input": str(input_path), "output": str(output_path), "variant": variant,
        "element_bytes": element_bytes, "lanes": lanes,
        "original_bound": original_bound, "new_bound": next_id,
        "target_bindings": [f"set{s}.binding{b}" for s, b in sorted(set(target_bindings))],
        "target_variables": target_vars,
        "payload_runtime_arrays": sorted(payload_arrays),
        "payload_pointer_types": sorted(payload_pointer_types),
        "cooperative_matrix_loads_rewritten": len(load_insts),
        "cooperative_matrix_stores_rewritten": len(store_insts),
        "payload_accesses_rewritten": len(payload_accesses),
        "unique_offsets": len(index_ids),
        "dynamic_offsets_scaled": sum(len(v) for v in dynamic_insert_after.values()),
        "minimum_guaranteed_offset_alignment_bytes": 1 << min(align_bits.values()),
        "workgroup_z_specialized_zero": True, "workgroup_z_replacements": z_replacements,
        "new_types": len(new_global_types), "new_constants": len(new_global_constants),
        "relocated_types": int(relocated_type is not None),
        "original_bytes": input_path.stat().st_size, "patched_bytes": output_path.stat().st_size,
        "transform": f"raw buffer u8 -> {variant}; byte offsets /{element_bytes}; matrix strides /{element_bytes}",
    }

def patch(source: Path, output: Path, report_path: Path | None = None) -> dict[str, object]:
    source, output = Path(source), Path(output)
    assets = Path(__file__).resolve().parents[4] / 'assets'
    if output.resolve().is_relative_to(assets.resolve()):
        raise RuntimeError('Do not write production assets directly')
    if output.exists():
        raise RuntimeError('Use a fresh shader output path')
    for tool in ("spirv-dis", "spirv-as", "spirv-val"):
        if not shutil.which(tool):
            raise RuntimeError(f"{tool} must be on PATH")

    with tempfile.TemporaryDirectory(prefix="fsr4-v16-") as td:
        td = Path(td)
        widened = td / "postpass-uvec2.spv"
        width_report = _widen(source, widened, "uvec2")
        if int(width_report["cooperative_matrix_loads_rewritten"]) != 51:
            raise RuntimeError("Widened-buffer rewrite did not find exactly 51 cooperative loads")
        if int(width_report["minimum_guaranteed_offset_alignment_bytes"]) < 16:
            raise RuntimeError("Widened-buffer alignment proof is weaker than 16 bytes")
        if int(width_report["workgroup_z_replacements"]) < 1:
            raise RuntimeError("Postpass foundation did not specialize WorkGroupID.z")
        text = run(["spirv-dis", "--raw-id", widened])

    u32 = one(r"^\s*(%\d+) = OpTypeInt 32 0\s*$", text, "uint32 type", re.M).group(1)
    u8 = one(r"^\s*(%\d+) = OpTypeInt 8 0\s*$", text, "uint8 type", re.M).group(1)
    v2 = one(rf"^\s*(%\d+) = OpTypeVector {re.escape(u32)} 2\s*$", text,
             "uvec2 type", re.M).group(1)

    const_by_value: dict[int, str] = {}
    for ident, value in re.findall(rf"^\s*(%\d+) = OpConstant {re.escape(u32)} (\d+)\s*$", text, re.M):
        const_by_value.setdefault(int(value), ident)
    inserted_constants: list[str] = []
    def c(value: int) -> str:
        if value in const_by_value:
            return const_by_value[value]
        ident = f"%v16_c{value}"
        const_by_value[value] = ident
        inserted_constants.append(f"{ident} = OpConstant {u32} {value}")
        return ident

    c0, c4, c8, c15, c16, c24 = (c(x) for x in (0,4,8,15,16,24))

    # Require one unambiguous lane source: SubgroupLocalInvocationId or LocalInvocationId.x.
    subgroup_lane_decs = list(re.finditer(
        r"^\s*OpDecorate (%\d+) BuiltIn SubgroupLocalInvocationId\s*$", text, re.M))
    local_id_decs = list(re.finditer(
        r"^\s*OpDecorate (%\d+) BuiltIn LocalInvocationId\s*$", text, re.M))

    lane_kind: str
    lane_var: str
    lane_load_lines: list[str]
    if len(subgroup_lane_decs) == 1 and len(local_id_decs) == 0:
        lane_kind = "SubgroupLocalInvocationId"
        lane_var = subgroup_lane_decs[0].group(1)
        lane_var_def = one(
            rf"^\s*{re.escape(lane_var)} = OpVariable (%\d+) Input\s*$", text,
            "SubgroupLocalInvocationId variable", re.M)
        lane_ptr = lane_var_def.group(1)
        lane_pointee = one(
            rf"^\s*{re.escape(lane_ptr)} = OpTypePointer Input (%\d+)\s*$", text,
            "SubgroupLocalInvocationId pointer type", re.M).group(1)
        if lane_pointee != u32:
            raise RuntimeError(
                f"SubgroupLocalInvocationId pointee {lane_pointee} is not uint32 {u32}")
        lane_load_lines = ["{lane} = OpLoad " + u32 + " " + lane_var]
    elif len(local_id_decs) == 1 and len(subgroup_lane_decs) == 0:
        lane_kind = "LocalInvocationId.x"
        lane_var = local_id_decs[0].group(1)
        lane_var_def = one(
            rf"^\s*{re.escape(lane_var)} = OpVariable (%\d+) Input\s*$", text,
            "LocalInvocationId variable", re.M)
        lane_ptr = lane_var_def.group(1)
        lane_vec = one(
            rf"^\s*{re.escape(lane_ptr)} = OpTypePointer Input (%\d+)\s*$", text,
            "LocalInvocationId pointer type", re.M).group(1)
        one(rf"^\s*{re.escape(lane_vec)} = OpTypeVector {re.escape(u32)} 3\s*$", text,
            "LocalInvocationId uvec3 type", re.M)
        lane_load_lines = [
            "{lid} = OpLoad " + lane_vec + " " + lane_var,
            "{lane} = OpCompositeExtract " + u32 + " {lid} 0",
        ]
    else:
        raise RuntimeError(
            "expected exactly one lane builtin: "
            f"SubgroupLocalInvocationId={len(subgroup_lane_decs)}, "
            f"LocalInvocationId={len(local_id_decs)}")

    # Require a 32x1x1 workgroup; the provider also forces a 32-lane subgroup.
    if not re.search(r"^\s*OpExecutionMode %\d+ LocalSize 32 1 1\s*$", text, re.M):
        raise RuntimeError("expected LocalSize 32 1 1")

    pointer_pointee: dict[str, tuple[str,str]] = {}
    for pid, storage, pointee in re.findall(r"^\s*(%\d+) = OpTypePointer (\w+) (%\d+)\s*$", text, re.M):
        pointer_pointee[pid] = (storage, pointee)

    matrix_types: dict[str, tuple[str,str,str,str,str]] = {}
    for m in re.finditer(r"^\s*(%\d+) = OpTypeCooperativeMatrixKHR (%\d+) (%\d+) (%\d+) (%\d+) (%\d+)\s*$", text, re.M):
        matrix_types[m.group(1)] = m.groups()[1:]

    fp8_types = set(re.findall(r"^\s*(%\d+) = OpTypeFloat 8\b.*$", text, re.M))
    if not fp8_types:
        raise RuntimeError("no FP8 scalar type found")

    access: dict[str, tuple[str,str,list[str]]] = {}
    for m in re.finditer(r"^\s*(%\d+) = OpAccessChain (%\d+) (%\d+)((?: %\d+)+)\s*$", text, re.M):
        result, ptype, parent, tail = m.groups()
        access[result] = (ptype, parent, tail.split())

    load_re = re.compile(
        r"^\s*(%\d+) = OpCooperativeMatrixLoadKHR (%\d+) (%\d+) (%\d+) (%\d+) NonPrivatePointer\s*$",
        re.M)
    loads = list(load_re.finditer(text))
    selected = []
    for m in loads:
        result, matrix, ptr, layout, stride = m.groups()
        if ptr not in access:
            continue
        ptype, parent, indices = access[ptr]
        storage, pointee = pointer_pointee.get(ptype, ("", ""))
        if storage != "StorageBuffer" or pointee != v2:
            continue
        selected.append(m)

    if len(selected) != 51:
        raise RuntimeError(f"expected 51 widened postpass cooperative loads, found {len(selected)}")

    # Rewrite only FP8 MatrixA/MatrixB fragments. Accumulators retain the widened-buffer layout.

    undefs: dict[str,str] = {}
    def value_of(cid: str) -> int:
        for value, ident in const_by_value.items():
            if ident == cid:
                return value
        mm = re.search(rf"^\s*{re.escape(cid)} = OpConstant {re.escape(u32)} (\d+)\s*$", text, re.M)
        if not mm:
            raise RuntimeError(f"constant {cid} is not a uint32 constant")
        return int(mm.group(1))

    ab_selected: list[re.Match[str]] = []
    ab_stride_counts: dict[int, int] = {}
    class_counts = {"fp8_a": 0, "fp8_b": 0, "fp8_acc": 0, "fp32_acc": 0, "other": 0}
    for m in selected:
        _, matrix, _, layout, stride = m.groups()
        if matrix not in matrix_types:
            raise RuntimeError(f"matrix type {matrix} not found")
        comp, scope, rows, cols, use = matrix_types[matrix]
        scope_v, rows_v, cols_v, use_v = map(value_of, (scope, rows, cols, use))
        if (scope_v, rows_v, cols_v) != (3, 16, 16):
            raise RuntimeError(
                f"unexpected cooperative matrix shape for {matrix}: "
                f"scope={scope_v} rows={rows_v} cols={cols_v} use={use_v}"
            )
        is_fp8 = comp in fp8_types
        if is_fp8 and use_v == 0:
            class_counts["fp8_a"] += 1
        elif is_fp8 and use_v == 1:
            class_counts["fp8_b"] += 1
        elif is_fp8 and use_v == 2:
            class_counts["fp8_acc"] += 1
        elif (not is_fp8) and use_v == 2:
            class_counts["fp32_acc"] += 1
        else:
            class_counts["other"] += 1

        if not (is_fp8 and use_v in (0, 1)):
            continue

        # Matrix A is row-major and Matrix B is column-major. Keep the stride in 8-byte units.
        if value_of(layout) != use_v:
            raise RuntimeError(f"matrix A/B layout/use mismatch for {matrix}")
        stride_value = value_of(stride)
        if stride_value < 2:
            raise RuntimeError(
                f"unexpected A/B uvec2 stride for {matrix}: {stride_value}; expected >=2")
        ab_stride_counts[stride_value] = ab_stride_counts.get(stride_value, 0) + 1
        ab_selected.append(m)
        undefs.setdefault(matrix, f"%v16_undef_{matrix[1:]}")

    expected_classes = {"fp8_a": 27, "fp8_b": 18, "fp8_acc": 2, "fp32_acc": 4, "other": 0}
    if class_counts != expected_classes:
        raise RuntimeError(
            f"unexpected postpass cooperative-load classes: {class_counts}; "
            f"expected {expected_classes}")
    if len(ab_selected) != 45:
        raise RuntimeError(f"expected 45 FP8 MatrixA/B loads, found {len(ab_selected)}")
    if sum(ab_stride_counts.values()) != 45:
        raise RuntimeError(
            f"A/B stride accounting mismatch: {ab_stride_counts}; expected 45 loads")

    selected = ab_selected

    first_fn = re.search(r"^\s*%\d+ = OpFunction\b", text, re.M)
    if not first_fn:
        raise RuntimeError("no function found")
    globals_text = ""
    if inserted_constants:
        globals_text += "\n".join(inserted_constants) + "\n"
    globals_text += "\n".join(f"{uid} = OpUndef {mty}" for mty,uid in undefs.items()) + "\n"
    text = text[:first_fn.start()] + globals_text + text[first_fn.start():]

    target_results = {m.group(1) for m in selected}
    replacements = 0
    expected_fragment_components = 8

    def repl(m: re.Match[str]) -> str:
        nonlocal replacements
        result, matrix, ptr, layout, stride = m.groups()
        if result not in target_results:
            return m.group(0)
        ptype, parent, indices = access[ptr]
        if not indices:
            raise RuntimeError(f"empty access chain for {ptr}")
        base = indices[-1]
        prefix_indices = indices[:-1]
        comp, scope, rows, cols, use = matrix_types[matrix]
        stride_value = value_of(stride)
        # Keep the original stride ID: it is already in uvec2 (8-byte) units.
        tag = f"v16_{result[1:]}"
        lane = f"%{tag}_lane"
        lid = f"%{tag}_lid"
        lines = [
            x.format(lane=lane, lid=lid) for x in lane_load_lines
        ] + [
            f"%{tag}_row = OpBitwiseAnd {u32} {lane} {c15}",
            f"%{tag}_half = OpShiftRightLogical {u32} %{tag}_lane {c4}",
            f"%{tag}_rowoff = OpIMul {u32} %{tag}_row {stride}",
            f"%{tag}_tmpidx = OpIAdd {u32} {base} %{tag}_rowoff",
            f"%{tag}_idx = OpIAdd {u32} %{tag}_tmpidx %{tag}_half",
            f"%{tag}_ptr = OpAccessChain {ptype} {parent} " + " ".join(prefix_indices + [f"%{tag}_idx"]),
            f"%{tag}_packed = OpLoad {v2} %{tag}_ptr Aligned 8",
            f"%{tag}_word0 = OpCompositeExtract {u32} %{tag}_packed 0",
            f"%{tag}_word1 = OpCompositeExtract {u32} %{tag}_packed 1",
        ]
        previous = undefs[matrix]
        shifts = {1:c8, 2:c16, 3:c24}
        for i in range(expected_fragment_components):
            word = f"%{tag}_word{i//4}"
            if i % 4:
                shifted = f"%{tag}_shift{i}"
                lines.append(f"{shifted} = OpShiftRightLogical {u32} {word} {shifts[i%4]}")
                word = shifted
            byte = f"%{tag}_byte{i}"
            f8 = f"%{tag}_fp8_{i}"
            lines.append(f"{byte} = OpUConvert {u8} {word}")
            lines.append(f"{f8} = OpBitcast {comp} {byte}")
            out = result if i == expected_fragment_components - 1 else f"%{tag}_mat{i}"
            lines.append(f"{out} = OpCompositeInsert {matrix} {f8} {previous} {i}")
            previous = out
        replacements += 1
        return "\n".join(lines)

    text = load_re.sub(repl, text)
    wanted = 45
    if replacements != wanted:
        raise RuntimeError(f"expected {wanted} explicit load replacements, got {replacements}")

    asm = output.with_suffix(".spvasm")
    output.parent.mkdir(parents=True, exist_ok=True)
    asm.write_text(text, encoding="utf-8")
    run(["spirv-as", "--target-env", "spv1.6", asm, "-o", output])

    run(["spirv-val", output])

    report = {
        "input": str(source),
        "output": str(output),
        "foundation": width_report,
        "foundation_cooperative_loads": 51,
        "cooperative_load_classes": class_counts,
        "explicit_ab_loads_available": 45,
        "explicit_ab_stride_counts": {str(k): v for k, v in sorted(ab_stride_counts.items())},
        "explicit_loads_rewritten": replacements,
        "expected_fragment_components": expected_fragment_components,
        "explicit_load_width_bytes": 8,
        "matrix_types": sorted(undefs),
        "lane_builtin": lane_kind,
        "lane_variable": lane_var,
        "transform": "Widened uvec2 buffers + actual-stride aligned 8-byte subgroup-lane loads for FP8 MatrixA/MatrixB + 8 FP8 composite inserts",
    }
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    print(json.dumps(patch(args.input,args.output,args.report),indent=2))

if __name__=='__main__':
    main()
