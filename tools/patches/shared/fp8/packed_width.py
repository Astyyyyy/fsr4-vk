#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Use wider integer elements for FP8 matrix buffers.

Scale offsets and strides to preserve byte addresses. Cooperative matrix
loads allow buffer elements wider than the matrix components.
"""

from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path

OP_TYPE_INT = 21
OP_TYPE_VECTOR = 23
OP_TYPE_RUNTIME_ARRAY = 29
OP_TYPE_STRUCT = 30
OP_TYPE_POINTER = 32
OP_CONSTANT = 43
OP_FUNCTION = 54
OP_VARIABLE = 59
OP_ACCESS_CHAIN = 65
OP_DECORATE = 71
OP_BITCAST = 124
OP_IADD = 128
OP_ISUB = 130
OP_IMUL = 132
OP_SELECT = 169
OP_SHIFT_RIGHT_LOGICAL = 194
OP_SHIFT_RIGHT_ARITHMETIC = 195
OP_SHIFT_LEFT_LOGICAL = 196
OP_BITWISE_OR = 197
OP_BITWISE_XOR = 198
OP_BITWISE_AND = 199
OP_COOP_MATRIX_LOAD_KHR = 4457
OP_COOP_MATRIX_STORE_KHR = 4458

DEC_ARRAY_STRIDE = 6
DEC_BINDING = 33
DEC_DESCRIPTOR_SET = 34
STORAGE_BUFFER = 12
SPIRV_MAGIC = 0x07230203

@dataclass
class Inst:
    op: int
    args: list[int]

    def words(self) -> list[int]:
        return [(((1 + len(self.args)) << 16) | self.op), *self.args]

def read_module(path: Path) -> tuple[list[int], list[Inst]]:
    raw = path.read_bytes()
    if len(raw) % 4:
        raise RuntimeError(f"{path}: SPIR-V size is not a multiple of four")
    words = list(struct.unpack(f"<{len(raw)//4}I", raw))
    if len(words) < 5 or words[0] != SPIRV_MAGIC:
        raise RuntimeError(f"{path}: not SPIR-V")
    header = words[:5]
    insts: list[Inst] = []
    i = 5
    while i < len(words):
        word0 = words[i]
        wc = word0 >> 16
        op = word0 & 0xFFFF
        if wc == 0 or i + wc > len(words):
            raise RuntimeError(f"{path}: malformed instruction at word {i}")
        insts.append(Inst(op, words[i+1:i+wc]))
        i += wc
    return header, insts

def write_module(path: Path, header: list[int], insts: list[Inst]) -> None:
    words = list(header)
    for inst in insts:
        words.extend(inst.words())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack(f"<{len(words)}I", *words))

def unique_one(values: list[int], what: str) -> int:
    values = list(dict.fromkeys(values))
    if len(values) != 1:
        raise RuntimeError(f"expected exactly one {what}, found {values}")
    return values[0]

def patch_module(input_path: Path, output_path: Path, variant: str, *, discover_bindings: bool = False) -> dict[str, object]:
    if variant not in {"u32", "uvec2", "uvec4"}:
        raise RuntimeError(f"unsupported variant: {variant}")
    lanes = {"u32": 1, "uvec2": 2, "uvec4": 4}[variant]
    element_bytes = 4 * lanes
    shift_amount = {4: 2, 8: 3, 16: 4}[element_bytes]

    header, insts = read_module(input_path)
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
        if inst.op == OP_TYPE_INT and len(a) == 3:
            if a[1:] == [32, 0]: uint32_types.append(a[0])
            elif a[1:] == [8, 0]: uint8_types.append(a[0])
        elif inst.op == OP_TYPE_VECTOR and len(a) == 3:
            vector_types[(a[1], a[2])] = a[0]
        elif inst.op == OP_TYPE_POINTER and len(a) == 3:
            pointer_types[a[0]] = (a[1], a[2])
        elif inst.op == OP_TYPE_RUNTIME_ARRAY and len(a) == 2:
            runtime_arrays[a[0]] = a[1]
        elif inst.op == OP_TYPE_STRUCT and a:
            structs[a[0]] = a[1:]
        elif inst.op == OP_CONSTANT and len(a) == 3:
            constants[a[1]] = (a[0], a[2])
        elif inst.op == OP_VARIABLE and len(a) >= 3:
            variable_types[a[1]] = (a[0], a[2])
        elif inst.op == OP_DECORATE and len(a) >= 2:
            decorations.setdefault(a[0], {}).setdefault(a[1], []).append(a[2:])

    uint32 = unique_one(uint32_types, "uint32 type")
    uint8 = unique_one(uint8_types, "uint8 type")

    # Move existing vector types before use; SPIR-V forbids duplicate vector types.
    new_global_types: list[Inst] = []
    relocated_type: Inst | None = None
    if lanes == 1:
        wide_type = uint32
    elif (uint32, lanes) in vector_types:
        wide_type = vector_types[(uint32, lanes)]
        relocated_type = next(
            x for x in insts if x.op == OP_TYPE_VECTOR and x.args[0] == wide_type
        )
    else:
        wide_type = alloc_id()
        new_global_types.append(Inst(OP_TYPE_VECTOR, [wide_type, uint32, lanes]))

    def decoration_scalar(target: int, decoration: int) -> int | None:
        entries = decorations.get(target, {}).get(decoration, [])
        if not entries:
            return None
        if len(entries) != 1 or len(entries[0]) != 1:
            raise RuntimeError(f"unexpected decoration shape on %{target}: {decoration} -> {entries}")
        return entries[0][0]

    target_bindings = [(1, 0), (4, 0)]
    if discover_bindings:
        target_bindings = []
        for var_id, (var_type, storage) in variable_types.items():
            if storage != STORAGE_BUFFER:
                continue
            _, array = pointer_types.get(var_type, (None, None))
            members = structs.get(runtime_arrays.get(array), [])
            if len(members) != 1 or runtime_arrays.get(members[0]) != uint8:
                continue
            descriptor_set = decoration_scalar(var_id, DEC_DESCRIPTOR_SET)
            binding = decoration_scalar(var_id, DEC_BINDING)
            if descriptor_set is None or binding is None:
                raise RuntimeError("raw buffer descriptor lacks set/binding")
            target_bindings.append((descriptor_set, binding))
        if not target_bindings:
            raise RuntimeError("no raw u8 buffer descriptors found")
    target_vars: list[int] = []
    payload_arrays: set[int] = set()
    payload_accesses: list[Inst] = []
    payload_pointer_types: set[int] = set()

    for descriptor_set, binding in target_bindings:
        candidates = [
            var_id for var_id in variable_types
            if decoration_scalar(var_id, DEC_DESCRIPTOR_SET) == descriptor_set
            and decoration_scalar(var_id, DEC_BINDING) == binding
        ]
        var_id = unique_one(candidates, f"set {descriptor_set} binding {binding} variable")
        target_vars.append(var_id)
        var_type, storage = variable_types[var_id]
        if storage != STORAGE_BUFFER:
            raise RuntimeError("target descriptor is not StorageBuffer")
        ptr_storage, descriptor_array_type = pointer_types.get(var_type, (None, None))
        if ptr_storage != STORAGE_BUFFER:
            raise RuntimeError("target descriptor pointer is not StorageBuffer")
        descriptor_struct = runtime_arrays.get(descriptor_array_type)
        if descriptor_struct is None:
            raise RuntimeError("target descriptor variable is not runtime array")
        members = structs.get(descriptor_struct)
        if not members or len(members) != 1:
            raise RuntimeError(f"unexpected buffer struct: {members}")
        payload_array = members[0]
        if runtime_arrays.get(payload_array) != uint8:
            raise RuntimeError("target payload is not u8")
        if decoration_scalar(payload_array, DEC_ARRAY_STRIDE) != 1:
            raise RuntimeError("target payload ArrayStride != 1")
        payload_arrays.add(payload_array)

        selected_descriptor_ptrs = {
            inst.args[1] for inst in insts
            if inst.op == OP_ACCESS_CHAIN and len(inst.args) >= 4 and inst.args[2] == var_id
        }
        if not selected_descriptor_ptrs:
            raise RuntimeError(f"no descriptor selection for set {descriptor_set} binding {binding}")
        for inst in insts:
            if inst.op == OP_ACCESS_CHAIN and len(inst.args) >= 5 and inst.args[2] in selected_descriptor_ptrs:
                payload_accesses.append(inst)
                payload_pointer_types.add(inst.args[0])

    if not payload_accesses:
        raise RuntimeError("no raw payload accesses found")
    for ptr_type in payload_pointer_types:
        storage, pointee = pointer_types.get(ptr_type, (None, None))
        if storage != STORAGE_BUFFER or pointee != uint8:
            raise RuntimeError(f"payload pointer %{ptr_type} is not StorageBuffer u8")

    payload_ptr_results = {inst.args[1] for inst in payload_accesses}
    load_insts = [
        inst for inst in insts
        if inst.op == OP_COOP_MATRIX_LOAD_KHR and len(inst.args) >= 3 and inst.args[2] in payload_ptr_results
    ]
    store_insts = [
        inst for inst in insts
        if inst.op == OP_COOP_MATRIX_STORE_KHR and inst.args and inst.args[0] in payload_ptr_results
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
        elif inst.op == OP_ACCESS_CHAIN and len(inst.args) >= 3 and inst.args[2] in payload_ptr_results:
            unexpected.append((inst.op, inst.args[2]))
    if unexpected:
        raise RuntimeError(f"raw payload pointers have unexpected uses: {unexpected[:8]}")

    value_to_const: dict[int,int] = {}
    for const_id, (type_id, value) in constants.items():
        if type_id == uint32:
            value_to_const.setdefault(value, const_id)
    shift_const = value_to_const.get(shift_amount)
    new_global_constants: list[Inst] = []
    if shift_const is None:
        shift_const = alloc_id()
        new_global_constants.append(Inst(OP_CONSTANT, [uint32, shift_const, shift_amount]))
        value_to_const[shift_amount] = shift_const

    result_def_index: dict[int,int] = {}
    result_type_ops = {
        61,65,66,124,126,127,128,129,130,131,132,133,134,
        194,195,196,197,198,199,200,201,202,203,204,205,
        OP_SELECT,245,4457,4459,4460,
    }
    for idx, inst in enumerate(insts):
        if inst.op in result_type_ops and len(inst.args) >= 2:
            result_def_index[inst.args[1]] = idx
        elif inst.op in (OP_CONSTANT, OP_VARIABLE) and len(inst.args) >= 2:
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
            if inst.op == OP_BITCAST and operands:
                out = guaranteed_tz(operands[0])
            elif inst.op in (OP_IADD, OP_ISUB, OP_BITWISE_OR, OP_BITWISE_XOR) and len(operands) >= 2:
                out = min(guaranteed_tz(operands[0]), guaranteed_tz(operands[1]))
            elif inst.op == OP_IMUL and len(operands) >= 2:
                out = min(32, guaranteed_tz(operands[0]) + guaranteed_tz(operands[1]))
            elif inst.op == OP_BITWISE_AND and len(operands) >= 2:
                out = max(guaranteed_tz(operands[0]), guaranteed_tz(operands[1]))
            elif inst.op == OP_SHIFT_LEFT_LOGICAL and len(operands) >= 2:
                s = constants.get(operands[1])
                if s is not None and s[0] == uint32: out = min(32, guaranteed_tz(operands[0]) + s[1])
            elif inst.op in (OP_SHIFT_RIGHT_LOGICAL, OP_SHIFT_RIGHT_ARITHMETIC) and len(operands) >= 2:
                s = constants.get(operands[1])
                if s is not None and s[0] == uint32: out = max(0, guaranteed_tz(operands[0]) - s[1])
            elif inst.op == OP_SELECT and len(operands) >= 3:
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
    dynamic_insert_after: dict[int,list[Inst]] = {}
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
                new_global_constants.append(Inst(OP_CONSTANT, [uint32, existing, scaled]))
                value_to_const[scaled] = existing
            replacement_index[old_index] = existing
        else:
            def_idx = result_def_index.get(old_index)
            if def_idx is None:
                raise RuntimeError(f"cannot find dynamic offset definition %{old_index}")
            new_id = alloc_id()
            replacement_index[old_index] = new_id
            dynamic_insert_after.setdefault(def_idx, []).append(
                Inst(OP_SHIFT_RIGHT_LOGICAL, [uint32, new_id, old_index, shift_const])
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
            new_global_constants.append(Inst(OP_CONSTANT, [uint32, existing, scaled]))
            value_to_const[scaled] = existing
        stride_replacements[stride_id] = existing

    for inst in insts:
        if inst.op == OP_TYPE_RUNTIME_ARRAY and inst.args[0] in payload_arrays:
            inst.args[1] = wide_type
        elif inst.op == OP_TYPE_POINTER and inst.args[0] in payload_pointer_types:
            inst.args[2] = wide_type
        elif inst.op == OP_DECORATE and len(inst.args) == 3 and inst.args[0] in payload_arrays and inst.args[1] == DEC_ARRAY_STRIDE:
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

    first_function = next((i for i,x in enumerate(insts) if x.op == OP_FUNCTION), None)
    if first_function is None:
        raise RuntimeError("module has no function")

    payload_type_indices = [
        i for i, x in enumerate(insts)
        if x.op == OP_TYPE_RUNTIME_ARRAY and x.args[0] in payload_arrays
    ]
    if not payload_type_indices:
        raise RuntimeError("cannot locate payload runtime-array declarations")
    first_payload_type = min(payload_type_indices)

    # Define uint32 before adding its vector type.
    uint32_type_index = next(
        (i for i, x in enumerate(insts)
         if x.op == OP_TYPE_INT and x.args and x.args[0] == uint32),
        None,
    )
    if uint32_type_index is None or uint32_type_index >= first_payload_type:
        raise RuntimeError("uint32 type is not defined before the payload arrays")

    rebuilt: list[Inst] = []
    for idx, inst in enumerate(insts):
        if inst.op == OP_TYPE_INT and inst.args[0] == uint32 and relocated_type is not None:
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
    write_module(output_path, header, rebuilt)
    out_header, _ = read_module(output_path)
    if out_header[3] != next_id: raise RuntimeError("output bound mismatch")

    return {
        "input": str(input_path), "output": str(output_path), "variant": variant,
        "element_bytes": element_bytes, "lanes": lanes,
        "original_bound": original_bound, "new_bound": next_id,
        "target_bindings": [f"set{s}.binding{b}" for s, b in target_bindings],
        "target_variables": target_vars,
        "payload_runtime_arrays": sorted(payload_arrays),
        "payload_pointer_types": sorted(payload_pointer_types),
        "cooperative_matrix_loads_rewritten": len(load_insts),
        "cooperative_matrix_stores_rewritten": len(store_insts),
        "payload_accesses_rewritten": len(payload_accesses),
        "unique_offsets": len(index_ids),
        "dynamic_offsets_scaled": sum(len(v) for v in dynamic_insert_after.values()),
        "minimum_guaranteed_offset_alignment_bytes": 1 << min(align_bits.values()),
        "new_types": len(new_global_types), "new_constants": len(new_global_constants),
        "relocated_types": int(relocated_type is not None),
        "original_bytes": input_path.stat().st_size, "patched_bytes": output_path.stat().st_size,
        "transform": f"raw buffer u8 -> {variant}; byte offsets /{element_bytes}; matrix strides /{element_bytes}",
    }

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--variant", choices=["u32","uvec2","uvec4"], required=True)
    p.add_argument("--report", type=Path)
    a = p.parse_args()
    r = patch_module(a.input, a.output, a.variant)
    text = json.dumps(r, indent=2)
    print(text)
    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(text+"\n", encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
