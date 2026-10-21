# SPDX-License-Identifier: GPL-3.0-only AND GPL-2.0-or-later
# Original glue: Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE and LICENSES/GPL-2.0.txt.

"""Simplify INT8 SPIR-V byte packing, extraction and requantization, and reuse
division results for remainder calculations without changing output values.
"""

import argparse, hashlib, importlib.util, json, re, struct, subprocess, tempfile
from collections import defaultdict
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "patch_driver", Path(__file__).resolve().parents[2] / "patch.py"
)
patch_driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch_driver)

class StructuralIR(patch_driver.IR):
    def __init__(self, text):
        # Address proofs stay local to blocks, pointers and expressions; do not cross calls.
        self.lines = text.splitlines()
        self.d, self.location, self.blocks = {}, {}, {}
        self.serial = 0
        self.globals, self.annotations, self.report = [], [], {}
        self.decorations = defaultdict(list)
        block = None
        for n, line in enumerate(self.lines):
            m = re.match(r"\s*(%\w+) = (Op\w+)(?: (.*))?$", line)
            if m:
                ident, op, args = m.groups()
                args = (args or "").split()
                if op == "OpFunction":
                    block = None
                elif op == "OpLabel":
                    block = ident
                    self.blocks[block] = []
                self.d[ident] = op, args
                self.location[ident] = block, n
            if block:
                self.blocks[block].append(n)
            if line.strip() == "OpFunctionEnd":
                block = None
            decoration = re.match(r"\s*OpDecorate (%\w+) (.*)", line)
            if decoration:
                self.decorations[decoration[1]].append(decoration[2])
        if not self.blocks:
            raise RuntimeError("No function blocks")
        self.entry = next(iter(self.blocks))

    def replace(self, ident, op, args):
        self.lines[self.location[ident][1]] = f"{ident} = {op} " + " ".join(args)
        self.d[ident] = op, args

    def literal(self, ident, seen=None):
        seen = set() if seen is None else seen
        if ident in seen:
            return None
        seen = seen | {ident}
        op, a = self.d.get(ident, ("", []))
        if op == "OpConstant" and self.d[a[0]][0] == "OpTypeInt":
            return int(a[1], 0) & ((1 << int(self.d[a[0]][1][0])) - 1)
        if op == "OpConstantComposite" or op == "OpCompositeConstruct":
            values = [self.literal(x, seen) for x in a[1:]]
            return values if all(x is not None for x in values) else None
        if op in ("OpCopyObject", "OpBitcast", "OpUConvert"):
            # Integer bit-preserving evaluation only; never evaluate floats.
            if self.d.get(a[0], ("", []))[0] != "OpTypeInt":
                return None
            value = self.literal(a[1], seen)
            return value if isinstance(value, int) else None
        if op == "OpCompositeExtract":
            co, ca = self.d.get(a[1], ("", []))
            if co in ("OpCompositeConstruct", "OpConstantComposite") and len(a) == 3:
                return self.literal(ca[1 + int(a[2])], seen)
            value = self.literal(a[1], seen)
            for index in a[2:]:
                if not isinstance(value, list) or int(index) >= len(value):
                    return None
                value = value[int(index)]
            return value
        funcs = {
            "OpIAdd": lambda x, y: x + y,
            "OpISub": lambda x, y: x - y,
            "OpIMul": lambda x, y: x * y,
            "OpBitwiseAnd": lambda x, y: x & y,
            "OpBitwiseOr": lambda x, y: x | y,
            "OpShiftLeftLogical": lambda x, y: x << y,
            "OpShiftRightLogical": lambda x, y: x >> y,
        }
        if op in funcs and self.d[a[0]] == ("OpTypeInt", ["32", "0"]):
            x, y = (self.literal(v, seen) for v in a[1:])
            if isinstance(x, int) and isinstance(y, int):
                if "Shift" in op and not 0 <= y < 32:
                    return None
                return funcs[op](x, y) & 0xFFFFFFFF
        return None

    def z(self, dispatch_z):
        if dispatch_z != 1:
            raise RuntimeError("Requires dispatch_z=1")
        local_modes = re.findall(
            r"OpExecutionMode %\w+ LocalSize (\d+) (\d+) (\d+)", self.finish()
        )
        local_z_one = len(local_modes) == 1 and local_modes[0][2] == "1"
        roots = dict(
            re.findall(
                r"OpDecorate (%\w+) BuiltIn (WorkgroupId|GlobalInvocationId|LocalInvocationId)\b",
                self.finish(),
            )
        )
        count = 0
        for ident, (op, a) in list(self.d.items()):
            if op != "OpLoad":
                continue
            ptr, component = a[1], None
            p_op, p_a = self.d.get(ptr, ("", []))
            root = ptr
            if p_op in ("OpAccessChain", "OpInBoundsAccessChain") and len(p_a) == 3:
                root, component = p_a[1], self.literal(p_a[2])
            if root not in roots or component not in (None, 2):
                continue
            if roots[root] != "WorkgroupId" and not local_z_one:
                raise RuntimeError(
                    "Global/LocalInvocationId Z requires literal LocalSize Z=1; specialization sizes unsupported"
                )
            if len(a) != 2:
                raise RuntimeError("Built-in load memory operands unsupported")
            if component == 2:
                self.replace(ident, "OpCopyObject", [a[0], self.constant(a[0], 0)])
            else:
                ty_op, ty_a = self.d[a[0]]
                if ty_op != "OpTypeVector" or ty_a[1] != "3":
                    raise RuntimeError("Expected three-component built-in")
                raw, x, y = self.fresh(), self.fresh(), self.fresh()
                zero = self.constant(ty_a[0], 0)
                self.lines[self.location[ident][1]] = "\n".join(
                    [
                        f"{raw} = OpLoad {a[0]} {ptr}",
                        f"{x} = OpCompositeExtract {ty_a[0]} {raw} 0",
                        f"{y} = OpCompositeExtract {ty_a[0]} {raw} 1",
                        f"{ident} = OpCompositeConstruct {a[0]} {x} {y} {zero}",
                    ]
                )
                self.d[ident] = "OpCompositeConstruct", [a[0], x, y, zero]
            count += 1
        if not count:
            raise RuntimeError("No supported Z loads")
        return {"z_loads": count, "dispatch_z": 1, "local_z_one": local_z_one}

    def divrem(self):
        """Compute signed remainders from the existing quotient.

        Preserve division guards, signed wrapping and control flow.
        """
        count = 0
        for block, locations in self.blocks.items():
            pairs = {}
            for n in locations:
                m = re.match(r"\s*(%\w+) = (OpSDiv|OpSRem) (.*)$", self.lines[n])
                if m:
                    pairs.setdefault(tuple(m[3].split()), {})[m[2]] = (m[1], n)
            for a, pair in pairs.items():
                if set(pair) != {"OpSDiv", "OpSRem"}:
                    continue
                ty, lhs, rhs = a
                if self.d[ty][0] != "OpTypeInt":
                    continue
                q, qn = pair["OpSDiv"]
                r, rn = pair["OpSRem"]
                product = self.fresh()
                if qn < rn:
                    prefix, quotient = [], q
                else:
                    quotient = self.fresh()
                    prefix = [f"{quotient} = OpSDiv {ty} {lhs} {rhs}"]
                    self.lines[qn] = f"{q} = OpCopyObject {ty} {quotient}"
                self.lines[rn] = "\n".join(
                    prefix
                    + [
                        f"{product} = OpIMul {ty} {quotient} {rhs}",
                        f"{r} = OpISub {ty} {lhs} {product}",
                    ]
                )
                count += 1
        if not count:
            raise RuntimeError("No same-block signed quotient/remainder pair")
        return {
            "divrem_pairs": count,
            "defined_domain": "nonzero divisor; no signed division overflow",
            "original_guard_control_flow_preserved": True,
        }

    def packed_mask(self):
        """Move a shared zero selection after byte packing.

        Only accept integer clamps that map zero to zero. Keep memory operations,
        barriers and branch guards in place.
        """
        count = 0
        for ident, (op, a) in list(self.d.items()):
            if op != "OpBitcast" or self.d[a[0]] != ("OpTypeInt", ["32", "0"]):
                continue
            uop, ua = self.d[a[1]]
            if uop != "OpUConvert":
                continue
            vo, va = self.d[ua[0]]
            if (
                vo != "OpTypeVector"
                or va[1] != "4"
                or self.d[va[0]][0] != "OpTypeInt"
                or self.d[va[0]][1][0] != "8"
            ):
                continue
            source = ua[1]
            clamp = None
            co, ca = self.d[source]
            if (
                co == "OpExtInst"
                and ca[2] == "SClamp"
                and self.d[ca[1]] == ("OpExtInstImport", ['"GLSL.std.450"'])
            ):
                lows, highs = self.literal(ca[4]), self.literal(ca[5])
                if (
                    not isinstance(lows, list)
                    or not isinstance(highs, list)
                    or len(lows) != 4
                    or len(highs) != 4
                ):
                    continue
                signed = lambda x: x if x < 2**31 else x - 2**32
                if not all(
                    signed(lo) <= 0 <= signed(hi) for lo, hi in zip(lows, highs)
                ):
                    continue
                clamp, source = ca, ca[3]
            co, ca = self.d[source]
            if co != "OpCompositeConstruct" or len(ca) != 5:
                continue
            vector_op, vector_a = self.d[ca[0]]
            if (
                vector_op != "OpTypeVector"
                or vector_a[1] != "4"
                or self.d[vector_a[0]][0] != "OpTypeInt"
                or self.d[vector_a[0]][1][0] != "32"
            ):
                continue
            selects = [self.d[x] for x in ca[1:]]
            if not all(
                o == "OpSelect"
                and len(s) == 4
                and s[0] == vector_a[0]
                and self.literal(s[3]) == 0
                for o, s in selects
            ):
                continue
            conditions = {s[1] for o, s in selects}
            if len(conditions) != 1:
                continue
            values = [s[2] for o, s in selects]
            raw, converted, packed = self.fresh(), self.fresh(), self.fresh()
            lines = [f"{raw} = OpCompositeConstruct {ca[0]} " + " ".join(values)]
            if clamp:
                clamped = self.fresh()
                lines.append(
                    f"{clamped} = OpExtInst {clamp[0]} {clamp[1]} SClamp {raw} {clamp[4]} {clamp[5]}"
                )
                raw = clamped
            lines += [
                f"{converted} = OpUConvert {ua[0]} {raw}",
                f"{packed} = OpBitcast {a[0]} {converted}",
                f"{ident} = OpSelect {a[0]} {next(iter(conditions))} {packed} {self.constant(a[0], 0)}",
            ]
            self.lines[self.location[ident][1]] = "\n".join(lines)
            count += 1
        if not count:
            raise RuntimeError("No proved four-channel zero-mask packing")
        return {
            "packed_masks": count,
            "preserves_rounding": True,
            "moves_memory_operations": False,
        }

class IR(StructuralIR):
    def constant(self, ty, value):
        op, args = self.d.get(ty, ("", []))
        if op == "OpTypeInt":
            bits = int(args[0])
            value &= (1 << bits) - 1
            if args[1] == "1" and value & (1 << (bits - 1)):
                value -= 1 << bits
        return super().constant(ty, value)

    INTEGER_PURE = {
        "OpIAdd",
        "OpISub",
        "OpIMul",
        "OpShiftLeftLogical",
        "OpShiftRightLogical",
        "OpShiftRightArithmetic",
        "OpBitwiseAnd",
        "OpBitwiseOr",
        "OpBitwiseXor",
        "OpNot",
        "OpUConvert",
        "OpSConvert",
        "OpBitcast",
        "OpCopyObject",
        "OpCompositeExtract",
        "OpCompositeConstruct",
        "OpVectorShuffle",
        "OpSelect",
        "OpIEqual",
        "OpINotEqual",
        "OpULessThan",
        "OpSLessThan",
        "OpUGreaterThan",
        "OpSGreaterThan",
        "OpLogicalAnd",
        "OpLogicalOr",
        "OpLogicalNot",
        "OpAccessChain",
        "OpInBoundsAccessChain",
    }
    # These operations preserve the load cache, not address equality.
    # Any unlisted instruction invalidates the cache.
    CACHE_PURE = INTEGER_PURE | {
        "OpFAdd",
        "OpFSub",
        "OpFMul",
        "OpFDiv",
        "OpFNegate",
        "OpSNegate",
        "OpConvertSToF",
        "OpConvertUToF",
        "OpConvertFToS",
        "OpConvertFToU",
        "OpFConvert",
        "OpSDot",
        "OpUDot",
        "OpSUDot",
        "OpCompositeInsert",
        "OpVectorTimesScalar",
        "OpFOrdLessThan",
        "OpFOrdGreaterThan",
        "OpFOrdEqual",
        "OpFOrdNotEqual",
        "OpCooperativeMatrixMulAddKHR",
        "OpCooperativeMatrixLoadKHR",
        "OpPhi",
        "OpNop",
        "OpLine",
        "OpNoLine",
    }

    def expression(self, ident, seen=None):
        """Compare types and expression structure without changing arithmetic."""
        seen = set() if seen is None else seen
        if ident in seen:
            return ("ssa", ident)
        seen = seen | {ident}
        op, a = self.d.get(ident, ("", []))
        if op == "OpConstant" and self.d.get(a[0], ("", []))[0] == "OpTypeInt":
            bits = int(self.d[a[0]][1][0])
            return (op, a[0], int(a[1], 0) & ((1 << bits) - 1))
        if (
            op == "OpLoad"
            and len(a) == 2
            and self.pointer_storage(a[1]) in ("PushConstant", "Input")
        ):
            return (op, a[0], self.expression(a[1], seen))
        if op not in self.INTEGER_PURE:
            return ("ssa", ident)
        # Preserve types and operand order.
        return (
            op,
            a[0],
            *(self.expression(x, seen) if x.startswith("%") else x for x in a[1:]),
        )

    def scalar_load_reuse(self):
        text = self.finish()
        if re.search(
            r"Op(?:Member)?Decorate .*\b(Volatile|Coherent|AliasedPointer)\b", text
        ):
            raise RuntimeError(
                "Volatile/coherent/aliased-pointer declarations unsupported"
            )
        count = 0
        for block, locations in self.blocks.items():
            cache = {}
            for n in locations:
                tokens = self.lines[n].strip().split()
                if not tokens:
                    continue
                result = tokens[0] if len(tokens) > 2 and tokens[1] == "=" else None
                op, a = (tokens[2], tokens[3:]) if result else (tokens[0], tokens[1:])
                if op == "OpLoad":
                    if len(a) != 2:
                        cache.clear()
                        continue
                    if self.pointer_storage(a[1]) != "StorageBuffer":
                        continue
                    to, ta = self.d.get(a[0], ("", []))
                    if to not in ("OpTypeInt", "OpTypeFloat", "OpTypeVector"):
                        continue
                    key = (a[0], self.expression(a[1]))
                    previous = cache.get(key)
                    if previous:
                        self.replace(result, "OpCopyObject", [a[0], previous])
                        count += 1
                    else:
                        cache[key] = result
                elif op == "OpStore" and self.pointer_storage(a[0]) == "Function":
                    # Function storage cannot alias a StorageBuffer; keep its loads opaque.
                    continue
                elif op not in self.CACHE_PURE:
                    cache.clear()
        if not count:
            raise RuntimeError(
                "No same-block equivalent scalar/vector SSBO loads proved"
            )
        return {
            "loads_reused": count,
            "contract": "same block; typed structural addresses; no intervening shared effect/unknown opcode",
            "cross_block_reuse": False,
            "external_concurrent_writes": "not permitted by the dispatch resource contract",
        }

    def byte_extract(self):
        """Extract the second byte directly from each 32-bit integer.

        Sign extension only changes discarded bits, so no value bound is needed.
        OpBitcast maps low bits to the first vector components.
        Keep source clamps, rounding and memory operations unchanged.
        """
        count = 0
        for ident, (op, a) in list(self.d.items()):
            if op != "OpUConvert" or len(a) != 2:
                continue
            vo, va = self.d.get(a[0], ("", []))
            if (
                vo != "OpTypeVector"
                or va[1] != "4"
                or self.d.get(va[0]) != ("OpTypeInt", ["8", "0"])
            ):
                continue
            src = a[1]
            so, sa = self.d.get(src, ("", []))
            if (
                so in ("OpBitcast", "OpCopyObject", "OpSConvert", "OpUConvert")
                and len(sa) == 2
            ):
                if so in ("OpSConvert", "OpUConvert"):
                    to, ta = self.d.get(sa[0], ("", []))
                    if (
                        to != "OpTypeVector"
                        or ta[1] != "4"
                        or self.d.get(ta[0], ("", []))[0] != "OpTypeInt"
                        or int(self.d[ta[0]][1][0]) < 8
                    ):
                        continue
                src = sa[1]
                so, sa = self.d.get(src, ("", []))
            if so != "OpCompositeConstruct" or len(sa) != 5:
                continue
            co, ca = self.d.get(sa[0], ("", []))
            if (
                co != "OpTypeVector"
                or ca[1] != "4"
                or self.d.get(ca[0], ("", []))[0] != "OpTypeInt"
                or self.d[ca[0]][1][0] != "32"
            ):
                continue
            # Conversion's immediate operand must also be a 4x32 integer vector.
            immediate_ty = self.d[a[1]][1][0]
            io, ia = self.d.get(immediate_ty, ("", []))
            if (
                io != "OpTypeVector"
                or ia[1] != "4"
                or self.d.get(ia[0], ("", []))[0] != "OpTypeInt"
                or int(self.d[ia[0]][1][0]) < 8
            ):
                continue
            words = []
            for r in sa[1:]:
                ro, ra = self.d.get(r, ("", []))
                guard = None
                if ro == "OpSelect" and len(ra) == 4 and ra[0] == ca[0]:
                    # Preserve the original overflow selection when simplifying the packed bytes.
                    for constant, shifted, polarity in (
                        (ra[2], ra[3], True),
                        (ra[3], ra[2], False),
                    ):
                        value = self.literal(constant)
                        if isinstance(value, int):
                            guard = (ra[1], value & 255, polarity)
                            ro, ra = self.d.get(shifted, ("", []))
                            break
                if (
                    ro != "OpShiftRightArithmetic"
                    or len(ra) != 3
                    or ra[0] != ca[0]
                    or self.literal(ra[2]) != 8
                ):
                    break
                words.append((ra[1], guard))
            if len(words) != 4:
                continue
            inserted, components = [], []
            for word, guard in words:
                packed, byte = self.fresh(), self.fresh()
                inserted += [
                    f"{packed} = OpBitcast {a[0]} {word}",
                    f"{byte} = OpCompositeExtract {va[0]} {packed} 1",
                ]
                if guard:
                    condition, value, polarity = guard
                    constant = self.constant(va[0], value)
                    result = self.fresh()
                    yes, no = (constant, byte) if polarity else (byte, constant)
                    inserted.append(
                        f"{result} = OpSelect {va[0]} {condition} {yes} {no}"
                    )
                    byte = result
                components.append(byte)
            self.replace(ident, "OpCompositeConstruct", [a[0], *components])
            n = self.location[ident][1]
            self.lines[n] = "\n".join([*inserted, self.lines[n]])
            count += 1
        if not count:
            raise RuntimeError(
                "No four-channel shift-by-8 to unsigned-byte packing matched; no prerequisite rounding rewrite implied"
            )
        return {
            "byte_pack_groups": count,
            "all_int32_patterns_exact": True,
            "rounding_and_clamps_unchanged": True,
            "specification": "https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpBitcast",
        }

    def requant_guarded(self):
        """Clamp before rounding to nearest even, preserving signed addition's wraparound.

        For shifts n from 1 to 23, values at or above 2**31 - 2**(n - 1) must still
        clamp to -128. This does not depend on model weights.
        """
        imports = {
            i
            for i, (o, a) in self.d.items()
            if o == "OpExtInstImport" and a == ['"GLSL.std.450"']
        }

        def round_pattern(r):
            op, a = self.d.get(r, ("", []))
            if (
                op != "OpShiftRightArithmetic"
                or len(a) != 3
                or self.d.get(a[0], ("", []))[0] != "OpTypeInt"
                or self.d[a[0]][1][0] != "32"
            ):
                return None
            n = self.literal(a[2])
            if not isinstance(n, int) or not 1 <= n <= 23:
                return None
            ao, aa = self.d.get(a[1], ("", []))
            if ao != "OpIAdd" or len(aa) != 3 or aa[0] != a[0]:
                return None
            for summed, parity in ((aa[1], aa[2]), (aa[2], aa[1])):
                so, sa = self.d.get(summed, ("", []))
                po, pa = self.d.get(parity, ("", []))
                if (
                    so != "OpIAdd"
                    or po != "OpBitwiseAnd"
                    or sa[0] != a[0]
                    or pa[0] != a[0]
                ):
                    continue
                for shifted, one in ((pa[1], pa[2]), (pa[2], pa[1])):
                    if self.literal(one) != 1:
                        continue
                    qo, qa = self.d.get(shifted, ("", []))
                    if (
                        qo != "OpShiftRightLogical"
                        or len(qa) != 3
                        or qa[0] != a[0]
                        or self.literal(qa[2]) != n
                    ):
                        continue
                    x = qa[1]
                    # This proof requires full-width integer arithmetic.
                    if any(
                        "RelaxedPrecision" in self.decorations[i]
                        for i in (r, a[1], summed, parity, shifted, x)
                    ):
                        continue
                    if any(
                        v == x and self.literal(k) == (1 << (n - 1)) - 1
                        for v, k in ((sa[1], sa[2]), (sa[2], sa[1]))
                    ):
                        return x, n, a[0]
            return None

        count, shifts = 0, []
        for ident, (op, a) in list(self.d.items()):
            if (
                op != "OpExtInst"
                or len(a) != 6
                or a[1] not in imports
                or a[2] != "SClamp"
            ):
                continue
            lo, hi = self.literal(a[4]), self.literal(a[5])
            if lo != [0xFFFFFF80] * 4 or hi != [127] * 4:
                # Native min/max vectors can themselves be signed int16.
                def signed_bounds(i):
                    co, ca = self.d.get(i, ("", []))
                    if co != "OpConstantComposite" or len(ca) != 5:
                        return None
                    result = []
                    for child in ca[1:]:
                        vo, va = self.d.get(child, ("", []))
                        if (
                            vo != "OpConstant"
                            or self.d.get(va[0], ("", []))[0] != "OpTypeInt"
                        ):
                            return None
                        bits = int(self.d[va[0]][1][0])
                        value = int(va[1], 0) & ((1 << bits) - 1)
                        result.append(
                            value - (1 << bits) if value & (1 << (bits - 1)) else value
                        )
                    return result

                if (
                    signed_bounds(a[4]) != [-128] * 4
                    or signed_bounds(a[5]) != [127] * 4
                ):
                    continue
            co, ca = self.d.get(a[3], ("", []))
            if co != "OpCompositeConstruct" or len(ca) != 5:
                continue
            parts = [round_pattern(r) for r in ca[1:]]
            if any(p is None for p in parts) or len({p[2] for p in parts}) != 1:
                continue
            vt, va = self.d.get(ca[0], ("", []))
            rt, ra = self.d.get(a[0], ("", []))
            if (
                vt != "OpTypeVector"
                or va != [parts[0][2], "4"]
                or rt != "OpTypeVector"
                or ra[1] != "4"
                or self.d.get(ra[0], ("", []))[0] != "OpTypeInt"
                or self.d[ra[0]][1][0] not in ("16", "32")
            ):
                continue
            boolean = self.type("OpTypeBool", [])
            inserted, results = [], []
            for x, n, ty in parts:
                lo_id, hi_id = self.constant(ty, -(128 << n)), self.constant(
                    ty, 127 << n
                )
                k, one, shift = (
                    self.constant(ty, (1 << (n - 1)) - 1),
                    self.constant(ty, 1),
                    self.constant(ty, n),
                )
                threshold, negative = self.constant(
                    ty, (1 << 31) - (1 << (n - 1))
                ), self.constant(ty, -128)
                clipped, shifted, parity, add, summed, rounded, overflow, result = [
                    self.fresh() for _ in range(8)
                ]
                inserted.extend(
                    [
                        f"{clipped} = OpExtInst {ty} {a[1]} SClamp {x} {lo_id} {hi_id}",
                        f"{shifted} = OpShiftRightLogical {ty} {clipped} {shift}",
                        f"{parity} = OpBitwiseAnd {ty} {shifted} {one}",
                        f"{add} = OpIAdd {ty} {clipped} {k}",
                        f"{summed} = OpIAdd {ty} {add} {parity}",
                        f"{rounded} = OpShiftRightArithmetic {ty} {summed} {shift}",
                        f"{overflow} = OpSGreaterThanEqual {boolean} {x} {threshold}",
                        f"{result} = OpSelect {ty} {overflow} {negative} {rounded}",
                    ]
                )
                results.append(result)
                shifts.append(n)
            vector = self.fresh()
            inserted.append(
                f"{vector} = OpCompositeConstruct {ca[0]} " + " ".join(results)
            )
            if a[0] == ca[0]:
                self.replace(ident, "OpCopyObject", [a[0], vector])
            elif self.d[ra[0]][1][0] == "32":
                self.replace(ident, "OpBitcast", [a[0], vector])
            else:
                self.replace(ident, "OpSConvert", [a[0], vector])
            line = self.location[ident][1]
            self.lines[line] = "\n".join([*inserted, self.lines[line]])
            count += 1
        if not count:
            raise RuntimeError(
                "No four-channel exact integer RTE/clamp pattern matched"
            )
        return {
            "requant_groups": count,
            "channel_shifts": shifts,
            "signed_overflow_guard": True,
            "all_int32_patterns_exact": True,
            "changed_rounding": False,
            "bank_bound_assumptions": False,
            "note": "Guarded adaptation; not the unguarded author transform or a performance claim",
        }

EXACT = ("packed_mask", "requant_guarded", "byte_extract", "divrem")

def apply(text, methods):
    reports = {}
    for method in methods:
        ir = IR(text)
        try:
            report = getattr(ir, method)()
        except RuntimeError as e:
            if not str(e).startswith("No "):
                raise
            reports[method] = {"matched": False, "reason": str(e)}
            continue
        # Preserve the source entry-point interface and SPIR-V version.
        interfaces = iter(l for l in text.splitlines() if "OpEntryPoint " in l)
        lines = ir.finish().splitlines()
        lines = [next(interfaces) if "OpEntryPoint " in l else l for l in lines]
        text = "\n".join(lines) + "\n"
        reports[method] = dict(report, matched=True)
    return text, reports

def command(args):
    proc = subprocess.run(
        list(map(str, args)), text=True, capture_output=True, timeout=120
    )
    if proc.returncode:
        raise RuntimeError(" ".join(map(str, args)) + "\n" + proc.stdout + proc.stderr)
    return proc.stdout

def generate(assets, output, spirv_tools, windows_variants=None):
    assets, output, spirv_tools = Path(assets), Path(output), Path(spirv_tools)
    if output.exists():
        raise RuntimeError(
            "INT8 patch output must be fresh (preserve previous build artifacts)"
        )
    if output.resolve() == assets.resolve() or assets.resolve().is_relative_to(
        output.resolve()
    ):
        raise RuntimeError("Output must not replace/enclose stock assets")
    if (
        output.resolve().is_relative_to(assets.resolve())
        and output.resolve() != (assets / "optimized").resolve()
    ):
        raise RuntimeError(
            "Inside the stock tree, only the separate optimized directory is allowed"
        )
    sources = [
        (p, p.relative_to(assets))
        for p in sorted(assets.rglob("*.spv"))
        if "optimized" not in p.relative_to(assets).parts
        and p.name != "auto-exposure.spv"
    ]
    if windows_variants:
        sources += [
            (p, p.relative_to(windows_variants))
            for p in sorted(Path(windows_variants).rglob("*.spv"))
        ]
    by_path = {}
    for source, relative in sources:
        if (
            relative in by_path
            and source.read_bytes() != by_path[relative].read_bytes()
        ):
            raise RuntimeError("Conflicting shader sources for " + relative.as_posix())
        by_path[relative] = source
    sources = [(source, relative) for relative, source in sorted(by_path.items())]
    if not sources:
        raise RuntimeError("No INT8 shader inputs")
    output.mkdir(parents=True)
    records, cache = [], {}
    with tempfile.TemporaryDirectory(prefix="fsr4-int8-patches-") as temporary:
        temp = Path(temporary)
        for source, relative in sources:
            payload = source.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            if digest not in cache:
                original = command([spirv_tools / "spirv-dis", "--raw-id", source])
                version = struct.unpack_from("<I", payload, 4)[0]
                target = f"spv{version >> 16}.{version >> 8 & 255}"
                exact_text, exact_report = apply(original, EXACT)
                if any(r["matched"] for r in exact_report.values()):
                    assembly, binary = temp / "current.spvasm", temp / "current.spv"
                    assembly.write_text(exact_text)
                    command(
                        [
                            spirv_tools / "spirv-as",
                            "--target-env",
                            target,
                            assembly,
                            "-o",
                            binary,
                        ]
                    )
                    clean = temp / "clean.spv"
                    command(
                        [
                            spirv_tools / "spirv-opt",
                            "--eliminate-dead-code-aggressive",
                            "--compact-ids",
                            binary,
                            "-o",
                            clean,
                        ]
                    )
                    data = clean.read_bytes()
                else:
                    data = payload
                check = temp / "validate.spv"
                check.write_bytes(data)
                command([spirv_tools / "spirv-val", "--target-env", "vulkan1.3", check])
                cache[digest] = data, exact_report
            data, exact_report = cache[digest]
            dest = output / "exact" / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            records.append(
                {
                    "file": dest.relative_to(output).as_posix(),
                    "profile": "exact",
                    "source": relative.as_posix(),
                    "source_sha256": digest,
                    "spirv_sha256": hashlib.sha256(data).hexdigest(),
                    "spirv_version": struct.unpack_from("<I", data, 4)[0],
                    "exact": exact_report,
                }
            )
    script_paths = [Path(__file__), Path(patch_driver.__file__)]
    report = {
        "schema": 2,
        "complete": True,
        "exact_operators": EXACT,
        "stock_assets_modified": False,
        "script_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in script_paths
        },
        "shaders": records,
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Generated {len(records)} exact INT8 patch variants ({len(cache)} unique inputs).",
        flush=True,
    )
    return output

def patch(source, output, mode):
    source, output = Path(source), Path(output)
    if (
        output.exists()
        or output.with_suffix(".spvasm").exists()
        or output.with_suffix(".json").exists()
    ):
        raise RuntimeError("Use a fresh output path")
    if output.resolve().is_relative_to(
        (Path(__file__).resolve().parents[4] / "assets").resolve()
    ):
        raise RuntimeError("Do not replace production assets")
    module = IR(command(["spirv-dis", "--raw-id", source]))
    methods = {
        "scalar-load-reuse": module.scalar_load_reuse,
        "byte-extract": module.byte_extract,
        "requant-guarded": module.requant_guarded,
    }
    report = methods[mode]() if mode in methods else None
    if report is None:
        raise RuntimeError("Unknown mode")
    output.parent.mkdir(parents=True, exist_ok=True)
    assembly = output.with_suffix(".spvasm")
    assembly.write_text(module.finish())
    command(["spirv-as", "--target-env", "spv1.6", assembly, "-o", output])
    command(["spirv-val", "--target-env", "vulkan1.3", output])
    report.update(
        mode=mode,
        classification="EXPERIMENTAL_EXACT_UNDER_RECORDED_CONTRACT",
        source=str(source.resolve()),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        gpu_validated=False,
    )
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stock", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--spirv-tools", type=Path, required=True)
    parser.add_argument("--windows-variants", type=Path)
    args = parser.parse_args()
    generate(args.stock, args.output, args.spirv_tools, args.windows_variants)

if __name__ == "__main__":
    main()
