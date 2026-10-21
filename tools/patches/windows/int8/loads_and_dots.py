# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Optimize Windows INT8 loads and verify dot accumulation against the model weights."""

import argparse, collections, hashlib, json, math, re, shutil, struct, subprocess
from pathlib import Path


class Audit:
    def __init__(self, text, weights):
        if not __debug__:
            raise RuntimeError(
                "Overflow verification requires Python assertions enabled"
            )
        assert re.search(
            r"OpExecutionMode %\w+ LocalSize \d+ \d+ 1\s*$", text, re.M
        ), "proof requires local z=1"
        self.defs = {}
        self.blocks = {}
        self.where = {}
        self.loops = {}
        self.edges = {}
        self.stores = collections.defaultdict(list)
        self.weights = weights
        self.cache = {}
        self.inductions = {}
        block = None
        for line in text.splitlines():
            m = re.match(r"\s*(%\w+) = (\w+)\s*(.*)", line)
            if m:
                ident, op, a = m[1], m[2], m[3].split()
                self.defs[ident] = (op, a)
                if op == "OpLabel":
                    block = ident
                    self.blocks[block] = []
                if block:
                    self.where[ident] = block
            else:
                a = line.split()
                if not a:
                    continue
                op, a = a[0], a[1:]
                ident = None
            if block:
                self.blocks[block].append((ident, op, a))
        for b, ins in self.blocks.items():
            for ident, op, a in ins:
                if op == "OpLoopMerge":
                    self.loops[b] = {"merge": a[0], "continue": a[1]}
                elif op == "OpBranch":
                    self.edges[b] = a
                elif op == "OpBranchConditional":
                    self.edges[b] = a[1:3]
                elif op == "OpStore":
                    self.stores[self.root(a[0])].append(a[1])
                elif op == "OpSwitch":
                    raise ValueError("unsupported switch")
        self.descriptor_sets = {
            i: int(n)
            for i, n in re.findall(r"OpDecorate (%\w+) DescriptorSet (\d+)", text)
        }
        self.strides = {
            i: int(n)
            for i, n in re.findall(r"OpDecorate (%\w+) ArrayStride (\d+)", text)
        }
        for h, loop in self.loops.items():
            region = set()
            pending = [h]
            while pending:
                b = pending.pop()
                if b == loop["merge"] or b in region:
                    continue
                region.add(b)
                pending.extend(self.edges.get(b, []))
            loop["region"] = region

    def root(self, p):
        op, a = self.defs[p]
        return (
            self.root(a[1]) if op in ("OpAccessChain", "OpInBoundsAccessChain") else p
        )

    def reaches_merge(self, b, merge):
        seen = set()
        while b not in seen:
            if b == merge:
                return True
            seen.add(b)
            ins = self.blocks[b]
            # The exit path must consist only of an unconditional branch.
            body = [(op, a) for _, op, a in ins if op != "OpLabel"]
            if len(body) != 1 or body[0][0] != "OpBranch":
                return False
            b = body[0][1][0]
        return False

    def induction(self, v):
        if v in self.inductions:
            return self.inductions[v]
        op, a = self.defs[v]
        assert op == "OpPhi" and len(a) == 5, (v, op, a)
        h = self.where[v]
        loop = self.loops[h]
        incoming = [(a[1], a[2]), (a[3], a[4])]
        steps = [
            (x, b)
            for x, b in incoming
            if self.defs[x] == ("OpIAdd", [a[0], v, "%uint_1"])
        ]
        assert len(steps) == 1, (v, incoming)
        step, back = steps[0]
        assert back == loop["continue"]
        initial, pre = next((x, b) for x, b in incoming if x != step)
        assert pre not in loop["region"]
        start = self.ev(initial)
        assert len(start) == 1
        start = next(iter(start))
        options = []
        for b in loop["region"]:
            for _, bop, ba in self.blocks[b]:
                if bop != "OpBranchConditional":
                    continue
                cop, ca = self.defs[ba[0]]
                if cop != "OpIEqual":
                    continue
                if not self.reaches_merge(ba[1], loop["merge"]):
                    continue
                for counter, limit in [(ca[1], ca[2]), (ca[2], ca[1])]:
                    if counter not in (v, step):
                        continue
                    # Every path to the backedge must pass the terminating comparison.
                    pending = [h]
                    seen = set()
                    while pending:
                        node = pending.pop()
                        if node in seen or node in (b, loop["merge"]):
                            continue
                        assert (
                            node != loop["continue"]
                        ), "exit comparison does not dominate backedge"
                        seen.add(node)
                        pending.extend(self.edges.get(node, []))
                    try:
                        values = self.ev(limit)
                    except (ValueError, AssertionError, KeyError):
                        continue
                    assert len(values) == 1
                    stop = next(iter(values))
                    last = stop - (1 if counter == step else 0)
                    if start <= last < 1024:
                        options.append((last, start, ba[0]))
        assert len(options) == 1, (v, options)
        last, start, condition = options[0]
        result = {
            "values": set(range(start, last + 1)),
            "iterations": last - start + 1,
            "header": h,
            "condition": condition,
        }
        self.inductions[v] = result
        return result

    def ev(self, v):
        if v in self.cache:
            return self.cache[v]
        op, a = self.defs[v]
        if op == "OpConstant":
            r = {int(a[1])}
        elif op == "OpPhi":
            r = self.induction(v)["values"]
        elif op == "OpLoad":
            po, pa = self.defs[a[1]]
            assert (
                po == "OpAccessChain"
                and pa[1] in ("%gl_GlobalInvocationID", "%gl_WorkGroupID")
                and self.ev(pa[-1]) == {2}
            ), (v, po, pa)
            r = {0}  # Dispatch and workgroup depth are both 1.
        elif op in (
            "OpIAdd",
            "OpISub",
            "OpIMul",
            "OpShiftLeftLogical",
            "OpShiftRightLogical",
            "OpBitwiseAnd",
            "OpBitwiseOr",
        ):
            funcs = {
                "OpIAdd": lambda x, y: x + y,
                "OpISub": lambda x, y: x - y,
                "OpIMul": lambda x, y: x * y,
                "OpShiftLeftLogical": lambda x, y: x << y,
                "OpShiftRightLogical": lambda x, y: x >> y,
                "OpBitwiseAnd": lambda x, y: x & y,
                "OpBitwiseOr": lambda x, y: x | y,
            }
            r = {
                funcs[op](x, y) & 0xFFFFFFFF
                for x in self.ev(a[1])
                for y in self.ev(a[2])
            }
            assert len(r) < 4096
        else:
            raise ValueError((v, op, a))
        self.cache[v] = r
        return r

    def run(self):
        starts = []
        dots = []
        for v, (op, a) in self.defs.items():
            if op == "OpSDot":
                dots.append(v)
            if op == "OpIAdd" and self.defs[a[2]][0] == "OpSDot":
                starts.append(a[1])
        seen = set()
        loads = set()
        constants = set()

        def trace(v):
            if v in seen:
                return
            seen.add(v)
            op, a = self.defs[v]
            if op == "OpSDot":
                return
            if op == "OpIAdd":
                assert sum(self.defs[c][0] == "OpSDot" for c in a[1:]) == 1, (
                    v,
                    "non-dot accumulator addition",
                )
                children = a[1:]
            elif op == "OpCompositeConstruct":
                children = a[1:]
            elif op == "OpPhi":
                children = a[1::2]
            elif op in (
                "OpCompositeExtract",
                "OpVectorExtractDynamic",
                "OpBitcast",
                "OpCopyObject",
            ):
                children = [a[1]]
            elif op == "OpExtInst" and a[2] == "SMax":
                children = a[3:]
            elif op == "OpLoad":
                rt = self.root(a[1])
                ro, ra = self.defs[rt]
                if ra[1] == "Function":
                    children = self.stores[rt]
                else:
                    assert self.descriptor_sets[rt] == 1
                    loads.add((v, rt))
                    return
            elif op == "OpConstant":
                constants.add(int(a[1]))
                return
            else:
                raise ValueError(("unbounded accumulator", v, op, a))
            for c in children:
                trace(c)

        for v in starts:
            trace(v)
        max_bias = 0
        bias_addresses = set()
        for v, rt in loads:
            ptr = self.defs[v][1][1]
            po, pa = self.defs[ptr]
            assert po == "OpAccessChain" and len(pa) == 4 and self.ev(pa[2]) == {0}
            varptr = self.defs[rt][1][0]
            descs = self.defs[varptr][1][1]
            block = self.defs[descs][1][0]
            array = self.defs[block][1][0]
            assert self.strides[array] == 16
            typ = self.defs[v][1][0]
            assert self.defs[typ] == ("OpTypeVector", ["%uint", "4"])
            for index in self.ev(pa[-1]):
                offset = index * 16
                assert offset + 16 <= len(self.weights)
                values = struct.unpack_from("<4i", self.weights, offset)
                max_bias = max(max_bias, *map(abs, values))
                bias_addresses.update(range(offset, offset + 16, 4))
        for c in constants:
            max_bias = max(max_bias, abs(c if c < 0x80000000 else c - 0x100000000))
        for h, loop in self.loops.items():
            if not any(self.where[d] in loop["region"] for d in dots):
                continue
            candidates = []
            for v, op, a in self.blocks[h]:
                if op != "OpPhi":
                    continue
                try:
                    candidates.append(self.induction(v))
                except (AssertionError, ValueError, KeyError):
                    pass
            assert len(candidates) == 1, ("unbounded dot loop", h, len(candidates))
            loop["iterations"] = candidates[0]["iterations"]
        max_dot_executions = sum(
            math.prod(
                l["iterations"]
                for l in self.loops.values()
                if self.where[d] in l["region"]
            )
            for d in dots
        )
        # A signed four-byte dot is bounded by 4 * 128 * 128. Copies, selects and SMax
        # preserve that bound; additions contribute one dot at a time.
        bound = max_bias + max_dot_executions * 65536
        assert bound < 2147483647, (
            "insufficient headroom",
            max_bias,
            max_dot_executions,
            bound,
        )
        return {
            "bias_max_abs": max_bias,
            "bias_words_checked": len(bias_addresses),
            "dot_executions_upper_bound": max_dot_executions,
            "accumulator_abs_upper_bound": bound,
            "dot_loop_iterations": [
                l["iterations"] for l in self.loops.values() if "iterations" in l
            ],
        }


def patch(text, width):
    lines = text.splitlines()
    defs = {}
    for line in lines:
        m = re.match(r"\s*(%\w+) = (\w+)\s*(.*)", line)
        if m:
            defs[m[1]] = (m[2], m[3].split())
    roots = re.findall(r"OpDecorate (%\w+) DescriptorSet 1\s*$", text, re.M)

    def scalar_payload(root):
        ptr = defs[root][1][0]
        descriptors = defs[ptr][1][1]
        struct = defs[descriptors][1][0]
        payload = defs[struct][1][0]
        element = defs[payload][1][0]
        return defs[element] == ("OpTypeInt", ["32", "0"])

    roots = [root for root in roots if scalar_payload(root)]
    if len(roots) != 1:
        raise ValueError("expected exactly one set 1 variable")
    root = roots[0]
    assert f"OpDecorate {root} NonWritable" in text
    var_ptr = defs[root][1][0]
    descriptor_array = defs[var_ptr][1][1]
    block = defs[descriptor_array][1][0]
    array = defs[block][1][0]
    uint = defs[array][1][0]
    assert defs[uint] == ("OpTypeInt", ["32", "0"])
    # This array must not also describe writable storage.
    assert sum(op == "OpTypeStruct" and array in a for op, a in defs.values()) == 1
    vec = next(
        (
            i
            for i, (op, a) in defs.items()
            if op == "OpTypeVector" and a == [uint, str(width)]
        ),
        "%weight_wide_vec",
    )
    vec_ptr = next(
        (
            i
            for i, (op, a) in defs.items()
            if op == "OpTypePointer" and a == ["StorageBuffer", vec]
        ),
        "%weight_wide_ptr",
    )
    shift = "%weight_wide_shift"
    mask = "%weight_wide_mask"
    origins = {root}
    pointers = {}
    for ident, (op, a) in defs.items():
        if op in ("OpAccessChain", "OpInBoundsAccessChain") and a[1] in origins:
            origins.add(ident)
            if defs[a[0]] == ("OpTypePointer", ["StorageBuffer", uint]):
                pointers[ident] = a
    assert pointers, "no weight scalar pointers"
    for ident in pointers:
        uses = [(op, a) for op, a in defs.values() if ident in a]
        assert all(op == "OpLoad" and a == [uint, ident] for op, a in uses), (
            ident,
            uses,
        )
    result = []
    count = 0
    for line in lines:
        line = re.sub(
            r"(OpDecorate " + re.escape(array) + r" ArrayStride) 4$",
            rf"\g<1> {width*4}",
            line,
        )
        m = re.match(r"\s*(%\w+) = (\w+)\s*(.*)", line)
        if m:
            ident, op, a = m[1], m[2], m[3].split()
            if ident in (vec, vec_ptr):
                continue
            if ident == array:
                result += [
                    f"{vec} = OpTypeVector {uint} {width}",
                    f"{vec_ptr} = OpTypePointer StorageBuffer {vec}",
                    f"{shift} = OpConstant {uint} {1 if width==2 else 2}",
                    f"{mask} = OpConstant {uint} {width-1}",
                ]
                line = f"{array} = OpTypeRuntimeArray {vec}"
            elif ident in pointers:
                index = a[-1]
                result += [
                    f"{ident}_block = OpShiftRightLogical {uint} {index} {shift}",
                    f"{ident}_lane = OpBitwiseAnd {uint} {index} {mask}",
                ]
                line = f"{ident} = {op} {vec_ptr} " + " ".join(
                    a[1:-1] + [ident + "_block"]
                )
            elif op == "OpLoad" and a[1] in pointers:
                pointer = a[1]
                result.append(f"{ident}_wide = OpLoad {vec} {pointer}")
                line = f"{ident} = OpVectorExtractDynamic {uint} {ident}_wide {pointer}_lane"
                count += 1
        result.append(line)
    assert count >= len(pointers)
    return "\n".join(result) + "\n", count


def fold_dot_acc(text):
    uses = collections.Counter(re.findall(r"%\w+", text))
    pattern = r"(?m)^\s*(%\w+) = OpSDot (%\w+) (%\w+) (%\w+) PackedVectorFormat4x8Bit\n\s*(%\w+) = OpIAdd \2 (%\w+) \1\s*$"
    count = 0

    def replace(match):
        nonlocal count
        dot, typ, a, b, result, acc = match.groups()
        if uses[dot] != 2:
            return match[0]
        count += 1
        return f"{result} = OpSDotAccSat {typ} {a} {b} {acc} PackedVectorFormat4x8Bit"

    result = re.sub(pattern, replace, text)
    assert count and count == text.count(" = OpSDot ")
    return result, count


def generate(assets, output, sdk_bin):
    import hashlib

    banks = list((assets / "weights").glob("*.bin"))
    assert banks and all(p.stat().st_size % 16 == 0 for p in banks)
    if output.exists():
        raise RuntimeError("Use a fresh Windows INT8 output directory")
    output.mkdir(parents=True)
    cache = {}
    acc_cache = {}
    proofs = []
    records = []
    tiers = ("1080", "2160", "4320") if (assets / "4320").is_dir() else ("1080", "2160")
    for tier in tiers:
        for preset in (
            "native",
            "quality",
            "balanced",
            "performance",
            "custom",
            "ultraperf",
        ):
            for index in range(2, 25, 2):
                relative = Path(tier) / preset / f"pass-{index:02}.spv"
                source = assets / relative
                digest = hashlib.sha256(source.read_bytes()).hexdigest()
                dest = output / relative.with_name(source.stem + ".win-dotacc.spv")
                dest.parent.mkdir(parents=True, exist_ok=True)
                if digest not in cache:
                    asm = subprocess.check_output(
                        [sdk_bin / "spirv-dis", source], text=True
                    )
                    patched, count = patch(asm, 4)
                    cache[digest] = (count, patched)
                else:
                    count, patched = cache[digest]
                # Prove the bound separately for each preset using its selected weights.
                weights = (assets / "weights" / f"weights_{preset}.bin").read_bytes()
                bound = Audit(patched, weights).run()
                bound.update(
                    {
                        "file": relative.as_posix(),
                        "weight_sha256": hashlib.sha256(weights).hexdigest(),
                    }
                )
                proofs.append(bound)
                acc_dest = dest
                if digest not in acc_cache:
                    accumulated, folded = fold_dot_acc(patched)
                    temp = output / "current.spvasm"
                    temp.write_text(accumulated)
                    subprocess.run(
                        [
                            sdk_bin / "spirv-as",
                            "--target-env",
                            "spv1.3",
                            temp,
                            "-o",
                            acc_dest,
                        ],
                        check=True,
                    )
                    subprocess.run(
                        [sdk_bin / "spirv-val", "--target-env", "vulkan1.3", acc_dest],
                        check=True,
                    )
                    acc_cache[digest] = (acc_dest, folded)
                else:
                    previous_acc, folded = acc_cache[digest]
                    shutil.copyfile(previous_acc, acc_dest)
                records.append(
                    {
                        "file": acc_dest.relative_to(output).as_posix(),
                        "source_sha256": digest,
                        "spirv_sha256": hashlib.sha256(
                            acc_dest.read_bytes()
                        ).hexdigest(),
                        "dot_accumulations_folded": folded,
                        "scalar_loads_rewritten": count,
                    }
                )
    (output / "dot-acc-overflow-proofs.json").write_text(
        json.dumps({"schema": 1, "dispatch_z": 1, "proofs": proofs}, indent=2) + "\n"
    )
    (output / "manifest-windows-combined.json").write_text(
        json.dumps({"schema": 1, "shaders": records}, indent=2) + "\n"
    )
    print(
        f"Generated {len(records)} Windows codegen variants ({len(cache)} unique inputs)."
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sdk-bin", type=Path, required=True)
    args = parser.parse_args()
    generate(args.assets, args.output, args.sdk_bin)
