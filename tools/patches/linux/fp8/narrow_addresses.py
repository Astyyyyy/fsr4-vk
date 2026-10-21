#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Narrow FP8 address calculations for RADV.

Select rewrites by logical pass; default to the 2160 tier. WorkGroupId.x must
fit in 16 bits. Validate SPIR-V before installing the outputs.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile

# Measured transforms by logical pass.

POLICY = {'pass0.spv': 'global-u16',
 'pass10.spv': 'shl8-u16',
 'pass11.spv': 'global-u16',
 'pass12.spv': 'global-u16',
 'pass4.spv': 'shl8-u16',
 'pass5.spv': 'shl8-extract16',
 'pass6.spv': 'global-extract16',
 'pass7.spv': 'shl8-u16',
 'pass8.spv': 'global-extract16',
 'pass9.spv': 'shl8-u16',
 'postpass.spv': 'global-extract16',
 'postpass_wmma.spv': 'global-u16'}

VALID_MODES = {
    "global-u16",
    "global-extract16",
    "shl8-u16",
    "shl8-extract16",
}

SPIRV_MAGIC = b"\x03\x02\x23\x07"

def sha256(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

def run(cmd):
    subprocess.run(
        [str(x) for x in cmd],
        check=True,
    )

def require_command(name: str):
    if shutil.which(name) is None:
        raise SystemExit(
            f"required command not found: {name}"
        )

def transform_shader(
    source: Path,
    destination: Path,
    mode: str,
):
    if mode not in VALID_MODES:
        raise RuntimeError(
            f"unknown mode: {mode}"
        )

    if source.read_bytes()[:4] != SPIRV_MAGIC:
        raise RuntimeError(
            f"not SPIR-V: {source}"
        )

    with tempfile.TemporaryDirectory(
        prefix="fsr4-address-"
    ) as td_raw:

        td = Path(td_raw)

        asm = td / "input.spvasm"
        patched_asm = td / "patched.spvasm"
        output_spv = td / "patched.spv"

        run(
            [
                "spirv-dis",
                "--raw-id",
                source,
                "-o",
                asm,
            ]
        )

        lines = asm.read_text(
            errors="replace"
        ).splitlines()

        # Refuse already-patched input.

        if any(
            "%fsr4patch_" in line
            for line in lines
        ):
            raise RuntimeError(
                f"{source.name}: input already contains fsr4patch IDs"
            )

        version = None

        for line in lines:
            m = re.match(
                r"^; Version:\s+([0-9]+\.[0-9]+)",
                line,
            )

            if m:
                version = m.group(1)
                break

        if version is None:
            raise RuntimeError(
                f"{source.name}: unable to determine SPIR-V version"
            )

        int_types = {}
        constants = {}
        wg_vars = set()
        defs = {}
        ops = {}

        result_re = re.compile(
            r"^\s*(%\S+)\s*=\s*(Op\S+)(?:\s+(.*))?$"
        )

        for line in lines:

            m = re.match(
                r"^\s*(%\S+)\s*=\s*"
                r"OpTypeInt\s+(\d+)\s+(\d+)\s*$",
                line,
            )

            if m:
                int_types[m.group(1)] = (
                    int(m.group(2)),
                    int(m.group(3)),
                )

            m = re.match(
                r"^\s*(%\S+)\s*=\s*"
                r"OpConstant\s+(%\S+)\s+(\S+)\s*$",
                line,
            )

            if m:
                try:
                    value = int(
                        m.group(3),
                        0,
                    )
                except ValueError:
                    value = None

                if value is not None:
                    constants[m.group(1)] = (
                        m.group(2),
                        value,
                    )

            m = re.match(
                r"^\s*OpDecorate\s+(%\S+)\s+"
                r"BuiltIn\s+WorkgroupId\s*$",
                line,
            )

            if m:
                wg_vars.add(
                    m.group(1)
                )

            m = result_re.match(line)

            if m:
                rid = m.group(1)
                op = m.group(2)
                rest = m.group(3) or ""

                defs[rid] = re.findall(
                    r"%\S+",
                    rest,
                )

                ops[rid] = op

        if not wg_vars:
            raise RuntimeError(
                f"{source.name}: no BuiltIn WorkgroupId"
            )

        def const_value(token):
            item = constants.get(token)

            return (
                item[1]
                if item is not None
                else None
            )

        def add_capability(name):
            nonlocal lines

            pattern = re.compile(
                rf"^\s*OpCapability\s+"
                rf"{re.escape(name)}\s*$"
            )

            if any(
                pattern.match(x)
                for x in lines
            ):
                return

            indexes = [
                i
                for i, x in enumerate(lines)
                if re.match(
                    r"^\s*OpCapability\b",
                    x,
                )
            ]

            if not indexes:
                raise RuntimeError(
                    f"{source.name}: capability section missing"
                )

            lines.insert(
                indexes[-1] + 1,
                f"OpCapability {name}",
            )

        def ensure_type_int(
            width,
            signed,
            name,
        ):
            nonlocal lines

            for ty, (w, s) in int_types.items():
                if (
                    w == width
                    and
                    s == signed
                ):
                    return ty

            first_type = next(
                (
                    i
                    for i, x in enumerate(lines)
                    if re.match(
                        r"^\s*%\S+\s*=\s*OpType",
                        x,
                    )
                ),
                None,
            )

            if first_type is None:
                raise RuntimeError(
                    f"{source.name}: type section missing"
                )

            lines.insert(
                first_type,
                f"{name} = OpTypeInt "
                f"{width} {signed}",
            )

            int_types[name] = (
                width,
                signed,
            )

            return name

        generated_constants = {}

        def ensure_constant(
            ty,
            value,
            tag,
        ):
            nonlocal lines

            for cid, (cty, cval) in constants.items():
                if (
                    cty == ty
                    and
                    cval == value
                ):
                    return cid

            key = (
                ty,
                value,
            )

            if key in generated_constants:
                return generated_constants[key]

            cid = (
                f"%fsr4patch_{tag}_"
                f"{len(generated_constants)}"
            )

            type_index = next(
                (
                    i
                    for i, x in enumerate(lines)
                    if re.match(
                        rf"^\s*{re.escape(ty)}\s*="
                        r"\s*OpTypeInt\b",
                        x,
                    )
                ),
                None,
            )

            if type_index is None:
                raise RuntimeError(
                    f"{source.name}: type not found: {ty}"
                )

            lines.insert(
                type_index + 1,
                f"{cid} = OpConstant "
                f"{ty} {value}",
            )

            constants[cid] = (
                ty,
                value,
            )

            generated_constants[key] = cid

            return cid

        # Find pointers to WorkGroupId.x.

        pointer_state = {
            var: "wgvec"
            for var in wg_vars
        }

        copy_re = re.compile(
            r"^\s*(%\S+)\s*=\s*"
            r"OpCopyObject\s+(%\S+)\s+(%\S+)\s*$"
        )

        access_re = re.compile(
            r"^\s*(%\S+)\s*=\s*"
            r"(OpAccessChain|"
            r"OpInBoundsAccessChain|"
            r"OpPtrAccessChain|"
            r"OpInBoundsPtrAccessChain)\s+"
            r"(%\S+)\s+(%\S+)"
            r"(?:\s+(.*))?$"
        )

        changed = True

        while changed:
            changed = False

            for line in lines:

                m = copy_re.match(line)

                if m:
                    result = m.group(1)
                    source_id = m.group(3)

                    if (
                        source_id in pointer_state
                        and
                        result not in pointer_state
                    ):
                        pointer_state[result] = (
                            pointer_state[source_id]
                        )

                        changed = True

                    continue

                m = access_re.match(line)

                if not m:
                    continue

                result = m.group(1)
                base = m.group(4)

                if (
                    base not in pointer_state
                    or
                    result in pointer_state
                ):
                    continue

                indices = (
                    m.group(5).split()
                    if m.group(5)
                    else []
                )

                if not indices:
                    continue

                if (
                    pointer_state[base] == "wgvec"
                    and
                    const_value(indices[0]) == 0
                ):
                    pointer_state[result] = "wgx"
                    changed = True

        # Find scalar loads of WorkGroupId.x.

        load_re = re.compile(
            r"^(\s*)"
            r"(%\S+)"
            r"\s*=\s*"
            r"OpLoad\s+"
            r"(%\S+)\s+"
            r"(%\S+)"
            r"(.*)$"
        )

        wg_seeds = set()
        wg_load_types = {}

        for line in lines:
            m = load_re.match(line)

            if not m:
                continue

            result = m.group(2)
            ty = m.group(3)
            ptr = m.group(4)

            if pointer_state.get(ptr) != "wgx":
                continue

            if ty not in int_types:
                continue

            width, signed = int_types[ty]

            if (
                width == 32
                and
                signed == 0
            ):
                wg_seeds.add(result)
                wg_load_types[result] = ty

        if not wg_seeds:
            raise RuntimeError(
                f"{source.name}: no scalar u32 WorkGroupId.x load"
            )

        # Track values derived from WorkGroupId.x.

        tainted = set(
            wg_seeds
        )

        changed = True

        while changed:
            changed = False

            for rid, operands in defs.items():

                if rid in tainted:
                    continue

                if set(operands) & tainted:
                    tainted.add(rid)
                    changed = True

        # Find values used to compute addresses.

        address_roots = set()

        for line in lines:

            m = access_re.match(line)

            if not m:
                continue

            indices = (
                m.group(5).split()
                if m.group(5)
                else []
            )

            for token in indices:

                if token in tainted:
                    address_roots.add(
                        token
                    )

        for rid, op in ops.items():

            if op != "OpConvertUToPtr":
                continue

            operands = defs.get(
                rid,
                [],
            )

            for token in operands[1:]:

                if token in tainted:
                    address_roots.add(
                        token
                    )

        if not address_roots:
            raise RuntimeError(
                f"{source.name}: WorkGroupId.x has no address roots"
            )

        # Trace address expressions back to their inputs.

        address_slice = set()

        def walk(node):

            if node in address_slice:
                return

            if node not in tainted:
                return

            address_slice.add(
                node
            )

            for src in defs.get(
                node,
                [],
            ):
                if src in tainted:
                    walk(src)

        for root in address_roots:
            walk(root)

        # Find address expressions that shift WorkGroupId.x left by 8 bits.

        shift_re = re.compile(
            r"^(\s*)"
            r"(%\S+)"
            r"\s*=\s*"
            r"OpShiftLeftLogical\s+"
            r"(%\S+)\s+"
            r"(%\S+)\s+"
            r"(%\S+)"
            r"(.*)$"
        )

        shl8_candidates = []

        for line in lines:

            m = shift_re.match(line)

            if not m:
                continue

            result = m.group(2)
            ty = m.group(3)
            base = m.group(4)
            shift = m.group(5)

            if result not in address_slice:
                continue

            if base not in wg_seeds:
                continue

            if const_value(shift) != 8:
                continue

            if ty not in int_types:
                continue

            width, signed = int_types[ty]

            if (
                width != 32
                or
                signed != 0
            ):
                continue

            shl8_candidates.append(
                result
            )


        if mode.startswith("shl8-"):

            if not shl8_candidates:
                raise RuntimeError(
                    f"{source.name}: selected {mode}, "
                    "but no address-slice WGID.x << 8 exists"
                )

        patched = 0


        if mode in (
            "global-u16",
            "global-extract16",
        ):

            if mode == "global-u16":

                add_capability(
                    "Int16"
                )

                u16_type = ensure_type_int(
                    16,
                    0,
                    "%fsr4patch_u16",
                )

            extract_constants = {}

            if mode == "global-extract16":

                for seed in wg_seeds:

                    ty = wg_load_types[
                        seed
                    ]

                    if ty not in extract_constants:

                        extract_constants[
                            ty
                        ] = (
                            ensure_constant(
                                ty,
                                0,
                                "extract_offset",
                            ),
                            ensure_constant(
                                ty,
                                16,
                                "extract_width",
                            ),
                        )

            out = []

            for line in lines:

                m = load_re.match(
                    line
                )

                if (
                    not m
                    or
                    m.group(2)
                    not in wg_seeds
                ):
                    out.append(
                        line
                    )

                    continue

                indent = m.group(1)
                result = m.group(2)
                ty = m.group(3)
                ptr = m.group(4)
                tail = m.group(5)

                raw = (
                    f"%fsr4patch_raw_"
                    f"{patched}"
                )

                out.append(
                    f"{indent}{raw} = "
                    f"OpLoad {ty} "
                    f"{ptr}{tail}"
                )

                if mode == "global-u16":

                    narrow = (
                        f"%fsr4patch_u16_"
                        f"{patched}"
                    )

                    out.append(
                        f"{indent}{narrow} = "
                        f"OpUConvert "
                        f"{u16_type} {raw}"
                    )

                    out.append(
                        f"{indent}{result} = "
                        f"OpUConvert "
                        f"{ty} {narrow}"
                    )

                else:

                    zero, sixteen = (
                        extract_constants[
                            ty
                        ]
                    )

                    out.append(
                        f"{indent}{result} = "
                        f"OpBitFieldUExtract "
                        f"{ty} {raw} "
                        f"{zero} {sixteen}"
                    )

                patched += 1

            lines = out


        else:

            candidate_set = set(
                shl8_candidates
            )

            if mode == "shl8-u16":

                add_capability(
                    "Int16"
                )

                u16_type = ensure_type_int(
                    16,
                    0,
                    "%fsr4patch_u16",
                )

            extract_constants = {}

            if mode == "shl8-extract16":

                for line in lines:

                    m = shift_re.match(
                        line
                    )

                    if (
                        not m
                        or
                        m.group(2)
                        not in candidate_set
                    ):
                        continue

                    ty = m.group(3)

                    if ty not in extract_constants:

                        extract_constants[
                            ty
                        ] = (
                            ensure_constant(
                                ty,
                                0,
                                "shl8_offset",
                            ),
                            ensure_constant(
                                ty,
                                16,
                                "shl8_width",
                            ),
                        )

            out = []

            for line in lines:

                m = shift_re.match(
                    line
                )

                if (
                    not m
                    or
                    m.group(2)
                    not in candidate_set
                ):
                    out.append(
                        line
                    )

                    continue

                indent = m.group(1)
                result = m.group(2)
                ty = m.group(3)
                base = m.group(4)
                shift = m.group(5)
                tail = m.group(6)

                if mode == "shl8-u16":

                    narrow = (
                        f"%fsr4patch_shl8_u16_"
                        f"{patched}"
                    )

                    wide = (
                        f"%fsr4patch_shl8_u32_"
                        f"{patched}"
                    )

                    out.append(
                        f"{indent}{narrow} = "
                        f"OpUConvert "
                        f"{u16_type} {base}"
                    )

                    out.append(
                        f"{indent}{wide} = "
                        f"OpUConvert "
                        f"{ty} {narrow}"
                    )

                    replacement = wide

                else:

                    extract = (
                        f"%fsr4patch_shl8_extract_"
                        f"{patched}"
                    )

                    zero, sixteen = (
                        extract_constants[
                            ty
                        ]
                    )

                    out.append(
                        f"{indent}{extract} = "
                        f"OpBitFieldUExtract "
                        f"{ty} {base} "
                        f"{zero} {sixteen}"
                    )

                    replacement = extract

                out.append(
                    f"{indent}{result} = "
                    f"OpShiftLeftLogical "
                    f"{ty} {replacement} "
                    f"{shift}{tail}"
                )

                patched += 1

            lines = out

        if patched == 0:
            raise RuntimeError(
                f"{source.name}: transformation produced zero patches"
            )

        patched_asm.write_text(
            "\n".join(
                lines
            )
            + "\n"
        )

        run(
            [
                "spirv-as",
                "--target-env",
                f"spv{version}",
                patched_asm,
                "-o",
                output_spv,
            ]
        )

        run(
            [
                "spirv-val",
                "--target-env",
                "vulkan1.3",
                output_spv,
            ]
        )

        destination.write_bytes(
            output_spv.read_bytes()
        )

        return {
            "mode": mode,
            "wgid_x_loads": len(
                wg_seeds
            ),
            "address_roots": len(
                address_roots
            ),
            "address_slice_values": len(
                address_slice
            ),
            "shl8_candidates": len(
                shl8_candidates
            ),
            "patches": patched,
            "spirv_version": version,
        }


def main():

    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "fp8_root",
        type=Path,
        nargs="?",
        default=Path(
            "assets/fp8"
        ),
        help=(
            "FP8 asset root containing resolution tiers "
            "(default: assets/fp8)"
        ),
    )

    parser.add_argument(
        "--tier",
        action="append",
        dest="tiers",
        help=(
            "tier to optimize; repeat for multiple tiers. "
            "Default: 2160"
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "transform and validate everything "
            "without modifying live shaders"
        ),
    )

    parser.add_argument(
        "--report",
        type=Path,
        help="JSON report path",
    )

    parser.add_argument(
        "--omit-unused-pass0", action="store_true",
        help="skip the legacy SDK pass0 artifact, which the DLL extractor does not produce",
    )
    args = parser.parse_args()

    for command in (
        "spirv-dis",
        "spirv-as",
        "spirv-val",
    ):
        require_command(
            command
        )

    root = args.fp8_root.resolve()

    if not root.is_dir():
        raise SystemExit(
            f"FP8 root does not exist: {root}"
        )

    tiers = (
        args.tiers
        if args.tiers
        else ["2160"]
    )

    report = {
        "schema": 1,
        "optimizer": (
            "structural-fp8-addressing"
        ),
        "policy_kind": (
            "logical-pass-name-plus-"
            "structural-preconditions"
        ),
        "hash_gated": False,
        "id_gated": False,
        "semantic_contract": (
            "relevant WorkGroupId.x "
            "requires no bits >= 16"
        ),
        "dry_run": args.dry_run,
        "tiers": {},
    }

    staged = []

    # Validate every output before replacing any files.

    with tempfile.TemporaryDirectory(
        prefix="fsr4-fp8-address-bundle-",
        dir=root.parent,
    ) as stage_raw:

        stage_root = Path(
            stage_raw
        )

        for tier in tiers:

            tier_dir = (
                root
                / tier
            )

            if not tier_dir.is_dir():
                raise SystemExit(
                    f"tier does not exist: "
                    f"{tier_dir}"
                )

            tier_report = {
                "shaders": [],
            }

            report[
                "tiers"
            ][tier] = tier_report

            for relative, mode in POLICY.items():

                if relative == "pass0.spv" and args.omit_unused_pass0:
                    continue

                source = (
                    tier_dir
                    / relative
                )

                if not source.is_file():
                    raise SystemExit(
                        f"{tier}: policy shader "
                        f"missing: {relative}"
                    )

                stage = (
                    stage_root
                    / tier
                    / relative
                )

                stage.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                before = sha256(
                    source
                )

                try:

                    structural = transform_shader(
                        source,
                        stage,
                        mode,
                    )

                except Exception as exc:

                    raise SystemExit(
                        "\n"
                        "STRUCTURAL POLICY MISMATCH\n"
                        "==========================\n"
                        f"tier:   {tier}\n"
                        f"shader: {relative}\n"
                        f"mode:   {mode}\n"
                        f"reason: {exc}\n\n"
                        "No shaders have been committed.\n"
                        "Treat this as a new FSR4 shader "
                        "revision requiring review."
                    )

                after = sha256(
                    stage
                )

                if before == after:
                    raise SystemExit(
                        f"{tier}/{relative}: "
                        "patch produced identical binary"
                    )

                entry = {
                    "shader": relative,
                    "before_sha256": before,
                    "after_sha256": after,
                    **structural,
                }

                tier_report[
                    "shaders"
                ].append(
                    entry
                )

                staged.append(
                    (
                        source,
                        stage,
                    )
                )

                print(
                    f"[OK] "
                    f"{tier}/{relative:<24} "
                    f"{mode:<20} "
                    f"patches="
                    f"{structural['patches']} "
                    f"roots="
                    f"{structural['address_roots']} "
                    f"shl8="
                    f"{structural['shl8_candidates']}"
                )


        if args.dry_run:

            print()
            print(
                "[DRY-RUN] all shaders "
                "structurally matched and validated; "
                "nothing committed"
            )

        else:

            for destination, source in staged:

                os.replace(
                    source,
                    destination,
                )

            print()
            print(
                f"[OK] committed "
                f"{len(staged)} validated "
                "shader transformations"
            )

    report_path = (
        args.report
        if args.report
        else
        root
        / "fp8-address-optimization.json"
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        )
        + "\n"
    )

    print(
        f"[OK] report: "
        f"{report_path}"
    )

if __name__ == "__main__":
    main()
