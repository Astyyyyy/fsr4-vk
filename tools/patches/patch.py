# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Apply shader patches and generate optimized RCAS."""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from collections import defaultdict

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "assets"))
sys.path.insert(0, str(HERE.parent / "convert"))
import workflow
from workflow import ROOT
from toolchain import find_tool, patching_tools

class IR:
    def __init__(self, text):
        self.lines = text.splitlines()
        self.d = {}
        self.location = {}
        self.blocks = {}
        self.serial = 0
        self.globals = []
        self.annotations = []
        self.report = {}
        block = None
        for n, line in enumerate(self.lines):
            m = re.match(r"\s*(%\w+) = (Op\w+)(?: (.*))?$", line)
            if not m:
                if block:
                    self.blocks[block].append(n)
                continue
            ident, op, args = m.groups()
            args = (args or "").split()
            self.d[ident] = (op, args)
            if op == "OpFunction":
                if self.blocks:
                    raise RuntimeError("Requires a single inlined function")
                block = None
            elif op == "OpLabel":
                block = ident
                self.blocks[block] = []
            self.location[ident] = (block, n)
            if block:
                self.blocks[block].append(n)
        if not self.blocks:
            raise RuntimeError("No function blocks")
        self.entry = next(iter(self.blocks))
        self.decorations = defaultdict(list)
        for line in self.lines:
            m = re.match(r"\s*OpDecorate (%\w+) (.*)", line)
            if m:
                self.decorations[m[1]].append(m[2])

    def fresh(self):
        while True:
            self.serial += 1
            ident = f"%redundancy{self.serial}"
            if ident not in self.d:
                self.d[ident] = ("", [])
                return ident

    def type(self, op, args):
        for ident, definition in self.d.items():
            if definition == (op, args):
                return ident
        ident = self.fresh()
        self.d[ident] = op, args
        self.globals.append(f"{ident} = {op} " + " ".join(args))
        return ident

    def constant(self, ty, value):
        args = [ty, str(value)]
        for ident, definition in self.d.items():
            if definition == ("OpConstant", args):
                return ident
        ident = self.fresh()
        self.d[ident] = "OpConstant", args
        self.globals.append(f"{ident} = OpConstant {ty} {value}")
        return ident

    def value(self, ident):
        op, args = self.d.get(ident, ("", []))
        if op == "OpConstant" and self.d[args[0]][0] == "OpTypeInt":
            return int(args[1], 0)
        if op in ("OpCopyObject", "OpBitcast"):
            return self.value(args[1])
        return None

    def pointer_storage(self, ident):
        op, args = self.d.get(ident, ("", []))
        if not args:
            return None
        ty_op, ty_args = self.d.get(args[0], ("", []))
        return ty_args[0] if ty_op == "OpTypePointer" else None

    def finish(self):
        lines = self.lines[:]
        # SPIR-V 1.4+ requires resource globals in the entry-point interface.
        globals_used = [
            i
            for i, (op, args) in self.d.items()
            if op == "OpVariable"
            and args[1] != "Function"
            and self.location.get(i, (None,))[0] is None
        ]
        for n, line in enumerate(lines):
            if "OpEntryPoint " in line:
                missing = [i for i in globals_used if i not in line.split()]
                lines[n] = line + " " + " ".join(missing)
        if self.annotations:
            first_type = next(n for n, l in enumerate(lines) if "= OpType" in l)
            lines[first_type:first_type] = self.annotations
        if self.globals:
            first_function = next(
                n for n, l in enumerate(lines) if "= OpFunction " in l
            )
            lines[first_function:first_function] = self.globals
        return "\n".join(lines) + "\n"

def converted_inputs(assets):
    if not (assets / "manifest.json").is_file():
        raise ValueError("Run convert.py successfully before patch.py")
    manifest = json.loads((assets / "manifest.json").read_text())
    expected = {row["file"]: row["sha256"] for row in manifest["payloads"] if workflow.baseline(row)}
    for name, index in manifest["aliases"].items():
        if "/stock/" in name:
            expected[name] = manifest["payloads"][index]["sha256"]
    inputs = {}
    for name, digest in expected.items():
        path = workflow.asset_path(assets, name)
        if not path.is_file():
            raise ValueError("Converted asset is missing; run convert.py: " + name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Converted asset changed: " + name)
        inputs.setdefault(digest, path)
    return inputs

def int8_inputs(reference, generated, output):
    manifest = json.loads((reference / "manifest.json").read_text())
    for alias, index in manifest["aliases"].items():
        parts = Path(alias).parts
        if (parts[0] != "int8" or parts[1] in ("models", "shaders")
                or "optimized" in parts or ".win-" in parts[-1]):
            continue
        digest = manifest["payloads"][index]["sha256"]
        if digest not in generated:
            raise ValueError("INT8 input was not regenerated: " + alias)
        destination = output.joinpath(*parts[1:])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(generated[digest], destination)

def shipping_patches(assets, generated, output, tools):
    manifest = json.loads((assets / "manifest.json").read_text())
    exposure = workflow.load("workflow_exposure", HERE / "shared/fp8/exposure_history.py")
    motion = workflow.load("workflow_motion", HERE / "shared/fp8/display_motion_vectors.py")
    for tier in ("1080", "2160", "4320"):
        alias = f"fp8/shaders/4.1.1/stock/{tier}/prepass_wmma.spv"
        row = manifest["payloads"][manifest["aliases"][alias]]
        source = generated[row["sha256"]]
        fixed = output / "fp8" / tier / "prepass_wmma.spv"
        exposure.patch_exposure(source, fixed)
        generated.update(workflow.generated_index(fixed.parent))
        motion.patch_motion(fixed, fixed.with_name("prepass_wmma.hrmv.spv"))
    windows_fp8 = workflow.load("workflow_windows_fp8", HERE / "windows/fp8/generate_variants.py")
    windows_fp8.generate(assets, output / "windows-fp8", generated=generated)
    inputs = output / "work/int8-inputs"
    int8_inputs(assets, generated, inputs)
    windows_int8 = workflow.load("workflow_windows_int8", HERE / "windows/int8/loads_and_dots.py")
    variants = windows_int8.generate(inputs, output / "windows-int8", tools)
    exact_int8 = workflow.load("workflow_exact_int8", HERE / "shared/int8/arithmetic_rewrites.py")
    exact_int8.generate(inputs, output / "int8-exact", tools, variants)

def patch(args):
    generated = converted_inputs(args.assets)
    extraction = args.input or args.assets / "extracted"
    workflow.validate_extraction(extraction)
    tools = patching_tools(args.spirv_tools)
    os.environ["FSR4_SPIRV_TOOLS"] = str(tools)
    os.environ["PATH"] = str(tools) + os.pathsep + os.environ.get("PATH", "")
    translator = find_tool("dxil-spirv", args.translator, "DXIL_SPIRV")
    (ROOT / "build").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="patch-", dir=ROOT / "build") as directory:
        temporary = Path(directory)
        patched = temporary / "patched"
        shipping_patches(args.assets, generated, patched, tools)
        raw_rcas = temporary / "rcas-raw"
        workflow.run(HERE.parent / "convert/rcas.py", extraction / "rcas", raw_rcas,
                     "--translator", translator)
        rcas = workflow.load("workflow_rcas", HERE / "shared/rcas/tile_4x64.py")
        for source in sorted(raw_rcas.glob("*.spv")):
            rcas.patch(source, patched / "rcas" / source.name)
        # Build the pack from verified conversion outputs, not previous patched files.
        stock = temporary / "stock"
        stock.mkdir()
        for digest, source in generated.items():
            shutil.copyfile(source, stock / (digest + source.suffix))
        reference = temporary / "reference"
        reference.mkdir()
        shutil.copyfile(args.assets / "manifest.json", reference / "manifest.json")
        sys.path.insert(0, str(ROOT / "tools/assets"))
        packer = workflow.load("workflow_asset_packer", ROOT / "tools/assets/manage.py")
        pack = temporary / "pack"
        packer.pack(SimpleNamespace(
            reference=reference, output=pack, generated=[stock, patched],
            native_only=False, omit_legacy_unused=True,
        ))
        workflow.run(ROOT / "tools/assets/manage.py", "verify", pack)
        files = {
            path.relative_to(pack).as_posix(): path
            for path in pack.rglob("*")
            if path.is_file() and path.suffix in (".spv", ".bin", ".delta")
        }
        workflow.install(files, args.assets)
        workflow.run(ROOT / "tools/assets/manage.py", "verify", args.assets)
    print("Patched assets:", args.assets.resolve())

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=ROOT / "assets")
    parser.add_argument("--input", type=Path, help="extraction directory (defaults to ASSETS/extracted)")
    parser.add_argument("--translator", type=Path)
    parser.add_argument("--spirv-tools", type=Path)
    args = parser.parse_args()
    try:
        patch(args)
    except (ValueError, RuntimeError, OSError, KeyError, subprocess.CalledProcessError) as error:
        parser.error(str(error))

if __name__ == "__main__":
    main()
