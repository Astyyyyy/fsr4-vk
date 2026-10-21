#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Convert extracted FSR 4.1.1 INT8 shaders to SPIR-V, reusing identical shaders."""

import argparse
import errno
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

def link_or_copy(source, destination):
    try:
        destination.hardlink_to(source)
    except OSError as error:
        if error.errno not in (errno.EXDEV, errno.EPERM, errno.EACCES,
                               errno.ENOTSUP, errno.EMLINK, errno.ENOSYS):
            raise
        shutil.copyfile(source, destination)

def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

def main() -> None:
    from toolchain import find_tool

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("assets/extracted/int8"),
        help="extracted INT8 shader directory (default: %(default)s)",
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path("build/converted-int8"),
        help="SPIR-V output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--dxil-spirv",
        type=Path,
        help="dxil-spirv executable (defaults to the bundled submodule)",
    )
    parser.add_argument("--spirv-val", help="SPIR-V validator (auto-detected by default)")
    parser.add_argument(
        "--auto-exposure",
        type=Path,
        help="validated shared SPD SPIR-V (not produced by this DXBC extractor)",
    )
    args = parser.parse_args()

    if args.output.exists():
        parser.error("use a fresh conversion output directory")

    inputs = sorted(args.input.rglob("pass-*.dxbc"))
    if not inputs:
        parser.error(f"no extracted shaders found below {args.input}")
    try:
        args.dxil_spirv = find_tool("dxil-spirv", args.dxil_spirv, "DXIL_SPIRV")
        args.spirv_val = find_tool("spirv-val", args.spirv_val)
    except ValueError as error:
        parser.error(str(error))

    exposure = args.auto_exposure or args.input / "auto-exposure.spv"
    if not exposure.is_file():
        parser.error(
            "provide --auto-exposure with the freshly converted shared SPD shader"
        )
    subprocess.run([args.spirv_val, str(exposure)], check=True)

    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / ".unique"
    cache.mkdir(exist_ok=True)
    translated: dict[str, Path] = {}
    records = []

    for number, source in enumerate(inputs, 1):
        dxbc_hash = digest(source)
        cached = translated.get(dxbc_hash)
        if cached is None:
            cached = cache / f"{dxbc_hash}.spv"
            subprocess.run(
                [
                    str(args.dxil_spirv),
                    str(source),
                    "--output",
                    str(cached),
                    "--bindless",
                    "--ssbo-uav",
                    "--ssbo-srv",
                    "--enable-shader-i8-dot",
                    "--assume-ssbo-32bit-wrapping",
                ],
                check=True,
            )
            subprocess.run([args.spirv_val, str(cached)], check=True)
            translated[dxbc_hash] = cached

        relative = source.relative_to(args.input).with_suffix(".spv")
        destination = args.output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        link_or_copy(cached, destination)
        records.append(
            {
                "file": relative.as_posix(),
                "dxbc_sha256": dxbc_hash,
                "spirv_sha256": digest(cached),
                "spirv_size": cached.stat().st_size,
            }
        )
        if number % 50 == 0:
            print(
                f"processed {number}/{len(inputs)} selections "
                f"({len(translated)} unique)"
            )

    weights = {}
    weights_output = args.output / "weights"
    for source in sorted((args.input / "weights").glob("*.bin")):
        weights_output.mkdir(exist_ok=True)
        destination = weights_output / source.name
        shutil.copyfile(source, destination)
        weights[source.stem.removeprefix("weights_")] = {
            "file": destination.relative_to(args.output).as_posix(),
            "sha256": digest(destination),
            "size": destination.stat().st_size,
        }

    auto_exposure_path = args.output / "auto-exposure.spv"
    if exposure.resolve() != auto_exposure_path.resolve():
        shutil.copyfile(exposure, auto_exposure_path)
    auto_exposure = {
        "file": auto_exposure_path.relative_to(args.output).as_posix(),
        "spirv_sha256": digest(auto_exposure_path),
        "spirv_size": auto_exposure_path.stat().st_size,
    }

    manifest = {
        "schema": 1,
        "source": str(args.input),
        "format": "FSR 4.1.1 INT8",
        "translator_flags": [
            "--bindless",
            "--ssbo-uav",
            "--ssbo-srv",
            "--enable-shader-i8-dot",
            "--assume-ssbo-32bit-wrapping",
        ],
        "selection_count": len(records),
        "unique_shader_count": len(translated),
        "shaders": records,
        "auto_exposure": auto_exposure,
        "weights": weights,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.rmtree(cache)
    print(
        f"converted and validated {len(records)} selections "
        f"({len(translated)} unique shaders) into {args.output}"
    )

if __name__ == "__main__":
    main()
