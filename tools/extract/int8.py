# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Extract INT8 shaders, weights and dispatch metadata from SDK tables."""

from pathlib import Path
import json, hashlib
from sdk import SDK, TIERS, PRESETS, write, SDK_HASH

WEIGHTS = (0xC05C00, 0xC45C20, 0xCA5C40, 0xC85C40, 0xC65C30, 0xC25C00)


def extract(args):
    sdk = SDK(args.dll)
    output = Path(args.output)
    if output.exists():
        raise ValueError("Use a fresh extraction directory")
    output.mkdir(parents=True)
    flags = range(16) if args.all_permutations else (args.flags,)
    rows = ["tier\tpreset\tpass\tsize\trva"]
    hashes = {}
    for preset, name in enumerate(PRESETS):
        bank = sdk.read(WEIGHTS[preset], 0x20000)
        write(output / "weights" / f"weights_{name}.bin", bank)
        for tier, tier_name in enumerate(TIERS):
            for flag in flags:
                directory = (
                    output
                    / (f"permutations/{flag:02x}" if args.all_permutations else "")
                    / tier_name
                    / name
                )
                for stage in range(27):
                    record = sdk.select("int8", stage, tier, preset, flag)
                    rva, data = sdk.shader(record)
                    path = directory / f"pass-{stage:02}.dxbc"
                    write(path, data)
                    write(
                        directory / f"pass-{stage:02}.selection.bin",
                        sdk.selection(record),
                    )
                    rows.append(f"{tier_name}\t{name}\t{stage}\t{len(data)}\t{rva:#x}")
                    hashes[path.relative_to(output).as_posix()] = hashlib.sha256(
                        data
                    ).hexdigest()
    (output / "selection.tsv").write_text("\n".join(rows) + "\n")
    (output / "manifest.json").write_text(
        json.dumps(dict(source_sha256=SDK_HASH, shaders=hashes), indent=2) + "\n"
    )
