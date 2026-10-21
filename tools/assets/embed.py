#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Verify assets and generate embedded lookups for Linux and Windows."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from catalog import shipping_aliases, model_files
from catalog import validate_stock_layout, select_pack
from storage import compress_shader
import storage as deltas

def generate(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("assets", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--sidecar-source", type=Path)
    p.add_argument("--native-only", action="store_true")
    p.add_argument("--include-8k", action="store_true")
    p.add_argument(
        "--elf",
        action="store_true",
        help="emit a read-only ELF asset blob and assembly wrapper",
    )
    p.add_argument(
        "--compress-shaders",
        action="store_true",
        help="compress embedded shader payloads independently",
    )
    a = p.parse_args(argv)
    a.output.mkdir(parents=True, exist_ok=True)
    data = json.loads((a.assets / "manifest.json").read_text())
    if data.get("schema") != 1 or data.get("encoding") != "uncompressed-copy-insert-v1":
        raise RuntimeError("Unsupported asset manifest")
    maximum_depth = data.get("max_delta_depth", 1)
    if not 1 <= maximum_depth <= 8:
        raise RuntimeError("Unsupported delta chain limit")
    blobs = []
    targets = []
    rows = []
    resources = []
    depths = []
    storage = []
    for expected, r in enumerate(data["payloads"]):
        if r["id"] != expected:
            raise RuntimeError("Non-contiguous asset IDs")
        relative = Path(r["file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("Unsafe asset path")
        path = (a.assets / relative).resolve()
        blob = path.read_bytes()
        if (
            len(blob) != r["stored_size"]
            or hashlib.sha256(blob).hexdigest() != r["stored_sha256"]
        ):
            raise RuntimeError("Stored asset changed: " + str(path))
        base = r["base"]
        if r["kind"] == "delta":
            if (
                base is None
                or not 0 <= base < expected
                or depths[base] >= maximum_depth
            ):
                raise RuntimeError("Invalid forward or excessive-depth delta base")
            target = deltas.reconstruct(targets[base], blob)
        elif r["kind"] == "full" and base is None:
            target = blob
        else:
            raise RuntimeError("Invalid asset encoding")
        if (
            len(target) != r["size"]
            or hashlib.sha256(target).hexdigest() != r["sha256"]
        ):
            raise RuntimeError("Reconstructed asset changed: " + str(path))
        targets.append(target)
        depths.append(0 if base is None else depths[base] + 1)
        packed, encoding = (
            compress_shader(blob)
            if a.compress_shaders and "shaders" in relative.parts
            else (blob, "none")
        )
        blobs.append(packed)
        if encoding == "zstd":
            path = (a.output / "compressed-assets" / f"{expected}.zst").resolve()
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(packed)
        storage.append(
            dict(
                id=expected,
                packed_size=len(packed),
                compression=encoding,
                packed_sha256=hashlib.sha256(packed).hexdigest(),
            )
        )
        rows.append(
            "    {"
            + f'{expected+100}, {len(packed)}, {r["stored_size"]}, {r["size"]}, {str(encoding == "zstd").lower()}, '
            + f"{base if base is not None else -1}, "
            + ", ".join(json.dumps(r[k]) for k in ("file", "stored_sha256", "sha256"))
            + "},"
        )
        resources.append(f"{expected+100} RCDATA {json.dumps(str(path))}")
    validate_stock_layout(data, targets, a.assets)
    for name, row in model_files(data).items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("Unsafe model path")
        if (a.assets / relative).read_bytes() != targets[row["payload"]]:
            raise RuntimeError("Named model file changed: " + name)
    aliases = []
    for key, index in sorted(data["aliases"].items()):
        if not 0 <= index < len(targets):
            raise RuntimeError("Invalid asset alias")
        aliases.append("    {" + json.dumps(key) + ", " + str(index) + "},")
    secondary = (
        "inline constexpr auto &sidecar_payload_index = payload_index;\n"
        + "inline constexpr auto &sidecar_asset_aliases = asset_aliases;\n"
        + "inline constexpr unsigned sidecar_max_delta_depth = max_delta_depth;\n"
    )
    if a.sidecar_source:
        # Sidecars retain the source catalog encoding; embedded deltas may be rebased.
        generate([str(a.sidecar_source), str(a.output / "sidecar-validation")])
        source = json.loads((a.sidecar_source / "manifest.json").read_text())
        source_aliases = shipping_aliases(
            source["aliases"], not a.include_8k, a.native_only
        )
        if source_aliases.keys() != data["aliases"].keys():
            raise RuntimeError("Sidecar/embedded alias mismatch")
        used = set(source_aliases.values())
        for i in list(used):
            base = source["payloads"][i]["base"]
            while base is not None:
                used.add(base)
                base = source["payloads"][base]["base"]
        ids = sorted(used)
        remap = {i: j for j, i in enumerate(ids)}
        source_rows = []
        for i in ids:
            r = source["payloads"][i]
            base = r["base"]
            source_rows.append(
                "    {"
                + f'{remap[i]+100}, {r["stored_size"]}, {r["stored_size"]}, {r["size"]}, false, '
                + f"{remap[base] if base is not None else -1}, "
                + ", ".join(
                    json.dumps(r[k]) for k in ("file", "stored_sha256", "sha256")
                )
                + "},"
            )
        source_names = []
        for key, i in sorted(source_aliases.items()):
            target = data["payloads"][data["aliases"][key]]
            if (target["sha256"], target["size"]) != (
                source["payloads"][i]["sha256"],
                source["payloads"][i]["size"],
            ):
                raise RuntimeError("Sidecar target mismatch: " + key)
            source_names.append("    {" + json.dumps(key) + ", " + str(remap[i]) + "},")
        secondary = (
            "inline constexpr Payload sidecar_payload_index[] = {\n"
            + "\n".join(source_rows)
            + "\n};\ninline constexpr Alias sidecar_asset_aliases[] = {\n"
            + "\n".join(source_names)
            + "\n};\ninline constexpr unsigned sidecar_max_delta_depth = "
            + str(source.get("max_delta_depth", 1))
            + ";\n"
        )
    header = (
        """#pragma once
#include <cstdint>
namespace fsr4assets {
struct Payload { unsigned resource, packed_size, stored_size, size; bool compressed; int base; const char *file, *stored_sha256, *sha256; };
struct Alias { const char *key; unsigned payload; };
inline constexpr Payload payload_index[] = {
"""
        + "\n".join(rows)
        + "\n};\ninline constexpr unsigned max_delta_depth = "
        + str(maximum_depth)
        + ";\ninline constexpr Alias asset_aliases[] = {\n"
        + "\n".join(aliases)
        + "\n};\n"
        + secondary
        + "}\n"
    )
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "asset-index.hpp").write_text(header)
    (a.output / "embedded-storage.json").write_text(
        json.dumps(dict(schema=1, payloads=storage), indent=2) + "\n"
    )
    (a.output / "assets.rc").write_text("\n".join(resources) + "\n")
    if a.elf:
        # Windows and Linux share the same payload format: decompress, then apply the delta.
        offsets = []
        total = 0
        for blob in blobs:
            offsets.append(total)
            total += len(blob)
        binary = (a.output / "embedded-assets.bin").resolve()
        binary.write_bytes(b"".join(blobs))
        (a.output / "embedded-assets.hpp").write_text(
            "#pragma once\n#include <cstddef>\nnamespace fsr4assets {\n"
            "inline constexpr std::size_t embedded_payload_offsets[] = {"
            + ",".join(map(str, offsets))
            + "};\n"
            f"inline constexpr std::size_t embedded_asset_bytes = {total};\n" + "}\n"
        )
        (a.output / "embedded-assets.S").write_text(
            '.section .rodata.fsr4_assets,"a",@progbits\n.balign 16\n'
            ".global fsr4_embedded_assets_begin\n.hidden fsr4_embedded_assets_begin\n"
            ".type fsr4_embedded_assets_begin,@object\nfsr4_embedded_assets_begin:\n"
            ".incbin " + json.dumps(str(binary)) + "\n"
            ".size fsr4_embedded_assets_begin, .-fsr4_embedded_assets_begin\n"
            '.section .note.GNU-stack,"",@progbits\n'
        )
    print(
        f'Verified {len(targets)} payloads ({data["delta_payloads"]} deltas), {len(aliases)} aliases; '
        + f"{sum(map(len,blobs)):,} stored bytes"
    )

def prepare(a):
    HERE = Path(__file__).resolve().parent
    source = a.assets.resolve()
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Validate the full source catalog even when reusing a cached embedded pack.
    signature = hashlib.sha256(
        (source / "manifest.json").read_bytes()
        + b"".join(
            (HERE / name).read_bytes()
            for name in ("embed.py", "catalog.py", "storage.py")
        )
        + bytes((a.native_only, a.include_8k))
    ).hexdigest()
    cache = output / "embedding" / signature[:16]
    pack = cache / "assets"
    if (cache / "complete").exists() and (
        cache / "complete"
    ).read_text().strip() != signature:
        raise RuntimeError("Generated pack signature collision")
    if not (cache / "complete").is_file():
        if pack.exists():
            raise RuntimeError(
                "Incomplete generated pack; select a fresh build directory"
            )
        select_pack(source, pack, omit_8k=not a.include_8k, native_only=a.native_only)
        (cache / "complete").write_text(signature + "\n")
    generation = (
        (["--compress-shaders"] if a.compress_shaders else [])
        + (["--native-only", "--elf"] if a.native_only else [])
        + (["--include-8k"] if a.include_8k else [])
    )
    generate([str(pack), str(output), "--sidecar-source", str(source), *generation])
    (output / "embedding-selection.json").write_text(
        json.dumps(
            dict(
                source=str(source),
                embedded_assets=str(pack),
                storage_index=str(output / "embedded-storage.json"),
                shader_compression="zstd-per-payload" if a.compress_shaders else "none",
                windows_assets_omitted=a.native_only,
                omitted_shader_tiers=[] if a.include_8k else ["4320"],
            ),
            indent=2,
        )
        + "\n"
    )

def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if "--prepare" not in args:
        return generate(args)
    args.remove("--prepare")
    parser = argparse.ArgumentParser(
        description="Prepare a cached build-only asset selection."
    )
    parser.add_argument("assets", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument("--include-8k", action="store_true")
    parser.add_argument("--compress-shaders", action="store_true")
    return prepare(parser.parse_args(args))

if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as error:
        sys.exit(f"Missing asset: {error.filename}. Run extraction, conversion and patching before building.")
