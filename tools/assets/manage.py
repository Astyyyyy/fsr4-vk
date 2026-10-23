# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Manage generated asset packs."""

import argparse, copy, hashlib, json, sys, tempfile
from collections import defaultdict
from pathlib import Path
import storage as deltas
import storage as d
import embed
from catalog import (
    shipping_aliases,
    model_files,
    is_stock_shader,
    shader_section,
    stock_base_alias,
    validate_stock_layout,
    select_pack,
)

ROOT = Path(__file__).resolve().parents[2]


def logical_contents(manifest):
    return {
        name: (
            manifest["payloads"][index]["sha256"],
            manifest["payloads"][index]["size"],
        )
        for name, index in manifest["aliases"].items()
    }


def adopt(pack, backup, root=ROOT):
    root = root.resolve()
    pack, backup = pack.resolve(), backup.resolve()
    build = root / "build"
    if not all(path.is_relative_to(build) and path != build for path in (pack, backup)):
        raise ValueError(
            "Pack and recovery backup must be under this project's build directory"
        )
    if pack.is_relative_to(backup) or backup.is_relative_to(pack):
        raise ValueError("Pack and backup must not overlap")
    if backup.exists():
        raise ValueError("Backup already exists; use a fresh exact path")
    current = root / "assets"
    old = json.loads((current / "manifest.json").read_text())
    new = json.loads((pack / "manifest.json").read_text())
    if logical_contents(old) != logical_contents(new):
        raise ValueError("Logical asset names or payload contents changed")
    embed.generate([str(pack), str(build / "adoption-check")])
    backup.parent.mkdir(parents=True, exist_ok=True)
    current.rename(backup)
    try:
        pack.rename(current)
    except BaseException:
        backup.rename(current)
        raise
    print("Adopted:", current)
    print("Recoverable previous assets:", backup)


def pack(args):
    if args.output.exists():
        raise ValueError("use a fresh output directory")
    reference = args.reference.resolve()
    for directory in args.generated:
        if directory.resolve().is_relative_to(reference) or reference.is_relative_to(
            directory.resolve()
        ):
            raise ValueError(
                "generated inputs must not include/enclose the reference pack"
            )

    original = json.loads((reference / "manifest.json").read_text())
    aliases = shipping_aliases(
        original["aliases"], native_only=args.native_only
    )
    omitted = set()
    if args.omit_legacy_unused:
        omitted = {
            row["id"]
            for row in original["payloads"]
            if row["file"].endswith("/initializers.bin")
            or (
                row["file"].startswith("fp8/") and Path(row["file"]).name == "pass0.spv"
            )
        }
        aliases = {key: index for key, index in aliases.items() if index not in omitted}

    needed = sorted(set(aliases.values()))
    expected_hashes = {original["payloads"][index]["sha256"] for index in needed}
    inputs = {}
    for directory in args.generated:
        if not directory.is_dir():
            raise ValueError("missing generated input directory: " + str(directory))
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in (".spv", ".bin"):
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest in expected_hashes:
                    inputs.setdefault(digest, path)

    missing = [
        original["payloads"][index]["file"]
        for index in needed
        if original["payloads"][index]["sha256"] not in inputs
    ]
    if missing:
        raise ValueError(
            f"{len(missing)} targets were not regenerated:\n" + "\n".join(missing)
        )

    remap = {previous: index for index, previous in enumerate(needed)}
    names = defaultdict(list)
    for key, previous in aliases.items():
        if is_stock_shader(key):
            names[previous].append(key)
    targets = [
        inputs[original["payloads"][index]["sha256"]].read_bytes() for index in needed
    ]
    rows = []
    args.output.mkdir(parents=True)
    for previous, target in zip(needed, targets):
        row = copy.deepcopy(original["payloads"][previous])
        row.update(id=remap[previous], kind="full", base=None)
        blob = target
        if previous not in names and shader_section(row["file"], "patched"):
            base_alias = stock_base_alias(row["file"])
            parent = remap[aliases[base_alias]]
            if parent >= row["id"]:
                raise ValueError("stock base must precede its patch")
            blob = deltas.generate(targets[parent], target)
            if deltas.reconstruct(targets[parent], blob) != target:
                raise ValueError("delta reconstruction mismatch")
            row.update(kind="delta", base=parent)
            row["file"] = Path(row["file"]).with_suffix(".delta").as_posix()
        row.update(
            stored_size=len(blob), stored_sha256=hashlib.sha256(blob).hexdigest()
        )
        destination = args.output / row["file"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(blob)
        rows.append(row)

    for previous, keys in names.items():
        for key in keys:
            destination = args.output / key
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(targets[remap[previous]])
    models = model_files(original)
    models = {
        key: dict(row, payload=remap[row["payload"]])
        for key, row in models.items()
        if row["payload"] in remap
    }
    for key, row in models.items():
        destination = args.output / key
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(targets[row["payload"]])

    manifest = copy.deepcopy(original)
    manifest.update(
        payloads=rows,
        aliases={key: remap[index] for key, index in aliases.items()},
        stored_payload_bytes=sum(row["stored_size"] for row in rows),
        source_payload_bytes=sum(row["size"] for row in rows),
        delta_payloads=sum(row["kind"] == "delta" for row in rows),
        max_delta_depth=1,
        omitted_shader_tiers=[],
        windows_assets_omitted=args.native_only,
    )
    validate_stock_layout(manifest, targets, args.output)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Packed {len(rows)} independently regenerated payloads; all locked hashes match."
    )


def unpack(a):
    if a.output.exists():
        raise ValueError("Use a fresh output directory")
    m = json.loads((a.assets / "manifest.json").read_text())
    cache = {}

    def payload(i):
        if i not in cache:
            r = m["payloads"][i]
            b = (a.assets / r["file"]).read_bytes()
            if hashlib.sha256(b).hexdigest() != r["stored_sha256"]:
                raise RuntimeError("Storage changed")
            if r["kind"] == "delta":
                b = d.reconstruct(payload(r["base"]), b)
            if hashlib.sha256(b).hexdigest() != r["sha256"]:
                raise RuntimeError("Payload changed")
            cache[i] = b
        return cache[i]

    for key, i in m["aliases"].items():
        parts = Path(key).parts
        if parts[0] != a.family or parts[1] in ("models", "shaders"):
            continue
        if "optimized" in parts and not a.include_patched:
            continue
        if not a.include_patched and ".win-" in key:
            continue
        dest = a.output / Path(*parts[1:])
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload(i))
    print(a.output.resolve())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser(
        "pack",
        help="Pack independently regenerated bytes against locked catalog hashes",
    )
    p.add_argument("reference", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--generated", type=Path, action="append", required=True)
    p.add_argument("--native-only", action="store_true")
    p.add_argument("--omit-legacy-unused", action="store_true")
    p = commands.add_parser("unpack", help="Materialize immutable tool inputs")
    p.add_argument("assets", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--family", choices=("fp8", "int8"), required=True)
    p.add_argument("--include-patched", action="store_true")
    p = commands.add_parser("select", help="Create a build-only selection")
    p.add_argument("assets", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--omit-8k", action="store_true")
    p.add_argument("--native-only", action="store_true")
    p = commands.add_parser(
        "verify", help="Validate all payloads, aliases and named models"
    )
    p.add_argument("assets", type=Path)
    p = commands.add_parser(
        "adopt", help="Replace assets with a verified logically identical pack"
    )
    p.add_argument("pack", type=Path)
    p.add_argument("backup", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "pack":
            return pack(args)
        if args.command == "unpack":
            return unpack(args)
        if args.command == "select":
            return select_pack(args.assets, args.output, args.omit_8k, args.native_only)
        if args.command == "adopt":
            return adopt(args.pack, args.backup)
        with tempfile.TemporaryDirectory(
            prefix="fsr4-asset-verification-"
        ) as temporary:
            embed.generate([str(args.assets), temporary])
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
