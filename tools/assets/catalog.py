# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Read the asset catalog and select assets for each build."""

from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict
import copy, hashlib, json, re
import storage as d

POLICY = "stock-full-patched-direct-v1"

SPIRV_MAGIC = b"\x03\x02\x23\x07"


def shader_section(key, section):
    parts = Path(key).parts
    return (
        len(parts) >= 5
        and ".." not in parts
        and parts[0] in ("fp8", "int8")
        and parts[1] == "shaders"
        and parts[3] == section
    )


def is_stock_shader(key):
    return shader_section(key, "stock") and key.endswith(".spv")


def stock_base_alias(patched):
    parts = Path(patched).parts
    if not shader_section(patched, "patched"):
        raise ValueError("Not a canonical patched shader: " + patched)
    tail = parts[4:]
    if tail[0] == "windows":
        tail = tail[1:]
    elif tail[:2] in (("common", "exact"), ("common", "display-motion"), ("common", "exposure")):
        tail = tail[2:]
    else:
        raise ValueError("Unknown patch family: " + patched)
    name = re.sub(r"\.(win-opt|win-dotacc|hrmv)(?=\.(spv|delta)$)", "", tail[-1])
    return Path(*parts[:3], "stock", *tail[:-1], name).with_suffix(".spv").as_posix()


def validate_stock_layout(manifest, targets, root=None):
    policy = manifest.get("shader_storage_policy")
    if policy is None:
        return
    if policy != POLICY or manifest.get("max_delta_depth") != 1:
        raise ValueError("Unsupported stock shader storage policy")
    rows = manifest["payloads"]
    aliases = manifest["aliases"]
    stock_ids = {index for key, index in aliases.items() if is_stock_shader(key)}
    for key, index in aliases.items():
        if (
            shader_section(key, "patched")
            and key.endswith(".spv")
            and index not in stock_ids
        ):
            if not shader_section(rows[index]["file"], "patched"):
                raise ValueError(
                    "Patched alias does not reference a patched payload: " + key
                )
        if not is_stock_shader(key):
            continue
        row = rows[index]
        if (
            row["kind"] != "full"
            or row["base"] is not None
            or not is_stock_shader(row["file"])
            or targets[index][:4] != SPIRV_MAGIC
        ):
            raise ValueError("Stock shader is not full SPIR-V: " + key)
        if root is not None and (root / key).read_bytes() != targets[index]:
            raise ValueError("Physical stock shader differs from locked target: " + key)
    for row in rows:
        if row["kind"] == "delta" and row["base"] not in stock_ids:
            raise ValueError("Delta base is not a full stock shader: " + row["file"])
        if not shader_section(row["file"], "patched"):
            continue
        base_alias = stock_base_alias(row["file"])
        if (
            row["kind"] != "delta"
            or not row["file"].endswith(".delta")
            or row["base"] != aliases.get(base_alias)
        ):
            raise ValueError(
                "Patched shader is not a direct delta of its stock base: " + row["file"]
            )
    if root is not None:
        stock_roots = {
            root / Path(*Path(key).parts[:4]) for key in aliases if is_stock_shader(key)
        }
        if any(next(directory.rglob("*.delta"), None) for directory in stock_roots):
            raise ValueError("Stock folders must not contain delta files")


_STANDARD = re.compile(
    r"/(1080|2160|4320)/(native|quality|balanced|performance|custom)/"
)


def standard_shader_path(path):
    if path.startswith("int8/") and "/shaders/" in path:
        return _STANDARD.sub(r"/\1/standard/", path)
    return path


def canonical_aliases(aliases):
    result = {}
    for path, payload in aliases.items():
        name = standard_shader_path(path)
        if name in result and result[name] != payload:
            raise ValueError("Standard INT8 shader aliases disagree: " + name)
        result[name] = payload
    return result


_MODEL = re.compile(r"^(?:fp8|int8)/models/[^/]+/")

_MODEL_FILE = re.compile(
    r"^(initializers\.bin|weights(?:_(?:native|quality|balanced|performance|custom|ultraperf|standard))?\.bin)$"
)


def shared_model_path(path):
    prefix = _MODEL.match(path)
    if not prefix:
        return path
    tail = path[prefix.end() :].split("/")
    if tail[0] in ("1080", "2160", "4320"):
        tail = tail[1:]
    if tail and tail[0] == "weights":
        tail = tail[1:]
    if len(tail) != 1 or not _MODEL_FILE.fullmatch(tail[0]):
        return path
    name = tail[0]
    if name.startswith("weights") and name != "weights_ultraperf.bin":
        name = "weights_standard.bin"
    return path[: prefix.end()] + name


def shared_models(mapping):
    result = {}
    for path, value in mapping.items():
        name = shared_model_path(path)
        if name in result and result[name] != value:
            raise ValueError("Model aliases disagree: " + name)
        result[name] = value
    return result


def model_files(manifest):
    result = {}
    for name, index in manifest["aliases"].items():
        if (_MODEL.match(name) and _MODEL_FILE.fullmatch(Path(name).name)
                and shared_model_path(name) == name):
            if not 0 <= index < len(manifest["payloads"]):
                raise ValueError("Invalid model alias: " + name)
            row = manifest["payloads"][index]
            result[name] = dict(payload=index, size=row["size"], sha256=row["sha256"])
    return result


def is_int8_nonexact_alias(key):
    if key.startswith("int8/optimized/"):
        return key.split("/")[2] != "exact"
    return key.startswith("int8/shaders/4.1.1/patched/common/lossy/")


def is_int8_vector_only_alias(key):
    return key.startswith("int8/") and ".win-vec4." in key.split("/")[-1]


def combined_int8_windows_path(key):
    if key.startswith("int8/shaders/"):
        return key.replace("/patched/windows/dot-accumulation/", "/patched/windows/")
    return key


def is_8k_shader_alias(key):
    return key.endswith(".spv") and "4320" in key.split("/")


def is_windows_asset_alias(key):
    parts = key.split("/")
    return "windows" in parts or ".win-" in parts[-1]


def shipping_aliases(aliases, omit_8k=False, native_only=False):
    result = {}
    for key, index in aliases.items():
        if (
            is_int8_nonexact_alias(key)
            or is_int8_vector_only_alias(key)
            or (omit_8k and is_8k_shader_alias(key))
            or (native_only and is_windows_asset_alias(key))
        ):
            continue
        name = combined_int8_windows_path(key)
        if name in result and result[name] != index:
            raise ValueError("Combined Windows alias collision: " + name)
        result[name] = index
    return result


def select_pack(assets, output, omit_8k=False, native_only=False):
    a = SimpleNamespace(
        assets=Path(assets),
        output=Path(output),
        omit_8k=omit_8k,
        native_only=native_only,
    )
    if a.output.exists():
        raise ValueError("Use a fresh output directory")
    old = json.loads((a.assets / "manifest.json").read_text())
    targets = []
    stored = []
    for row in old["payloads"]:
        blob = (a.assets / row["file"]).read_bytes()
        if hashlib.sha256(blob).hexdigest() != row["stored_sha256"]:
            raise ValueError("Stored payload changed")
        target = (
            d.reconstruct(targets[row["base"]], blob)
            if row["base"] is not None
            else blob
        )
        if hashlib.sha256(target).hexdigest() != row["sha256"]:
            raise ValueError("Target changed")
        stored.append(blob)
        targets.append(target)
    aliases = shipping_aliases(old["aliases"], a.omit_8k, a.native_only)
    ids = sorted(set(aliases.values()))
    remap = {i: j for j, i in enumerate(ids)}
    names = defaultdict(list)
    for key, i in aliases.items():
        names[i].append(key)
    rows = []
    depths = []
    buckets = defaultdict(list)
    rebased = 0
    a.output.mkdir(parents=True)
    for previous in ids:
        row = copy.deepcopy(old["payloads"][previous])
        target = targets[previous]
        parent = row["base"]
        blob = stored[previous]
        match = re.search(
            r"(pass-?\d+|prepass(?:_wmma)?|postpass(?:_wmma)?)", Path(row["file"]).name
        )
        bucket = (row["file"].split("/")[0], match[0]) if match else None
        if parent is not None and parent not in remap:
            rebased += 1
            blob = target
            parent = None
            ancestor = old["payloads"][previous]["base"]
            while ancestor is not None and ancestor not in remap:
                ancestor = old["payloads"][ancestor]["base"]
            candidates = ([] if ancestor is None else [ancestor]) + (
                buckets[bucket][-8:] if bucket else []
            )
            for candidate in dict.fromkeys(candidates):
                if depths[remap[candidate]] >= old.get("max_delta_depth", 1):
                    continue
                encoded = d.generate(targets[candidate], target)
                if len(encoded) < len(blob):
                    blob = encoded
                    parent = candidate
        if parent is not None and d.reconstruct(targets[parent], blob) != target:
            raise ValueError("Rebased delta changed target")
        row["id"] = remap[previous]
        row["base"] = None if parent is None else remap[parent]
        row["kind"] = "full" if parent is None else "delta"
        depth = 0 if parent is None else depths[remap[parent]] + 1
        if depth > old.get("max_delta_depth", 1):
            raise ValueError("Chain depth increased")
        if (
            is_int8_nonexact_alias(row["file"])
            or is_int8_vector_only_alias(row["file"])
            or (a.omit_8k and "4320" in Path(row["file"]).parts)
            or (a.native_only and is_windows_asset_alias(row["file"]))
        ):
            row["file"] = next(k for k in names[previous] if "/shaders/" in k)
        row["file"] = combined_int8_windows_path(row["file"])
        if target[:4] == b"\x03\x02\x23\x07":
            row["file"] = Path(row["file"]).with_suffix(
                ".spv" if parent is None else ".delta"
            ).as_posix()
        row["stored_size"] = len(blob)
        row["stored_sha256"] = hashlib.sha256(blob).hexdigest()
        dest = a.output / row["file"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(blob)
        rows.append(row)
        depths.append(depth)
        if bucket:
            buckets[bucket].append(previous)
        if len(rows) % 100 == 0:
            print(
                f"{len(rows)}/{len(ids)} retained payloads; rebased {rebased}",
                flush=True,
            )
    m = copy.deepcopy(old)
    m["payloads"] = rows
    m["aliases"] = {k: remap[i] for k, i in aliases.items()}
    m["stored_payload_bytes"] = sum(r["stored_size"] for r in rows)
    m["source_payload_bytes"] = sum(r["size"] for r in rows)
    m["delta_payloads"] = sum(r["kind"] == "delta" for r in rows)
    m.pop("int8_lossy_profiles", None)
    m["int8_windows_variants"] = ["combined-vector-dot"]
    if a.omit_8k:
        m["omitted_shader_tiers"] = ["4320"]
    if a.native_only:
        m["windows_assets_omitted"] = True
    if old.get("shader_storage_policy"):
        for key, previous in aliases.items():
            if is_stock_shader(key):
                dest = a.output / key
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(targets[previous])
        validate_stock_layout(m, [targets[i] for i in ids], a.output)
    for path, row in model_files(m).items():
        dest = a.output / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(targets[ids[row["payload"]]])
    (a.output / "manifest.json").write_text(json.dumps(m, indent=2) + "\n")
    print(
        f'Kept {len(rows)} payloads, {len(aliases)} aliases, {m["stored_payload_bytes"]:,} bytes; '
        f'removed {len(old["payloads"])-len(rows)} payloads, rebased {rebased}.',
        flush=True,
    )
