# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Shared asset workflow helpers."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run(script, *arguments):
    subprocess.run([sys.executable, str(script), *map(str, arguments)], check=True)


def asset_path(root, name):
    path = root / name
    if Path(name).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Invalid asset path: " + name)
    return path


def prepare_assets(assets, reference=None):
    reference = reference or ROOT / "tools/assets/reference"
    if (assets.resolve().is_relative_to(reference.resolve())
            or reference.resolve().is_relative_to(assets.resolve())):
        raise ValueError("Generated assets must be separate from the reference catalog")
    manifest = reference / "manifest.json"
    if not manifest.is_file():
        raise ValueError("Reference catalog is missing; restore tools/assets/reference from the repository")
    data = manifest.read_bytes()
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "manifest.json").write_bytes(data)


def validate_extraction(directory):
    metadata = json.loads((directory / "manifest.json").read_text())
    sdk = load("workflow_sdk", ROOT / "tools/extract/sdk.py")
    if (not metadata.get("complete") or metadata.get("model") != "all"
            or metadata.get("source_sha256") != sdk.SDK_HASH):
        raise ValueError("Run extract.py for a complete SDK 4.1.1 extraction first")
    for family in ("fp8", "int8"):
        for tier in sdk.TIERS:
            if not (directory / "endpoints" / ("sdk-" + family) / tier).is_dir():
                raise ValueError("Extraction is missing " + family + " " + tier
                                 + " endpoints; rerun extract.py in a fresh directory")
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path != directory / "manifest.json"
    }
    if actual != set(metadata["files"]):
        raise ValueError("Extraction files changed; rerun extract.py in a fresh directory")
    for name, expected in metadata["files"].items():
        path = asset_path(directory, name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Extraction changed: " + name)


def generated_index(directory):
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix in (".spv", ".bin"):
            result.setdefault(hashlib.sha256(path.read_bytes()).hexdigest(), path)
    return result


def baseline(row):
    parts = Path(row["file"]).parts
    return "patched" not in parts and "windows" not in parts and "rcas" not in parts


def install(files, assets):
    for name, source in files.items():
        destination = asset_path(assets, name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


