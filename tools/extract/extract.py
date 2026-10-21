# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Extract all FSR 4.1.1 model weights and shaders."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]

import fp8, int8
from sdk import SDK, SDK_HASH, extract_endpoints, extract_rcas


def extract(dll, output, model="all"):
    sdk = SDK(dll)
    if output.exists():
        raise ValueError("Use a fresh extraction output directory")
    output.mkdir(parents=True)
    if model in ("all", "fp8"):
        fp8.extract(SimpleNamespace(
            dll=dll, output=output / "fp8", all_blobs=False, endpoint_profile="shipping"
        ))
    if model in ("all", "int8"):
        int8.extract(SimpleNamespace(
            dll=dll, output=output / "int8", all_permutations=True, flags=0
        ))
    if model in ("all", "endpoints"):
        extract_endpoints(sdk, output / "endpoints")
    if model in ("all", "rcas"):
        extract_rcas(sdk, output / "rcas")
    files = {
        path.relative_to(output).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    (output / "manifest.json").write_text(json.dumps({
        "schema": 1, "complete": True, "sdk_version": "4.1.1.2740",
        "source_sha256": SDK_HASH, "model": model, "files": files,
    }, indent=2) + "\n")
    print("Extracted:", output.resolve())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "assets/extracted")
    parser.add_argument("--model", choices=("all", "fp8", "int8", "endpoints", "rcas"), default="all")
    args = parser.parse_args(argv)
    try:
        extract(args.dll, args.output, args.model)
    except (ValueError, RuntimeError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
