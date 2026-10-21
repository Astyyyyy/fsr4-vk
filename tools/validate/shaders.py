#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Validate reconstructed SPIR-V modules."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "convert"))
from toolchain import find_tool

spec=importlib.util.spec_from_file_location('delta',Path(__file__).resolve().parent.parent/'assets/storage.py')
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('assets',type=Path);p.add_argument('--spirv-val');a=p.parse_args()
try:
    a.spirv_val = find_tool("spirv-val", a.spirv_val)
except ValueError as error:
    p.error(str(error))
m=json.loads((a.assets/'manifest.json').read_text());cache={};count=0
def payload(i):
    if i not in cache:
        r=m['payloads'][i];b=(a.assets/r['file']).read_bytes()
        if hashlib.sha256(b).hexdigest()!=r['stored_sha256']:raise RuntimeError('Storage changed')
        if r['kind']=='delta':b=d.reconstruct(payload(r['base']),b)
        if hashlib.sha256(b).hexdigest()!=r['sha256']:raise RuntimeError('Payload changed')
        cache[i]=b
    return cache[i]
with tempfile.TemporaryDirectory(prefix='fsr4vk-spirv-validation-') as directory:
    module=Path(directory)/'shader.spv'
    for r in m['payloads']:
        b=payload(r['id'])
        if b[:4]!=b'\x03\x02\x23\x07':continue
        module.write_bytes(b)
        try:
            subprocess.run([a.spirv_val, str(module)], check=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError('SPIR-V validation failed: ' + r['file']) from error
        count += 1
print(f'Validated {count} unique reconstructed SPIR-V modules')
