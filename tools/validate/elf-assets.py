#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Verify embedded Linux assets directly from the ELF file."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent/'assets'
sys.path.insert(0, str(TOOLS))
from storage import storage_index, unpack_payload
from catalog import is_windows_asset_alias, is_8k_shader_alias
spec = importlib.util.spec_from_file_location('deltas', TOOLS/'storage.py')
d = importlib.util.module_from_spec(spec); spec.loader.exec_module(d)
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('binary', type=Path)
p.add_argument('assets', type=Path, help='selected native embedding pack, not the full source catalog')
p.add_argument('output', type=Path)
p.add_argument('--storage-index', type=Path)
a = p.parse_args()
binary = a.binary.read_bytes()
if binary[:6] != b'\x7fELF\x02\x01': raise RuntimeError('Expected little-endian ELF64')
shoff = struct.unpack_from('<Q', binary, 40)[0]
entsize, count, _ = struct.unpack_from('<HHH', binary, 58)
sections = [struct.unpack_from('<IIQQQQIIQQ', binary, shoff+i*entsize) for i in range(count)]
symbols = []
for section in sections:
    if section[1] != 2: continue # SHT_SYMTAB, available before stripping.
    strings = sections[section[6]]
    for pos in range(section[4], section[4]+section[5], section[9]):
        name, info, other, index, value, size = struct.unpack_from('<IBBHQQ', binary, pos)
        start = strings[4]+name
        text = binary[start:binary.index(b'\0', start)].decode()
        if text == 'fsr4_embedded_assets_begin': symbols.append((index, value, size, other, info >> 4))
if len(symbols) != 1: raise RuntimeError('Missing/ambiguous ELF embedded asset symbol')
index, value, size, visibility, binding = symbols[0]
section = sections[index]
if section[2] & 1 or not section[2] & 2: raise RuntimeError('Asset section must be allocated and read-only')
# GNU ld may localize a hidden global into a LOCAL/DEFAULT final symbol.
if visibility & 3 != 2 and binding != 0: raise RuntimeError('Embedded data symbol must be hidden/local')
raw = section[4]+value-section[3]
blob = binary[raw:raw+size]
m = json.loads((a.assets/'manifest.json').read_text())
if not m.get('windows_assets_omitted'): raise RuntimeError('Not a native-only selection')
if any(is_windows_asset_alias(k) for k in m['aliases']): raise RuntimeError('Windows alias retained')
if any(is_windows_asset_alias(r['file']) for r in m['payloads']): raise RuntimeError('Windows payload retained')
if '4320' in m.get('omitted_shader_tiers', []) and any(is_8k_shader_alias(k) for k in m['aliases']):
    raise RuntimeError('8K shader alias retained')
pos = 0; targets = []; spirv = 0
storage = storage_index(a.storage_index, m)
for row, encoded in zip(m['payloads'], storage):
    packed = blob[pos:pos+encoded['packed_size']]; pos += encoded['packed_size']
    stored = unpack_payload(packed, encoded, row)
    if hashlib.sha256(stored).hexdigest() != row['stored_sha256']: raise RuntimeError('ELF storage mismatch')
    if stored != (a.assets/row['file']).read_bytes(): raise RuntimeError('ELF/catalog mismatch')
    target = stored if row['base'] is None else d.reconstruct(targets[row['base']], stored)
    if len(target) != row['size'] or hashlib.sha256(target).hexdigest() != row['sha256']:
        raise RuntimeError('ELF reconstructed target mismatch')
    targets.append(target); spirv += target[:4] == b'\x03\x02\x23\x07'
if pos != size or len(blob) != size: raise RuntimeError('ELF blob extent mismatch')
report = dict(complete=True, binary=str(a.binary.resolve()), binary_bytes=len(binary),
              binary_sha256=hashlib.sha256(binary).hexdigest(), payloads_verified=len(targets),
              spirv_payloads=spirv, embedded_stored_bytes=size, compressed_payloads=sum(r['compression']=='zstd' for r in storage), windows_payloads=0,
              omitted_shader_tiers=m.get('omitted_shader_tiers',[]),
              aliases=len(m['aliases']), read_only=True, private_data_symbol=True)
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
