#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Verify embedded Windows assets and provider exports."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
sys.path.insert(0,str(Path(__file__).parents[1]/"assets"))
from storage import storage_index, unpack_payload

spec = importlib.util.spec_from_file_location(
    'deltas', Path(__file__).parents[1]/'assets/storage.py')
deltas = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deltas)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('binary', type=Path)
    p.add_argument('assets', type=Path)
    p.add_argument('--baseline-manifest', type=Path)
    p.add_argument('--omit-8k', action='store_true')
    p.add_argument('--embedding-selection', type=Path,
                   help='generated build selection; the assets argument remains the full source root')
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    storage_path = None
    source_assets = a.assets
    if a.embedding_selection:
        selection = json.loads(a.embedding_selection.read_text())
        if Path(selection['source']).resolve() != a.assets.resolve():
            raise ValueError('Embedding source root mismatch')
        a.assets = Path(selection['embedded_assets'])
        if selection.get('storage_index'):
            storage_path = Path(selection['storage_index'])
    data = a.binary.read_bytes()
    if data[:2] != b'MZ': raise ValueError('Not a PE binary')
    pe = struct.unpack_from('<I', data, 0x3c)[0]
    if data[pe:pe+4] != b'PE\0\0': raise ValueError('Invalid PE signature')
    count = struct.unpack_from('<H', data, pe+6)[0]
    opt_size = struct.unpack_from('<H', data, pe+20)[0]
    opt = pe+24
    if struct.unpack_from('<H', data, opt)[0] != 0x20b:
        raise ValueError('Expected PE32+')
    sections = [struct.unpack_from('<IIII', data, opt+opt_size+40*i+8)
                for i in range(count)]

    def offset(rva):
        for virtual_size, address, raw_size, raw in sections:
            if address <= rva < address+max(virtual_size, raw_size):
                return raw+rva-address
        raise ValueError('Unmapped PE RVA')

    base = offset(struct.unpack_from('<I', data, opt+112+16)[0])

    def entries(relative):
        pos = base+relative
        named, numeric = struct.unpack_from('<HH', data, pos+12)
        return [struct.unpack_from('<II', data, pos+16+8*i)
                for i in range(named+numeric)]

    resource_type = next(v & 0x7fffffff for k, v in entries(0) if k == 10)
    resources = {}
    for resource_id, directory in entries(resource_type):
        languages = entries(directory & 0x7fffffff)
        if len(languages) != 1: raise ValueError('Ambiguous resource language')
        _, entry = languages[0]
        address, size, _, _ = struct.unpack_from('<IIII', data, base+entry)
        pos = offset(address)
        resources[resource_id] = data[pos:pos+size]

    catalog = json.loads((a.assets/'manifest.json').read_text())
    targets = []
    storage = storage_index(storage_path, catalog)
    for row, encoded in zip(catalog['payloads'], storage):
        blob = unpack_payload(resources.pop(row['id']+100), encoded, row)
        if len(blob) != row['stored_size'] or hashlib.sha256(blob).hexdigest() != row['stored_sha256']:
            raise ValueError('Embedded stored hash mismatch: '+row['file'])
        target = deltas.reconstruct(targets[row['base']], blob) if row['kind'] == 'delta' else blob
        if len(target) != row['size'] or hashlib.sha256(target).hexdigest() != row['sha256']:
            raise ValueError('Embedded target hash mismatch: '+row['file'])
        targets.append(target)
    if resources: raise ValueError('Unexpected extra RCDATA resources')
    source_8k_targets = None
    if a.omit_8k:
        spec = importlib.util.spec_from_file_location('policy', Path(__file__).parents[1]/'assets/catalog.py')
        policy = importlib.util.module_from_spec(spec); spec.loader.exec_module(policy)
        if any(policy.is_8k_shader_alias(k) for k in catalog['aliases']):
            raise ValueError('8K shader alias is still embedded')
        source = json.loads((source_assets/'manifest.json').read_text())
        high = {i for k,i in source['aliases'].items() if policy.is_8k_shader_alias(k)}
        low = {i for k,i in source['aliases'].items() if not policy.is_8k_shader_alias(k)}
        exclusive = {source['payloads'][i]['sha256'] for i in high-low}
        if exclusive & {hashlib.sha256(b).hexdigest() for b in targets}:
            raise ValueError('8K-only shader payload is still embedded')
        source_8k_targets = len(exclusive)
    baseline_paths = None
    if a.baseline_manifest:
        original = json.loads(a.baseline_manifest.read_text())['entries']
        if a.omit_8k:
            spec = importlib.util.spec_from_file_location('policy', Path(__file__).parents[1]/'assets/catalog.py')
            policy = importlib.util.module_from_spec(spec); spec.loader.exec_module(policy)
            original = [r for r in original if
                        not (a.omit_8k and policy.is_8k_shader_alias(r['path']))]
        for row in original:
            target = targets[catalog['aliases'][row['path']]]
            if len(target) != row['size'] or hashlib.sha256(target).hexdigest() != row['sha256']:
                raise ValueError('Baseline logical path changed: '+row['path'])
        baseline_paths = len(original)

    export = offset(struct.unpack_from('<I', data, opt+112)[0])
    names_count = struct.unpack_from('<I', data, export+24)[0]
    names_table = offset(struct.unpack_from('<I', data, export+32)[0])
    names = []
    for i in range(names_count):
        pos = offset(struct.unpack_from('<I', data, names_table+4*i)[0])
        names.append(data[pos:data.index(b'\0', pos)].decode('ascii'))
    expected = ['ffxConfigure', 'ffxCreateContext', 'ffxDestroyContext', 'ffxDispatch', 'ffxQuery']
    if sorted(names) != expected: raise ValueError('Unexpected exported API: '+repr(names))
    report = dict(binary_bytes=len(data), binary_sha256=hashlib.sha256(data).hexdigest(),
                  payloads_verified=len(targets), compressed_payloads=sum(r['compression']=='zstd' for r in storage), deltas=catalog['delta_payloads'],
                  logical_aliases=len(catalog['aliases']), exports=sorted(names), complete=True)
    if baseline_paths is not None: report['baseline_paths_verified'] = baseline_paths
    if source_8k_targets is not None:
        report['source_8k_targets_preserved'] = source_8k_targets
        report['embedded_8k_only_targets'] = 0
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
