#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Generate Windows FP8 model and endpoint shaders using the pinned SPIRV-Tools."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def model_recipe(tier, pass_number):
    if tier not in ('1080', '2160', '4320') or pass_number not in range(1, 13):
        raise ValueError('Unsupported model tier/pass')
    options = dict(components_first=False,
                   preserve_names=pass_number not in (3, 6))
    if pass_number in (7, 8):
        return 'loop-control-none', dict(loop_control='none')
    if pass_number in (9, 11):
        return 'accumulator-io', dict(accumulator_io=True, packed_a_words=True)
    if tier == '2160' and pass_number in (1, 2, 12):
        return 'loop-control-unroll-no-optimizer-unroll', dict(loop_pass=False, optimize=False)
    return 'baseline', options

def windows_postpass_alias(source):
    prefix = 'fp8/shaders/4.1.1/color-endpoints/'
    parts = source.removeprefix(prefix).split('/')
    if (not source.startswith(prefix) or len(parts) != 4 or
            parts[0] not in ('1080', '2160', '4320') or parts[1] != 'postpass' or
            parts[2] not in ('linear', 'nonlinear', 'srgb', 'pq') or
            parts[3] not in ('auto.spv', 'external.spv', 'auto-debug.spv', 'external-debug.spv')):
        raise ValueError('Not an FP8 colour postpass: ' + source)
    return source.replace('/color-endpoints/', '/windows/color-endpoints/', 1)

class Catalog:
    def __init__(self, root, generated=None):
        self.root = root.resolve()
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        self.deltas = load('fp8_recipe_deltas', HERE.parents[2] / 'assets/storage.py')
        self.cache = {}
        self.generated = generated

    def payload(self, index):
        if index not in self.cache:
            row = self.manifest['payloads'][index]
            relative = Path(row['file'])
            if relative.is_absolute() or '..' in relative.parts or 'extracted' in relative.parts:
                raise ValueError('Shader input must come from the active asset catalog')
            stored = (self.root / relative).read_bytes()
            if hashlib.sha256(stored).hexdigest() != row['stored_sha256']:
                raise ValueError('Stored shader changed: ' + row['file'])
            target = (stored if row['base'] is None else
                      self.deltas.reconstruct(self.payload(row['base']), stored))
            if len(target) != row['size'] or hashlib.sha256(target).hexdigest() != row['sha256']:
                raise ValueError('Shader target changed: ' + row['file'])
            self.cache[index] = target
        return self.cache[index]

    def expected_hash(self, alias):
        return self.manifest['payloads'][self.manifest['aliases'][alias]]['sha256']

    def read(self, alias):
        if self.generated is not None:
            digest = self.expected_hash(alias)
            if digest not in self.generated:
                raise ValueError('Source shader was not regenerated: ' + alias)
            data = self.generated[digest].read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError('Regenerated source changed: ' + alias)
            return data
        return self.payload(self.manifest['aliases'][alias])

def generate(assets, output, stage='all', generated=None):
    if output.exists():
        raise ValueError('Use a fresh generation output directory')
    if output.resolve().is_relative_to(assets.resolve()):
        raise ValueError('Output must be outside source assets')
    catalog = Catalog(assets, generated)
    matrix = load('windows_fp8_matrix', HERE / 'matrix_io.py')
    postpass = load('windows_fp8_postpass', HERE / 'packed_postpass_loads.py')
    vector_conversions = load('windows_fp8_vector_conversions', HERE / 'vector_conversions.py')
    output.mkdir(parents=True)
    records, mismatches = [], []
    report = dict(schema=1, complete=False, source=str(assets.resolve()),
                  model_toolchain='Vulkan SDK 1.4.328.1',
                  spirv_tools=subprocess.check_output(['spirv-opt', '--version'], text=True, stderr=subprocess.STDOUT).strip(),
                  shaders=records)

    def save():
        (output / 'generation-report.json').write_text(json.dumps(report, indent=2) + '\n')

    def emit(source_alias, target_alias, recipe, patcher, options, locked):
        source = output / 'work' / f'input-{len(records):03d}.spv'
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(catalog.read(source_alias))
        generated = output / 'work' / f'output-{len(records):03d}.spv'
        generated.parent.mkdir(parents=True, exist_ok=True)
        if patcher is None:
            shutil.copyfile(source, generated)
            patch_report = {}
        else:
            patch_report = patcher(source, generated, **options)
        patch_report.update(vector_conversions.patch(generated, generated))
        blob = generated.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        record = dict(source_alias=source_alias, alias=target_alias, recipe=recipe,
                      source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      sha256=digest, size=len(blob), patch=patch_report)
        if locked:
            expected = catalog.expected_hash(target_alias)
            record.update(expected_sha256=expected, matches_lock=digest == expected)
            if digest != expected:
                mismatches.append(target_alias)
        destination = output / 'shaders' / target_alias
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(generated, destination)
        record['file'] = destination.relative_to(output).as_posix()
        records.append(record)
        save()
        print(target_alias, 'MISMATCH' if record.get('matches_lock') is False else 'OK', flush=True)

    if stage in ('all', 'models'):
        for tier in ('1080', '2160', '4320'):
            for index in range(1, 13):
                recipe, options = model_recipe(tier, index)
                emit(f'fp8/{tier}/pass{index}.spv', f'fp8/{tier}/pass{index}.win-opt.spv',
                     recipe, matrix.patch, options, True)
        if mismatches:
            raise ValueError('Model shaders differ from the production lock. Use the historical '
                             'Vulkan SDK 1.4.328.1 tools; do not replace assets. Mismatches: ' +
                             ', '.join(mismatches))
    if stage in ('all', 'prepasses'):
        sources = sorted(key for key in catalog.manifest['aliases']
                         if key.startswith('fp8/shaders/4.1.1/color-endpoints/') and
                         '/prepass/' in key)
        if len(sources) != 144:
            raise ValueError('Expected all three tiers and colour/motion/exposure/jitter prepasses')
        for source in sources:
            target = source.replace('/color-endpoints/', '/windows/color-endpoints/', 1)
            emit(source, target, 'prepass', None, {}, False)
    if stage in ('all', 'postpasses'):
        sources = sorted(key for key in catalog.manifest['aliases']
                         if key.startswith('fp8/shaders/4.1.1/color-endpoints/') and
                         '/postpass/' in key)
        if len(sources) != 48:
            raise ValueError('Expected all three tiers and colour/exposure/debug postpasses')
        for source in sources:
            emit(source, windows_postpass_alias(source), 'packed-postpass', postpass.patch, {}, False)
    report['complete'] = True
    save()
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('assets', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--spirv-tools', type=Path, required=True,
                        help='directory with spirv-as/dis/opt/val (SDK 1.4.328.1 for model hashes)')
    parser.add_argument('--stage', choices=('all', 'models', 'prepasses', 'postpasses'), default='all')
    args = parser.parse_args()
    os.environ['PATH'] = str(args.spirv_tools.resolve()) + os.pathsep + os.environ.get('PATH', '')
    for name in ('spirv-as', 'spirv-dis', 'spirv-opt', 'spirv-val'):
        executable = args.spirv_tools / (name + ('.exe' if os.name == 'nt' else ''))
        if not executable.is_file():
            parser.error('Missing SPIR-V tool in the supplied directory: ' + str(executable))
    try:
        generate(args.assets, args.output, args.stage)
    except ValueError as error:
        parser.error(str(error))

if __name__ == '__main__':
    main()
