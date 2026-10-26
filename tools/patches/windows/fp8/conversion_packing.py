#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Pack Windows FP8 conversions in groups of four, retaining finite clamps.

Used after matrix_io for all twelve passes in the 1080, 2160 and 4320 tiers. Omitting
their exceptional-value repairs allows the Windows driver to shorten the
conversion/packing schedule. NaN/Inf canonicalization therefore differs from
the original recipe; this is not an equivalent rewrite for arbitrary values.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
VECTOR_WIDTH = 4
ELEMENTS_PER_LANE = 8


def vector_type(module, component):
    """Find or declare a four-component vector type."""
    args = [component, str(VECTOR_WIDTH)]
    matches = [ident for ident, definition in module.d.items()
               if definition == ('OpTypeVector', args)]
    if len(matches) > 1:
        raise ValueError('Ambiguous FP8 conversion vector type')
    if matches:
        return matches[0]
    ident = module.fresh()
    module.d[ident] = ('OpTypeVector', args)
    module.globals.append(f'{ident} = OpTypeVector {component} {VECTOR_WIDTH}')
    return ident


def pack_matrix_conversions(module):
    """Convert each lane's eight FP32 matrix elements as two four-wide vectors."""
    float_vector = vector_type(module, module.float)
    fp8_vector = vector_type(module, module.fp8)
    conversions = 0

    def convert(match):
        nonlocal conversions
        result, target, value = match.groups()
        if target not in module.mats or module.mats[target][0] != module.fp8:
            return match[0]
        source_type = module.d[value][1][0]
        if (source_type not in module.mats or module.mats[source_type][0] != module.float
                or module.mats[source_type][1:4] != module.mats[target][1:4]):
            raise ValueError('Unsupported FP32-to-FP8 matrix conversion')
        conversions += 1
        lines = []
        previous = module.undef(target)
        for start in range(0, ELEMENTS_PER_LANE, VECTOR_WIDTH):
            elements = [module.emit(lines, 'OpCompositeExtract', module.float, value, i)
                        for i in range(start, start + VECTOR_WIDTH)]
            vector = module.emit(lines, 'OpCompositeConstruct', float_vector, *elements)
            converted = module.emit(lines, 'OpFConvert', fp8_vector, vector)
            for component in range(VECTOR_WIDTH):
                scalar = module.emit(lines, 'OpCompositeExtract', module.fp8, converted, component)
                index = start + component
                # Preserve the original result ID on the final matrix insertion.
                ident = result if index == ELEMENTS_PER_LANE - 1 else module.fresh()
                lines.append(f'{ident} = OpCompositeInsert {target} {scalar} {previous} {index}')
                previous = ident
        return '\n'.join(lines)

    module.text = re.sub(
        r'^\s*(%\w+) = OpFConvert (%\w+) (%\w+)\s*$', convert, module.text, flags=re.M)
    return conversions


def omit_exceptional_value_repairs(module):
    """Replace select(isnan(x) || isinf(x), NaN, clamp(x)) with the existing clamp."""

    repairs = 0

    def finite_clamp(match):
        nonlocal repairs
        result, result_type, condition, nan, clamped = match.groups()
        if (result_type != module.float
                or module.d[nan] != ('OpConstant', [module.float, '0x1.8p+128'])):
            return match[0]
        predicate_op, predicate_args = module.d[condition]
        if (predicate_op != 'OpLogicalOr' or len(predicate_args) != 3
                or predicate_args[0] != module.bool):
            raise ValueError('Unknown FP8 special-value predicate')
        nan_op, nan_args = module.d[predicate_args[1]]
        inf_op, inf_args = module.d[predicate_args[2]]
        if nan_op != 'OpIsNan' or inf_op != 'OpIsInf' or nan_args != inf_args:
            raise ValueError('Unknown FP8 NaN/Inf predicate')

        # Only omit the repair when its other branch clamps the same input to +/-448.
        clamp_op, clamp_args = module.d[clamped]
        if (clamp_op != 'OpExtInst' or len(clamp_args) != 6 or clamp_args[0] != module.float
                or module.d[clamp_args[1]] != ('OpExtInstImport', ['"GLSL.std.450"'])
                or clamp_args[2:4] != ['NClamp', nan_args[1]]
                or module.d[clamp_args[4]] != ('OpConstant', [module.float, '-448'])
                or module.d[clamp_args[5]] != ('OpConstant', [module.float, '448'])):
            raise ValueError('Expected the original FP8 +/-448 finite clamp')
        repairs += 1
        return f'{result} = OpCopyObject {result_type} {clamped}'

    module.text = re.sub(
        r'(%\w+) = OpSelect (%\w+) (%\w+) (%\w+) (%\w+)', finite_clamp, module.text)
    return repairs


def patch(source, output):
    source, output = Path(source), Path(output)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    spec = importlib.util.spec_from_file_location('fp8_conversion_matrix', HERE / 'matrix_io.py')
    matrix = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = matrix
    spec.loader.exec_module(matrix)
    module = matrix.Module(matrix.run(['spirv-dis', '--raw-id', source]))
    if 'SaturatedToLargestFloat8NormalConversionEXT' in module.text:
        raise ValueError('Expected explicit finite clamps before unsaturated FP8 conversion')

    conversions = pack_matrix_conversions(module)
    if not conversions:
        raise ValueError('Expected FP32-to-FP8 matrix conversions')
    repairs = omit_exceptional_value_repairs(module)
    if not 0 < repairs <= conversions * ELEMENTS_PER_LANE:
        raise ValueError(f'Unexpected scalar conversion repair count: {repairs}')

    output.parent.mkdir(parents=True, exist_ok=True)
    assembly = output.with_suffix('.spvasm')
    assembly.write_text(module.finish(), encoding='utf-8', newline='\n')
    matrix.run(['spirv-as', assembly, '--target-env', 'vulkan1.3', '-o', output])
    matrix.run(['spirv-val', output, '--target-env', 'vulkan1.3'])
    return dict(matrix_conversions=conversions, vector_width=VECTOR_WIDTH, finite_clamps='retained',
                exceptional_value_repairs_omitted=repairs, source_sha256=digest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(json.dumps(patch(args.input, args.output), indent=2))


if __name__ == '__main__':
    main()
