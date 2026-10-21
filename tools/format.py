#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Format project C/C++ files or check their formatting."""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FORMAT_MAJOR = 22
SOURCE_ROOTS = ('src', 'tools')
SUFFIXES = frozenset(('.c', '.cc', '.cpp', '.cxx', '.h', '.hh', '.hpp', '.hxx', '.inl'))


def source_files(root):
    """Find project C/C++ files without traversing generated or third-party code."""
    return sorted(path for name in SOURCE_ROOTS for path in (root/name).rglob('*')
                  if path.is_file() and not path.is_symlink() and path.suffix in SUFFIXES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='exit nonzero if formatting differs')
    parser.add_argument('--clang-format', default=os.environ.get('CLANG_FORMAT'),
                        help='formatter executable (otherwise CLANG_FORMAT or PATH)')
    args = parser.parse_args()
    formatter = args.clang_format or shutil.which(f'clang-format-{FORMAT_MAJOR}') or shutil.which('clang-format')
    if not formatter:
        parser.error(f'clang-format {FORMAT_MAJOR} is required; install it or set CLANG_FORMAT')
    try:
        version = subprocess.check_output([formatter, '--version'], text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as error:
        parser.error(f'Cannot run clang-format: {error}')
    match = re.search(r'clang-format version (\d+)\.', version)
    if not match or int(match[1]) != FORMAT_MAJOR:
        parser.error(f'Use clang-format {FORMAT_MAJOR} to avoid style churn; found {version.strip()}')
    files = source_files(ROOT)
    if not files:
        parser.error('No owned C/C++ sources found')
    mode = ['--dry-run', '--Werror', '--ferror-limit=1'] if args.check else ['-i']
    command = [formatter, f'--style=file:{ROOT/".clang-format"}',
               '--fail-on-incomplete-format', *mode, *map(str, files)]
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode:
        if args.check:
            print('Formatting check failed. Run: python3 tools/format.py', file=sys.stderr)
        return result.returncode
    print(f'{"Checked" if args.check else "Formatted"} {len(files)} owned C/C++ files with clang-format {FORMAT_MAJOR}.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
