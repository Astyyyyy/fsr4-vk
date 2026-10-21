#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mingw_location=""
cmake_args=()
while (($#)); do
    case "$1" in
        --mingw-location)
            if (($# < 2)) || [[ -z "$2" ]]; then
                echo "--mingw-location requires the directory containing the MinGW tools" >&2
                exit 1
            fi
            mingw_location="$2"
            shift 2
            ;;
        --mingw-location=*)
            mingw_location="${1#*=}"
            if [[ -z "$mingw_location" ]]; then
                echo "--mingw-location requires the directory containing the MinGW tools" >&2
                exit 1
            fi
            shift
            ;;
        -h|--help)
            echo "Usage: bash tools/build/windows.sh [--mingw-location /path/to/bin] [CMake arguments...]"
            echo "Requires MinGW-w64 on PATH, or provide its tools directory with --mingw-location."
            exit 0
            ;;
        --) shift; cmake_args+=("$@"); break ;;
        *) cmake_args+=("$1"); shift ;;
    esac
done

require_tool() {
    local executable
    if ! executable="$(command -v "$1")"; then
        echo "Required tool not found: $1" >&2
        exit 1
    fi
    if [[ "$executable" != /* ]]; then
        executable="$PWD/$executable"
    fi
    printf '%s\n' "$executable"
}

mingw_prefix="x86_64-w64-mingw32-"
if [[ -n "$mingw_location" ]]; then
    mingw_prefix="${mingw_location%/}/$mingw_prefix"
fi
build_dir="$repo_root/build/windows"
compiler="$(require_tool "${mingw_prefix}g++")"
resource_compiler="$(require_tool "${mingw_prefix}windres")"
dlltool="$(require_tool "${mingw_prefix}dlltool")"
require_tool cmake >/dev/null
cmake -S "$repo_root" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_SYSTEM_NAME=Windows \
    -DCMAKE_CXX_COMPILER="$compiler" \
    -DCMAKE_RC_COMPILER="$resource_compiler" \
    -DCMAKE_DLLTOOL="$dlltool" \
    -DFSR4_ASSET_ROOT="$repo_root/assets" "${cmake_args[@]}"
cmake --build "$build_dir" --parallel "${FSR4_BUILD_JOBS:-4}"
