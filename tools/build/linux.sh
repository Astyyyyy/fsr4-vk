#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cmake -S "$repo_root" -B "$repo_root/build/linux" -DCMAKE_BUILD_TYPE=Release -DFSR4_EMBEDDED_ASSETS=ON \
    -DFSR4_ASSET_ROOT="$repo_root/assets" "$@"
cmake --build "$repo_root/build/linux" --parallel "${FSR4_BUILD_JOBS:-4}"
