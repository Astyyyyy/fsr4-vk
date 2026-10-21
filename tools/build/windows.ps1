# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

param([string]$OutputDir='', [string]$AssetRoot='')
$ErrorActionPreference='Stop'
$repo=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if (!$AssetRoot) { $AssetRoot=Join-Path $repo 'assets' }
if (!$OutputDir) { $OutputDir=Join-Path $repo 'build/windows-native' }
& cmake -S $repo -B $OutputDir -A x64 "-DFSR4_ASSET_ROOT=$AssetRoot"
if ($LASTEXITCODE -ne 0) { throw 'Configuration/asset validation failed.' }
& cmake --build $OutputDir --config Release --parallel
if ($LASTEXITCODE -ne 0) { throw 'Provider build failed.' }
