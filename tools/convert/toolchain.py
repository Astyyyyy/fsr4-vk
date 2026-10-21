# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Build pinned host tools and resolve explicit overrides."""

import ctypes.util
from functools import lru_cache
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]

def find_tool(name, override=None, environment=None):
    selected = override or (os.environ.get(environment) if environment else None)
    if selected:
        found = shutil.which(str(Path(selected).expanduser()))
        if found:
            return str(Path(found).resolve())
        raise ValueError(f"Executable not found: {selected}")

    if name == "dxil-spirv" and (ROOT / "third_party/dxil-spirv/CMakeLists.txt").is_file():
        return build_translator()

    if name in ("spirv-as", "spirv-dis", "spirv-opt", "spirv-val"):
        return str(patching_tools() / host_executable(name))
    if name in ("glslang", "glslangValidator"):
        return str(build_glslang() / host_executable("glslang"))

    found = shutil.which(name)
    if found:
        return str(Path(found).resolve())

    directories = []
    if os.environ.get("VULKAN_SDK"):
        sdk = Path(os.environ["VULKAN_SDK"]).expanduser()
        directories.extend((sdk / "bin", sdk / "Bin"))
    for base in (ROOT / "toolchains", ROOT.parent / "toolchains"):
        if name == "dxil-spirv":
            tool = base / "dxil-spirv"
            directories.extend((tool, tool / "bin", tool / "build", tool / "build/cli"))
    executable = name + (".exe" if os.name == "nt" else "")
    for directory in directories:
        found = shutil.which(str(directory / executable))
        if found:
            return str(Path(found).resolve())
    if name == "dxil-spirv":
        raise ValueError(
            "dxil-spirv submodule is missing; run git submodule update --init --recursive"
        )
    hint = f" or set {environment}" if environment else ""
    raise ValueError(f"{name} not found; install it on PATH{hint}, or provide its path")

def build_translator():
    source = ROOT / "third_party/dxil-spirv"
    build = ROOT / "build/tools/dxil-spirv"
    cmake = shutil.which("cmake")
    if not cmake:
        raise ValueError("CMake is required to build the dxil-spirv submodule")
    if not (source / "third_party/SPIRV-Tools/CMakeLists.txt").is_file():
        raise ValueError("Initialize dxil-spirv dependencies: git submodule update --init --recursive")
    try:
        if not (build / "CMakeCache.txt").is_file():
            subprocess.run(
                [cmake, "-S", str(source), "-B", str(build),
                 "-DCMAKE_BUILD_TYPE=Release", "-DDXIL_SPIRV_CLI=ON",
                 "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"],
                check=True,
            )
        subprocess.run(
            [cmake, "--build", str(build), "--config", "Release", "--parallel", "4",
             "--target", "dxil-spirv", "dxil-spirv-c-shared"],
            check=True,
        )
    except subprocess.CalledProcessError as error:
        raise ValueError("Building the bundled dxil-spirv translator failed") from error
    executable = "dxil-spirv.exe" if os.name == "nt" else "dxil-spirv"
    for directory in (build / "Release", build):
        path = directory / executable
        if path.is_file():
            return str(path.resolve())
    raise ValueError("dxil-spirv build did not produce its executable")

def find_library(override=None, translator=None):
    selected = override or os.environ.get("DXIL_SPIRV_LIBRARY")
    if selected:
        path = Path(selected).expanduser()
        if not path.is_file():
            raise ValueError(f"Shared library not found: {selected}")
        return str(path.resolve())

    directories = []
    if os.environ.get("DXIL_SPIRV_LIBRARY_DIR"):
        directories.append(Path(os.environ["DXIL_SPIRV_LIBRARY_DIR"]).expanduser())
    if translator is None:
        try:
            translator = find_tool("dxil-spirv", environment="DXIL_SPIRV")
        except ValueError:
            pass
    if translator:
        binary = Path(translator).resolve().parent
        directories.extend(
            (binary, binary.parent, binary.parent / "lib", binary.parent / "lib64")
        )
    pattern = (
        "*dxil-spirv*c*shared*.dll" if os.name == "nt"
        else "libdxil-spirv-c-shared*.dylib" if sys.platform == "darwin"
        else "libdxil-spirv-c-shared.so*"
    )
    for directory in directories:
        for candidate in sorted(directory.glob(pattern)):
            if candidate.is_file():
                return str(candidate.resolve())
    found = ctypes.util.find_library("dxil-spirv-c-shared")
    if found:
        return found
    raise ValueError(
        "dxil-spirv C shared library not found; install it beside the translator "
        "or in its ../lib directory, or set DXIL_SPIRV_LIBRARY"
    )

def host_executable(name):
    return name + (".exe" if os.name == "nt" else "")


@lru_cache(maxsize=None)
def build_host_tools(name, options, targets):
    source = ROOT / "third_party" / name
    if not (source / "CMakeLists.txt").is_file():
        raise ValueError(f"{name} submodule is missing; run git submodule update --init --recursive")
    cmake = shutil.which("cmake")
    if not cmake:
        raise ValueError(f"CMake is required to build {name}")
    build = ROOT / "build/tools" / name
    binary = build / "bin"
    try:
        # Configure separately so tools run on the host during Windows cross-builds.
        subprocess.run([
            cmake, "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5", "-DBUILD_SHARED_LIBS=OFF",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={binary}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE={binary}", *options,
        ], check=True)
        subprocess.run([
            cmake, "--build", str(build), "--config", "Release", "--parallel", "4",
            "--target", *targets,
        ], check=True)
    except subprocess.CalledProcessError as error:
        raise ValueError(f"Building {name} host tools failed") from error
    return binary.resolve()


def patching_tools(override=None):
    """Build the pinned assembler, disassembler, optimizer and validator."""
    names = ("spirv-as", "spirv-dis", "spirv-opt", "spirv-val")
    override = override or os.environ.get("FSR4_SPIRV_TOOLS")
    if override:
        binary = Path(override)
    else:
        headers = ROOT / "third_party/SPIRV-Headers"
        if not (headers / "CMakeLists.txt").is_file():
            raise ValueError("SPIRV-Headers submodule is missing; run git submodule update --init --recursive")
        binary = build_host_tools("SPIRV-Tools", (
            "-DSPIRV_SKIP_TESTS=ON", "-DSPIRV_WERROR=OFF", "-DSPIRV_SKIP_EXECUTABLES=OFF",
            f"-DSPIRV-Headers_SOURCE_DIR={headers}",
        ), names)
    if not all((binary / host_executable(name)).is_file() for name in names):
        raise ValueError("Incomplete SPIRV-Tools directory: " + str(binary))
    return binary.resolve()


def build_glslang():
    binary = build_host_tools("glslang", (
        "-DBUILD_EXTERNAL=OFF", "-DGLSLANG_TESTS=OFF", "-DGLSLANG_ENABLE_INSTALL=OFF",
        "-DENABLE_GLSLANG_BINARIES=ON", "-DENABLE_OPT=OFF", "-DENABLE_HLSL=OFF",
    ), ("glslang-standalone",))
    if not (binary / host_executable("glslang")).is_file():
        raise ValueError("glslang build did not produce its executable")
    return binary
