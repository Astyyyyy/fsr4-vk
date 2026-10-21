# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Convert the auto-exposure shader using dxil-spirv's C API."""

import argparse
import ctypes as c
from pathlib import Path

U = c.c_uint
B = c.c_ubyte
P = c.c_void_p


class D3DBinding(c.Structure):
    _fields_ = [
        (name, U)
        for name in (
            "stage",
            "kind",
            "resource_index",
            "register_space",
            "register_index",
            "range_size",
            "alignment",
        )
    ]


class Heap(c.Structure):
    _fields_ = [("heap_root_offset", U), ("use_heap", B)]


class VulkanBinding(c.Structure):
    _fields_ = [
        ("set", U),
        ("binding", U),
        ("root_constant_index", U),
        ("bindless", Heap),
        ("descriptor_type", U),
    ]


class UAVSource(c.Structure):
    _fields_ = [("d3d_binding", D3DBinding), ("has_counter", B)]


class SRVTarget(c.Structure):
    _fields_ = [("buffer_binding", VulkanBinding), ("offset_binding", VulkanBinding)]


class UAVTarget(c.Structure):
    _fields_ = [
        ("buffer_binding", VulkanBinding),
        ("counter_binding", VulkanBinding),
        ("offset_binding", VulkanBinding),
    ]


class CBVTarget(c.Structure):
    # Size the C union for its largest member, VulkanBinding.
    _fields_ = [("uniform_binding", VulkanBinding), ("push_constant", B)]


class Compiled(c.Structure):
    _fields_ = [("data", P), ("size", c.c_size_t)]


class BoolOption(c.Structure):
    _fields_ = [("type", U), ("enabled", B)]


class SourceOption(c.Structure):
    _fields_ = [("type", U), ("name", c.c_char_p)]


def convert(source, output, library=None):
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError("Use a fresh auto-exposure output path")
    from toolchain import find_library

    library = library or find_library()
    api = c.CDLL(str(library))

    def function(name, result, *arguments):
        f = getattr(api, "dxil_spv_" + name)
        f.restype, f.argtypes = result, list(arguments)
        return f

    version = function("get_version", None, c.POINTER(U), c.POINTER(U), c.POINTER(U))
    major, minor, patch = U(), U(), U()
    version(c.byref(major), c.byref(minor), c.byref(patch))
    if major.value != 2 or minor.value < 72:
        raise ValueError(
            "Unsupported dxil-spirv C ABI; requires version 2.72 or later in major 2"
        )
    parse = function("parse_dxil_blob", c.c_int, P, c.c_size_t, c.POINTER(P))
    create = function("create_converter", c.c_int, P, c.POINTER(P))
    free_blob = function("parsed_blob_free", None, P)
    free_converter = function("converter_free", None, P)
    add_option = function("converter_add_option", c.c_int, P, P)
    run = function("converter_run", c.c_int, P)
    compiled = function("converter_get_compiled_spirv", c.c_int, P, c.POINTER(Compiled))

    def check(status):
        if status != 0:
            raise RuntimeError(f"dxil-spirv operation failed: {status}")

    def image(source_binding, root):
        return VulkanBinding(1, 1, root, Heap(source_binding.register_index, 1), 0)

    srv_type = c.CFUNCTYPE(B, P, c.POINTER(D3DBinding), c.POINTER(SRVTarget))
    uav_type = c.CFUNCTYPE(B, P, c.POINTER(UAVSource), c.POINTER(UAVTarget))
    cbv_type = c.CFUNCTYPE(B, P, c.POINTER(D3DBinding), c.POINTER(CBVTarget))

    @srv_type
    def srv(_, source_binding, target):
        c.memset(target, 0, c.sizeof(SRVTarget))
        target.contents.buffer_binding = image(source_binding.contents, 1)
        return 1

    @uav_type
    def uav(_, source_binding, target):
        c.memset(target, 0, c.sizeof(UAVTarget))
        target.contents.buffer_binding = image(source_binding.contents.d3d_binding, 0)
        return 1

    @cbv_type
    def cbv(_, source_binding, target):
        if (
            source_binding.contents.register_space
            or source_binding.contents.register_index
        ):
            return 0
        c.memset(target, 0, c.sizeof(CBVTarget))
        target.contents.uniform_binding.descriptor_type = 3  # BUFFER_DEVICE_ADDRESS
        return 1

    parsed, converter = P(), P()
    data = source.read_bytes()
    if not data:
        raise ValueError("Empty input DXBC")
    buffer = c.create_string_buffer(data)
    try:
        check(parse(buffer, len(data), c.byref(parsed)))
        check(create(parsed, c.byref(converter)))
        # Keep Python callbacks alive while the C API uses them.
        for name, callback, callback_type in [
            ("srv", srv, srv_type),
            ("uav", uav, uav_type),
            ("cbv", cbv, cbv_type),
        ]:
            function("converter_set_" + name + "_remapper", None, P, callback_type, P)(
                converter, callback, None
            )
        function("converter_set_root_constant_word_count", None, P, U)(converter, 2)
        function("converter_set_root_descriptor_count", None, P, U)(converter, 1)
        for option in (
            BoolOption(7, 1),
            BoolOption(10, 1),
            SourceOption(11, b"996136f00380aba8.dxil"),
        ):
            check(add_option(converter, c.byref(option)))
        check(run(converter))
        result = Compiled()
        check(compiled(converter, c.byref(result)))
        if not result.data or not result.size:
            raise RuntimeError("Converter returned no SPIR-V")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(c.string_at(result.data, result.size))
    finally:
        if converter:
            free_converter(converter)
        if parsed:
            free_blob(parsed)


def main():
    from toolchain import find_library

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--library", type=Path)
    args = parser.parse_args()
    try:
        convert(args.source, args.output, find_library(args.library))
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
