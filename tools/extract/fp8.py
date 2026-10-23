#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Extract FP8 shaders and weights from the pinned FSR 4.1.1 DX12 DLL.

Presets share a standard weight bank; Ultra Performance uses a separate bank.
"""

import hashlib
import json
import struct
import subprocess
from pathlib import Path

PASS_NAMES = [
    "prepass",
    "pass0_post",
    "pass1",
    "pass1_post",
    "pass2",
    "pass2_post",
    "pass3",
    "pass3_post",
    "pass4",
    "pass4_post",
    "pass5",
    "pass5_post",
    "pass6",
    "pass6_post",
    "pass7",
    "pass7_post",
    "pass8",
    "pass8_post",
    "pass9",
    "pass9_post",
    "pass10",
    "pass10_post",
    "pass11",
    "pass11_post",
    "pass12",
    "pass12_post",
    "postpass",
    "prepass_wmma",
    "postpass_wmma",
]

PRESET_OFFSETS = {
    "native": 0x156EA60,
    "quality": 0x158EA70,
    "balanced": 0x15AEA90,
    "performance": 0x152EA50,
    "custom": 0x154EA60,
    "ultraperf": 0x150EA50,
}

def extract(args):
    data = args.dll.read_bytes()
    source_hash = hashlib.sha256(data).hexdigest()
    if (
        source_hash
        != "d0dcccc74a43c44ba435b7a369b456e0970d8a4464e4bd683119b374f2c9fb46"
    ):
        raise ValueError(
            "unsupported DLL: this extractor uses the FSR 4.1.1.2740 layout"
        )
    if args.output.exists():
        raise ValueError("use a fresh extraction output directory")

    blobs = []
    start = 0
    while True:
        pos = data.find(b"DXBC", start)
        if pos == -1 or pos + 28 > len(data):
            break
        size = struct.unpack("<I", data[pos + 24 : pos + 28])[0]
        if size > 0 and pos + size <= len(data):
            blobs.append((pos, size))
        start = pos + 1

    print(f"Found {len(blobs)} DXBC blobs in {args.dll.name}")

    rdata_raw_start = 0xA7E00
    rdata_vma = 0x1800A9000
    text_raw_start = 0x400
    text_vma = 0x180001000

    blob_map = {}
    for idx, (pos, size) in enumerate(blobs):
        va = rdata_vma + (pos - rdata_raw_start)
        blob_map[va] = (idx, pos, size)

    jt_raw = text_raw_start + (0x1800993E8 - text_vma)
    targets = [
        0x180000000 + struct.unpack("<I", data[jt_raw + i * 4 : jt_raw + i * 4 + 4])[0]
        for i in range(27)
    ]

    def read_blob_by_data_va(data_va):
        raw_off = 0x1AFDE00 + (data_va - 0x181AFF000)
        blob_va = struct.unpack("<Q", data[raw_off : raw_off + 8])[0]
        return blob_map[blob_va]

    def get_table_blob(idx_rva, shader_rva, comb=6):
        idx_raw = rdata_raw_start + (idx_rva - (rdata_vma - 0x180000000))
        shader_raw = 0x1AFDE00 + (shader_rva - 0x181AFF000)
        entry_idx = struct.unpack(
            "<I", data[idx_raw + comb * 4 : idx_raw + comb * 4 + 4]
        )[0]
        entry_raw = shader_raw + entry_idx * 0x130
        blob_va = struct.unpack("<Q", data[entry_raw : entry_raw + 8])[0]
        return blob_map[blob_va]

    tier_blobs = {"1080": {}, "2160": {}, "4320": {}}

    for p in range(1, 26):
        target = targets[p]
        cmd = [
            "llvm-objdump",
            "-d",
            f"--start-address={target}",
            f"--stop-address={target+0x700}",
            str(args.dll),
        ]
        out = subprocess.check_output(cmd).decode()
        lines = [
            l.strip()
            for l in out.splitlines()
            if l.strip() and not l.startswith(("amd_", "Disassembly", "00000001"))
        ]

        jes = []
        for line in lines[:10]:
            if "je\t" in line or "je " in line:
                for t in line.split():
                    if t.startswith("0x180"):
                        jes.append(int(t, 16))

        je_1080 = jes[0] if len(jes) > 0 else None
        je_2160 = jes[1] if len(jes) > 1 else None

        def find_shader_va(start_addr):
            for i, l in enumerate(lines):
                if f"{start_addr:x}:" in l:
                    for next_l in lines[i : i + 10]:
                        if "movq" in next_l and "# 0x181b" in next_l:
                            return int(next_l.split("#")[-1].strip(), 16)
            return None

        va_4320 = None
        for l in lines[:15]:
            if "movq" in l and "# 0x181b" in l:
                va_4320 = int(l.split("#")[-1].strip(), 16)
                break

        pname = PASS_NAMES[p]
        tier_blobs["1080"][pname] = read_blob_by_data_va(find_shader_va(je_1080))
        tier_blobs["2160"][pname] = read_blob_by_data_va(find_shader_va(je_2160))
        tier_blobs["4320"][pname] = read_blob_by_data_va(va_4320)

    # Load baseline endpoints before applying the shipping profile overrides.
    tier_blobs["1080"]["prepass"] = get_table_blob(0xE98360, 0x181B2AAE8, 6)
    tier_blobs["2160"]["prepass"] = get_table_blob(0x14EE480, 0x181B32188, 6)
    tier_blobs["4320"]["prepass"] = get_table_blob(0xEC0700, 0x181B34B88, 6)

    tier_blobs["1080"]["prepass_wmma"] = get_table_blob(0xE98360, 0x181B2AAE8, 7)
    tier_blobs["2160"]["prepass_wmma"] = get_table_blob(0x14EE480, 0x181B32188, 7)
    tier_blobs["4320"]["prepass_wmma"] = get_table_blob(0xEC0700, 0x181B34B88, 7)

    tier_blobs["1080"]["postpass"] = get_table_blob(0x12D6C30, 0x181B2FED8, 4)
    tier_blobs["2160"]["postpass"] = get_table_blob(0x13C3850, 0x181B2D1C8, 4)
    tier_blobs["4320"]["postpass"] = get_table_blob(0x1210B40, 0x181B25FA8, 4)

    tier_blobs["1080"]["postpass_wmma"] = get_table_blob(0x12D6C30, 0x181B2FED8, 4)
    tier_blobs["2160"]["postpass_wmma"] = get_table_blob(0x13C3850, 0x181B2D1C8, 4)
    tier_blobs["4320"]["postpass_wmma"] = get_table_blob(0x1210B40, 0x181B25FA8, 4)

    # Match the endpoint keys expected by the asset catalog.
    if args.endpoint_profile == "shipping":
        tier_blobs["1080"]["prepass"] = get_table_blob(0xE98360, 0x181B2AAE8, 38)
        tier_blobs["1080"]["prepass_wmma"] = get_table_blob(0xE98360, 0x181B2AAE8, 39)
        tier_blobs["2160"]["prepass"] = get_table_blob(0x14EE480, 0x181B32188, 38)
        tier_blobs["1080"]["postpass"] = get_table_blob(0x12D6C30, 0x181B2FED8, 6)
        tier_blobs["1080"]["postpass_wmma"] = get_table_blob(0x12D6C30, 0x181B2FED8, 7)
        tier_blobs["2160"]["postpass"] = get_table_blob(0x13C3850, 0x181B2D1C8, 6)

    # This legacy rcas entry is a debug shader. sdk.extract_rcas extracts sharpening.
    rcas = next((blob for blob in blobs if blob[0] == 0xDCF00), None)
    if rcas is None:
        raise ValueError("missing archived debug DXBC")
    rcas_index = blobs.index(rcas)
    for selected in tier_blobs.values():
        selected["rcas"] = (rcas_index, *rcas)

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": str(args.dll),
        "source_sha256": source_hash,
        "endpoint_profile": args.endpoint_profile,
        "tiers": {},
    }
    if args.all_blobs:
        archive = args.output / "all-blobs"
        archive.mkdir()
        manifest["all_blobs"] = []
        for offset, size in blobs:
            name = f"all-blobs/{offset:08x}.dxbc"
            (args.output / name).write_bytes(data[offset : offset + size])
            manifest["all_blobs"].append(
                {"file": name, "offset_in_dll": offset, "size": size}
            )

    spd_offset = 0xD6FD0
    spd_size = struct.unpack_from("<I", data, spd_offset + 24)[0]
    spd = data[spd_offset : spd_offset + spd_size]
    if spd[:4] != b"DXBC" or len(spd) != spd_size:
        raise ValueError("invalid shared auto-exposure DXBC")
    (args.output / "auto-exposure.dxbc").write_bytes(spd)
    manifest["auto_exposure"] = {
        "file": "auto-exposure.dxbc",
        "offset_in_dll": spd_offset,
        "size": spd_size,
        "sha256": hashlib.sha256(spd).hexdigest(),
    }

    for tier in ["1080", "2160", "4320"]:
        tier_dir = args.output / "sm66" / tier
        tier_dir.mkdir(parents=True, exist_ok=True)
        manifest["tiers"][tier] = []
        print(f"Writing {len(tier_blobs[tier])} passes to {tier_dir}/")

        for pname in (*PASS_NAMES, "rcas"):
            idx, pos, size = tier_blobs[tier][pname]
            blob = data[pos : pos + size]
            out_file = tier_dir / f"{pname}.dxbc"
            out_file.write_bytes(blob)
            digest = hashlib.sha256(blob).hexdigest()
            manifest["tiers"][tier].append(
                {
                    "pass": pname,
                    "file": out_file.relative_to(args.output).as_posix(),
                    "blob_index": idx,
                    "offset_in_dll": pos,
                    "size": size,
                    "sha256": digest,
                }
            )

    weights_dir = args.output / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    manifest["weights"] = {}
    print(f"\nExtracting 6 quality preset weight banks (128 KB each):")

    for pname, offset in PRESET_OFFSETS.items():
        wblob = data[offset : offset + 0x20000]
        wpath = weights_dir / f"weights_{pname}.bin"
        wpath.write_bytes(wblob)
        digest = hashlib.sha256(wblob).hexdigest()
        manifest["weights"][pname] = {
            "file": wpath.relative_to(args.output).as_posix(),
            "offset_in_dll": offset,
            "size": len(wblob),
            "sha256": digest,
        }
        print(
            f"  {pname:12}: {len(wblob):,} bytes at 0x{offset:x} (sha256 {digest[:8]}...)"
        )

    manifest["initializers"] = {"note": "merged with weights_*.bin in FSR 4.1.1"}

    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nExtraction complete. Manifest saved to {args.output}/manifest.json")
