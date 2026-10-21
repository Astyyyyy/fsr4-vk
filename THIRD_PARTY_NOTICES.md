# Third-party notices

Original fsr4-vk code is licensed under GPL-3.0-only; see [LICENSE](LICENSE)
and the [README licensing section](README.md#licensing). Imported material
retains its applicable terms, copyright notices and attribution.

## AMD FidelityFX SDK: shader and model assets

The FSR 4.1.1 shaders, weights and initializers originate from
`Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_dx12.dll` in
[SDK 2.3.0](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/releases/tag/v2.3.0),
commit `60f4ea81909200d8542eca14dccb2628b763a9a3`. The official release DLL was
verified against the extractor's required version **4.1.1.2740** and SHA-256:

```text
d0dcccc74a43c44ba435b7a369b456e0970d8a4464e4bd683119b374f2c9fb46
```

The [SDK licence at that commit](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/blob/60f4ea81909200d8542eca14dccb2628b763a9a3/Kits/FidelityFX/docs/license.md)
explicitly includes this DLL in the list of MIT-licensed exceptions. A complete,
unmodified copy is in [LICENSES/AMD-FidelityFX-SDK-license.md](LICENSES/AMD-FidelityFX-SDK-license.md).
The applicable MIT copyright and permission notice is also reproduced in
[LICENSES/AMD-FidelityFX-SDK-MIT.md](LICENSES/AMD-FidelityFX-SDK-MIT.md).
The SDK's restrictive default is not the licence for this explicitly listed DLL.

## FFX API headers and earlier implementation

The headers under `third_party/fidelityfx-api/` were imported from FidelityFX SDK
commit `01446e6a74888bf349652fcf2cbf5f642d30c2bf`. They retain their inline AMD MIT
notices. These older header sources are separate from the SDK 2.3.0 DLL used
for the 4.1.1 shader/model assets; it is not an attribution of those assets to
SDK 2.0.0.

AMD, FidelityFX and FSR are trademarks of Advanced Micro Devices, Inc. This
project is independent and does not claim AMD sponsorship or endorsement.

## dxil-spirv

`third_party/dxil-spirv` is a pinned submodule of
[HansKristian-Work/dxil-spirv](https://github.com/HansKristian-Work/dxil-spirv).
It is used for build-time shader conversion and is not linked into the provider.
Its MIT notice is reproduced in [LICENSES/dxil-spirv-MIT.txt](LICENSES/dxil-spirv-MIT.txt).
Nested dependencies retain their own licence files in the recursive submodules.

## Vulkan headers and loader exports

`third_party/Vulkan-Headers` and `third_party/Vulkan-Loader` are official Khronos
submodules pinned to `vulkan-sdk-1.4.357.0`. The headers are used under their MIT
option; see [LICENSES/Vulkan-Headers-MIT.txt](LICENSES/Vulkan-Headers-MIT.txt).
The loader's Apache-2.0 export definitions generate the Windows import library;
see [LICENSES/Vulkan-Loader-Apache-2.0.txt](LICENSES/Vulkan-Loader-Apache-2.0.txt).
No loader implementation is compiled or bundled. The provider uses the Vulkan
loader installed on the user's system.

## Shader build tools

`third_party/SPIRV-Tools` and `third_party/SPIRV-Headers` are pinned independently
of dxil-spirv's nested dependencies to reproduce the shader catalog. Their
licence texts are in [LICENSES/SPIRV-Tools-Apache-2.0.txt](LICENSES/SPIRV-Tools-Apache-2.0.txt)
and [LICENSES/SPIRV-Headers.txt](LICENSES/SPIRV-Headers.txt).

`third_party/glslang` is pinned to 16.4.0 and compiles the overlay shader. Its
licence texts are in [LICENSES/glslang.txt](LICENSES/glslang.txt).
These tools are built for the host and are not linked into the provider.

## Overlay font

`third_party/fonts/DejaVuSans-Bold.ttf` is the unmodified DejaVu Sans Bold font,
used to bake the diagnostic overlay's ASCII distance field. The complete
Bitstream/DejaVu/Arev copyright and font terms are reproduced in
[LICENSES/DejaVu-fonts.txt](LICENSES/DejaVu-fonts.txt) and apply to the font and
derived atlas data. The project GPL default does not replace those terms.

`tools/overlay/build-font.py` produces `src/diagnostics/overlay_font.hpp`.
Pillow, NumPy and SciPy are external regeneration tools, not runtime dependencies.

## Zstandard

`third_party/zstd/` vendors the upstream
[v1.5.7](https://github.com/facebook/zstd/tree/v1.5.7) headers and generated
standalone decoder under its BSD-3-Clause licence option. The complete notice,
redistribution conditions, non-endorsement clause and disclaimer are in
[LICENSES/Zstandard-BSD.txt](LICENSES/Zstandard-BSD.txt); the vendored README
records generation and hashes. Only the decoder is compiled into the provider.
Python's `compression.zstd` or the `zstandard` package compresses assets at build
time and is not a runtime dependency.
