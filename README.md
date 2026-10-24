# fsr4-vk

An independent Vulkan implementation of FSR 4.1.1 FP8 and INT8 for Windows
and Linux, using AMD's model weights and shaders translated to SPIR-V.

This project is not affiliated with or endorsed by AMD.

## Requirements

- Python 3.10+, CMake 3.22.1+ and a C++20 compiler.
- `llvm-objdump`.
- Python 3.14+ for built-in Zstandard support, or the `zstandard` package
  (`python3 -m pip install zstandard`).

## Prepare assets

Get `Samples/Upscalers/FidelityFX_FSR/dx12/x64/Release/amd_fidelityfx_upscaler_dx12.dll` from
[AMD's FidelityFX SDK 2.3.0](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/releases/tag/v2.3.0).
The tools require 4.1.1.2740 and verify the DLL hash automatically.

```sh
git submodule update --init --recursive
python3 tools/setup.py --dll /path/to/amd_fidelityfx_upscaler_dx12.dll
```

Setup extracts the models, translates the shaders and applies the patches under
`assets/`. These generated assets are not included in the repository. Later runs
can omit `--dll` to reuse the extracted files. Required shader tools are built
automatically from the pinned submodules into `build/tools/`.

## Build

Prepare the assets before building. The scripts configure CMake and build a
Release library with the models and compressed shaders embedded, so the
`assets/` folder is not needed at runtime. Rerun the same command after source
changes for an incremental build.

### Linux

Build the native shared library using the system C++ compiler:

```sh
bash tools/build/linux.sh
```

Output: `build/linux/amd_fidelityfx_upscaler_vk.so`.

### Windows

With Visual Studio C++ support installed, run from a developer PowerShell.
The script configures an x64 build and builds the Release configuration:

```powershell
.\tools\build\windows.ps1
```

Output: `build/windows-native/Release/amd_fidelityfx_upscaler_vk.dll`.

### Windows from Linux

Cross-compile using MinGW-w64. The script expects
`x86_64-w64-mingw32-g++`, `x86_64-w64-mingw32-windres` and
`x86_64-w64-mingw32-dlltool`:

```sh
bash tools/build/windows.sh
```

Output: `build/windows/amd_fidelityfx_upscaler_vk.dll`.
Shader tools run on the build host, a separate Windows Vulkan SDK is not required.
If the tools are in another directory, pass it with `--mingw-location`:

```sh
bash tools/build/windows.sh --mingw-location /path/to/mingw/bin
```

## Licensing

Code written for this project is licensed under [GPL-3.0-only](LICENSE).
Third-party code and assets retain their respective licences. See
[LICENSES/](LICENSES/) for the licence texts and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and details.
