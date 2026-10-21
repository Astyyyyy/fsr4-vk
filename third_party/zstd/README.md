# Zstandard decoder

Pinned upstream: https://github.com/facebook/zstd/tree/v1.5.7
License selected for this project: BSD-3-Clause, in `LICENSES/Zstandard-BSD.txt`.

`zstd.h` and `zstd_errors.h` are unchanged upstream. `zstddeclib.c` is the unchanged output of the
upstream single-file decoder generator, compiled as C++ by this project:

```sh
cd zstd/build/single_file_libs
python3 combine.py -r ../../lib -x legacy/zstd_legacy.h -o zstddeclib.c zstddeclib-in.c
```

No compressor, legacy decoder or shared-library dependency is built into the provider.

Upstream v1.5.7 source archive SHA256: `37d7284556b20954e56e1ca85b80226768902e2edabd3b649e9e72c0c9012ee3`
Generated decoder SHA256: `5eb38abe0a7eea13674b312a006df72002d4192f0f0b4d09e6f03fb927a1f3d0`
