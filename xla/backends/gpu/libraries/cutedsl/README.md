# CuTeDSL FFI for OpenXLA's CUDA PJRT plugin

OpenXLA's CUDA PJRT plugin registers only the
`__xla_gpu_cutedsl_collective_v3` FFI target. CuTeDSL owns and registers its
non-collective FFI targets.

The collective target accepts top-level `module` bytes and their 32-byte
SHA-256 `key`, plus a `config_format="protobuf"` discriminator and a `config`
string containing the serialized `CollectiveCallConfigV3` wire message. The
complete collective configuration and module digest are validated during
Instantiate. The config records the clique width used to compile the
region-major address table. The first FFI result is an internal
`U64[peer_region_count, abi_clique_size]` scratch buffer allocated by XLA in
device memory; remaining results are the generated function's ordinary
results. Prepare rejects a different runtime clique width before loading the
module or requesting collective resources. Initialize resolves the absolute
addresses, and Execute copies them into the scratch result immediately before
launching the generated function on the same stream. Symmetric rows contain
one peer address per rank. Multimem rows contain the rank-local LSA multimem
alias from the NCCL symmetric window, repeated across the row to preserve the
v3 table shape.

The generated-function frame carries one pointer to a fixed 24-byte host
`CollectiveContextAbi` descriptor instead of one argument per peer address.
The descriptor contains identical pinned-host and device views of the address
table, plus rank and clique size. Generated host code can use the host view for
TMA descriptor initialization and constructs a row-major CuTe global-memory
tensor over the device view. Device kernels index the tensor to load an
absolute peer or multimem address; independent regions do not need to share an
allocation layout.

## Runtime linkage

OSS XLA compiles against `CuteDSLRuntime.h` from the pinned CUTLASS release.
The function table used by the module loader is private to XLA; the CuTeDSL
runtime does not expose or negotiate a separate API table.

Google builds select the combined static runtime through the
`cutedsl_runtime_static` label flag. OSS builds always use the ABI header from
the pinned CUTLASS release. The header is byte-identical across the release's
supported architectures and CUDA major versions. CuTeDSL support can be
removed from the GPU plugin explicitly with:

```
--//xla/backends/gpu/libraries/cutedsl:enable_cutedsl_support=false
```

`third_party/cutlass_cutedsl_runtime/workspace.bzl` to bump the release version
and artifact digest together. CUTLASS 4.6.1's static archive is not linked in
OSS: it contains process-load initializers and depends on a newer private
libstdc++ ABI than XLA's hermetic toolchain provides.

By default, OSS builds compile the handler without a link-time CuTeDSL
dependency. On first use, XLA loads `libcute_dsl_runtime.so` with
`RTLD_NOW | RTLD_LOCAL`, resolves the six required symbols, and retains the
library for the process lifetime. The platform dynamic-loader search path
locates the library, so a package or deployment can provide it through
`LD_LIBRARY_PATH` or an equivalent mechanism. Loading remains lazy and does
not affect users that do not use CuTeDSL.

Both runtime variants register CUDA helper functions directly with their ORC
JIT, so the plugin passes no runtime path in `shared_libs`. That argument
remains available for actual dependencies of generated modules.
