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

The generated-function frame carries one pointer to a fixed 16-byte host
`CollectiveContextAbi` descriptor instead of one argument per peer address.
The descriptor contains the device-table pointer, rank, and clique size.
Generated host code loads only that descriptor and constructs a row-major CuTe
global-memory tensor over the table. Device kernels index the tensor to load an
absolute peer or multimem address; independent regions do not need to share an
allocation layout.

## Runtime linkage

XLA owns the runtime ABI used by this FFI handler and declares its six
required C entry points in `runtime_api.h`. The function table used by the
module loader is private to XLA; the CuTeDSL runtime does not expose or
negotiate a separate API table.

Google builds select the combined static runtime through the
`cutedsl_runtime_static` label flag. OSS builds can compile against the ABI
header from a pinned CUTLASS release by selecting the target matching the
build architecture and CUDA major version. For example, a CUDA 13 x86_64
build uses:

```
--//xla/backends/gpu/libraries/cutedsl:cutedsl_runtime_headers=@cutlass_cutedsl_runtime_x86_64_cuda13//:headers
```

The selected release target supplies `CuteDSLRuntime.h`. Update
`third_party/cutlass_cutedsl_runtime/workspace.bzl` to bump the release version
and artifact digests. CUTLASS 4.6.1's static archive is not linked in OSS: it
contains process-load initializers and depends on a newer private libstdc++ ABI
than XLA's hermetic toolchain provides.

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
