# CuTeDSL FFI for OpenXLA's CUDA PJRT plugin

The test-only `ffi` library statically registers CuTeDSL 4.6.1's current v2
non-collective handlers and state type. It lets external tests link a runtime
archive and exercise the built-in targets without Python. The CuTeDSL runtime
owns those symbols, their call ABI, and their lifecycle state; XLA contains
registration only.

The CUDA PJRT plugin does not link the non-collective registration library.
Production CuTeDSL users load and register the runtime dynamically through the
CuTeDSL Python package.

The `collective_ffi` library implements and statically registers
`__xla_gpu_cutedsl_collective_v3`. The collective handler remains in XLA
because it acquires XLA-owned communicator and symmetric-memory resources.

The collective target accepts top-level `module` bytes and their 32-byte
SHA-256 `key`, plus a `config_format="protobuf"` discriminator and a `config`
string containing the serialized `CollectiveCallConfigV3` wire message. The
complete collective configuration and module digest are validated during
Instantiate. The config records the clique width used to compile the
region-major address table. Prepare rejects a different runtime clique width
before loading the module or requesting collective resources. Initialize
resolves the absolute addresses into host metadata. Symmetric rows contain one
peer address per rank. Multimem rows contain the rank-local LSA multimem alias
from the NCCL symmetric window, repeated across the row to preserve the v3
table shape.

The generated-function frame carries one pointer to a fixed 16-byte host
`CollectiveContextAbi` descriptor instead of one argument per peer address.
The descriptor contains a host address-table view plus rank and clique size.
Generated host code resolves peer or multimem pointers while constructing TMA
descriptors or kernel arguments. Device kernels never access the metadata
table, so collective execution requires no device scratch result, host-to-device
metadata copy, or completion event. Independent regions do not need to share
an allocation layout.

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
