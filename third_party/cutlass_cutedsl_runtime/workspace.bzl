"""Pinned CuTeDSL runtime artifacts published with CUTLASS releases."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

CUTLASS_CUTEDSL_RUNTIME_VERSION = "4.6.1"

# CuteDSLRuntime.h is byte-identical in every architecture and CUDA variant of
# this release. Use one canonical archive so the ABI is pinned without making
# callers select a binary platform for a platform-independent header.
_ARCHITECTURE = "x86_64"
_CUDA_MAJOR = "13"
_SHA256 = "94b356f853ea409fb0aa857a962b18c0c756aee24947e430101bf4f2657c5247"

def repo():
    archive = "cutlass-install-{}-cu{}-{}.tar.gz".format(
        _ARCHITECTURE,
        _CUDA_MAJOR,
        CUTLASS_CUTEDSL_RUNTIME_VERSION,
    )
    tf_http_archive(
        name = "cutlass_cutedsl_runtime",
        build_file = "//third_party/cutlass_cutedsl_runtime:cutlass_cutedsl_runtime.BUILD",
        sha256 = _SHA256,
        strip_prefix = "{}/cu{}".format(_ARCHITECTURE, _CUDA_MAJOR),
        urls = tf_mirror_urls(
            "https://github.com/NVIDIA/cutlass/releases/download/v{}/{}".format(
                CUTLASS_CUTEDSL_RUNTIME_VERSION,
                archive,
            ),
        ),
    )
