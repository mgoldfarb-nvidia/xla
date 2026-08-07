"""Pinned CuTeDSL runtime artifacts published with CUTLASS releases."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

CUTLASS_CUTEDSL_RUNTIME_VERSION = "4.6.1"

_ARTIFACTS = [
    struct(
        architecture = "aarch64",
        cuda_major = "12",
        sha256 = "6d09d85ade64973d91cc70b05a2067ba00873874c6ecc3095c1309216ce904c2",
    ),
    struct(
        architecture = "aarch64",
        cuda_major = "13",
        sha256 = "8dfe0548abf69e96db2fc6fa5c9cc253de43122d8b10a79f476e0e46380a0c2e",
    ),
    struct(
        architecture = "x86_64",
        cuda_major = "12",
        sha256 = "f3480065a0ed5beb916577b44b2853511d0a9a6dad1846a5941324986947bc24",
    ),
    struct(
        architecture = "x86_64",
        cuda_major = "13",
        sha256 = "94b356f853ea409fb0aa857a962b18c0c756aee24947e430101bf4f2657c5247",
    ),
]

def repo():
    for artifact in _ARTIFACTS:
        platform = "{}_cuda{}".format(
            artifact.architecture,
            artifact.cuda_major,
        )
        archive = "cutlass-install-{}-cu{}-{}.tar.gz".format(
            artifact.architecture,
            artifact.cuda_major,
            CUTLASS_CUTEDSL_RUNTIME_VERSION,
        )
        tf_http_archive(
            name = "cutlass_cutedsl_runtime_{}".format(platform),
            build_file = "//third_party/cutlass_cutedsl_runtime:cutlass_cutedsl_runtime.BUILD",
            sha256 = artifact.sha256,
            strip_prefix = "{}/cu{}".format(
                artifact.architecture,
                artifact.cuda_major,
            ),
            urls = tf_mirror_urls(
                "https://github.com/NVIDIA/cutlass/releases/download/v{}/{}".format(
                    CUTLASS_CUTEDSL_RUNTIME_VERSION,
                    archive,
                ),
            ),
        )
