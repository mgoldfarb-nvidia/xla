/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_BACKENDS_GPU_TESTS_GPU_COLLECTIVES_FFI_PLUGIN_KERNEL_H_
#define XLA_BACKENDS_GPU_TESTS_GPU_COLLECTIVES_FFI_PLUGIN_KERNEL_H_

#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>

// Launches the public-FFI DSO test's NCCL LSA all-reduce. The NCCL types stay
// opaque at this boundary so the host-side FFI handler only depends on packed
// bytes returned by the public API.
cudaError_t LaunchGpuCollectivesFfiTestKernel(cudaStream_t stream,
                                              const void* device_comm,
                                              size_t device_comm_size,
                                              const void* symmetric_memory,
                                              size_t symmetric_memory_size,
                                              uint64_t offset, size_t count);

#endif  // XLA_BACKENDS_GPU_TESTS_GPU_COLLECTIVES_FFI_PLUGIN_KERNEL_H_
