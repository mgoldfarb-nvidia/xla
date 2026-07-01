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

#include "xla/backends/gpu/tests/gpu_collectives_ffi_plugin_kernel.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "third_party/nccl/nccl.h"
#include "third_party/nccl/nccl_device.h"
#include <cuda_runtime.h>

namespace {

// This is the public-FFI variant of the existing NcclDevAllReduce test kernel
// in collective_ops_ffi_kernels.cu.cc. It is compiled into the external DSO so
// launching it requires no StreamExecutor or XLA GPU runtime symbols.
template <typename T>
__global__ void NcclDevAllReduce(ncclDevComm dev_comm, ncclWindow_t window,
                                 size_t offset, size_t count) {
  ncclLsaBarrierSession<ncclCoopCta> barrier(ncclCoopCta(), dev_comm,
                                             ncclTeamTagLsa(), blockIdx.x);
  barrier.sync(ncclCoopCta(), cuda::memory_order_relaxed);

  const int rank = dev_comm.lsaRank;
  const int num_ranks = dev_comm.lsaSize;
  const int global_thread =
      threadIdx.x + blockDim.x * (rank + blockIdx.x * num_ranks);
  const int global_thread_count = blockDim.x * gridDim.x * num_ranks;

  for (size_t i = global_thread; i < count; i += global_thread_count) {
    T value = 0;
    for (int peer = 0; peer < num_ranks; ++peer) {
      T* input = static_cast<T*>(ncclGetLsaPointer(window, offset, peer));
      value += input[i];
    }
    for (int peer = 0; peer < num_ranks; ++peer) {
      T* output = static_cast<T*>(ncclGetLsaPointer(window, offset, peer));
      output[i] = value;
    }
  }

  // Public FFI handlers must collectively quiesce remote accesses before their
  // device work returns. This final barrier is what makes XLA's subsequent
  // local stream completion sufficient to release the registered allocations.
  barrier.sync(ncclCoopCta(), cuda::memory_order_release);
}

}  // namespace

cudaError_t LaunchGpuCollectivesFfiTestKernel(cudaStream_t stream,
                                              const void* device_comm,
                                              size_t device_comm_size,
                                              const void* symmetric_memory,
                                              size_t symmetric_memory_size,
                                              uint64_t offset, size_t count) {
  if (stream == nullptr || device_comm == nullptr ||
      device_comm_size != sizeof(ncclDevComm) || symmetric_memory == nullptr ||
      symmetric_memory_size != sizeof(ncclWindow_t) || count == 0) {
    return cudaErrorInvalidValue;
  }

  ncclDevComm unpacked_device_comm;
  ncclWindow_t unpacked_window;
  std::memcpy(&unpacked_device_comm, device_comm, sizeof(unpacked_device_comm));
  std::memcpy(&unpacked_window, symmetric_memory, sizeof(unpacked_window));

  NcclDevAllReduce<uint32_t><<<1, 8, 0, stream>>>(
      unpacked_device_comm, unpacked_window, offset, count);
  return cudaPeekAtLastError();
}

cudaError_t LaunchGpuCollectivesFfiTwoBufferTestKernel(
    cudaStream_t stream, const void* device_comm, size_t device_comm_size,
    const void* first_symmetric_memory, size_t first_symmetric_memory_size,
    uint64_t first_offset, size_t first_count,
    const void* second_symmetric_memory, size_t second_symmetric_memory_size,
    uint64_t second_offset, size_t second_count) {
  if (stream == nullptr || device_comm == nullptr ||
      device_comm_size != sizeof(ncclDevComm) ||
      first_symmetric_memory == nullptr ||
      first_symmetric_memory_size != sizeof(ncclWindow_t) || first_count == 0 ||
      second_symmetric_memory == nullptr ||
      second_symmetric_memory_size != sizeof(ncclWindow_t) ||
      second_count == 0) {
    return cudaErrorInvalidValue;
  }

  ncclDevComm unpacked_device_comm;
  ncclWindow_t first_window;
  ncclWindow_t second_window;
  std::memcpy(&unpacked_device_comm, device_comm, sizeof(unpacked_device_comm));
  std::memcpy(&first_window, first_symmetric_memory, sizeof(first_window));
  std::memcpy(&second_window, second_symmetric_memory, sizeof(second_window));

  NcclDevAllReduce<uint32_t><<<1, 8, 0, stream>>>(
      unpacked_device_comm, first_window, first_offset, first_count);
  cudaError_t launch_status = cudaPeekAtLastError();
  if (launch_status != cudaSuccess) return launch_status;

  NcclDevAllReduce<uint32_t><<<1, 8, 0, stream>>>(
      unpacked_device_comm, second_window, second_offset, second_count);
  return cudaPeekAtLastError();
}
