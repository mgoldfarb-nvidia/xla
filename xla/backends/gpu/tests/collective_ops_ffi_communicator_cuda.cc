/* Copyright 2025 The OpenXLA Authors.

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

#include <cstddef>
#include <cstdint>

#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "third_party/gpus/cuda/include/driver_types.h"
#include "third_party/nccl/nccl.h"         // IWYU pragma: keep
#include "third_party/nccl/nccl_device.h"  // IWYU pragma: keep
#include "xla/backends/gpu/tests/collective_ops_ffi_kernels.h"
#include "xla/ffi/api/collectives_api.h"
#include "xla/ffi/api/collectives_c_api.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"

namespace xla::gpu {

absl::Status DeviceCommunicatorAllReduceU32(
    stream_executor::Stream* stream, ffi::DeviceCommunicatorLookup communicator,
    ffi::WindowLookup src, ffi::WindowLookup dst, size_t count) {
  TF_RET_CHECK(communicator.device_communicator != nullptr);
  TF_RET_CHECK(communicator.byte_size == sizeof(ncclDevComm));
  TF_RET_CHECK(communicator.version == NCCL_VERSION_CODE);
  TF_RET_CHECK(src.window != nullptr && dst.window != nullptr);

  const auto* dev_comm =
      reinterpret_cast<const ncclDevComm*>(communicator.device_communicator);
  ABSL_ASSIGN_OR_RETURN(
      auto kernel, stream_executor::gpu::GpuKernelRegistry::GetGlobalRegistry()
                       .LoadKernel<SymmetricAllReduce>(stream->parent()));
  auto args = stream_executor::PackKernelArgs(
      /*shmem_bytes=*/0, *dev_comm,
      stream_executor::DeviceAddressBase(reinterpret_cast<void*>(src.window)),
      stream_executor::DeviceAddressBase(reinterpret_cast<void*>(dst.window)),
      src.offset, dst.offset, count);
  ABSL_RETURN_IF_ERROR(kernel->Launch(stream_executor::ThreadDim(8),
                                      stream_executor::BlockDim(1), stream,
                                      *args));
  return stream->BlockHostUntilDone();
}

absl::Status CommunicatorAllReduceU32(stream_executor::Stream* stream,
                                      XLA_FFI_Communicator* communicator,
                                      const void* send_buffer,
                                      void* recv_buffer, int64_t count) {
  ncclComm_t nccl_comm = reinterpret_cast<ncclComm_t>(communicator);
  cudaStream_t cuda_stream =
      absl::bit_cast<cudaStream_t>(stream->platform_specific_handle().stream);

  ncclResult_t result =
      ncclAllReduce(send_buffer, recv_buffer, count, ncclUint32, ncclSum,
                    nccl_comm, cuda_stream);
  TF_RET_CHECK(result == ncclSuccess)
      << "ncclAllReduce failed: " << ncclGetErrorString(result);
  return stream->BlockHostUntilDone();
}

absl::StatusOr<void*> GetWindowPeerDevicePointer(XLA_FFI_Window* window,
                                                 size_t window_offset,
                                                 int peer) {
#if (NCCL_VERSION_CODE >= 22902) || defined(USE_NCCL_HOST_API)
  ncclWindow_t nccl_win = reinterpret_cast<ncclWindow_t>(window);
  void* ptr = nullptr;
  // NOLINTBEGIN(misc-include-cleaner): comes from <nccl_device.h>.
  ncclResult_t r =
      ncclGetPeerDevicePointer(nccl_win, window_offset, peer, &ptr);
  // NOLINTEND(misc-include-cleaner)
  TF_RET_CHECK(r == ncclSuccess) << "ncclGetPeerDevicePointer(peer=" << peer
                                 << ") failed: " << ncclGetErrorString(r);
  return ptr;
#else
  return absl::UnimplementedError(
      "GetWindowPeerDevicePointer requires NCCL >= 2.29.2");
#endif
}

}  // namespace xla::gpu
