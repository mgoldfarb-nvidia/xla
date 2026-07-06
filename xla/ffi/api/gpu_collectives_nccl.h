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

#ifndef XLA_FFI_API_GPU_COLLECTIVES_NCCL_H_
#define XLA_FFI_API_GPU_COLLECTIVES_NCCL_H_

#include <cstdint>

#include "xla/ffi/api/c_api_gpu_collectives_nccl.h"
#include "xla/ffi/api/gpu_collectives.h"

namespace xla::ffi {

// NCCL specialization of ProviderDeviceComm. Callers supply ncclDevComm and
// NCCL_VERSION_CODE from the NCCL headers used to compile their device kernel.
//
// Example:
//   Ffi::Bind()
//       .Ctx<NcclDeviceComm<ncclDevComm, NCCL_VERSION_CODE>>()
//       .To([](ncclDevComm comm) { ... });
template <typename T, uint64_t abi_version>
using NcclDeviceComm =
    ProviderDeviceComm<T, XLA_FFI_GpuCollective_NCCL_DEVICE_COMM_ABI_SCHEMA,
                       abi_version>;

}  // namespace xla::ffi

#endif  // XLA_FFI_API_GPU_COLLECTIVES_NCCL_H_
