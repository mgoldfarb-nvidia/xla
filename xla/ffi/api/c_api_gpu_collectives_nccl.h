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

#ifndef XLA_FFI_API_C_API_GPU_COLLECTIVES_NCCL_H_
#define XLA_FFI_API_C_API_GPU_COLLECTIVES_NCCL_H_

#include <stdint.h>
#include "xla/ffi/api/c_api_gpu_collectives.h"

// Stable schemas for NCCL device kernel arguments returned by the generic GPU
// device-communication API. These constants identify the concrete device ABI;
// they do not select NCCL or configure communication resources.
//
// Callers include nccl.h and nccl_device.h, and use NCCL_VERSION_CODE as the
// expected ABI version. The device-communicator destination is an ncclDevComm
// with destination_size == sizeof(ncclDevComm). The symmetric-memory
// destination is an ncclWindow_t with destination_size ==
// sizeof(ncclWindow_t).
//
// Team barrier logical slot i maps to NCCL barrier index i. Local barrier
// logical slots are placed after all team slots; use the helper below to map a
// local logical slot to its NCCL LSA barrier index. Notification slot i maps to
// GIN signal i and completion slot i maps to GIN counter i within each selected
// GIN context. XLA enables GIN when the semantic requirements need it and
// registers every tagged symmetric window during resource preparation; no
// FFI-side GIN registration is required.
#define XLA_FFI_GpuCollective_NCCL_DEVICE_COMM_ABI_SCHEMA \
  UINT64_C(0x4e43434c44433031)  // ASCII "NCCLDC01"
#define XLA_FFI_GpuCollective_NCCL_SYMMETRIC_MEMORY_ABI_SCHEMA \
  UINT64_C(0x4e43434c574e3031)  // ASCII "NCCLWN01"

// The caller must validate logical_local_slot < info->local_barrier_count.
static inline uint32_t XLA_FFI_GpuCollective_NcclLocalBarrierIndex(
    const XLA_FFI_GpuDeviceCommunication_Info* info,
    uint32_t logical_local_slot) {
  return info->team_barrier_count + logical_local_slot;
}

#endif  // XLA_FFI_API_C_API_GPU_COLLECTIVES_NCCL_H_
