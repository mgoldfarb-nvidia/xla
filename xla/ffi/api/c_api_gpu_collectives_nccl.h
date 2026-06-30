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

// Stable schemas for NCCL packed kernel arguments returned by the generic GPU
// device-communication API. These constants identify the concrete device ABI;
// they do not select NCCL or configure communication resources.
#define XLA_FFI_GpuCollective_NCCL_DEVICE_COMM_ABI_SCHEMA \
  UINT64_C(0x4e43434c44433031)  // ASCII "NCCLDC01"
#define XLA_FFI_GpuCollective_NCCL_SYMMETRIC_MEMORY_ABI_SCHEMA \
  UINT64_C(0x4e43434c574e3031)  // ASCII "NCCLWN01"

#endif  // XLA_FFI_API_C_API_GPU_COLLECTIVES_NCCL_H_
