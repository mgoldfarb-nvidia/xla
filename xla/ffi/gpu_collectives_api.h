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

#ifndef XLA_FFI_GPU_COLLECTIVES_API_H_
#define XLA_FFI_GPU_COLLECTIVES_API_H_

#include "absl/status/status.h"
#include "xla/ffi/api/c_api_gpu_collectives.h"

namespace xla::ffi {

// Backend-neutral interface used by the public GPU device-communication C API.
// A GPU execution installs an execution-scoped implementation in the FFI
// execution context; the generic C callbacks validate the argument struct size
// and execution stage before forwarding resource-copy operations.
class GpuCollectivesApi {
 public:
  virtual ~GpuCollectivesApi() = default;

  virtual absl::Status GetDeviceComm(
      XLA_FFI_GpuCollectives_GetDeviceComm_Args* args) = 0;
  virtual absl::Status GetDeviceMemory(
      XLA_FFI_GpuCollectives_GetDeviceMemory_Args* args) = 0;
};

}  // namespace xla::ffi

#endif  // XLA_FFI_GPU_COLLECTIVES_API_H_
