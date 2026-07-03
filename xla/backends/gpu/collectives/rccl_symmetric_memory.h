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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_RCCL_SYMMETRIC_MEMORY_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_RCCL_SYMMETRIC_MEMORY_H_

#include <memory>
#include <string>
#include <thread>

#include "absl/status/statusor.h"
#include "xla/backends/gpu/collectives/rccl_types.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/executor.h"

namespace xla::gpu {

// An RCCL window registration handle that makes local buffers accessible from
// remote peers via symmetric memory registration. Analogous to
// NcclSymmetricMemory for the ROCm/RCCL platform.
class RcclSymmetricMemory final : public SymmetricMemory {
 public:
  ~RcclSymmetricMemory() final;

  static absl::StatusOr<std::unique_ptr<RcclSymmetricMemory>> Create(
      std::shared_ptr<RcclCommState> comm,
      stream_executor::DeviceAddressBase addr,
      std::shared_ptr<tsl::Executor> executor,
      stream_executor::StreamExecutor* stream_executor);

  stream_executor::DeviceAddressBase addr() const final;

  // multimem_addr() and peer_addr() are not supported by RCCL; the base-class
  // defaults (returning Unimplemented) are used.

  ncclWindow_t win() const { return win_; }

  std::string ToString() const final;

  PackedKernelArg PackKernelArg() const final;

 private:
  RcclSymmetricMemory(std::shared_ptr<RcclCommState> comm, ncclWindow_t win,
                      stream_executor::DeviceAddressBase addr,
                      std::shared_ptr<tsl::Executor> executor,
                      stream_executor::StreamExecutor* stream_executor);

  std::shared_ptr<RcclCommState> comm_;
  ncclWindow_t win_;
  stream_executor::DeviceAddressBase addr_;
  std::shared_ptr<tsl::Executor> executor_;
  stream_executor::StreamExecutor* stream_executor_;
  std::thread::id owner_thread_id_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_RCCL_SYMMETRIC_MEMORY_H_
