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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_NCCL_SYMMETRIC_MEMORY_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_NCCL_SYMMETRIC_MEMORY_H_

#include <cstdint>
#include <memory>
#include <string>
#include <thread>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "third_party/nccl/nccl.h"
#include "xla/backends/gpu/collectives/nccl_types.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/executor.h"

namespace xla::gpu {

// Stable schema identifier for a packed ncclWindow_t kernel argument.
inline constexpr uint64_t kNcclWindowAbiSchema =
    uint64_t{0x4e43434c574e3031};  // "NCCLWN01"

absl::Status ValidateNcclWindowDeviceAbi(uint64_t compile_time_version,
                                         uint64_t runtime_version);

// A NCCL window registration handle that makes local buffers accessible from
// remote peers via symmetric memory registration process.
class NcclSymmetricMemory final : public SymmetricMemory {
 public:
  ~NcclSymmetricMemory() final;

  static absl::StatusOr<std::unique_ptr<NcclSymmetricMemory>> Create(
      std::shared_ptr<NcclCommState> comm_state,
      stream_executor::DeviceAddressBase addr,
      std::shared_ptr<tsl::Executor> executor,
      stream_executor::StreamExecutor* stream_executor);

  stream_executor::DeviceAddressBase addr() const final;
  absl::StatusOr<stream_executor::DeviceAddressBase> multimem_addr()
      const final;

  absl::StatusOr<stream_executor::DeviceAddressBase> peer_addr(
      RankId peer) const final;

  ncclWindow_t win() const { return win_; }

  std::string ToString() const final;

  PackedKernelArg PackKernelArg() const final;

 private:
  NcclSymmetricMemory(std::shared_ptr<NcclCommState> comm_state,
                      ncclWindow_t win, stream_executor::DeviceAddressBase addr,
                      std::shared_ptr<tsl::Executor> executor,
                      stream_executor::StreamExecutor* stream_executor,
                      uint64_t validated_device_abi_version);

  std::shared_ptr<NcclCommState> comm_state_;
  ncclWindow_t win_;
  stream_executor::DeviceAddressBase addr_;
  std::shared_ptr<tsl::Executor> executor_;
  stream_executor::StreamExecutor* stream_executor_;
  std::thread::id owner_thread_id_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_NCCL_SYMMETRIC_MEMORY_H_
