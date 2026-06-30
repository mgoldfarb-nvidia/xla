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

#ifndef XLA_BACKENDS_GPU_RUNTIME_FFI_COLLECTIVE_RESOURCES_H_
#define XLA_BACKENDS_GPU_RUNTIME_FFI_COLLECTIVE_RESOURCES_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_memory.h"
#include "xla/backends/gpu/runtime/collective_memory_requests.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/thunk_id.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/c_api_gpu_collectives.h"
#include "xla/ffi/gpu_collectives_api.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/shaped_slice.h"

namespace xla::gpu {

// Execution-scoped implementation of the public GPU device-communication API.
//
// A handler declares device communication once with an FFI trait. XLA then
// acquires a default full-execution communication team and registers every FFI
// operand/result assigned to collective memory. The public API is deliberately
// read-only: handlers can retrieve opaque kernel arguments, but cannot select
// providers or configure backend resources. Device work must collectively
// quiesce remote accesses before returning so XLA can safely release resources
// after local stream completion.
class FfiCollectiveResources final : public ffi::GpuCollectivesApi {
 public:
  FfiCollectiveResources(absl::string_view target_name,
                         absl::string_view profile_annotation, ThunkId thunk_id,
                         absl::Span<const NullableShapedSlice> operands,
                         absl::Span<const NullableShapedSlice> results);

  FfiCollectiveResources(const FfiCollectiveResources&) = delete;
  FfiCollectiveResources& operator=(const FfiCollectiveResources&) = delete;
  FfiCollectiveResources(FfiCollectiveResources&&) = default;
  FfiCollectiveResources& operator=(FfiCollectiveResources&&) = default;
  ~FfiCollectiveResources() override = default;

  // Installs resources borrowed from the current thunk lifecycle invocation.
  absl::Status BeginInvocation(
      XLA_FFI_ExecutionStage stage, const BufferAllocations* buffer_allocations,
      const CollectiveParams* collective_params,
      CollectiveCliqueRequests* collective_clique_requests,
      CollectiveMemoryRequests* collective_memory_requests,
      const CollectiveCliques* collective_cliques,
      const CollectiveMemory* collective_memory);

  // Declares all resources for the minimal initial contract. This is invoked
  // by the thunk based on handler metadata, before the handler's optional
  // Prepare callback. It provisions one load/store-accessible synchronization
  // slot per callsite.
  absl::Status PrepareDeviceCommunication();

  absl::Status GetDeviceComm(
      XLA_FFI_GpuCollectives_GetDeviceComm_Args* args) override;
  absl::Status GetDeviceMemory(
      XLA_FFI_GpuCollectives_GetDeviceMemory_Args* args) override;

  uint64_t resource_domain() const { return resource_domain_; }

 private:
  friend class FfiCollectiveResourcesTestPeer;

  struct StaticBuffer {
    BufferAllocation::Index allocation;
    int64_t offset;
    int64_t size;
    int64_t allocation_size;
    int64_t memory_space;
  };

  struct BufferView {
    BufferAllocation::Index allocation;
    uint64_t offset;
    uint64_t size;
    uint64_t allocation_size;
  };

  struct CollectiveRecord {
    GpuCliqueKey key;
    GpuDeviceCommunicator::Requirements requirements;
  };

  absl::Status CheckStageForGet(absl::string_view operation) const;

  absl::StatusOr<BufferView> FindBufferView(const XLA_FFI_Buffer& buffer) const;
  static absl::Status PreparePackedDestination(
      XLA_FFI_GpuCollective_PackedKernelArg* output, uint64_t schema,
      uint64_t abi_version, size_t size, size_t alignment);

  uint64_t resource_domain_;
  bool has_valid_resource_domain_;

  std::vector<StaticBuffer> static_buffers_;
  std::optional<CollectiveRecord> collective_;
  std::vector<BufferAllocation::Index> registered_allocations_;
  bool prepared_ = false;

  XLA_FFI_ExecutionStage stage_ = XLA_FFI_ExecutionStage_INSTANTIATE;
  const BufferAllocations* buffer_allocations_ = nullptr;
  const CollectiveParams* collective_params_ = nullptr;
  CollectiveCliqueRequests* collective_clique_requests_ = nullptr;
  CollectiveMemoryRequests* collective_memory_requests_ = nullptr;
  const CollectiveCliques* collective_cliques_ = nullptr;
  const CollectiveMemory* collective_memory_ = nullptr;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_FFI_COLLECTIVE_RESOURCES_H_
