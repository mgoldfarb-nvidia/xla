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
#include <memory>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
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

// A normalized device-communication profile frozen for one FFI callsite.
// CustomCallThunk owns one instance and shares it with every execution-scoped
// FfiCollectiveResources object created for that thunk. This makes a dynamic
// Prepare request stable across invocations while allowing concurrent prepares.
class FfiDeviceCommunicationProfile {
 public:
  FfiDeviceCommunicationProfile() = default;
  FfiDeviceCommunicationProfile(const FfiDeviceCommunicationProfile&) = delete;
  FfiDeviceCommunicationProfile& operator=(
      const FfiDeviceCommunicationProfile&) = delete;

  absl::Status Freeze(const GpuDeviceCommunicator::Requirements& requirements);

 private:
  absl::Mutex mutex_;
  std::optional<GpuDeviceCommunicator::Requirements> requirements_
      ABSL_GUARDED_BY(mutex_);
};

// Execution-scoped implementation of the public GPU device-communication API.
//
// A handler declares device communication once with an FFI trait. XLA then
// acquires a default full-execution communication team and registers every FFI
// operand/result assigned to collective memory. During Prepare, handlers may
// request a provider-neutral resource profile. XLA freezes the normalized
// profile per callsite and realizes it before Initialize. Device work must
// collectively quiesce remote accesses before returning so XLA can safely
// release resources after local stream completion.
class FfiCollectiveResources final : public ffi::GpuCollectivesApi {
 public:
  FfiCollectiveResources(
      absl::string_view target_name, absl::string_view profile_annotation,
      ThunkId thunk_id, absl::Span<const NullableShapedSlice> operands,
      absl::Span<const NullableShapedSlice> results,
      bool uses_device_communication,
      std::shared_ptr<FfiDeviceCommunicationProfile> profile = nullptr);

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

  // Finalizes the explicit request made by the handler during Prepare, or the
  // legacy default when no explicit request was made, then declares all
  // resources. This must run after the handler's optional Prepare callback.
  absl::Status FinalizeDeviceCommunication();

  absl::Status RequestDeviceCommunication(
      XLA_FFI_GpuCollectives_RequestDeviceCommunication_Args* args) override;
  absl::Status GetDeviceCommunicationInfo(
      XLA_FFI_GpuCollectives_GetDeviceCommunicationInfo_Args* args) override;

  absl::Status GetDeviceComm(
      XLA_FFI_GpuCollectives_GetDeviceComm_Args* args) override;
  absl::Status GetRegisteredMemoryHandle(
      XLA_FFI_GpuCollectives_GetRegisteredMemoryHandle_Args* args) override;

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

  static GpuDeviceCommunicator::Requirements DefaultRequirements();
  static absl::StatusOr<GpuDeviceCommunicator::Requirements>
  NormalizeRequirements(
      const XLA_FFI_GpuDeviceCommunication_Requirements& requirements);
  static absl::Status ValidateResolvedInfo(
      const GpuDeviceCommunicator::Requirements& requirements,
      const GpuDeviceCommunicator::Info& info);

  absl::Status CheckStageForGet(absl::string_view operation) const;
  absl::StatusOr<GpuDeviceCommunicator*> GetCommunicator(
      absl::string_view operation) const;

  absl::StatusOr<BufferView> FindBufferView(const XLA_FFI_Buffer& buffer) const;
  static absl::Status ValidateKernelArgDestination(void* destination,
                                                   size_t destination_size,
                                                   size_t packed_size);

  uint64_t resource_domain_;
  bool has_valid_resource_domain_;
  bool uses_device_communication_;
  std::shared_ptr<FfiDeviceCommunicationProfile> profile_;

  std::vector<StaticBuffer> static_buffers_;
  std::optional<GpuDeviceCommunicator::Requirements> requested_requirements_;
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
