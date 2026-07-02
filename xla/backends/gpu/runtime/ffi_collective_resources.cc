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

#include "xla/backends/gpu/runtime/ffi_collective_resources.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_execution.h"
#include "xla/backends/gpu/runtime/collective_memory.h"
#include "xla/backends/gpu/runtime/collective_memory_requests.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/c_api_gpu_collectives.h"
#include "xla/hlo/ir/collective_op_group_mode.h"
#include "xla/primitive_util.h"
#include "xla/runtime/device_id.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/computation_placer.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/util.h"
#include "tsl/platform/fingerprint.h"

namespace xla::gpu {
namespace {

constexpr int64_t kCollectiveMemorySpace = 1;
constexpr uint64_t kFfiResourceDomainNamespace = uint64_t{1} << 63;

static_assert(GpuDeviceCommunicator::kLocalMulticast ==
              XLA_FFI_GPU_DEVICE_COMM_FEATURE_LOCAL_MULTICAST);
static_assert(GpuDeviceCommunicator::kNetworkDeviceOperations ==
              XLA_FFI_GPU_DEVICE_COMM_FEATURE_NETWORK_DEVICE_OPERATIONS);

uint64_t ResourceDomain(absl::string_view target_name,
                        absl::string_view profile_annotation,
                        ThunkId thunk_id) {
  // Length prefixes make the fingerprint input unambiguous. The high bit keeps
  // FFI callsite domains out of the native collective/P2P channel namespace.
  std::string key =
      absl::StrCat("xla.ffi.gpu.device_comm.callsite.v1|", target_name.size(),
                   ":", target_name, "|", profile_annotation.size(), ":",
                   profile_annotation, "|", thunk_id.value());
  return tsl::Fingerprint64(key) | kFfiResourceDomainNamespace;
}

absl::Status CheckStructSize(absl::string_view name, size_t expected,
                             size_t actual) {
  if (actual < expected) {
    return InvalidArgument("Unexpected %s size: expected at least %d, got %d",
                           name, expected, actual);
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status FfiDeviceCommunicationProfile::Freeze(
    const GpuDeviceCommunicator::Requirements& requirements) {
  absl::MutexLock lock(mutex_);
  if (!requirements_.has_value()) {
    requirements_ = requirements;
    return absl::OkStatus();
  }
  if (!(*requirements_ == requirements)) {
    return FailedPrecondition(
        "Device-communication requirements changed across invocations: "
        "frozen=%v, requested=%v. Request a stable upper bound instead.",
        *requirements_, requirements);
  }
  return absl::OkStatus();
}

FfiCollectiveResources::FfiCollectiveResources(
    absl::string_view target_name, absl::string_view profile_annotation,
    ThunkId thunk_id, absl::Span<const NullableShapedSlice> operands,
    absl::Span<const NullableShapedSlice> results,
    bool uses_device_communication,
    std::shared_ptr<FfiDeviceCommunicationProfile> profile)
    : resource_domain_(
          ResourceDomain(target_name, profile_annotation, thunk_id)),
      has_valid_resource_domain_(thunk_id.value() != 0),
      uses_device_communication_(uses_device_communication),
      profile_(profile ? std::move(profile)
                       : std::make_shared<FfiDeviceCommunicationProfile>()) {
  static_buffers_.reserve(operands.size() + results.size());
  auto add_static_buffer = [&](const NullableShapedSlice& shaped_slice) {
    if (!shaped_slice.has_value() ||
        shaped_slice->slice.allocation() == nullptr) {
      return;
    }
    const BufferAllocation::Slice& slice = shaped_slice->slice;
    int64_t memory_space = shaped_slice->shape.has_layout()
                               ? shaped_slice->shape.layout().memory_space()
                               : 0;
    static_buffers_.push_back(
        StaticBuffer{slice.index(), slice.offset(), slice.size(),
                     slice.allocation()->size(), memory_space});
  };
  absl::c_for_each(operands, add_static_buffer);
  absl::c_for_each(results, add_static_buffer);
}

absl::Status FfiCollectiveResources::BeginInvocation(
    XLA_FFI_ExecutionStage stage, const BufferAllocations* buffer_allocations,
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* collective_clique_requests,
    CollectiveMemoryRequests* collective_memory_requests,
    const CollectiveCliques* collective_cliques,
    const CollectiveMemory* collective_memory) {
  stage_ = stage;
  buffer_allocations_ = buffer_allocations;
  collective_params_ = collective_params;
  collective_clique_requests_ = collective_clique_requests;
  collective_memory_requests_ = collective_memory_requests;
  collective_cliques_ = collective_cliques;
  collective_memory_ = collective_memory;
  return absl::OkStatus();
}

GpuDeviceCommunicator::Requirements
FfiCollectiveResources::DefaultRequirements() {
  GpuDeviceCommunicator::Requirements requirements;
  requirements.team_barrier_count = 1;
  return requirements;
}

absl::StatusOr<GpuDeviceCommunicator::Requirements>
FfiCollectiveResources::NormalizeRequirements(
    const XLA_FFI_GpuDeviceCommunication_Requirements& requirements) {
  RETURN_IF_ERROR(CheckStructSize(
      "XLA_FFI_GpuDeviceCommunication_Requirements",
      XLA_FFI_GpuDeviceCommunication_Requirements_STRUCT_SIZE_V1_1,
      requirements.struct_size));

  GpuDeviceCommunicator::Requirements normalized;
  switch (requirements.peer_access) {
    case XLA_FFI_GPU_PEER_ACCESS_LOCAL_DOMAIN:
      normalized.peer_access = GpuDeviceCommunicator::PeerAccess::kLocalDomain;
      break;
    case XLA_FFI_GPU_PEER_ACCESS_HIERARCHICAL:
      normalized.peer_access = GpuDeviceCommunicator::PeerAccess::kHierarchical;
      break;
    case XLA_FFI_GPU_PEER_ACCESS_DIRECT_ANY_PEER:
      normalized.peer_access =
          GpuDeviceCommunicator::PeerAccess::kDirectAnyPeer;
      break;
    default:
      return InvalidArgument("Unknown GPU peer access value: %d",
                             static_cast<int>(requirements.peer_access));
  }

  GpuDeviceCommunicator::Features unknown_required =
      requirements.required_features & ~GpuDeviceCommunicator::kKnownFeatures;
  if (unknown_required != 0) {
    return Unimplemented(
        "Unknown required GPU device-communication feature bits: 0x%x",
        unknown_required);
  }
  normalized.required_features = requirements.required_features;
  normalized.preferred_features = (requirements.preferred_features &
                                   GpuDeviceCommunicator::kKnownFeatures) &
                                  ~normalized.required_features;

  auto checked_count = [](absl::string_view name,
                          uint32_t count) -> absl::StatusOr<int32_t> {
    if (count > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
      return InvalidArgument("%s exceeds the supported maximum: %d", name,
                             count);
    }
    return static_cast<int32_t>(count);
  };
  ASSIGN_OR_RETURN(
      normalized.local_barrier_count,
      checked_count("local_barrier_count", requirements.local_barrier_count));
  ASSIGN_OR_RETURN(
      normalized.team_barrier_count,
      checked_count("team_barrier_count", requirements.team_barrier_count));
  ASSIGN_OR_RETURN(normalized.notification_slot_count,
                   checked_count("notification_slot_count",
                                 requirements.notification_slot_count));
  ASSIGN_OR_RETURN(normalized.completion_slot_count,
                   checked_count("completion_slot_count",
                                 requirements.completion_slot_count));
  return normalized;
}

absl::Status FfiCollectiveResources::RequestDeviceCommunication(
    XLA_FFI_GpuCollectives_RequestDeviceCommunication_Args* args) {
  if (args == nullptr) {
    return InvalidArgument("RequestDeviceCommunication args must not be null");
  }
  RETURN_IF_ERROR(CheckStructSize(
      "XLA_FFI_GpuCollectives_RequestDeviceCommunication_Args",
      XLA_FFI_GpuCollectives_RequestDeviceCommunication_Args_STRUCT_SIZE_V1_1,
      args->struct_size));
  if (stage_ != XLA_FFI_ExecutionStage_PREPARE) {
    return FailedPrecondition(
        "RequestDeviceCommunication is only available during FFI Prepare");
  }
  if (!uses_device_communication_) {
    return FailedPrecondition(
        "RequestDeviceCommunication requires a handler declared with "
        "USES_DEVICE_COMMUNICATION");
  }
  if (args->requirements == nullptr) {
    return InvalidArgument(
        "RequestDeviceCommunication requires non-null requirements");
  }

  ASSIGN_OR_RETURN(GpuDeviceCommunicator::Requirements normalized,
                   NormalizeRequirements(*args->requirements));
  if (requested_requirements_.has_value() &&
      !(*requested_requirements_ == normalized)) {
    return InvalidArgument(
        "Conflicting device-communication requests in one FFI Prepare: "
        "first=%v, requested=%v",
        *requested_requirements_, normalized);
  }
  requested_requirements_ = normalized;
  return absl::OkStatus();
}

absl::Status FfiCollectiveResources::ValidateResolvedInfo(
    const GpuDeviceCommunicator::Requirements& requirements,
    const GpuDeviceCommunicator::Info& info) {
  GpuDeviceCommunicator::Features missing_required =
      requirements.required_features & ~info.enabled_features;
  if (missing_required != 0) {
    return Internal(
        "Device communicator omitted required feature bits from resolved "
        "information: 0x%x",
        missing_required);
  }
  if (info.team_barrier_count < 0 || info.local_barrier_count < 0 ||
      info.notification_slot_count < 0 || info.completion_slot_count < 0) {
    return Internal("Device communicator returned negative resource counts");
  }
  if (info.team_barrier_count < requirements.team_barrier_count ||
      info.local_barrier_count < requirements.local_barrier_count ||
      info.notification_slot_count < requirements.notification_slot_count ||
      info.completion_slot_count < requirements.completion_slot_count) {
    return Internal(
        "Device communicator under-provisioned logical resources: "
        "requested=%v, resolved={team_barriers: %d, local_barriers: %d, "
        "notifications: %d, completions: %d}",
        requirements, info.team_barrier_count, info.local_barrier_count,
        info.notification_slot_count, info.completion_slot_count);
  }
  if (info.team_size <= 0 || info.rank < 0 || info.rank >= info.team_size ||
      info.local_domain_size <= 0 || info.local_domain_size > info.team_size ||
      info.local_domain_count <= 0 || info.local_rank < 0 ||
      info.local_rank >= info.local_domain_size) {
    return Internal("Device communicator returned invalid topology dimensions");
  }

  bool local_domain_covers_team =
      info.local_domain_size == info.team_size && info.local_domain_count == 1;
  if ((info.local_domain_size == info.team_size) !=
      (info.local_domain_count == 1)) {
    return Internal(
        "Device communicator returned inconsistent local-domain topology");
  }
  switch (requirements.peer_access) {
    case GpuDeviceCommunicator::PeerAccess::kLocalDomain:
      break;
    case GpuDeviceCommunicator::PeerAccess::kHierarchical:
      if (!local_domain_covers_team &&
          info.topology == GpuDeviceCommunicator::Topology::kLocalDomain) {
        return Internal(
            "Device communicator did not satisfy hierarchical peer access");
      }
      break;
    case GpuDeviceCommunicator::PeerAccess::kDirectAnyPeer:
      if (!local_domain_covers_team &&
          info.topology != GpuDeviceCommunicator::Topology::kAllPeers) {
        return Internal(
            "Device communicator did not satisfy direct-any-peer access");
      }
      break;
  }
  return absl::OkStatus();
}

absl::Status FfiCollectiveResources::FinalizeDeviceCommunication() {
  if (stage_ != XLA_FFI_ExecutionStage_PREPARE) {
    return FailedPrecondition(
        "Device communication can only be prepared during FFI Prepare");
  }
  if (prepared_) return absl::OkStatus();
  if (!uses_device_communication_) {
    return FailedPrecondition(
        "Device communication requires a handler declared with "
        "USES_DEVICE_COMMUNICATION");
  }
  if (!has_valid_resource_domain_) {
    return FailedPrecondition(
        "Device communication requires a non-zero thunk id for callsite "
        "isolation");
  }
  if (collective_params_ == nullptr ||
      collective_params_->device_assn == nullptr ||
      collective_clique_requests_ == nullptr) {
    return FailedPrecondition(
        "Device communication requires GPU collective Prepare resources");
  }

  std::vector<BufferAllocation::Index> allocations;
  allocations.reserve(static_buffers_.size());
  for (const StaticBuffer& buffer : static_buffers_) {
    if (buffer.memory_space != kCollectiveMemorySpace) continue;
    if (buffer.allocation < 0 || buffer.offset < 0 || buffer.size < 0 ||
        buffer.allocation_size < 0 || buffer.offset > buffer.allocation_size ||
        buffer.size > buffer.allocation_size - buffer.offset) {
      return InvalidArgument(
          "Invalid collective-memory FFI buffer allocation or slice");
    }
    // Empty FFI buffers do not need a provider window. GetDeviceMemory rejects
    // empty views because providers do not define a useful packed argument for
    // them, but their presence must not make Prepare fail.
    if (buffer.size == 0) continue;
    if (std::find(allocations.begin(), allocations.end(), buffer.allocation) ==
        allocations.end()) {
      allocations.push_back(buffer.allocation);
    }
  }
  if (!allocations.empty() && collective_memory_requests_ == nullptr) {
    return FailedPrecondition(
        "Tagged device-communication buffers require collective-memory "
        "Prepare resources");
  }

  const DeviceAssignment& device_assignment = *collective_params_->device_assn;
  if (device_assignment.replica_count() <= 0 ||
      device_assignment.computation_count() <= 0) {
    return InvalidArgument("Device communication team must not be empty");
  }
  if (device_assignment.replica_count() >
      std::numeric_limits<int64_t>::max() /
          device_assignment.computation_count()) {
    return InvalidArgument("Device communication team size overflows int64");
  }
  int64_t team_size =
      device_assignment.replica_count() * device_assignment.computation_count();

  ReplicaGroup full_team;
  for (int64_t id = 0; id < team_size; ++id) {
    full_team.add_replica_ids(id);
  }
  std::vector<ReplicaGroup> replica_groups = {std::move(full_team)};
  constexpr CollectiveOpGroupMode kGroupMode =
      CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_FLATTENED_ID;

  ASSIGN_OR_RETURN(std::vector<std::vector<GlobalDeviceId>> device_groups,
                   GetParticipatingDevicesGroups(device_assignment,
                                                 replica_groups, kGroupMode));
  absl::c_for_each(device_groups, [](auto& group) { absl::c_sort(group); });
  absl::c_sort(device_groups);

  ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetGpuCliqueKey(*collective_params_, replica_groups, kGroupMode,
                      CommunicationId(resource_domain_)));

  GpuDeviceCommunicator::Requirements requirements =
      requested_requirements_.value_or(DefaultRequirements());
  RETURN_IF_ERROR(profile_->Freeze(requirements));

  CollectiveCliqueRequests::CliqueRequirements clique_requirements;
  clique_requirements.barrier_reqs =
      CollectiveCliqueRequests::BarrierRequirements{
          /*module_execution_barrier=*/true};
  clique_requirements.dev_comm = requirements;
  RETURN_IF_ERROR(collective_clique_requests_->RequestClique(
      clique_key, device_groups, clique_requirements));

  for (BufferAllocation::Index allocation : allocations) {
    RETURN_IF_ERROR(collective_memory_requests_->RequestSymmetricAllocation(
        clique_key, allocation));
  }

  collective_ = CollectiveRecord{std::move(clique_key), requirements};
  registered_allocations_ = std::move(allocations);
  prepared_ = true;
  return absl::OkStatus();
}

absl::Status FfiCollectiveResources::CheckStageForGet(
    absl::string_view operation) const {
  if (stage_ != XLA_FFI_ExecutionStage_INITIALIZE &&
      stage_ != XLA_FFI_ExecutionStage_EXECUTE) {
    return FailedPrecondition(
        "%s is only available during FFI Initialize or Execute", operation);
  }
  if (!prepared_ || !collective_.has_value()) {
    return FailedPrecondition(
        "%s requires a handler declared with USES_DEVICE_COMMUNICATION",
        operation);
  }
  return absl::OkStatus();
}

absl::StatusOr<GpuDeviceCommunicator*> FfiCollectiveResources::GetCommunicator(
    absl::string_view operation) const {
  RETURN_IF_ERROR(CheckStageForGet(operation));
  if (collective_params_ == nullptr || collective_cliques_ == nullptr) {
    return FailedPrecondition("%s requires acquired collective resources",
                              operation);
  }

  const CollectiveRecord& record = *collective_;
  return collective_cliques_->GetDeviceComm(
      record.key, collective_params_->global_device_id, record.requirements);
}

absl::Status FfiCollectiveResources::GetDeviceCommunicationInfo(
    XLA_FFI_GpuCollectives_GetDeviceCommunicationInfo_Args* args) {
  if (args == nullptr) {
    return InvalidArgument("GetDeviceCommunicationInfo args must not be null");
  }
  RETURN_IF_ERROR(CheckStructSize(
      "XLA_FFI_GpuCollectives_GetDeviceCommunicationInfo_Args",
      XLA_FFI_GpuCollectives_GetDeviceCommunicationInfo_Args_STRUCT_SIZE_V1_1,
      args->struct_size));
  if (args->info == nullptr) {
    return InvalidArgument("GetDeviceCommunicationInfo info must not be null");
  }
  RETURN_IF_ERROR(
      CheckStructSize("XLA_FFI_GpuDeviceCommunication_Info",
                      XLA_FFI_GpuDeviceCommunication_Info_STRUCT_SIZE_V1_1,
                      args->info->struct_size));

  ASSIGN_OR_RETURN(GpuDeviceCommunicator * communicator,
                   GetCommunicator("GetDeviceCommunicationInfo"));
  const GpuDeviceCommunicator::Info& source = communicator->info();
  const CollectiveRecord& record = *collective_;
  RETURN_IF_ERROR(ValidateResolvedInfo(record.requirements, source));

  XLA_FFI_GpuCommunicationTopology topology;
  switch (source.topology) {
    case GpuDeviceCommunicator::Topology::kLocalDomain:
      topology = XLA_FFI_GPU_TOPOLOGY_LOCAL_DOMAIN;
      break;
    case GpuDeviceCommunicator::Topology::kHierarchical:
      topology = XLA_FFI_GPU_TOPOLOGY_HIERARCHICAL;
      break;
    case GpuDeviceCommunicator::Topology::kAllPeers:
      topology = XLA_FFI_GPU_TOPOLOGY_ALL_PEERS;
      break;
    default:
      return Internal("Device communicator returned an unknown topology");
  }

  XLA_FFI_GpuDeviceCommunication_Info* destination = args->info;
  destination->rank = source.rank;
  destination->team_size = source.team_size;
  destination->local_rank = source.local_rank;
  destination->local_domain_size = source.local_domain_size;
  destination->local_domain_count = source.local_domain_count;
  destination->topology = topology;
  destination->enabled_features = source.enabled_features;
  destination->team_barrier_count = source.team_barrier_count;
  destination->local_barrier_count = source.local_barrier_count;
  destination->notification_slot_count = source.notification_slot_count;
  destination->completion_slot_count = source.completion_slot_count;
  return absl::OkStatus();
}

absl::Status FfiCollectiveResources::ValidateKernelArgDestination(
    void* destination, size_t destination_size, size_t packed_size) {
  if (destination == nullptr) {
    return InvalidArgument("Kernel argument destination must not be null");
  }
  if (packed_size == 0) {
    return FailedPrecondition(
        "Device-communication provider returned an empty kernel argument");
  }
  if (destination_size != packed_size) {
    return InvalidArgument(
        "Kernel argument destination size mismatch: expected %d bytes, got %d",
        packed_size, destination_size);
  }
  return absl::OkStatus();
}

absl::Status FfiCollectiveResources::GetDeviceComm(
    XLA_FFI_GpuCollectives_GetDeviceComm_Args* args) {
  if (args == nullptr) {
    return InvalidArgument("GetDeviceComm args must not be null");
  }

  ASSIGN_OR_RETURN(GpuDeviceCommunicator * communicator,
                   GetCommunicator("GetDeviceComm"));
  if (args->destination == nullptr) {
    return InvalidArgument("Kernel argument destination must not be null");
  }
  RETURN_IF_ERROR(communicator->CheckKernelArgAbi(args->expected_abi_schema,
                                                  args->expected_abi_version));

  se::PackedKernelArg packed = communicator->PackKernelArg();
  RETURN_IF_ERROR(ValidateKernelArgDestination(
      args->destination, args->destination_size, packed.size_bytes()));
  std::memcpy(args->destination, packed.data(), packed.size_bytes());
  return absl::OkStatus();
}

absl::StatusOr<FfiCollectiveResources::BufferView>
FfiCollectiveResources::FindBufferView(const XLA_FFI_Buffer& buffer) const {
  RETURN_IF_ERROR(CheckStructSize("XLA_FFI_Buffer", XLA_FFI_Buffer_STRUCT_SIZE,
                                  buffer.struct_size));
  if (buffer.data == nullptr) {
    return InvalidArgument("Device-communication buffer must not be null");
  }
  if (buffer.rank < 0 || (buffer.rank > 0 && buffer.dims == nullptr)) {
    return InvalidArgument("Invalid device-communication buffer shape");
  }

  PrimitiveType element_type = static_cast<PrimitiveType>(buffer.dtype);
  if (!primitive_util::IsArrayType(element_type)) {
    return InvalidArgument(
        "Invalid device-communication buffer element type: %d",
        static_cast<int>(buffer.dtype));
  }
  uint64_t byte_size = primitive_util::ByteWidth(element_type);
  for (int64_t i = 0; i < buffer.rank; ++i) {
    if (buffer.dims[i] < 0) {
      return InvalidArgument(
          "Device-communication buffer dimension is negative");
    }
    uint64_t dim = static_cast<uint64_t>(buffer.dims[i]);
    if (dim != 0 && byte_size > std::numeric_limits<uint64_t>::max() / dim) {
      return InvalidArgument("Device-communication buffer byte size overflows");
    }
    byte_size *= dim;
  }
  if (byte_size == 0) {
    return InvalidArgument("Device-communication buffer must not be empty");
  }
  if (buffer_allocations_ == nullptr) {
    return FailedPrecondition(
        "GetDeviceMemory requires runtime buffer allocations");
  }

  uintptr_t view_begin = reinterpret_cast<uintptr_t>(buffer.data);
  if (byte_size > std::numeric_limits<uintptr_t>::max() - view_begin) {
    return InvalidArgument("Device-communication buffer range overflows");
  }
  uintptr_t view_end = view_begin + byte_size;

  std::optional<BufferView> match;
  bool overlaps_static_slice_out_of_bounds = false;
  bool found_untagged_view = false;
  for (const StaticBuffer& buffer_info : static_buffers_) {
    if (buffer_info.allocation < 0 ||
        static_cast<size_t>(buffer_info.allocation) >=
            buffer_allocations_->size() ||
        buffer_info.offset < 0 || buffer_info.size < 0 ||
        buffer_info.allocation_size <= 0) {
      continue;
    }

    se::DeviceAddressBase allocation =
        buffer_allocations_->GetDeviceAddress(buffer_info.allocation);
    if (allocation.opaque() == nullptr || allocation.size() == 0) continue;
    uintptr_t allocation_begin =
        reinterpret_cast<uintptr_t>(allocation.opaque());
    if (allocation.size() >
        std::numeric_limits<uintptr_t>::max() - allocation_begin) {
      return InvalidArgument("Device allocation address range overflows");
    }
    uintptr_t allocation_end = allocation_begin + allocation.size();

    uint64_t slice_offset = static_cast<uint64_t>(buffer_info.offset);
    uint64_t slice_size = static_cast<uint64_t>(buffer_info.size);
    uint64_t static_allocation_size =
        static_cast<uint64_t>(buffer_info.allocation_size);
    if (slice_offset > static_allocation_size ||
        slice_size > static_allocation_size - slice_offset ||
        slice_offset > allocation.size() ||
        slice_size > allocation.size() - slice_offset) {
      return InvalidArgument(
          "Static FFI buffer slice is outside its allocation");
    }
    uintptr_t slice_begin = allocation_begin + slice_offset;
    uintptr_t slice_end = slice_begin + slice_size;
    if (view_begin < slice_end && view_end > slice_begin &&
        (view_begin < slice_begin || view_end > slice_end)) {
      overlaps_static_slice_out_of_bounds = true;
    }
    if (view_begin < slice_begin || view_end > slice_end ||
        view_end > allocation_end) {
      continue;
    }
    if (buffer_info.memory_space != kCollectiveMemorySpace) {
      found_untagged_view = true;
      continue;
    }

    BufferView candidate{buffer_info.allocation,
                         static_cast<uint64_t>(view_begin - allocation_begin),
                         byte_size, static_cast<uint64_t>(allocation.size())};
    if (match.has_value() &&
        (match->allocation != candidate.allocation ||
         match->offset != candidate.offset || match->size != candidate.size ||
         match->allocation_size != candidate.allocation_size)) {
      return InvalidArgument(
          "Device-communication buffer aliases multiple allocations");
    }
    match = candidate;
  }

  if (match.has_value()) return *match;
  if (overlaps_static_slice_out_of_bounds) {
    return InvalidArgument(
        "Device-communication buffer view exceeds an FFI operand or result "
        "slice");
  }
  if (found_untagged_view) {
    return FailedPrecondition(
        "FFI buffer is not tagged for device communication");
  }
  return NotFound(
      "Device-communication buffer is not an FFI operand or result view");
}

absl::Status FfiCollectiveResources::GetDeviceMemory(
    XLA_FFI_GpuCollectives_GetDeviceMemory_Args* args) {
  RETURN_IF_ERROR(CheckStageForGet("GetDeviceMemory"));
  if (args == nullptr || args->buffer == nullptr) {
    return InvalidArgument("GetDeviceMemory requires a buffer");
  }
  if (collective_memory_ == nullptr) {
    return FailedPrecondition(
        "GetDeviceMemory requires acquired collective memory");
  }

  ASSIGN_OR_RETURN(BufferView view, FindBufferView(*args->buffer));
  if (std::find(registered_allocations_.begin(), registered_allocations_.end(),
                view.allocation) == registered_allocations_.end()) {
    return FailedPrecondition(
        "FFI buffer allocation was not registered for device communication");
  }

  auto [symmetric_memory, symmetric_offset] =
      collective_memory_->FindSymmetricMemory(collective_->key,
                                              view.allocation);
  if (symmetric_memory == nullptr) {
    return NotFound("Registered device memory was not acquired");
  }
  if (view.offset > std::numeric_limits<uint64_t>::max() - symmetric_offset) {
    return Internal("Device-memory view offset overflows");
  }
  uint64_t offset = symmetric_offset + view.offset;

  if (args->destination == nullptr) {
    return InvalidArgument("Kernel argument destination must not be null");
  }
  RETURN_IF_ERROR(symmetric_memory->CheckKernelArgAbi(
      args->expected_abi_schema, args->expected_abi_version));

  SymmetricMemory::PackedKernelArg packed = symmetric_memory->PackKernelArg();
  RETURN_IF_ERROR(ValidateKernelArgDestination(
      args->destination, args->destination_size, sizeof(packed)));
  std::memcpy(args->destination, &packed, sizeof(packed));
  args->offset = offset;
  return absl::OkStatus();
}

}  // namespace xla::gpu
