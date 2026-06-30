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
constexpr int32_t kDeviceCommunicationBarrierSlots = 1;
constexpr uint64_t kFfiResourceDomainNamespace = uint64_t{1} << 63;

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

bool IsPowerOfTwo(size_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

}  // namespace

FfiCollectiveResources::FfiCollectiveResources(
    absl::string_view target_name, absl::string_view profile_annotation,
    ThunkId thunk_id, absl::Span<const NullableShapedSlice> operands,
    absl::Span<const NullableShapedSlice> results)
    : resource_domain_(
          ResourceDomain(target_name, profile_annotation, thunk_id)),
      has_valid_resource_domain_(thunk_id.value() != 0) {
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

absl::Status FfiCollectiveResources::PrepareDeviceCommunication() {
  if (stage_ != XLA_FFI_ExecutionStage_PREPARE) {
    return FailedPrecondition(
        "Device communication can only be prepared during FFI Prepare");
  }
  if (prepared_) return absl::OkStatus();
  if (!has_valid_resource_domain_) {
    return FailedPrecondition(
        "Device communication requires a non-zero thunk id for callsite "
        "isolation");
  }
  if (collective_params_ == nullptr ||
      collective_params_->device_assn == nullptr ||
      collective_clique_requests_ == nullptr ||
      collective_memory_requests_ == nullptr) {
    return FailedPrecondition(
        "Device communication requires GPU collective Prepare resources");
  }

  std::vector<BufferAllocation::Index> allocations;
  allocations.reserve(static_buffers_.size());
  for (const StaticBuffer& buffer : static_buffers_) {
    if (buffer.memory_space != kCollectiveMemorySpace) continue;
    if (buffer.allocation < 0 || buffer.offset < 0 || buffer.size <= 0 ||
        buffer.allocation_size <= 0 || buffer.offset > buffer.allocation_size ||
        buffer.size > buffer.allocation_size - buffer.offset) {
      return InvalidArgument(
          "Invalid collective-memory FFI buffer allocation or slice");
    }
    if (std::find(allocations.begin(), allocations.end(), buffer.allocation) ==
        allocations.end()) {
      allocations.push_back(buffer.allocation);
    }
  }
  if (allocations.empty()) {
    return FailedPrecondition(
        "Device-communication handler has no collective-memory tagged "
        "operands or results");
  }

  const DeviceAssignment& device_assignment = *collective_params_->device_assn;
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

  GpuDeviceCommunicator::Requirements requirements{
      kDeviceCommunicationBarrierSlots};
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

absl::Status FfiCollectiveResources::PreparePackedDestination(
    XLA_FFI_GpuCollective_PackedKernelArg* output, uint64_t schema,
    uint64_t abi_version, size_t size, size_t alignment) {
  if (output == nullptr) {
    return InvalidArgument("Packed kernel argument output must not be null");
  }
  RETURN_IF_ERROR(CheckStructSize(
      "XLA_FFI_GpuCollective_PackedKernelArg",
      XLA_FFI_GpuCollective_PackedKernelArg_STRUCT_SIZE, output->struct_size));

  output->size = size;
  output->alignment = alignment;
  output->schema = schema;
  output->abi_version = abi_version;

  if (size == 0 || !IsPowerOfTwo(alignment) || schema == 0 ||
      abi_version == 0) {
    return FailedPrecondition(
        "Device-communication provider returned invalid kernel argument "
        "metadata");
  }
  if (output->destination == nullptr && output->capacity == 0) {
    return absl::OkStatus();
  }
  if (output->destination == nullptr) {
    return InvalidArgument(
        "Packed kernel argument destination is null with non-zero capacity");
  }
  if (output->capacity < size) {
    return ResourceExhausted(
        "Packed kernel argument requires %d bytes, but capacity is %d", size,
        output->capacity);
  }
  if (reinterpret_cast<uintptr_t>(output->destination) % alignment != 0) {
    return InvalidArgument(
        "Packed kernel argument destination does not satisfy %d-byte "
        "alignment",
        alignment);
  }
  return absl::OkStatus();
}

absl::Status FfiCollectiveResources::GetDeviceComm(
    XLA_FFI_GpuCollectives_GetDeviceComm_Args* args) {
  RETURN_IF_ERROR(CheckStageForGet("GetDeviceComm"));
  if (args == nullptr) {
    return InvalidArgument("GetDeviceComm args must not be null");
  }
  if (collective_params_ == nullptr || collective_cliques_ == nullptr) {
    return FailedPrecondition(
        "GetDeviceComm requires acquired collective resources");
  }

  const CollectiveRecord& record = *collective_;
  ASSIGN_OR_RETURN(GpuDeviceCommunicator * communicator,
                   collective_cliques_->GetDeviceComm(
                       record.key, collective_params_->global_device_id,
                       record.requirements));
  const GpuDeviceCommunicator::PackedKernelArgMetadata& metadata =
      communicator->GetKernelArgMetadata();
  RETURN_IF_ERROR(PreparePackedDestination(
      args->kernel_arg, metadata.device_abi_schema, metadata.device_abi_version,
      metadata.size_bytes, metadata.alignment));

  bool query = args->kernel_arg->destination == nullptr &&
               args->kernel_arg->capacity == 0;
  if (!query) {
    se::PackedKernelArg packed = communicator->PackKernelArg();
    if (packed.size_bytes() != metadata.size_bytes) {
      return Internal(
          "Packed device communicator size %d does not match metadata size %d",
          packed.size_bytes(), metadata.size_bytes);
    }
    std::memcpy(args->kernel_arg->destination, packed.data(),
                packed.size_bytes());
  }
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
  args->offset = symmetric_offset + view.offset;

  SymmetricMemory::PackedKernelArgMetadata metadata =
      symmetric_memory->GetKernelArgMetadata();
  RETURN_IF_ERROR(PreparePackedDestination(
      args->kernel_arg, metadata.device_abi_schema, metadata.device_abi_version,
      metadata.size_bytes, metadata.alignment));

  bool query = args->kernel_arg->destination == nullptr &&
               args->kernel_arg->capacity == 0;
  if (!query) {
    SymmetricMemory::PackedKernelArg packed = symmetric_memory->PackKernelArg();
    if (metadata.size_bytes != sizeof(packed)) {
      return Internal(
          "Packed device-memory size %d does not match metadata size %d",
          sizeof(packed), metadata.size_bytes);
    }
    std::memcpy(args->kernel_arg->destination, &packed, sizeof(packed));
  }
  return absl::OkStatus();
}

}  // namespace xla::gpu
