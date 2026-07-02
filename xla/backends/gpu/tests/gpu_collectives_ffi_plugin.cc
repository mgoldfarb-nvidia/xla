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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "xla/backends/gpu/tests/gpu_collectives_ffi_plugin_kernel.h"
#include "xla/ffi/api/c_api_gpu_collectives_nccl.h"
#include "xla/ffi/api/gpu_collectives.h"

// Include NCCL after XLA headers.
#include "third_party/nccl/nccl.h"
#include "third_party/nccl/nccl_device.h"

namespace {

using xla::ffi::AnyBuffer;
using xla::ffi::CommunicationTopology;
using xla::ffi::DeviceCommunicationInfo;
using xla::ffi::DeviceCommunicationRequirements;
using xla::ffi::Error;
using xla::ffi::ErrorCode;
using xla::ffi::ErrorOr;
using xla::ffi::Ffi;
using xla::ffi::GpuCollectives;
using xla::ffi::GpuDeviceCommunication;
using xla::ffi::Initialized;
using xla::ffi::PlatformStream;
using xla::ffi::Result;
using xla::ffi::TypeId;
using xla::ffi::TypeInfo;
using xla::ffi::Unexpected;

Error FailedPrecondition(std::string message) {
  return Error(ErrorCode::kFailedPrecondition, std::move(message));
}

struct DeviceMemorySnapshot {
  ncclWindow_t window{};
  uint64_t offset = 0;
};

ErrorOr<ncclDevComm> SnapshotDeviceComm(const GpuCollectives& collectives) {
  ncclDevComm device_comm{};
  Error error = collectives.GetDeviceComm(
      XLA_FFI_GpuCollective_NCCL_DEVICE_COMM_ABI_SCHEMA, NCCL_VERSION_CODE,
      &device_comm, sizeof(device_comm));
  if (error.failure()) return Unexpected(std::move(error));
  return device_comm;
}

ErrorOr<DeviceMemorySnapshot> SnapshotDeviceMemory(
    const GpuCollectives& collectives, AnyBuffer buffer) {
  AnyBuffer::Dimensions dimensions = buffer.dimensions();
  if (buffer.element_type() != xla::ffi::U32 || dimensions.size() != 1 ||
      dimensions[0] < 2) {
    return Unexpected(FailedPrecondition(
        "device-memory test view requires a U32 vector with at least two "
        "elements"));
  }

  // Request a strict subview so the end-to-end test exercises view offsets even
  // when buffer assignment gives the FFI operand its own allocation.
  int64_t view_elements = dimensions[0] - 1;
  XLA_FFI_Buffer view = {
      XLA_FFI_Buffer_STRUCT_SIZE,
      /*extension_start=*/nullptr,
      XLA_FFI_DataType_U32,
      static_cast<uint32_t*>(buffer.untyped_data()) + 1,
      /*rank=*/1,
      &view_elements,
  };

  DeviceMemorySnapshot snapshot;
  Error error = collectives.GetDeviceMemory(
      &view, XLA_FFI_GpuCollective_NCCL_SYMMETRIC_MEMORY_ABI_SCHEMA,
      NCCL_VERSION_CODE, &snapshot.window, sizeof(snapshot.window),
      &snapshot.offset);
  if (error.failure()) return Unexpected(std::move(error));
  if (snapshot.offset == 0) {
    return Unexpected(FailedPrecondition(
        "device-memory view did not preserve a nonzero allocation offset"));
  }
  return snapshot;
}

bool SameDeviceComm(const ncclDevComm& lhs, const ncclDevComm& rhs) {
  return std::memcmp(&lhs, &rhs, sizeof(ncclDevComm)) == 0;
}

bool SameDeviceMemorySnapshot(const DeviceMemorySnapshot& lhs,
                              const DeviceMemorySnapshot& rhs) {
  return lhs.offset == rhs.offset &&
         std::memcmp(&lhs.window, &rhs.window, sizeof(ncclWindow_t)) == 0;
}

struct CollectiveState {
  static TypeId id;
  static TypeInfo info;

  ncclDevComm device_comm{};
  std::vector<DeviceMemorySnapshot> device_memories;
};

TypeId CollectiveState::id = {};
TypeInfo CollectiveState::info = xla::ffi::MakeTypeInfo<CollectiveState>();

ErrorOr<std::vector<DeviceMemorySnapshot>> SnapshotDeviceMemories(
    const GpuCollectives& collectives,
    std::initializer_list<AnyBuffer> buffers) {
  std::vector<DeviceMemorySnapshot> device_memories;
  device_memories.reserve(buffers.size());
  for (AnyBuffer buffer : buffers) {
    ErrorOr<DeviceMemorySnapshot> device_memory =
        SnapshotDeviceMemory(collectives, buffer);
    if (!device_memory.has_value()) return Unexpected(device_memory.error());
    device_memories.push_back(*std::move(device_memory));
  }
  return device_memories;
}

bool SameDeviceMemorySnapshots(const std::vector<DeviceMemorySnapshot>& lhs,
                               const std::vector<DeviceMemorySnapshot>& rhs) {
  if (lhs.size() != rhs.size()) return false;
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (!SameDeviceMemorySnapshot(lhs[i], rhs[i])) return false;
  }
  return true;
}

ErrorOr<std::unique_ptr<CollectiveState>> SnapshotCollectiveState(
    const GpuCollectives& collectives,
    std::initializer_list<AnyBuffer> buffers) {
  ErrorOr<ncclDevComm> device_comm = SnapshotDeviceComm(collectives);
  if (!device_comm.has_value()) return Unexpected(device_comm.error());
  ErrorOr<std::vector<DeviceMemorySnapshot>> device_memories =
      SnapshotDeviceMemories(collectives, buffers);
  if (!device_memories.has_value()) {
    return Unexpected(device_memories.error());
  }

  auto state = std::make_unique<CollectiveState>();
  state->device_comm = *std::move(device_comm);
  state->device_memories = *std::move(device_memories);
  return state;
}

bool SameCollectiveState(const CollectiveState& lhs,
                         const CollectiveState& rhs) {
  return SameDeviceComm(lhs.device_comm, rhs.device_comm) &&
         SameDeviceMemorySnapshots(lhs.device_memories, rhs.device_memories);
}

Error ValidateExactAlias(AnyBuffer operand, Result<AnyBuffer> result) {
  if (operand.untyped_data() != result->untyped_data() ||
      !(operand.dimensions() == result->dimensions())) {
    return FailedPrecondition(
        "custom-call result does not alias its operand exactly");
  }
  return Error::Success();
}

Error Prepare(GpuDeviceCommunication communication) {
  DeviceCommunicationRequirements requirements;
  requirements.team_barriers = 1;
  return communication.Request(requirements);
}

Error ValidateCommunicationInfo(const GpuDeviceCommunication& communication) {
  DeviceCommunicationInfo info;
  Error error = communication.GetInfo(&info);
  if (error.failure()) return error;

  if (info.rank < 0 || info.rank >= info.team_size || info.local_rank < 0 ||
      info.local_rank >= info.local_domain_size || info.team_size <= 0 ||
      info.local_domain_size <= 0 || info.local_domain_count <= 0) {
    return FailedPrecondition(
        "device communication returned invalid resolved topology");
  }
  if (info.topology != CommunicationTopology::kLocalDomain ||
      !info.enabled_features.empty() || info.team_barrier_count < 1 ||
      info.local_barrier_count != 0 || info.notification_slot_count != 0 ||
      info.completion_slot_count != 0) {
    return FailedPrecondition(
        "device communication did not realize the requested default profile");
  }
  return Error::Success();
}

ErrorOr<std::unique_ptr<CollectiveState>> Initialize(
    AnyBuffer operand, Result<AnyBuffer>, GpuCollectives collectives) {
  Error error = ValidateCommunicationInfo(collectives);
  if (error.failure()) return Unexpected(std::move(error));
  return SnapshotCollectiveState(collectives, {operand});
}

Error Execute(AnyBuffer operand, Result<AnyBuffer> result,
              CollectiveState* initialized, GpuCollectives collectives,
              cudaStream_t stream) {
  if (initialized == nullptr) {
    return FailedPrecondition("FFI initialize state is missing");
  }
  Error error = ValidateExactAlias(operand, result);
  if (error.failure()) return error;

  ErrorOr<std::unique_ptr<CollectiveState>> current =
      SnapshotCollectiveState(collectives, {operand});
  if (!current.has_value()) return current.error();
  const CollectiveState& current_state = **current;
  if (!SameCollectiveState(*initialized, current_state)) {
    return FailedPrecondition(
        "device communication resources changed between Initialize and "
        "Execute");
  }

  if (operand.element_type() != xla::ffi::U32) {
    return FailedPrecondition("test kernel requires a U32 operand");
  }
  const DeviceMemorySnapshot& device_memory =
      current_state.device_memories.front();
  cudaError_t launch_status = LaunchGpuCollectivesFfiTestKernel(
      stream, &current_state.device_comm, sizeof(current_state.device_comm),
      &device_memory.window, sizeof(device_memory.window), device_memory.offset,
      operand.element_count() - 1);
  if (launch_status != cudaSuccess) {
    return FailedPrecondition(
        std::string("failed to launch NCCL LSA kernel: ") +
        cudaGetErrorString(launch_status));
  }
  return Error::Success();
}

ErrorOr<std::unique_ptr<CollectiveState>> InitializeTwoBuffers(
    AnyBuffer first_operand, AnyBuffer second_operand, Result<AnyBuffer>,
    Result<AnyBuffer>, GpuCollectives collectives) {
  return SnapshotCollectiveState(collectives, {first_operand, second_operand});
}

Error ExecuteTwoBuffers(AnyBuffer first_operand, AnyBuffer second_operand,
                        Result<AnyBuffer> first_result,
                        Result<AnyBuffer> second_result,
                        CollectiveState* initialized,
                        GpuCollectives collectives, cudaStream_t stream) {
  if (initialized == nullptr) {
    return FailedPrecondition("FFI initialize state is missing");
  }
  Error error = ValidateExactAlias(first_operand, first_result);
  if (error.failure()) return error;
  error = ValidateExactAlias(second_operand, second_result);
  if (error.failure()) return error;

  ErrorOr<std::unique_ptr<CollectiveState>> current =
      SnapshotCollectiveState(collectives, {first_operand, second_operand});
  if (!current.has_value()) return current.error();
  const CollectiveState& current_state = **current;
  if (!SameCollectiveState(*initialized, current_state)) {
    return FailedPrecondition(
        "device communication resources changed between Initialize and "
        "Execute");
  }
  if (first_operand.element_type() != xla::ffi::U32 ||
      second_operand.element_type() != xla::ffi::U32) {
    return FailedPrecondition("test kernel requires U32 operands");
  }

  const DeviceMemorySnapshot& first_memory = current_state.device_memories[0];
  const DeviceMemorySnapshot& second_memory = current_state.device_memories[1];
  cudaError_t launch_status = LaunchGpuCollectivesFfiTwoBufferTestKernel(
      stream, &current_state.device_comm, sizeof(current_state.device_comm),
      &first_memory.window, sizeof(first_memory.window), first_memory.offset,
      first_operand.element_count() - 1, &second_memory.window,
      sizeof(second_memory.window), second_memory.offset,
      second_operand.element_count() - 1);
  if (launch_status != cudaSuccess) {
    return FailedPrecondition(
        std::string("failed to launch two-buffer NCCL LSA kernel: ") +
        cudaGetErrorString(launch_status));
  }
  return Error::Success();
}

XLA_FFI_DEFINE_HANDLER(kPrepare, Prepare,
                       Ffi::BindPrepare().Ctx<GpuDeviceCommunication>());

XLA_FFI_DEFINE_HANDLER(kInitialize, Initialize,
                       Ffi::BindInitialize()
                           .Arg<AnyBuffer>()
                           .Ret<AnyBuffer>()
                           .Ctx<GpuCollectives>());

XLA_FFI_DEFINE_HANDLER(kExecute, Execute,
                       Ffi::BindExecute()
                           .Arg<AnyBuffer>()
                           .Ret<AnyBuffer>()
                           .Ctx<Initialized<CollectiveState>>()
                           .Ctx<GpuCollectives>()
                           .Ctx<PlatformStream<cudaStream_t>>());

XLA_FFI_DEFINE_HANDLER(kInitializeTwoBuffers, InitializeTwoBuffers,
                       Ffi::BindInitialize()
                           .Arg<AnyBuffer>()
                           .Arg<AnyBuffer>()
                           .Ret<AnyBuffer>()
                           .Ret<AnyBuffer>()
                           .Ctx<GpuCollectives>());

XLA_FFI_DEFINE_HANDLER(kExecuteTwoBuffers, ExecuteTwoBuffers,
                       Ffi::BindExecute()
                           .Arg<AnyBuffer>()
                           .Arg<AnyBuffer>()
                           .Ret<AnyBuffer>()
                           .Ret<AnyBuffer>()
                           .Ctx<Initialized<CollectiveState>>()
                           .Ctx<GpuCollectives>()
                           .Ctx<PlatformStream<cudaStream_t>>());

}  // namespace

extern "C" __attribute__((visibility("default"))) XLA_FFI_Error*
XlaGpuCollectivesFfiRegister(const XLA_FFI_Api* api) {
  if (XLA_FFI_Error* error = Ffi::RegisterTypeId(
          api, "xla.gpu.collectives.public_dso_test.state.v1",
          &CollectiveState::id, &CollectiveState::info)) {
    return error;
  }

  if (XLA_FFI_Error* error = Ffi::RegisterStaticHandler(
          api, "__xla_test_gpu_collectives_public_dso", "gpu",
          XLA_FFI_Handler_Bundle{
              /*instantiate=*/nullptr,
              /*prepare=*/kPrepare,
              /*initialize=*/kInitialize,
              /*execute=*/kExecute,
          },
          static_cast<XLA_FFI_Handler_Traits>(
              xla::ffi::Traits::kUsesDeviceCommunication))) {
    return error;
  }

  return Ffi::RegisterStaticHandler(
      api, "__xla_test_gpu_collectives_public_dso_two_buffers", "gpu",
      XLA_FFI_Handler_Bundle{
          /*instantiate=*/nullptr,
          /*prepare=*/nullptr,
          /*initialize=*/kInitializeTwoBuffers,
          /*execute=*/kExecuteTwoBuffers,
      },
      static_cast<XLA_FFI_Handler_Traits>(
          xla::ffi::Traits::kUsesDeviceCommunication));
}
