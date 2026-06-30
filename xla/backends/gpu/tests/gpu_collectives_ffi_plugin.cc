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
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "third_party/nccl/nccl.h"
#include "xla/backends/gpu/tests/gpu_collectives_ffi_plugin_kernel.h"
#include "xla/ffi/api/c_api_gpu_collectives_nccl.h"
#include "xla/ffi/api/gpu_collectives.h"

namespace {

using xla::ffi::AnyBuffer;
using xla::ffi::Error;
using xla::ffi::ErrorCode;
using xla::ffi::ErrorOr;
using xla::ffi::Ffi;
using xla::ffi::GpuCollectiveDeviceMemory;
using xla::ffi::GpuCollectivePackedKernelArgMetadata;
using xla::ffi::GpuCollectives;
using xla::ffi::Initialized;
using xla::ffi::PlatformStream;
using xla::ffi::Result;
using xla::ffi::TypeId;
using xla::ffi::TypeInfo;
using xla::ffi::Unexpected;

Error FailedPrecondition(std::string message) {
  return Error(ErrorCode::kFailedPrecondition, std::move(message));
}

bool SameMetadata(const GpuCollectivePackedKernelArgMetadata& lhs,
                  const GpuCollectivePackedKernelArgMetadata& rhs) {
  return lhs.size == rhs.size && lhs.alignment == rhs.alignment &&
         lhs.schema == rhs.schema && lhs.abi_version == rhs.abi_version;
}

Error ValidateMetadata(const GpuCollectivePackedKernelArgMetadata& metadata,
                       uint64_t expected_schema) {
  if (metadata.size == 0) {
    return FailedPrecondition("packed kernel argument has zero size");
  }
  if (metadata.alignment == 0 ||
      (metadata.alignment & (metadata.alignment - 1)) != 0) {
    return FailedPrecondition(
        "packed kernel argument alignment is not a power of two");
  }
  if (metadata.schema != expected_schema) {
    return FailedPrecondition("unexpected packed kernel argument schema");
  }
  if (metadata.abi_version != NCCL_VERSION_CODE) {
    return FailedPrecondition("unexpected NCCL device ABI version");
  }
  return Error::Success();
}

struct AlignedStorage {
  std::vector<uint8_t> storage;
  void* data = nullptr;
};

ErrorOr<AlignedStorage> MakeAlignedStorage(size_t size, size_t alignment) {
  if (size == 0 || alignment == 0 || (alignment & (alignment - 1)) != 0) {
    return Unexpected(FailedPrecondition(
        "cannot allocate storage for invalid size or alignment"));
  }
  if (size > std::numeric_limits<size_t>::max() - (alignment - 1)) {
    return Unexpected(
        FailedPrecondition("packed kernel argument storage size overflows"));
  }

  AlignedStorage result;
  result.storage.resize(size + alignment - 1);
  uintptr_t begin = reinterpret_cast<uintptr_t>(result.storage.data());
  if (begin > std::numeric_limits<uintptr_t>::max() - (alignment - 1)) {
    return Unexpected(FailedPrecondition(
        "packed kernel argument alignment calculation overflows"));
  }
  uintptr_t aligned =
      (begin + alignment - 1) & ~(static_cast<uintptr_t>(alignment) - 1);
  result.data = reinterpret_cast<void*>(aligned);
  return result;
}

struct DeviceCommSnapshot {
  GpuCollectivePackedKernelArgMetadata metadata;
  std::vector<uint8_t> bytes;
};

struct DeviceMemorySnapshot {
  GpuCollectiveDeviceMemory result;
  std::vector<uint8_t> bytes;
};

ErrorOr<DeviceCommSnapshot> PackDeviceComm(const GpuCollectives& collectives) {
  GpuCollectivePackedKernelArgMetadata query;
  Error error = collectives.GetDeviceComm(
      /*destination=*/nullptr, /*capacity=*/0, &query);
  if (error.failure()) return Unexpected(std::move(error));

  error = ValidateMetadata(query,
                           XLA_FFI_GpuCollective_NCCL_DEVICE_COMM_ABI_SCHEMA);
  if (error.failure()) return Unexpected(std::move(error));

  ErrorOr<AlignedStorage> storage =
      MakeAlignedStorage(query.size, query.alignment);
  if (!storage.has_value()) return Unexpected(storage.error());

  GpuCollectivePackedKernelArgMetadata packed;
  error = collectives.GetDeviceComm(storage->data, query.size, &packed);
  if (error.failure()) return Unexpected(std::move(error));
  if (!SameMetadata(query, packed)) {
    return Unexpected(FailedPrecondition(
        "device communicator query and pack metadata are inconsistent"));
  }

  DeviceCommSnapshot snapshot;
  snapshot.metadata = packed;
  snapshot.bytes.resize(packed.size);
  std::memcpy(snapshot.bytes.data(), storage->data, snapshot.bytes.size());
  return snapshot;
}

ErrorOr<DeviceMemorySnapshot> PackDeviceMemory(
    const GpuCollectives& collectives, AnyBuffer buffer) {
  AnyBuffer::Dimensions dimensions = buffer.dimensions();
  if (buffer.element_type() != xla::ffi::U32 || dimensions.size() != 1 ||
      dimensions[0] < 2) {
    return Unexpected(FailedPrecondition(
        "device-memory test view requires a U32 vector with at least two "
        "elements"));
  }

  // Query a strict subview so the end-to-end test exercises view offsets even
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

  GpuCollectiveDeviceMemory query;
  Error error = collectives.GetDeviceMemory(&view, /*destination=*/nullptr,
                                            /*capacity=*/0, &query);
  if (error.failure()) return Unexpected(std::move(error));

  error = ValidateMetadata(
      query.metadata, XLA_FFI_GpuCollective_NCCL_SYMMETRIC_MEMORY_ABI_SCHEMA);
  if (error.failure()) return Unexpected(std::move(error));
  if (query.offset == 0) {
    return Unexpected(FailedPrecondition(
        "device-memory view did not preserve a nonzero allocation offset"));
  }

  ErrorOr<AlignedStorage> storage =
      MakeAlignedStorage(query.metadata.size, query.metadata.alignment);
  if (!storage.has_value()) return Unexpected(storage.error());

  GpuCollectiveDeviceMemory packed;
  error = collectives.GetDeviceMemory(&view, storage->data,
                                      query.metadata.size, &packed);
  if (error.failure()) return Unexpected(std::move(error));
  if (!SameMetadata(query.metadata, packed.metadata) ||
      query.offset != packed.offset) {
    return Unexpected(FailedPrecondition(
        "device-memory query and pack metadata are inconsistent"));
  }

  DeviceMemorySnapshot snapshot;
  snapshot.result = packed;
  snapshot.bytes.resize(packed.metadata.size);
  std::memcpy(snapshot.bytes.data(), storage->data, snapshot.bytes.size());
  return snapshot;
}

bool SameDeviceCommSnapshot(const DeviceCommSnapshot& lhs,
                            const DeviceCommSnapshot& rhs) {
  return SameMetadata(lhs.metadata, rhs.metadata) && lhs.bytes == rhs.bytes;
}

bool SameDeviceMemorySnapshot(const DeviceMemorySnapshot& lhs,
                              const DeviceMemorySnapshot& rhs) {
  return SameMetadata(lhs.result.metadata, rhs.result.metadata) &&
         lhs.result.offset == rhs.result.offset && lhs.bytes == rhs.bytes;
}

struct CollectiveState {
  static TypeId id;
  static TypeInfo info;

  DeviceCommSnapshot device_comm;
  DeviceMemorySnapshot device_memory;
};

TypeId CollectiveState::id = {};
TypeInfo CollectiveState::info = xla::ffi::MakeTypeInfo<CollectiveState>();

ErrorOr<std::unique_ptr<CollectiveState>> Initialize(
    AnyBuffer operand, Result<AnyBuffer>, GpuCollectives collectives) {
  ErrorOr<DeviceCommSnapshot> device_comm = PackDeviceComm(collectives);
  if (!device_comm.has_value()) return Unexpected(device_comm.error());
  ErrorOr<DeviceMemorySnapshot> device_memory =
      PackDeviceMemory(collectives, operand);
  if (!device_memory.has_value()) return Unexpected(device_memory.error());

  auto state = std::make_unique<CollectiveState>();
  state->device_comm = *std::move(device_comm);
  state->device_memory = *std::move(device_memory);
  return state;
}

Error Execute(AnyBuffer operand, Result<AnyBuffer> result,
              CollectiveState* initialized, GpuCollectives collectives,
              cudaStream_t stream) {
  if (initialized == nullptr) {
    return FailedPrecondition("FFI initialize state is missing");
  }
  if (operand.untyped_data() != result->untyped_data() ||
      !(operand.dimensions() == result->dimensions())) {
    return FailedPrecondition(
        "custom-call result does not alias its operand exactly");
  }

  ErrorOr<DeviceCommSnapshot> device_comm = PackDeviceComm(collectives);
  if (!device_comm.has_value()) return device_comm.error();
  ErrorOr<DeviceMemorySnapshot> device_memory =
      PackDeviceMemory(collectives, operand);
  if (!device_memory.has_value()) return device_memory.error();

  if (!SameDeviceCommSnapshot(initialized->device_comm, *device_comm)) {
    return FailedPrecondition(
        "device communicator changed between Initialize and Execute");
  }
  if (!SameDeviceMemorySnapshot(initialized->device_memory, *device_memory)) {
    return FailedPrecondition(
        "device memory changed between Initialize and Execute");
  }

  if (operand.element_type() != xla::ffi::U32) {
    return FailedPrecondition("test kernel requires a U32 operand");
  }
  cudaError_t launch_status = LaunchGpuCollectivesFfiTestKernel(
      stream, device_comm->bytes.data(), device_comm->bytes.size(),
      device_memory->bytes.data(), device_memory->bytes.size(),
      device_memory->result.offset, operand.element_count() - 1);
  if (launch_status != cudaSuccess) {
    return FailedPrecondition(
        std::string("failed to launch NCCL LSA kernel: ") +
        cudaGetErrorString(launch_status));
  }
  return Error::Success();
}

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

}  // namespace

extern "C" __attribute__((visibility("default"))) XLA_FFI_Error*
XlaGpuCollectivesFfiRegister(const XLA_FFI_Api* api) {
  if (XLA_FFI_Error* error = Ffi::RegisterTypeId(
          api, "xla.gpu.collectives.public_dso_test.state.v1",
          &CollectiveState::id, &CollectiveState::info)) {
    return error;
  }

  return Ffi::RegisterStaticHandler(
      api, "__xla_test_gpu_collectives_public_dso", "gpu",
      XLA_FFI_Handler_Bundle{
          /*instantiate=*/nullptr,
          /*prepare=*/nullptr,
          /*initialize=*/kInitialize,
          /*execute=*/kExecute,
      },
      static_cast<XLA_FFI_Handler_Traits>(
          xla::ffi::Traits::kUsesDeviceCommunication));
}
