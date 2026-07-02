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

#include "xla/ffi/api/gpu_collectives.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>

#include <gtest/gtest.h>
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/c_api_gpu_collectives.h"
#include "xla/ffi/api/c_api_gpu_collectives_nccl.h"
#include "xla/ffi/api/ffi.h"

namespace xla::ffi {
namespace {

static_assert(offsetof(XLA_FFI_Api, XLA_FFI_Error_GetCode) >=
              XLA_FFI_STRUCT_SIZE(XLA_FFI_Api, XLA_FFI_DeviceOrdinal_Get));
static_assert(XLA_FFI_GpuDeviceCommunication_Requirements_STRUCT_SIZE_V1_1 <=
              XLA_FFI_GpuDeviceCommunication_Requirements_STRUCT_SIZE);
static_assert(
    XLA_FFI_GpuCollectives_RequestDeviceCommunication_Args_STRUCT_SIZE_V1_1 <=
    XLA_FFI_GpuCollectives_RequestDeviceCommunication_Args_STRUCT_SIZE);
static_assert(XLA_FFI_GpuDeviceCommunication_Info_STRUCT_SIZE_V1_1 <=
              XLA_FFI_GpuDeviceCommunication_Info_STRUCT_SIZE);
static_assert(
    XLA_FFI_GpuCollectives_GetDeviceCommunicationInfo_Args_STRUCT_SIZE_V1_1 <=
    XLA_FFI_GpuCollectives_GetDeviceCommunicationInfo_Args_STRUCT_SIZE);
static_assert(XLA_FFI_GpuCollectives_GetDeviceComm_Args_STRUCT_SIZE_V1_0 <=
              XLA_FFI_GpuCollectives_GetDeviceComm_Args_STRUCT_SIZE);
static_assert(XLA_FFI_GpuCollectives_GetDeviceMemory_Args_STRUCT_SIZE_V1_0 <=
              XLA_FFI_GpuCollectives_GetDeviceMemory_Args_STRUCT_SIZE);

constexpr uint64_t kNcclAbiVersion = 22907;
constexpr size_t kDeviceCommSize = 64;
constexpr size_t kDeviceMemorySize = 8;

struct DeviceCommCall {
  uint64_t expected_abi_schema = 0;
  uint64_t expected_abi_version = 0;
  void* destination = nullptr;
  size_t destination_size = 0;
};

struct DeviceMemoryCall {
  XLA_FFI_DataType dtype = XLA_FFI_DataType_INVALID;
  void* buffer_data = nullptr;
  int64_t buffer_rank = 0;
  uint64_t expected_abi_schema = 0;
  uint64_t expected_abi_version = 0;
  void* destination = nullptr;
  size_t destination_size = 0;
};

struct DeviceCommunicationRequestCall {
  XLA_FFI_ExecutionContext* ctx = nullptr;
  XLA_FFI_GpuDeviceCommunication_Requirements requirements = {};
};

struct DeviceCommunicationInfoCall {
  XLA_FFI_ExecutionContext* ctx = nullptr;
  bool was_zero_initialized = false;
};

DeviceCommCall device_comm_call;
DeviceMemoryCall device_memory_call;
DeviceCommunicationRequestCall device_communication_request_call;
DeviceCommunicationInfoCall device_communication_info_call;

XLA_FFI_Error* CreateError(XLA_FFI_Error_Code errc, const char* message) {
  XLA_FFI_Error_Create_Args args = {};
  args.struct_size = XLA_FFI_Error_Create_Args_STRUCT_SIZE;
  args.message = message;
  args.errc = errc;
  return GetXlaFfiApi()->XLA_FFI_Error_Create(&args);
}

XLA_FFI_Error* GetDeviceCommSuccess(
    XLA_FFI_GpuCollectives_GetDeviceComm_Args* args) {
  device_comm_call = DeviceCommCall{
      args->expected_abi_schema,
      args->expected_abi_version,
      args->destination,
      args->destination_size,
  };
  static_cast<uint8_t*>(args->destination)[0] = 0x5a;
  return nullptr;
}

XLA_FFI_Error* GetDeviceCommUnavailable(
    XLA_FFI_GpuCollectives_GetDeviceComm_Args*) {
  return CreateError(XLA_FFI_Error_Code_UNAVAILABLE,
                     "device communicator unavailable");
}

XLA_FFI_Error* GetDeviceMemorySuccess(
    XLA_FFI_GpuCollectives_GetDeviceMemory_Args* args) {
  device_memory_call = DeviceMemoryCall{
      args->buffer->dtype,       args->buffer->data,         args->buffer->rank,
      args->expected_abi_schema, args->expected_abi_version, args->destination,
      args->destination_size,
  };
  static_cast<uint8_t*>(args->destination)[0] = 0xa5;
  args->offset = 128;
  return nullptr;
}

XLA_FFI_Error* RequestDeviceCommunicationSuccess(
    XLA_FFI_GpuCollectives_RequestDeviceCommunication_Args* args) {
  device_communication_request_call = {
      args->ctx,
      *args->requirements,
  };
  return nullptr;
}

XLA_FFI_Error* GetDeviceCommunicationInfoSuccess(
    XLA_FFI_GpuCollectives_GetDeviceCommunicationInfo_Args* args) {
  device_communication_info_call = {
      args->ctx,
      args->info->extension_start == nullptr && args->info->rank == 0 &&
          args->info->team_size == 0 && args->info->enabled_features == 0,
  };
  args->info->rank = 3;
  args->info->team_size = 8;
  args->info->local_rank = 1;
  args->info->local_domain_size = 4;
  args->info->local_domain_count = 2;
  args->info->topology = XLA_FFI_GPU_TOPOLOGY_HIERARCHICAL;
  args->info->enabled_features =
      XLA_FFI_GPU_DEVICE_COMM_FEATURE_LOCAL_MULTICAST |
      XLA_FFI_GPU_DEVICE_COMM_FEATURE_NETWORK_DEVICE_OPERATIONS;
  args->info->team_barrier_count = 2;
  args->info->local_barrier_count = 3;
  args->info->notification_slot_count = 5;
  args->info->completion_slot_count = 7;
  return nullptr;
}

XLA_FFI_GpuCollectives_Extension MakeExtension() {
  XLA_FFI_GpuCollectives_Extension extension = {};
  extension.extension_base.struct_size =
      XLA_FFI_GpuCollectives_Extension_STRUCT_SIZE;
  extension.extension_base.type = XLA_FFI_Extension_GpuCollectives;
  extension.api_major_version = XLA_FFI_GPU_COLLECTIVES_API_MAJOR;
  extension.api_minor_version = XLA_FFI_GPU_COLLECTIVES_API_MINOR;
  extension.get_device_comm = GetDeviceCommSuccess;
  extension.get_device_memory = GetDeviceMemorySuccess;
  extension.request_device_communication = RequestDeviceCommunicationSuccess;
  extension.get_device_communication_info = GetDeviceCommunicationInfoSuccess;
  return extension;
}

XLA_FFI_Api MakeApi(XLA_FFI_Extension_Base* extension) {
  XLA_FFI_Api api = *GetXlaFfiApi();
  api.extension_start = extension;
  return api;
}

TEST(GpuCollectivesTest, FindsExtensionInChainAndForwardsBothOperations) {
  XLA_FFI_GpuCollectives_Extension extension = MakeExtension();
  XLA_FFI_Extension_Base other = {
      XLA_FFI_Extension_Base_STRUCT_SIZE,
      XLA_FFI_Extension_Metadata,
      &extension.extension_base,
  };
  XLA_FFI_Api api = MakeApi(&other);
  GpuCollectives collectives(
      &api, reinterpret_cast<XLA_FFI_ExecutionContext*>(uintptr_t{1}));
  ASSERT_TRUE(collectives.available());

  std::array<uint8_t, kDeviceCommSize> comm = {};
  Error error = collectives.GetDeviceComm(
      XLA_FFI_GpuCollective_NCCL_DEVICE_COMM_ABI_SCHEMA, kNcclAbiVersion,
      comm.data(), comm.size());
  EXPECT_TRUE(error.success());
  EXPECT_EQ(device_comm_call.expected_abi_schema,
            XLA_FFI_GpuCollective_NCCL_DEVICE_COMM_ABI_SCHEMA);
  EXPECT_EQ(device_comm_call.expected_abi_version, kNcclAbiVersion);
  EXPECT_EQ(device_comm_call.destination, comm.data());
  EXPECT_EQ(device_comm_call.destination_size, comm.size());
  EXPECT_EQ(comm[0], 0x5a);

  int64_t dims[] = {4};
  XLA_FFI_Buffer buffer = {
      XLA_FFI_Buffer_STRUCT_SIZE,
      /*extension_start=*/nullptr,
      XLA_FFI_DataType_F32,
      reinterpret_cast<void*>(uintptr_t{0x1000}),
      /*rank=*/1,
      dims,
  };
  std::array<uint8_t, kDeviceMemorySize> memory = {};
  uint64_t offset = 0;
  error = collectives.GetDeviceMemory(
      AnyBuffer(&buffer),
      XLA_FFI_GpuCollective_NCCL_SYMMETRIC_MEMORY_ABI_SCHEMA, kNcclAbiVersion,
      memory.data(), memory.size(), &offset);
  EXPECT_TRUE(error.success());
  EXPECT_EQ(device_memory_call.dtype, XLA_FFI_DataType_F32);
  EXPECT_EQ(device_memory_call.buffer_data, buffer.data);
  EXPECT_EQ(device_memory_call.buffer_rank, 1);
  EXPECT_EQ(device_memory_call.expected_abi_schema,
            XLA_FFI_GpuCollective_NCCL_SYMMETRIC_MEMORY_ABI_SCHEMA);
  EXPECT_EQ(device_memory_call.expected_abi_version, kNcclAbiVersion);
  EXPECT_EQ(device_memory_call.destination, memory.data());
  EXPECT_EQ(device_memory_call.destination_size, memory.size());
  EXPECT_EQ(memory[0], 0xa5);
  EXPECT_EQ(offset, 128);
}

TEST(GpuCollectivesTest, RequestsRequirementsAndReturnsResolvedInfo) {
  static_assert(std::is_same_v<GpuDeviceCommunication, GpuCollectives>);

  XLA_FFI_GpuCollectives_Extension extension = MakeExtension();
  XLA_FFI_Api api = MakeApi(&extension.extension_base);
  XLA_FFI_ExecutionContext* ctx =
      reinterpret_cast<XLA_FFI_ExecutionContext*>(uintptr_t{1});
  GpuDeviceCommunication communication(&api, ctx);

  DeviceCommunicationRequirements requirements;
  requirements.peer_access = PeerAccess::kHierarchical;
  requirements.required_features =
      DeviceCommunicationFeature::kNetworkDeviceOperations;
  requirements.preferred_features =
      DeviceCommunicationFeature::kLocalMulticast |
      DeviceCommunicationFeature::kNetworkDeviceOperations;
  requirements.local_barriers = 2;
  requirements.team_barriers = 1;
  requirements.notification_slots = 8;
  requirements.completion_slots = 4;

  Error error = communication.Request(requirements);
  ASSERT_TRUE(error.success()) << error.message();
  EXPECT_EQ(device_communication_request_call.ctx, ctx);
  const XLA_FFI_GpuDeviceCommunication_Requirements& c_requirements =
      device_communication_request_call.requirements;
  EXPECT_EQ(c_requirements.struct_size,
            XLA_FFI_GpuDeviceCommunication_Requirements_STRUCT_SIZE);
  EXPECT_EQ(c_requirements.extension_start, nullptr);
  EXPECT_EQ(c_requirements.peer_access, XLA_FFI_GPU_PEER_ACCESS_HIERARCHICAL);
  EXPECT_EQ(c_requirements.required_features,
            XLA_FFI_GPU_DEVICE_COMM_FEATURE_NETWORK_DEVICE_OPERATIONS);
  EXPECT_EQ(c_requirements.preferred_features,
            XLA_FFI_GPU_DEVICE_COMM_FEATURE_LOCAL_MULTICAST |
                XLA_FFI_GPU_DEVICE_COMM_FEATURE_NETWORK_DEVICE_OPERATIONS);
  EXPECT_EQ(c_requirements.local_barrier_count, 2);
  EXPECT_EQ(c_requirements.team_barrier_count, 1);
  EXPECT_EQ(c_requirements.notification_slot_count, 8);
  EXPECT_EQ(c_requirements.completion_slot_count, 4);

  DeviceCommunicationInfo info;
  error = communication.GetInfo(&info);
  ASSERT_TRUE(error.success()) << error.message();
  EXPECT_EQ(device_communication_info_call.ctx, ctx);
  EXPECT_TRUE(device_communication_info_call.was_zero_initialized);
  EXPECT_EQ(info.rank, 3);
  EXPECT_EQ(info.team_size, 8);
  EXPECT_EQ(info.local_rank, 1);
  EXPECT_EQ(info.local_domain_size, 4);
  EXPECT_EQ(info.local_domain_count, 2);
  EXPECT_EQ(info.topology, CommunicationTopology::kHierarchical);
  EXPECT_TRUE(info.enabled_features.contains(
      DeviceCommunicationFeature::kLocalMulticast));
  EXPECT_TRUE(info.enabled_features.contains(
      DeviceCommunicationFeature::kNetworkDeviceOperations));
  EXPECT_EQ(info.team_barrier_count, 2);
  EXPECT_EQ(info.local_barrier_count, 3);
  EXPECT_EQ(info.notification_slot_count, 5);
  EXPECT_EQ(info.completion_slot_count, 7);
}

TEST(GpuCollectivesTest, UsesAllZeroRequirementsAsLocalDomainProfile) {
  XLA_FFI_GpuCollectives_Extension extension = MakeExtension();
  XLA_FFI_Api api = MakeApi(&extension.extension_base);
  GpuDeviceCommunication communication(
      &api, reinterpret_cast<XLA_FFI_ExecutionContext*>(uintptr_t{1}));

  Error error = communication.Request(DeviceCommunicationRequirements{});
  ASSERT_TRUE(error.success()) << error.message();
  const XLA_FFI_GpuDeviceCommunication_Requirements& requirements =
      device_communication_request_call.requirements;
  EXPECT_EQ(requirements.peer_access, XLA_FFI_GPU_PEER_ACCESS_LOCAL_DOMAIN);
  EXPECT_EQ(requirements.required_features, 0);
  EXPECT_EQ(requirements.preferred_features, 0);
  EXPECT_EQ(requirements.local_barrier_count, 0);
  EXPECT_EQ(requirements.team_barrier_count, 0);
  EXPECT_EQ(requirements.notification_slot_count, 0);
  EXPECT_EQ(requirements.completion_slot_count, 0);
}

TEST(GpuCollectivesTest, PreservesExactErrorCodeAndMessage) {
  XLA_FFI_GpuCollectives_Extension extension = MakeExtension();
  extension.get_device_comm = GetDeviceCommUnavailable;
  XLA_FFI_Api api = MakeApi(&extension.extension_base);
  GpuCollectives collectives(
      &api, reinterpret_cast<XLA_FFI_ExecutionContext*>(uintptr_t{1}));

  std::array<uint8_t, kDeviceCommSize> destination = {};
  Error error = collectives.GetDeviceComm(
      XLA_FFI_GpuCollective_NCCL_DEVICE_COMM_ABI_SCHEMA, kNcclAbiVersion,
      destination.data(), destination.size());

  EXPECT_EQ(error.errc(), ErrorCode::kUnavailable);
  EXPECT_EQ(error.message(), "device communicator unavailable");
}

TEST(GpuCollectivesTest, RejectsInvalidBufferDestinationAndOffset) {
  XLA_FFI_GpuCollectives_Extension extension = MakeExtension();
  XLA_FFI_Api api = MakeApi(&extension.extension_base);
  GpuCollectives collectives(
      &api, reinterpret_cast<XLA_FFI_ExecutionContext*>(uintptr_t{1}));

  Error error = collectives.GetDeviceComm(
      XLA_FFI_GpuCollective_NCCL_DEVICE_COMM_ABI_SCHEMA, kNcclAbiVersion,
      /*destination=*/nullptr, kDeviceCommSize);
  EXPECT_EQ(error.errc(), ErrorCode::kInvalidArgument);

  std::array<uint8_t, kDeviceCommSize> comm = {};
  error = collectives.GetDeviceComm(
      XLA_FFI_GpuCollective_NCCL_DEVICE_COMM_ABI_SCHEMA, kNcclAbiVersion,
      comm.data(), /*destination_size=*/0);
  EXPECT_EQ(error.errc(), ErrorCode::kInvalidArgument);

  XLA_FFI_Buffer buffer = {XLA_FFI_Buffer_STRUCT_SIZE};
  std::array<uint8_t, kDeviceMemorySize> memory = {};
  uint64_t offset = 0;
  error = collectives.GetDeviceMemory(
      /*buffer=*/nullptr,
      XLA_FFI_GpuCollective_NCCL_SYMMETRIC_MEMORY_ABI_SCHEMA, kNcclAbiVersion,
      memory.data(), memory.size(), &offset);
  EXPECT_EQ(error.errc(), ErrorCode::kInvalidArgument);

  error = collectives.GetDeviceMemory(
      &buffer, XLA_FFI_GpuCollective_NCCL_SYMMETRIC_MEMORY_ABI_SCHEMA,
      kNcclAbiVersion, memory.data(), memory.size(), /*offset=*/nullptr);
  EXPECT_EQ(error.errc(), ErrorCode::kInvalidArgument);

  error = collectives.GetInfo(/*info=*/nullptr);
  EXPECT_EQ(error.errc(), ErrorCode::kInvalidArgument);
}

TEST(GpuCollectivesTest, DecodesWhenExtensionIsUnavailable) {
  XLA_FFI_Api api = MakeApi(/*extension=*/nullptr);
  DiagnosticEngine diagnostic;
  std::optional<GpuCollectives> collectives =
      CtxDecoding<GpuCollectives>::Decode(
          &api, reinterpret_cast<XLA_FFI_ExecutionContext*>(uintptr_t{1}),
          diagnostic);
  ASSERT_TRUE(collectives.has_value());
  EXPECT_FALSE(collectives->available());

  std::array<uint8_t, kDeviceCommSize> destination = {};
  Error error = collectives->GetDeviceComm(
      XLA_FFI_GpuCollective_NCCL_DEVICE_COMM_ABI_SCHEMA, kNcclAbiVersion,
      destination.data(), destination.size());
  EXPECT_EQ(error.errc(), ErrorCode::kUnimplemented);
}

TEST(GpuCollectivesTest, ChecksCompatibilityPerOperation) {
  XLA_FFI_GpuCollectives_Extension extension = MakeExtension();
  XLA_FFI_Api api = MakeApi(&extension.extension_base);
  GpuCollectives collectives(
      &api, reinterpret_cast<XLA_FFI_ExecutionContext*>(uintptr_t{1}));

  extension.extension_base.struct_size = XLA_FFI_Extension_Base_STRUCT_SIZE;
  EXPECT_FALSE(collectives.available());

  extension.extension_base.struct_size =
      XLA_FFI_STRUCT_SIZE(XLA_FFI_GpuCollectives_Extension, get_device_comm);
  extension.api_minor_version = 0;
  std::array<uint8_t, kDeviceCommSize> comm = {};
  Error error = collectives.GetDeviceComm(
      XLA_FFI_GpuCollective_NCCL_DEVICE_COMM_ABI_SCHEMA, kNcclAbiVersion,
      comm.data(), comm.size());
  EXPECT_TRUE(error.success()) << error.message();
  EXPECT_FALSE(collectives.available());

  XLA_FFI_Buffer buffer = {XLA_FFI_Buffer_STRUCT_SIZE};
  std::array<uint8_t, kDeviceMemorySize> memory = {};
  uint64_t offset = 0;
  error = collectives.GetDeviceMemory(
      &buffer, XLA_FFI_GpuCollective_NCCL_SYMMETRIC_MEMORY_ABI_SCHEMA,
      kNcclAbiVersion, memory.data(), memory.size(), &offset);
  EXPECT_EQ(error.errc(), ErrorCode::kUnimplemented);

  extension.extension_base.struct_size =
      XLA_FFI_STRUCT_SIZE(XLA_FFI_GpuCollectives_Extension, get_device_memory);
  EXPECT_TRUE(collectives.available());
  error = collectives.GetDeviceMemory(
      &buffer, XLA_FFI_GpuCollective_NCCL_SYMMETRIC_MEMORY_ABI_SCHEMA,
      kNcclAbiVersion, memory.data(), memory.size(), &offset);
  EXPECT_TRUE(error.success()) << error.message();

  error = collectives.Request(DeviceCommunicationRequirements{});
  EXPECT_EQ(error.errc(), ErrorCode::kUnimplemented);
  DeviceCommunicationInfo info;
  error = collectives.GetInfo(&info);
  EXPECT_EQ(error.errc(), ErrorCode::kUnimplemented);

  extension.extension_base.struct_size = XLA_FFI_STRUCT_SIZE(
      XLA_FFI_GpuCollectives_Extension, request_device_communication);
  extension.api_minor_version = 1;
  error = collectives.Request(DeviceCommunicationRequirements{});
  EXPECT_TRUE(error.success()) << error.message();
  error = collectives.GetInfo(&info);
  EXPECT_EQ(error.errc(), ErrorCode::kUnimplemented);

  extension.extension_base.struct_size =
      XLA_FFI_GpuCollectives_Extension_STRUCT_SIZE;
  extension.api_major_version = 0;  // Legacy pre-1.0 callback table.
  EXPECT_FALSE(collectives.available());

  extension.api_major_version = XLA_FFI_GPU_COLLECTIVES_API_MAJOR + 1;
  EXPECT_FALSE(collectives.available());

  extension.api_major_version = XLA_FFI_GPU_COLLECTIVES_API_MAJOR;
  extension.api_minor_version = 0;
  EXPECT_TRUE(collectives.available());
  error = collectives.GetDeviceComm(
      XLA_FFI_GpuCollective_NCCL_DEVICE_COMM_ABI_SCHEMA, kNcclAbiVersion,
      comm.data(), comm.size());
  EXPECT_TRUE(error.success()) << error.message();
  error = collectives.Request(DeviceCommunicationRequirements{});
  EXPECT_EQ(error.errc(), ErrorCode::kUnimplemented);
}

TEST(GpuCollectivesTest, RejectsTruncatedExtensionBase) {
  struct TruncatedExtension {
    size_t struct_size;
  } truncated_extension{/*struct_size=*/sizeof(size_t)};
  XLA_FFI_Api api =
      MakeApi(reinterpret_cast<XLA_FFI_Extension_Base*>(&truncated_extension));

  GpuCollectives collectives(
      &api, reinterpret_cast<XLA_FFI_ExecutionContext*>(uintptr_t{1}));
  EXPECT_FALSE(collectives.available());
}

TEST(GpuCollectivesTest, AcceptsFutureMinorAndLargerExtension) {
  struct FutureGpuCollectivesExtension {
    XLA_FFI_GpuCollectives_Extension extension;
    void* future_field;
  } future_extension{MakeExtension(), /*future_field=*/nullptr};
  future_extension.extension.extension_base.struct_size =
      sizeof(FutureGpuCollectivesExtension);
  future_extension.extension.api_minor_version =
      XLA_FFI_GPU_COLLECTIVES_API_MINOR + 1;
  XLA_FFI_Api api = MakeApi(&future_extension.extension.extension_base);

  GpuCollectives collectives(
      &api, reinterpret_cast<XLA_FFI_ExecutionContext*>(uintptr_t{1}));
  EXPECT_TRUE(collectives.available());
}

}  // namespace
}  // namespace xla::ffi
