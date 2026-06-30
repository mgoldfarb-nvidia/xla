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

#include <cstddef>
#include <cstdint>
#include <optional>

#include <gtest/gtest.h>
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/c_api_gpu_collectives.h"
#include "xla/ffi/api/c_api_gpu_collectives_nccl.h"
#include "xla/ffi/api/ffi.h"

namespace xla::ffi {
namespace {

static_assert(offsetof(XLA_FFI_Api, XLA_FFI_Error_GetCode) >=
              XLA_FFI_STRUCT_SIZE(XLA_FFI_Api, XLA_FFI_DeviceOrdinal_Get));

XLA_FFI_Error* CreateError(XLA_FFI_Error_Code errc, const char* message) {
  XLA_FFI_Error_Create_Args args = {};
  args.struct_size = XLA_FFI_Error_Create_Args_STRUCT_SIZE;
  args.message = message;
  args.errc = errc;
  return GetXlaFfiApi()->XLA_FFI_Error_Create(&args);
}

XLA_FFI_Error* GetDeviceCommSuccess(
    XLA_FFI_GpuCollectives_GetDeviceComm_Args* args) {
  args->kernel_arg->size = 64;
  args->kernel_arg->alignment = 16;
  args->kernel_arg->schema = XLA_FFI_GpuCollective_NCCL_DEVICE_COMM_ABI_SCHEMA;
  args->kernel_arg->abi_version = 22907;
  return nullptr;
}

XLA_FFI_Error* GetDeviceCommResourceExhausted(
    XLA_FFI_GpuCollectives_GetDeviceComm_Args*) {
  return CreateError(XLA_FFI_Error_Code_RESOURCE_EXHAUSTED,
                     "resource capacity exhausted");
}

XLA_FFI_Error* GetDeviceMemorySuccess(
    XLA_FFI_GpuCollectives_GetDeviceMemory_Args* args) {
  args->kernel_arg->size = 8;
  args->kernel_arg->alignment = 8;
  args->kernel_arg->schema =
      XLA_FFI_GpuCollective_NCCL_SYMMETRIC_MEMORY_ABI_SCHEMA;
  args->kernel_arg->abi_version = 22907;
  args->offset = 128;
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
  return extension;
}

XLA_FFI_Api MakeApi(XLA_FFI_Extension_Base* extension) {
  XLA_FFI_Api api = *GetXlaFfiApi();
  api.extension_start = extension;
  return api;
}

TEST(GpuCollectivesTest, FindsExtensionInChainAndForwardsBothQueries) {
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

  GpuCollectivePackedKernelArgMetadata comm;
  Error error = collectives.GetDeviceComm(
      /*destination=*/nullptr, /*capacity=*/0, &comm);
  EXPECT_TRUE(error.success());
  EXPECT_EQ(comm.size, 64);
  EXPECT_EQ(comm.alignment, 16);
  EXPECT_EQ(comm.schema, XLA_FFI_GpuCollective_NCCL_DEVICE_COMM_ABI_SCHEMA);
  EXPECT_EQ(comm.abi_version, 22907);

  int64_t dims[] = {4};
  XLA_FFI_Buffer buffer = {
      XLA_FFI_Buffer_STRUCT_SIZE,
      /*extension_start=*/nullptr,
      XLA_FFI_DataType_F32,
      reinterpret_cast<void*>(uintptr_t{0x1000}),
      /*rank=*/1,
      dims,
  };
  GpuCollectiveDeviceMemory memory;
  error = collectives.GetDeviceMemory(
      AnyBuffer(&buffer), /*destination=*/nullptr, /*capacity=*/0, &memory);
  EXPECT_TRUE(error.success());
  EXPECT_EQ(memory.metadata.size, 8);
  EXPECT_EQ(memory.metadata.alignment, 8);
  EXPECT_EQ(memory.metadata.schema,
            XLA_FFI_GpuCollective_NCCL_SYMMETRIC_MEMORY_ABI_SCHEMA);
  EXPECT_EQ(memory.metadata.abi_version, 22907);
  EXPECT_EQ(memory.offset, 128);
}

TEST(GpuCollectivesTest, PreservesExactErrorCodeAndMessage) {
  XLA_FFI_GpuCollectives_Extension extension = MakeExtension();
  extension.get_device_comm = GetDeviceCommResourceExhausted;
  XLA_FFI_Api api = MakeApi(&extension.extension_base);
  GpuCollectives collectives(
      &api, reinterpret_cast<XLA_FFI_ExecutionContext*>(uintptr_t{1}));

  GpuCollectivePackedKernelArgMetadata metadata;
  Error error = collectives.GetDeviceComm(
      /*destination=*/nullptr, /*capacity=*/0, &metadata);

  EXPECT_EQ(error.errc(), ErrorCode::kResourceExhausted);
  EXPECT_EQ(error.message(), "resource capacity exhausted");
}

TEST(GpuCollectivesTest, RejectsNullResultPointers) {
  XLA_FFI_GpuCollectives_Extension extension = MakeExtension();
  XLA_FFI_Api api = MakeApi(&extension.extension_base);
  GpuCollectives collectives(
      &api, reinterpret_cast<XLA_FFI_ExecutionContext*>(uintptr_t{1}));

  Error error = collectives.GetDeviceComm(
      /*destination=*/nullptr, /*capacity=*/0, /*metadata=*/nullptr);
  EXPECT_EQ(error.errc(), ErrorCode::kInvalidArgument);

  XLA_FFI_Buffer buffer = {XLA_FFI_Buffer_STRUCT_SIZE};
  error = collectives.GetDeviceMemory(&buffer, /*destination=*/nullptr,
                                      /*capacity=*/0, /*result=*/nullptr);
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

  GpuCollectivePackedKernelArgMetadata metadata;
  Error error = collectives->GetDeviceComm(
      /*destination=*/nullptr, /*capacity=*/0, &metadata);
  EXPECT_EQ(error.errc(), ErrorCode::kUnimplemented);
}

TEST(GpuCollectivesTest, RejectsTruncatedOrWrongMajorExtension) {
  XLA_FFI_GpuCollectives_Extension extension = MakeExtension();
  XLA_FFI_Api api = MakeApi(&extension.extension_base);
  GpuCollectives collectives(
      &api, reinterpret_cast<XLA_FFI_ExecutionContext*>(uintptr_t{1}));

  extension.extension_base.struct_size = XLA_FFI_Extension_Base_STRUCT_SIZE;
  EXPECT_FALSE(collectives.available());

  extension.extension_base.struct_size =
      XLA_FFI_GpuCollectives_Extension_STRUCT_SIZE;
  extension.api_major_version = 0;  // Legacy pre-1.0 callback table.
  EXPECT_FALSE(collectives.available());

  extension.api_major_version = XLA_FFI_GPU_COLLECTIVES_API_MAJOR + 1;
  EXPECT_FALSE(collectives.available());

  extension.api_major_version = XLA_FFI_GPU_COLLECTIVES_API_MAJOR;
  extension.api_minor_version = XLA_FFI_GPU_COLLECTIVES_API_MINOR - 1;
  EXPECT_FALSE(collectives.available());
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
