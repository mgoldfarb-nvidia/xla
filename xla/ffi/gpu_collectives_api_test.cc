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

#include "xla/ffi/gpu_collectives_api.h"

#include <cstddef>
#include <variant>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/c_api_gpu_collectives.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_registry.h"
#include "xla/ffi/ffi_structs.h"
#include "xla/ffi/invoke.h"

namespace xla::ffi {
namespace {

class TestGpuCollectivesApi final : public GpuCollectivesApi {
 public:
  absl::Status GetDeviceComm(
      XLA_FFI_GpuCollectives_GetDeviceComm_Args*) override {
    ++device_comm_gets;
    return device_comm_status;
  }

  absl::Status GetDeviceMemory(
      XLA_FFI_GpuCollectives_GetDeviceMemory_Args*) override {
    ++device_memory_gets;
    return device_memory_status;
  }

  absl::Status device_comm_status = absl::OkStatus();
  absl::Status device_memory_status = absl::OkStatus();
  int device_comm_gets = 0;
  int device_memory_gets = 0;
};

const XLA_FFI_GpuCollectives_Extension* GpuCollectivesExtension() {
  XLA_FFI_Extension_Base* extension = GetXlaFfiApi()->extension_start;
  while (extension != nullptr) {
    if (extension->type == XLA_FFI_Extension_GpuCollectives) {
      return reinterpret_cast<const XLA_FFI_GpuCollectives_Extension*>(
          extension);
    }
    extension = extension->next;
  }
  return nullptr;
}

XLA_FFI_Error_Code TakeErrorCode(XLA_FFI_Error* error) {
  EXPECT_NE(error, nullptr);
  XLA_FFI_Error_GetCode_Args get_code = {};
  get_code.struct_size = XLA_FFI_Error_GetCode_Args_STRUCT_SIZE;
  get_code.error = error;
  GetXlaFfiApi()->XLA_FFI_Error_GetCode(&get_code);

  XLA_FFI_Error_Destroy_Args destroy = {};
  destroy.struct_size = XLA_FFI_Error_Destroy_Args_STRUCT_SIZE;
  destroy.error = error;
  GetXlaFfiApi()->XLA_FFI_Error_Destroy(&destroy);
  return get_code.errc;
}

XLA_FFI_ExecutionContext GpuContext(XLA_FFI_ExecutionStage stage,
                                    GpuCollectivesApi* api) {
  XLA_FFI_ExecutionContext ctx;
  ctx.stage = stage;
  ctx.backend_context = XLA_FFI_ExecutionContext::GpuContext{};
  std::get<XLA_FFI_ExecutionContext::GpuContext>(ctx.backend_context)
      .gpu_collectives = api;
  return ctx;
}

bool legacy_handler_invoked = false;

XLA_FFI_Error* LegacyV03Handler(XLA_FFI_CallFrame* call_frame) {
  if (call_frame->extension_start != nullptr &&
      call_frame->extension_start->type == XLA_FFI_Extension_Metadata) {
    auto* extension = reinterpret_cast<XLA_FFI_Metadata_Extension*>(
        call_frame->extension_start);
    extension->metadata->api_version.struct_size =
        XLA_FFI_Api_Version_STRUCT_SIZE;
    extension->metadata->api_version.extension_start = nullptr;
    extension->metadata->api_version.major_version = 0;
    extension->metadata->api_version.minor_version = 3;
    extension->metadata->traits = 0;
    extension->metadata->state_type_id = XLA_FFI_UNKNOWN_TYPE_ID;
    return nullptr;
  }

  legacy_handler_invoked = true;
  return nullptr;
}

TEST(GpuCollectivesApiTest, RootApiPublishesMinimalVersionedExtension) {
  const XLA_FFI_Api* api = GetXlaFfiApi();
  ASSERT_EQ(api->api_version.major_version, 0);
  ASSERT_EQ(api->api_version.minor_version, XLA_FFI_API_MINOR);
  ASSERT_NE(api->XLA_FFI_Error_GetCode, nullptr);

  const XLA_FFI_GpuCollectives_Extension* extension = GpuCollectivesExtension();
  ASSERT_NE(extension, nullptr);
  EXPECT_EQ(extension->extension_base.struct_size,
            XLA_FFI_GpuCollectives_Extension_STRUCT_SIZE);
  EXPECT_EQ(extension->api_major_version, XLA_FFI_GPU_COLLECTIVES_API_MAJOR);
  EXPECT_EQ(extension->api_minor_version, XLA_FFI_GPU_COLLECTIVES_API_MINOR);
  EXPECT_NE(extension->get_device_comm, nullptr);
  EXPECT_NE(extension->get_device_memory, nullptr);
}

TEST(GpuCollectivesApiTest, RegistersAndInvokesLegacyV03Handler) {
  legacy_handler_invoked = false;
  XLA_FFI_Handler_Bundle bundle = {
      /*instantiate=*/nullptr,
      /*prepare=*/nullptr,
      /*initialize=*/nullptr,
      /*execute=*/LegacyV03Handler,
  };
  absl::Status registration = RegisterHandler(
      GetXlaFfiApi(), "gpu-collectives-v03-compat", "Host", bundle,
      /*traits=*/0);
  ASSERT_TRUE(registration.ok()) << registration;

  absl::StatusOr<HandlerRegistration> registered =
      FindHandler("gpu-collectives-v03-compat", "Host");
  ASSERT_TRUE(registered.ok()) << registered.status();
  EXPECT_EQ(registered->metadata.api_version.major_version, 0);
  EXPECT_EQ(registered->metadata.api_version.minor_version, 3);

  CallFrame call_frame =
      CallFrameBuilder(/*num_args=*/0, /*num_rets=*/0).Build();
  absl::Status invocation =
      Invoke(GetXlaFfiApi(), registered->bundle.execute, call_frame);
  EXPECT_TRUE(invocation.ok()) << invocation;
  EXPECT_TRUE(legacy_handler_invoked);
}

TEST(GpuCollectivesApiTest, ValidatesTopLevelStructSize) {
  TestGpuCollectivesApi adapter;
  XLA_FFI_ExecutionContext ctx =
      GpuContext(XLA_FFI_ExecutionStage_INITIALIZE, &adapter);
  XLA_FFI_GpuCollectives_GetDeviceComm_Args args = {};
  args.struct_size = XLA_FFI_GpuCollectives_GetDeviceComm_Args_STRUCT_SIZE - 1;
  args.ctx = &ctx;

  XLA_FFI_Error* error = GpuCollectivesExtension()->get_device_comm(&args);
  EXPECT_EQ(TakeErrorCode(error), XLA_FFI_Error_Code_INVALID_ARGUMENT);
  EXPECT_EQ(adapter.device_comm_gets, 0);

  args.struct_size =
      XLA_FFI_GpuCollectives_GetDeviceComm_Args_STRUCT_SIZE + sizeof(void*);
  error = GpuCollectivesExtension()->get_device_comm(&args);
  EXPECT_EQ(error, nullptr);
  EXPECT_EQ(adapter.device_comm_gets, 1);
}

TEST(GpuCollectivesApiTest, RejectsNullOperationArgs) {
  const XLA_FFI_GpuCollectives_Extension* extension = GpuCollectivesExtension();
  ASSERT_NE(extension, nullptr);

  EXPECT_EQ(TakeErrorCode(extension->get_device_comm(nullptr)),
            XLA_FFI_Error_Code_INVALID_ARGUMENT);
  EXPECT_EQ(TakeErrorCode(extension->get_device_memory(nullptr)),
            XLA_FFI_Error_Code_INVALID_ARGUMENT);
}

TEST(GpuCollectivesApiTest, RejectsWrongStageBeforeForwarding) {
  TestGpuCollectivesApi adapter;
  XLA_FFI_ExecutionContext ctx =
      GpuContext(XLA_FFI_ExecutionStage_PREPARE, &adapter);
  XLA_FFI_GpuCollectives_GetDeviceComm_Args args = {};
  args.struct_size = XLA_FFI_GpuCollectives_GetDeviceComm_Args_STRUCT_SIZE;
  args.ctx = &ctx;

  XLA_FFI_Error* error = GpuCollectivesExtension()->get_device_comm(&args);
  EXPECT_EQ(TakeErrorCode(error), XLA_FFI_Error_Code_FAILED_PRECONDITION);
  EXPECT_EQ(adapter.device_comm_gets, 0);
}

TEST(GpuCollectivesApiTest, RejectsCpuAndMissingGpuAdapter) {
  XLA_FFI_ExecutionContext cpu;
  cpu.stage = XLA_FFI_ExecutionStage_INITIALIZE;
  cpu.backend_context = XLA_FFI_ExecutionContext::CpuContext{};
  XLA_FFI_GpuCollectives_GetDeviceComm_Args args = {};
  args.struct_size = XLA_FFI_GpuCollectives_GetDeviceComm_Args_STRUCT_SIZE;
  args.ctx = &cpu;

  XLA_FFI_Error* error = GpuCollectivesExtension()->get_device_comm(&args);
  EXPECT_EQ(TakeErrorCode(error), XLA_FFI_Error_Code_UNIMPLEMENTED);

  XLA_FFI_ExecutionContext gpu =
      GpuContext(XLA_FFI_ExecutionStage_INITIALIZE, /*api=*/nullptr);
  args.ctx = &gpu;
  error = GpuCollectivesExtension()->get_device_comm(&args);
  EXPECT_EQ(TakeErrorCode(error), XLA_FFI_Error_Code_FAILED_PRECONDITION);
}

TEST(GpuCollectivesApiTest, ForwardsBothOperationsAtAuthoritativeStages) {
  TestGpuCollectivesApi adapter;
  XLA_FFI_ExecutionContext initialize =
      GpuContext(XLA_FFI_ExecutionStage_INITIALIZE, &adapter);

  XLA_FFI_GpuCollectives_GetDeviceComm_Args device_comm = {};
  device_comm.struct_size =
      XLA_FFI_GpuCollectives_GetDeviceComm_Args_STRUCT_SIZE;
  device_comm.ctx = &initialize;
  EXPECT_EQ(GpuCollectivesExtension()->get_device_comm(&device_comm), nullptr);

  XLA_FFI_ExecutionContext execute =
      GpuContext(XLA_FFI_ExecutionStage_EXECUTE, &adapter);
  XLA_FFI_GpuCollectives_GetDeviceMemory_Args device_memory = {};
  device_memory.struct_size =
      XLA_FFI_GpuCollectives_GetDeviceMemory_Args_STRUCT_SIZE;
  device_memory.ctx = &execute;
  EXPECT_EQ(GpuCollectivesExtension()->get_device_memory(&device_memory),
            nullptr);

  EXPECT_EQ(adapter.device_comm_gets, 1);
  EXPECT_EQ(adapter.device_memory_gets, 1);
}

TEST(GpuCollectivesApiTest, PreservesAdapterStatusCode) {
  TestGpuCollectivesApi adapter;
  adapter.device_comm_status = absl::ResourceExhaustedError("no resources");
  XLA_FFI_ExecutionContext ctx =
      GpuContext(XLA_FFI_ExecutionStage_EXECUTE, &adapter);
  XLA_FFI_GpuCollectives_GetDeviceComm_Args args = {};
  args.struct_size = XLA_FFI_GpuCollectives_GetDeviceComm_Args_STRUCT_SIZE;
  args.ctx = &ctx;

  XLA_FFI_Error* error = GpuCollectivesExtension()->get_device_comm(&args);
  EXPECT_EQ(TakeErrorCode(error), XLA_FFI_Error_Code_RESOURCE_EXHAUSTED);
  EXPECT_EQ(adapter.device_comm_gets, 1);
}

}  // namespace
}  // namespace xla::ffi
