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

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "xla/backends/gpu/tests/collective_ops_e2e_test_base.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_registry.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/load_library.h"
#include "tsl/platform/path.h"

namespace xla::gpu {
namespace {

constexpr int64_t kNumReplicas = 2;
constexpr char kHandlerName[] = "__xla_test_gpu_collectives_public_dso";
constexpr char kTwoBufferHandlerName[] =
    "__xla_test_gpu_collectives_public_dso_two_buffers";
constexpr char kRegisterSymbol[] = "XlaGpuCollectivesFfiRegister";

using RegisterFn = XLA_FFI_Error*(const XLA_FFI_Api*);

std::string TakeErrorMessage(const XLA_FFI_Api* api, XLA_FFI_Error* error) {
  XLA_FFI_Error_GetMessage_Args message_args = {};
  message_args.struct_size = XLA_FFI_Error_GetMessage_Args_STRUCT_SIZE;
  message_args.error = error;
  api->XLA_FFI_Error_GetMessage(&message_args);
  std::string message = message_args.message == nullptr
                            ? "unknown XLA FFI error"
                            : message_args.message;

  XLA_FFI_Error_Destroy_Args destroy_args = {};
  destroy_args.struct_size = XLA_FFI_Error_Destroy_Args_STRUCT_SIZE;
  destroy_args.error = error;
  api->XLA_FFI_Error_Destroy(&destroy_args);
  return message;
}

class GpuCollectivesFfiDsoTest : public CollectiveOpsE2ETestBase {
 public:
  GpuCollectivesFfiDsoTest()
      : CollectiveOpsE2ETestBase(/*memory_size=*/32 * kMB,
                                 /*collectives_memory_size=*/32 * kMB) {}

  static void SetUpTestSuite() {
    const char* plugin_runfile =
        std::getenv("XLA_GPU_COLLECTIVES_FFI_PLUGIN_PATH");
    ASSERT_NE(plugin_runfile, nullptr)
        << "XLA_GPU_COLLECTIVES_FFI_PLUGIN_PATH must be set by the test rule";
    const char* test_srcdir = std::getenv("TEST_SRCDIR");
    ASSERT_NE(test_srcdir, nullptr) << "TEST_SRCDIR must be set by Bazel";
    std::string plugin_path = tsl::io::JoinPath(test_srcdir, plugin_runfile);

    ASSERT_OK(
        tsl::internal::LoadDynamicLibrary(plugin_path.c_str(), &library_));
    ASSERT_NE(library_, nullptr);

    void* symbol = nullptr;
    ASSERT_OK(tsl::internal::GetSymbolFromLibrary(library_, kRegisterSymbol,
                                                  &symbol));
    ASSERT_NE(symbol, nullptr);
    RegisterFn* register_plugin = reinterpret_cast<RegisterFn*>(symbol);

    const XLA_FFI_Api* api = ffi::GetXlaFfiApi();
    ASSERT_NE(api, nullptr);
    if (XLA_FFI_Error* error = register_plugin(api)) {
      FAIL() << "Failed to register public GPU collectives FFI plugin: "
             << TakeErrorMessage(api, error);
    }
  }

 protected:
  static void* library_;
};

void* GpuCollectivesFfiDsoTest::library_ = nullptr;

void VerifyResults(const std::vector<Literal>& results) {
  ASSERT_EQ(results.size(), kNumReplicas);
  for (const Literal& result : results) {
    LiteralTestUtil::ExpectR1Equal<uint32_t>({11, 24},
                                             LiteralSlice(result, {0}));
    LiteralTestUtil::ExpectR1Equal<uint32_t>({14, 30},
                                             LiteralSlice(result, {1}));
  }
}

void VerifyTwoBufferResults(const std::vector<Literal>& results) {
  ASSERT_EQ(results.size(), kNumReplicas);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({10, 112, 114, 116},
                                           LiteralSlice(results[0], {0}));
  LiteralTestUtil::ExpectR1Equal<uint32_t>({20, 222, 224, 226, 228},
                                           LiteralSlice(results[0], {1}));
  LiteralTestUtil::ExpectR1Equal<uint32_t>({100, 112, 114, 116},
                                           LiteralSlice(results[1], {0}));
  LiteralTestUtil::ExpectR1Equal<uint32_t>({200, 222, 224, 226, 228},
                                           LiteralSlice(results[1], {1}));
}

void VerifyHandlerRegistration(const char* handler_name, bool has_prepare) {
  ASSERT_OK_AND_ASSIGN(ffi::HandlerRegistration registration,
                       ffi::FindHandler(handler_name, "gpu"));
  EXPECT_EQ(registration.metadata.api_version.major_version, 0);
  EXPECT_EQ(registration.metadata.api_version.minor_version, XLA_FFI_API_MINOR);
  EXPECT_FALSE(ffi::IsCommandBufferCompatible(registration.metadata));
  EXPECT_NE(
      registration.metadata.traits & static_cast<XLA_FFI_Handler_Traits>(
                                         ffi::Traits::kUsesDeviceCommunication),
      0);
  EXPECT_EQ(registration.bundle.instantiate, nullptr);
  if (has_prepare) {
    EXPECT_NE(registration.bundle.prepare, nullptr);
  } else {
    EXPECT_EQ(registration.bundle.prepare, nullptr);
  }
  EXPECT_NE(registration.bundle.initialize, nullptr);
  EXPECT_NE(registration.bundle.execute, nullptr);
}

TEST_F(GpuCollectivesFfiDsoTest,
       PublicApiPluginRunsTwoCallsitesAcrossRepeatedExecutions) {
  VerifyHandlerRegistration(kHandlerName, /*has_prepare=*/true);

  if (device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << device_count() << " available)";
  }
  if (!IsHopperAndHigher()) {
    GTEST_SKIP() << "NCCL symmetric memory requires Hopper+";
  }

  constexpr char kHlo[] = R"(
    HloModule public_gpu_collectives_ffi_dso, replica_count=2

    ENTRY main {
      p0 = u32[8]{0} parameter(0)
      collective_buffer = u32[8]{0} copy(p0)
      slice0 = u32[2]{0} slice(collective_buffer), slice={[1:3]}
      call0 = u32[2]{0} custom-call(slice0),
        custom_call_target="__xla_test_gpu_collectives_public_dso",
        api_version=API_VERSION_TYPED_FFI,
        custom_call_has_side_effect=true,
        output_to_operand_aliasing={{}: (0, {})},
        frontend_attributes={operands_memory_spaces="{0:1}", results_memory_spaces="{0:1}"}
      slice1 = u32[2]{0} slice(collective_buffer), slice={[4:6]}
      call1 = u32[2]{0} custom-call(slice1),
        custom_call_target="__xla_test_gpu_collectives_public_dso",
        api_version=API_VERSION_TYPED_FFI,
        custom_call_has_side_effect=true,
        output_to_operand_aliasing={{}: (0, {})},
        frontend_attributes={operands_memory_spaces="{0:1}", results_memory_spaces="{0:1}"}
      out0 = u32[2]{0} copy(call0)
      out1 = u32[2]{0} copy(call1)
      ROOT result = (u32[2]{0}, u32[2]{0}) tuple(out0, out1)
    }
  )";

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kHlo, /*replica_count=*/kNumReplicas));
  Literal argument =
      LiteralUtil::CreateR1<uint32_t>({10, 11, 12, 13, 14, 15, 16, 17});
  std::vector<std::vector<Literal*>> arguments(
      kNumReplicas, std::vector<Literal*>{&argument});

  ASSERT_OK_AND_ASSIGN(ExecutionResult first,
                       ExecuteReplicated(std::move(module), arguments,
                                         /*run_hlo_passes=*/false));
  VerifyResults(first.results);

  ASSERT_OK_AND_ASSIGN(std::vector<Literal> second,
                       ExecuteReplicated(first.executable.get(), arguments,
                                         /*run_hlo_passes=*/false));
  VerifyResults(second);
}

TEST_F(GpuCollectivesFfiDsoTest,
       PublicApiPluginAccessesTwoTaggedBuffersInOneCall) {
  VerifyHandlerRegistration(kTwoBufferHandlerName, /*has_prepare=*/false);

  if (device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << device_count() << " available)";
  }
  if (!IsHopperAndHigher()) {
    GTEST_SKIP() << "NCCL symmetric memory requires Hopper+";
  }

  constexpr char kHlo[] = R"(
    HloModule public_gpu_collectives_ffi_two_buffers, replica_count=2

    ENTRY main {
      p0 = u32[4]{0} parameter(0)
      p1 = u32[5]{0} parameter(1)
      collective_buffer0 = u32[4]{0} copy(p0)
      collective_buffer1 = u32[5]{0} copy(p1)
      call = (u32[4]{0}, u32[5]{0}) custom-call(collective_buffer0, collective_buffer1),
        custom_call_target="__xla_test_gpu_collectives_public_dso_two_buffers",
        api_version=API_VERSION_TYPED_FFI,
        custom_call_has_side_effect=true,
        output_to_operand_aliasing={{0}: (0, {}), {1}: (1, {})},
        frontend_attributes={operands_memory_spaces="{0:1,1:1}", results_memory_spaces="{0:1,1:1}"}
      result0 = u32[4]{0} get-tuple-element(call), index=0
      result1 = u32[5]{0} get-tuple-element(call), index=1
      out0 = u32[4]{0} copy(result0)
      out1 = u32[5]{0} copy(result1)
      ROOT result = (u32[4]{0}, u32[5]{0}) tuple(out0, out1)
    }
  )";

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kHlo, /*replica_count=*/kNumReplicas));
  Literal first_argument0 = LiteralUtil::CreateR1<uint32_t>({10, 11, 12, 13});
  Literal second_argument0 =
      LiteralUtil::CreateR1<uint32_t>({20, 21, 22, 23, 24});
  Literal first_argument1 =
      LiteralUtil::CreateR1<uint32_t>({100, 101, 102, 103});
  Literal second_argument1 =
      LiteralUtil::CreateR1<uint32_t>({200, 201, 202, 203, 204});
  std::vector<std::vector<Literal*>> arguments{
      {&first_argument0, &second_argument0},
      {&first_argument1, &second_argument1}};

  ASSERT_OK_AND_ASSIGN(ExecutionResult first,
                       ExecuteReplicated(std::move(module), arguments,
                                         /*run_hlo_passes=*/false));
  VerifyTwoBufferResults(first.results);

  ASSERT_OK_AND_ASSIGN(std::vector<Literal> second,
                       ExecuteReplicated(first.executable.get(), arguments,
                                         /*run_hlo_passes=*/false));
  VerifyTwoBufferResults(second);
}

}  // namespace
}  // namespace xla::gpu
