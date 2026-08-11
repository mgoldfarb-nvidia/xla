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

#include <memory>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_registry.h"
#include "xla/ffi/type_registry.h"

namespace {

namespace ffi = ::xla::ffi;

struct TestCallState {
  static ffi::TypeId id;
};

ffi::TypeId TestCallState::id = XLA_FFI_UNKNOWN_TYPE_ID;
constexpr ffi::TypeInfo kTestCallStateTypeInfo =
    ffi::MakeTypeInfo<TestCallState>();

ffi::ErrorOr<std::unique_ptr<TestCallState>> Instantiate() {
  return std::make_unique<TestCallState>();
}

ffi::Error NoOp() { return ffi::Error::Success(); }

}  // namespace

extern "C" XLA_FFI_TypeId* CuteDSLRT_NvJaxCutlassCallStateTypeId_v2() {
  return &TestCallState::id;
}

extern "C" const XLA_FFI_TypeInfo*
CuteDSLRT_NvJaxCutlassCallStateTypeInfo_v2() {
  return &kTestCallStateTypeInfo;
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CuteDSLRT_NvJaxCutlassCallInstantiate_v2,
                              Instantiate, ffi::Ffi::BindInstantiate());
XLA_FFI_DEFINE_HANDLER_SYMBOL(CuteDSLRT_NvJaxCutlassCallPrepare_v2, NoOp,
                              ffi::Ffi::BindPrepare());
XLA_FFI_DEFINE_HANDLER_SYMBOL(CuteDSLRT_NvJaxCutlassCallExecute_v2, NoOp,
                              ffi::Ffi::Bind(),
                              {ffi::Traits::kCmdBufferCompatible});
XLA_FFI_DEFINE_HANDLER_SYMBOL(CuteDSLRT_NvJaxCutlassCallExecuteNoCudaGraph_v2,
                              NoOp, ffi::Ffi::Bind());

namespace xla::gpu::cutedsl {
namespace {

TEST(CuteDslFfiRegistrationTest, RegistersRuntimeTypeAndHandlers) {
  absl::StatusOr<ffi::TypeRegistry::TypeId> state_type =
      ffi::TypeRegistry::GetTypeId("CuteDSLRT_NvJaxCutlassCallTypes_v2");
  ASSERT_TRUE(state_type.ok()) << state_type.status();
  EXPECT_EQ(state_type->value(), TestCallState::id.type_id);

  absl::StatusOr<ffi::HandlerRegistration> cuda_graph =
      ffi::FindHandler("CuteDSLRT_NvJaxCutlassCall_v2", "CUDA");
  ASSERT_TRUE(cuda_graph.ok()) << cuda_graph.status();
  EXPECT_EQ(cuda_graph->metadata.state_type_id.type_id,
            TestCallState::id.type_id);
  EXPECT_EQ(cuda_graph->metadata.traits,
            XLA_FFI_HANDLER_TRAITS_COMMAND_BUFFER_COMPATIBLE);

  absl::StatusOr<ffi::HandlerRegistration> no_cuda_graph = ffi::FindHandler(
      "CuteDSLRT_NvJaxCutlassCallNoCudaGraph_v2", "CUDA");
  ASSERT_TRUE(no_cuda_graph.ok()) << no_cuda_graph.status();
  EXPECT_EQ(no_cuda_graph->metadata.state_type_id.type_id,
            TestCallState::id.type_id);
  EXPECT_EQ(no_cuda_graph->metadata.traits, 0);
  EXPECT_EQ(no_cuda_graph->bundle.instantiate,
            cuda_graph->bundle.instantiate);
  EXPECT_EQ(no_cuda_graph->bundle.prepare, cuda_graph->bundle.prepare);
  EXPECT_NE(no_cuda_graph->bundle.execute, cuda_graph->bundle.execute);
}

}  // namespace
}  // namespace xla::gpu::cutedsl
