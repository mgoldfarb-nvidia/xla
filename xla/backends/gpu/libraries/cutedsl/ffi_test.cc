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

#include "xla/ffi/ffi.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_pjrt_client.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/cuda_platform.h"  // IWYU pragma: keep
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/resource_loader.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/xla_data.pb.h"

XLA_FFI_DECLARE_HANDLER_SYMBOL(CuteDSLRT_NvJaxCutlassCallInstantiate_v2);
XLA_FFI_DECLARE_HANDLER_SYMBOL(CuteDSLRT_NvJaxCutlassCallPrepare_v2);
XLA_FFI_DECLARE_HANDLER_SYMBOL(CuteDSLRT_NvJaxCutlassCallExecute_v2);

namespace xla::gpu::cutedsl {
namespace {

namespace ffi = ::xla::ffi;

// The CuTeDSL runtime owns these handlers. Register its exported lifecycle
// bundle only in this test process to exercise the runtime-owned call target.
XLA_FFI_REGISTER_HANDLER(
    ffi::GetXlaFfiApi(), "__xla_gpu_cutedsl_call_v3", "CUDA",
    (XLA_FFI_Handler_Bundle{
        /*instantiate=*/CuteDSLRT_NvJaxCutlassCallInstantiate_v2,
        /*prepare=*/CuteDSLRT_NvJaxCutlassCallPrepare_v2,
        /*initialize=*/nullptr,
        /*execute=*/CuteDSLRT_NvJaxCutlassCallExecute_v2,
        /*record=*/nullptr}),
    XLA_FFI_HANDLER_TRAITS_COMMAND_BUFFER_COMPATIBLE);

TEST(CuteDslCustomCallTest, RunVectorAdd) {
  std::string hlo_text;
  ASSERT_OK(tsl::ReadFileToString(
      tsl::Env::Default(),
      tsl::GetDataDependencyFilepath(
          "xla/backends/gpu/libraries/cutedsl/vector_add.hlo"),
      &hlo_text));
  ASSERT_OK_AND_ASSIGN(auto module,
                       xla::ParseAndReturnUnverifiedModule(hlo_text));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtClient> client,
                       xla::GetXlaPjrtGpuClient(/*options=*/{}));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtLoadedExecutable> executable,
      client->CompileAndLoad(xla::XlaComputation(module->ToProto()),
                             /*options=*/{}));

  constexpr size_t kElementCount = 1024;
  std::vector<float> lhs(kElementCount);
  std::vector<float> rhs(kElementCount);
  for (size_t i = 0; i < kElementCount; ++i) {
    lhs[i] = static_cast<float>(i);
    rhs[i] = static_cast<float>(2 * i);
  }

  xla::Shape shape = xla::ShapeUtil::MakeShape(
      xla::F32, {static_cast<int64_t>(kElementCount)});
  ASSERT_FALSE(client->addressable_devices().empty());
  ASSERT_OK_AND_ASSIGN(
      xla::PjRtMemorySpace * memory_space,
      client->addressable_devices().front()->default_memory_space());
  ASSERT_OK_AND_ASSIGN(
      auto lhs_buffer,
      client->BufferFromHostBuffer(
          lhs.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          xla::PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/nullptr, memory_space,
          /*device_layout=*/nullptr));
  ASSERT_OK_AND_ASSIGN(
      auto rhs_buffer,
      client->BufferFromHostBuffer(
          rhs.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          xla::PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/nullptr, memory_space,
          /*device_layout=*/nullptr));

  ASSERT_OK_AND_ASSIGN(
      auto results, executable->Execute({{lhs_buffer.get(), rhs_buffer.get()}},
                                        /*options=*/{}));
  ASSERT_EQ(results.size(), 1);
  ASSERT_EQ(results[0].size(), 1);
  ASSERT_OK_AND_ASSIGN(auto result, results[0][0]->ToLiteral().Await());
  for (size_t i = 0; i < kElementCount; ++i) {
    EXPECT_FLOAT_EQ(result->Get<float>({static_cast<int64_t>(i)}),
                    lhs[i] + rhs[i]);
  }
}

}  // namespace
}  // namespace xla::gpu::cutedsl
