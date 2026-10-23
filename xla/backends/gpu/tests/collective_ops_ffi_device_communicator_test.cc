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
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "third_party/nccl/nccl.h"
#include "third_party/nccl/nccl_device.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/backends/gpu/tests/collective_ops_e2e_test_base.h"
#include "xla/backends/gpu/tests/collective_ops_ffi_kernels.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/collectives_api.h"
#include "xla/ffi/api/collectives_c_api.h"
#include "xla/ffi/collectives_ffi.h"
#include "xla/ffi/ffi.h"
#include "xla/literal.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"
#include "xla/tests/literal_test_util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

constexpr int64_t kCommunicationId = 0;
constexpr int64_t kNumReplicas = 2;

absl::Status Instantiate(ffi::Communicator comm, int64_t version_offset,
                         int32_t /*lsa_barrier_count*/) {
  ABSL_ASSIGN_OR_RETURN(ffi::GpuCollectivesVersion version,
                        comm.GetGpuCollectivesVersion());
  int runtime_version;
  TF_RET_CHECK(ncclGetVersion(&runtime_version) == ncclSuccess);
  TF_RET_CHECK(version.runtime_version == runtime_version);
  if (version.build_version != NCCL_VERSION_CODE + version_offset) {
    return absl::FailedPreconditionError(absl::Substitute(
        "NCCL header version mismatch: caller=$0, XLA=$1",
        NCCL_VERSION_CODE + version_offset, version.build_version));
  }
  return absl::OkStatus();
}

absl::Status Prepare(ffi::BufferR0<U32> src,
                     ffi::Result<ffi::BufferR0<U32>> dst,
                     ffi::Communicator comm, int64_t version_offset,
                     int32_t lsa_barrier_count) {
  ffi::DeviceCommunicatorRequirements requirements;
  requirements.lsa_barrier_count = lsa_barrier_count;
  ABSL_RETURN_IF_ERROR(comm.RequestDeviceCommunicator(
      ffi::GroupMode::kFlattenedId, {{0, 1}}, kCommunicationId, requirements,
      NCCL_VERSION_CODE + version_offset));
  return comm.RequestWindow(
      ffi::GroupMode::kFlattenedId, {{0, 1}}, kCommunicationId,
      {{src.device_memory().opaque(), src.device_memory().size()},
       {dst->device_memory().opaque(), dst->device_memory().size()}});
}

absl::Status Execute(se::Stream* stream, ffi::BufferR0<U32> src,
                     ffi::Result<ffi::BufferR0<U32>> dst,
                     ffi::Communicator comm, int64_t version_offset,
                     int32_t lsa_barrier_count) {
  ffi::DeviceCommunicatorRequirements requirements;
  requirements.lsa_barrier_count = lsa_barrier_count;
  ABSL_ASSIGN_OR_RETURN(
      const XLA_FFI_DeviceCommunicator* descriptor,
      comm.GetDeviceCommunicator(ffi::GroupMode::kFlattenedId, {{0, 1}},
                                 kCommunicationId, requirements,
                                 NCCL_VERSION_CODE + version_offset));
  ABSL_ASSIGN_OR_RETURN(
      ffi::WindowLookup src_window,
      comm.GetWindow(ffi::GroupMode::kFlattenedId, {{0, 1}}, kCommunicationId,
                     src.device_memory().opaque()));
  ABSL_ASSIGN_OR_RETURN(
      ffi::WindowLookup dst_window,
      comm.GetWindow(ffi::GroupMode::kFlattenedId, {{0, 1}}, kCommunicationId,
                     dst->device_memory().opaque()));
  TF_RET_CHECK(descriptor != nullptr);
  const ncclDevComm& dev_comm =
      *reinterpret_cast<const ncclDevComm*>(descriptor);
  TF_RET_CHECK(dev_comm.lsaSize == kNumReplicas);

  ABSL_ASSIGN_OR_RETURN(auto kernel,
                        se::gpu::GpuKernelRegistry::GetGlobalRegistry()
                            .LoadKernel<SymmetricAllReduce>(stream->parent()));
  // The borrowed handle addresses host memory; the kernel takes its value.
  se::KernelArgsPackedTuple<ncclDevComm, se::DeviceAddressBase,
                            se::DeviceAddressBase, size_t, size_t, size_t>
      args(dev_comm, se::DeviceAddressBase(src_window.window),
           se::DeviceAddressBase(dst_window.window), src_window.offset,
           dst_window.offset, src.element_count(), /*shared_memory_bytes=*/0);
  ABSL_RETURN_IF_ERROR(
      kernel->Launch(se::ThreadDim(32), se::BlockDim(1), stream, args));
  return stream->BlockHostUntilDone();
}

XLA_FFI_DEFINE_HANDLER(kInstantiate, Instantiate,
                       ffi::Ffi::BindInstantiate()
                           .Ctx<ffi::Extension<ffi::Collectives>>()
                           .Attr<int64_t>("version_offset")
                           .Attr<int32_t>("lsa_barrier_count"));

XLA_FFI_DEFINE_HANDLER(kPrepare, Prepare,
                       ffi::Ffi::BindPrepare()
                           .Arg<ffi::BufferR0<U32>>()
                           .Ret<ffi::BufferR0<U32>>()
                           .Ctx<ffi::Extension<ffi::Collectives>>()
                           .Attr<int64_t>("version_offset")
                           .Attr<int32_t>("lsa_barrier_count"));

XLA_FFI_DEFINE_HANDLER(kExecute, Execute,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Arg<ffi::BufferR0<U32>>()
                           .Ret<ffi::BufferR0<U32>>()
                           .Ctx<ffi::Extension<ffi::Collectives>>()
                           .Attr<int64_t>("version_offset")
                           .Attr<int32_t>("lsa_barrier_count"));

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(),
                         "__xla_test$$public_api_device_communicator", "gpu",
                         XLA_FFI_Handler_Bundle{kInstantiate, kPrepare, nullptr,
                                                kExecute});

class DeviceCommunicatorFfiTest : public CollectiveOpsE2ETestBase {
 public:
  DeviceCommunicatorFfiTest()
      : CollectiveOpsE2ETestBase(/*memory_size=*/32 * kMB,
                                 /*collectives_memory_size=*/32 * kMB) {}

  void SetUp() override {
    CollectiveOpsE2ETestBase::SetUp();
    if (device_count() < kNumReplicas || !IsHopperAndHigher()) {
      GTEST_SKIP() << "Requires two Hopper or newer GPUs with NCCL LSA access";
    }
  }

 protected:
  absl::StatusOr<ExecutionResult> Run(int64_t version_offset = 0,
                                      int32_t lsa_barrier_count = 1) {
    std::string hlo = absl::Substitute(
        R"hlo(
      HloModule m, replica_count=2
      ENTRY main {
        id = u32[] replica-id()
        src = u32[]{:S(1)} copy(id)
        result = u32[]{:S(1)} custom-call(src),
          custom_call_target="__xla_test$$$$public_api_device_communicator",
          api_version=API_VERSION_TYPED_FFI,
          backend_config={version_offset = $0 : i64,
                          lsa_barrier_count = $1 : i32}
        ROOT output = u32[] copy(result)
      }
    )hlo",
        version_offset, lsa_barrier_count);
    ABSL_ASSIGN_OR_RETURN(auto module,
                          ParseAndReturnVerifiedModule(hlo, kNumReplicas));
    return ExecuteReplicated(std::move(module),
                             /*arguments=*/std::vector<Literal*>(),
                             /*run_hlo_passes=*/false);
  }
};

TEST_F(DeviceCommunicatorFfiTest, LaunchesLsaAllReduceWithBorrowedDescriptor) {
  ASSERT_OK_AND_ASSIGN(ExecutionResult result, Run());
  ASSERT_EQ(result.results.size(), kNumReplicas);
  for (const Literal& output : result.results) {
    LiteralTestUtil::ExpectR0Equal<uint32_t>(1, output);
  }
}

TEST_F(DeviceCommunicatorFfiTest, RejectsMismatchedHeaders) {
  EXPECT_THAT(
      Run(/*version_offset=*/1),
      StatusIs(absl::StatusCode::kFailedPrecondition, HasSubstr("version")));
}

TEST_F(DeviceCommunicatorFfiTest, RejectsNegativeBarrierCount) {
  EXPECT_THAT(
      Run(/*version_offset=*/0, /*lsa_barrier_count=*/-1),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("non-negative")));
}

}  // namespace
}  // namespace xla::gpu
