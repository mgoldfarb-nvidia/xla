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
#include <limits>
#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/gpu_collectives_stub.h"
#include "xla/backends/gpu/runtime/custom_call_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/core/collectives/collectives_registry.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/collectives_api.h"
#include "xla/ffi/collectives_ffi.h"
#include "xla/ffi/ffi.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {
namespace {

using ::absl_testing::StatusIs;

constexpr int64_t kBuildVersion = 22907;
constexpr int64_t kRuntimeVersion = 23004;
constexpr char kInstantiated[] = "Instantiate queried collective versions";

class VersionedCollectives : public GpuCollectivesStub {
 public:
  absl::StatusOr<GpuCollectivesVersion> GetGpuCollectivesVersion()
      const override {
    return GpuCollectivesVersion{kBuildVersion, kRuntimeVersion};
  }
};

absl::Status QueryVersion(ffi::Communicator comm) {
  ABSL_ASSIGN_OR_RETURN(ffi::GpuCollectivesVersion version,
                        comm.GetGpuCollectivesVersion());
  EXPECT_EQ(version.build_version, kBuildVersion);
  EXPECT_EQ(version.runtime_version, kRuntimeVersion);
  // Distinguish a completed query from an incorrectly skipped Instantiate call.
  return absl::UnknownError(kInstantiated);
}

XLA_FFI_DEFINE_HANDLER(
    kQueryVersion, QueryVersion,
    ffi::Ffi::BindInstantiate().Ctx<ffi::Extension<ffi::Collectives>>());
XLA_FFI_DEFINE_HANDLER(
    kExecute, [] { return absl::OkStatus(); }, ffi::Ffi::Bind());

enum class HandlerKind { kC, kOwned };

class CustomCallThunkInstantiateTest
    : public ::testing::TestWithParam<HandlerKind> {
 public:
  static void SetUpTestSuite() {
    ASSERT_OK(CollectivesRegistry::Register(
        "CUDA", "instantiate_test", std::numeric_limits<int32_t>::max(),
        std::make_unique<VersionedCollectives>()));
  }
};

TEST_P(CustomCallThunkInstantiateTest, QueriesCollectivesVersionWithoutGpu) {
  const se::GpuComputeCapability capability(se::CudaComputeCapability(9, 0));
  if (GetParam() == HandlerKind::kC) {
    XLA_FFI_Handler_Bundle bundle = {};
    bundle.instantiate = kQueryVersion;
    bundle.execute = kExecute;
    EXPECT_THAT(CustomCallThunk::Create(
                    Thunk::ThunkInfo(), "query_collectives_version", bundle,
                    /*operands=*/{}, /*results=*/{}, /*attributes=*/{},
                    /*called_computation=*/nullptr, capability),
                StatusIs(absl::StatusCode::kUnknown, kInstantiated));
  } else {
    CustomCallThunk::OwnedHandlerBundle bundle;
    bundle.instantiate =
        ffi::Ffi::BindInstantiate().Ctx<ffi::Extension<ffi::Collectives>>().To(
            QueryVersion);
    bundle.execute = ffi::Ffi::Bind().To([] { return absl::OkStatus(); });
    EXPECT_THAT(
        CustomCallThunk::Create(
            Thunk::ThunkInfo(), "query_collectives_version", std::move(bundle),
            /*operands=*/{}, /*results=*/{},
            /*attributes=*/{}, /*called_computation=*/nullptr, capability),
        StatusIs(absl::StatusCode::kUnknown, kInstantiated));
  }
}

INSTANTIATE_TEST_SUITE_P(HandlerBundles, CustomCallThunkInstantiateTest,
                         ::testing::Values(HandlerKind::kC,
                                           HandlerKind::kOwned));

}  // namespace
}  // namespace xla::gpu
