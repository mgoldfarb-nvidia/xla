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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/collectives/nccl_communicator.h"
#include "xla/backends/gpu/collectives/nccl_symmetric_memory.h"
#include "xla/ffi/api/c_api_gpu_collectives_nccl.h"
#include "xla/tsl/lib/core/status_test_util.h"

namespace xla::gpu {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using Requirements = GpuDeviceCommunicator::Requirements;

NcclCapabilities Capabilities(
    int lsa_team_count,
    NcclCapabilities::GinConnectionType gin_connection_type =
        NcclCapabilities::GinConnectionType::kNone) {
  NcclCapabilities capabilities;
  capabilities.supports_device_comm = true;
  capabilities.lsa_team_count = lsa_team_count;
  capabilities.gin_connection_type = gin_connection_type;
  return capabilities;
}

static_assert(kNcclDeviceCommAbiSchema ==
              XLA_FFI_GpuCollective_NCCL_DEVICE_COMM_ABI_SCHEMA);
static_assert(kNcclWindowAbiSchema ==
              XLA_FFI_GpuCollective_NCCL_SYMMETRIC_MEMORY_ABI_SCHEMA);

TEST(NcclDeviceRequirementsTest, MapsLsaBarrierCount) {
  ASSERT_OK_AND_ASSIGN(
      ncclDevCommRequirements reqs,
      BuildNcclDeviceCommRequirements(Requirements{/*lsa_barrier_count=*/2},
                                      Capabilities(/*lsa_team_count=*/1)));

  EXPECT_EQ(reqs.lsaBarrierCount, 2);
#if NCCL_VERSION_CODE >= 22907
  EXPECT_EQ(reqs.barrierCount, 0);
  EXPECT_EQ(reqs.ginConnectionType, NCCL_GIN_CONNECTION_NONE);
#endif
}

TEST(NcclDeviceRequirementsTest, RejectsNegativeLsaBarrierCount) {
  EXPECT_THAT(
      BuildNcclDeviceCommRequirements(Requirements{/*lsa_barrier_count=*/-1},
                                      Capabilities(/*lsa_team_count=*/1)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("lsa_barrier_count must be non-negative")));
}

TEST(NcclDeviceRequirementsTest, RejectsNegativeGlobalBarrierCount) {
  EXPECT_THAT(
      BuildNcclDeviceCommRequirements(Requirements{/*lsa_barrier_count=*/0,
                                                   /*global_barrier_count=*/-1},
                                      Capabilities(/*lsa_team_count=*/1)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("global_barrier_count must be non-negative")));
}

#if NCCL_VERSION_CODE >= 22907
TEST(NcclDeviceRequirementsTest, FoldsGlobalBarrierIntoSingleLsaTeam) {
  ASSERT_OK_AND_ASSIGN(
      ncclDevCommRequirements reqs,
      BuildNcclDeviceCommRequirements(Requirements{/*lsa_barrier_count=*/2,
                                                   /*global_barrier_count=*/3},
                                      Capabilities(/*lsa_team_count=*/1)));

  EXPECT_EQ(reqs.lsaBarrierCount, 3);
  EXPECT_EQ(reqs.barrierCount, 0);
  EXPECT_EQ(reqs.ginConnectionType, NCCL_GIN_CONNECTION_NONE);
}

TEST(NcclDeviceRequirementsTest, MapsGlobalBarrierToFullGin) {
  ASSERT_OK_AND_ASSIGN(
      ncclDevCommRequirements reqs,
      BuildNcclDeviceCommRequirements(
          Requirements{/*lsa_barrier_count=*/0,
                       /*global_barrier_count=*/2},
          Capabilities(/*lsa_team_count=*/2,
                       NcclCapabilities::GinConnectionType::kFull)));

  EXPECT_EQ(reqs.lsaBarrierCount, 0);
  EXPECT_EQ(reqs.barrierCount, 2);
  EXPECT_EQ(reqs.ginConnectionType, NCCL_GIN_CONNECTION_FULL);
  EXPECT_EQ(reqs.ginSignalCount, 0);
  EXPECT_EQ(reqs.ginCounterCount, 0);
}

TEST(NcclDeviceRequirementsTest, MapsGlobalBarrierToRailGin) {
  ASSERT_OK_AND_ASSIGN(
      ncclDevCommRequirements reqs,
      BuildNcclDeviceCommRequirements(
          Requirements{/*lsa_barrier_count=*/0,
                       /*global_barrier_count=*/2},
          Capabilities(/*lsa_team_count=*/2,
                       NcclCapabilities::GinConnectionType::kRail)));

  EXPECT_EQ(reqs.barrierCount, 2);
  EXPECT_EQ(reqs.ginConnectionType, NCCL_GIN_CONNECTION_RAIL);
}

TEST(NcclDeviceRequirementsTest, RejectsGlobalBarrierWithoutLsaTopology) {
  EXPECT_THAT(
      BuildNcclDeviceCommRequirements(Requirements{/*lsa_barrier_count=*/0,
                                                   /*global_barrier_count=*/1},
                                      Capabilities(/*lsa_team_count=*/0)),
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("does not report its LSA topology")));
}

TEST(NcclDeviceRequirementsTest, RejectsMultiLsaGlobalBarrierWithoutGin) {
  EXPECT_THAT(
      BuildNcclDeviceCommRequirements(Requirements{/*lsa_barrier_count=*/0,
                                                   /*global_barrier_count=*/1},
                                      Capabilities(/*lsa_team_count=*/2)),
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("reports no GIN connectivity")));
}
#endif

TEST(NcclDeviceAbiTest, AcceptsExactVersion) {
  EXPECT_OK(ValidateNcclDeviceAbi(/*compile_time_version=*/22907,
                                  /*runtime_version=*/22907));
}

TEST(NcclDeviceAbiTest, RejectsVersionMismatch) {
  EXPECT_THAT(
      ValidateNcclDeviceAbi(/*compile_time_version=*/22907,
                            /*runtime_version=*/23000),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("XLA compile-time version=22907, loaded runtime "
                         "version=23000")));
}

TEST(NcclWindowDeviceAbiTest, RequiresExactRuntimeVersion) {
  EXPECT_OK(ValidateNcclWindowDeviceAbi(/*compile_time_version=*/22907,
                                        /*runtime_version=*/22907));
  EXPECT_THAT(
      ValidateNcclWindowDeviceAbi(/*compile_time_version=*/22907,
                                  /*runtime_version=*/23000),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("XLA compile-time version=22907, loaded runtime "
                         "version=23000")));
}

}  // namespace
}  // namespace xla::gpu
