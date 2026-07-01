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

#include "xla/backends/gpu/collectives/gpu_communicator.h"

#include <cstdint>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/btree_set.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/stream_executor/kernel_args.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using Requirements = GpuDeviceCommunicator::Requirements;

class FakeGpuDeviceCommunicator final : public GpuDeviceCommunicator {
 public:
  FakeGpuDeviceCommunicator(uint64_t device_abi_schema,
                            uint64_t device_abi_version)
      : GpuDeviceCommunicator(device_abi_schema, device_abi_version) {}

  int64_t lsa_size() const final { return 0; }
  std::string ToString() const final { return "FakeGpuDeviceCommunicator"; }
  se::PackedKernelArg PackKernelArg() const final {
    return se::PackedKernelArg(/*size_bytes=*/0, [](absl::Span<char>) {});
  }
};

TEST(GpuDeviceCommunicatorRequirementsTest,
     PreservesLegacyAggregateInitialization) {
  Requirements requirements{8};

  EXPECT_EQ(requirements.lsa_barrier_count, 8);
  EXPECT_EQ(requirements.global_barrier_count, 0);
}

TEST(GpuDeviceCommunicatorRequirementsTest, EqualityUsesAllBarrierCounts) {
  Requirements one_global{8, 1};
  Requirements two_global{8, 2};

  EXPECT_EQ(Requirements{8}, Requirements{8});
  EXPECT_FALSE(Requirements{8} == Requirements{7});
  EXPECT_FALSE(one_global == two_global);
}

TEST(GpuDeviceCommunicatorRequirementsTest,
     OrdersGlobalThenLsaCountsDescending) {
  absl::btree_set<Requirements> requirements = {
      Requirements{1, 0}, Requirements{3, 0}, Requirements{2, 1},
      Requirements{1, 2}};

  ASSERT_EQ(requirements.size(), 4);
  auto it = requirements.begin();
  EXPECT_EQ((it)->global_barrier_count, 2);
  EXPECT_EQ((it++)->lsa_barrier_count, 1);
  EXPECT_EQ((it)->global_barrier_count, 1);
  EXPECT_EQ((it++)->lsa_barrier_count, 2);
  EXPECT_EQ((it++)->lsa_barrier_count, 3);
  EXPECT_EQ((it++)->lsa_barrier_count, 1);
}

TEST(GpuDeviceCommunicatorRequirementsTest, StringifiesBarrierCounts) {
  EXPECT_EQ(absl::StrCat(Requirements{8, 2}),
            "{lsa_barrier_count: 8, global_barrier_count: 2}");
}

TEST(GpuDeviceCommunicatorKernelArgAbiTest, AcceptsExactProviderAbi) {
  FakeGpuDeviceCommunicator communicator(/*device_abi_schema=*/123,
                                         /*device_abi_version=*/456);

  EXPECT_THAT(communicator.CheckKernelArgAbi(/*expected_schema=*/123,
                                             /*expected_version=*/456),
              IsOk());
}

TEST(GpuDeviceCommunicatorKernelArgAbiTest, RejectsZeroProviderSchema) {
  FakeGpuDeviceCommunicator communicator(/*device_abi_schema=*/0,
                                         /*device_abi_version=*/456);

  EXPECT_THAT(communicator.CheckKernelArgAbi(/*expected_schema=*/123,
                                             /*expected_version=*/456),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("does not expose")));
}

TEST(GpuDeviceCommunicatorKernelArgAbiTest, RejectsZeroProviderVersion) {
  FakeGpuDeviceCommunicator communicator(/*device_abi_schema=*/123,
                                         /*device_abi_version=*/0);

  EXPECT_THAT(communicator.CheckKernelArgAbi(/*expected_schema=*/123,
                                             /*expected_version=*/456),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("does not expose")));
}

TEST(GpuDeviceCommunicatorKernelArgAbiTest, RejectsSchemaMismatch) {
  FakeGpuDeviceCommunicator communicator(/*device_abi_schema=*/123,
                                         /*device_abi_version=*/456);

  EXPECT_THAT(communicator.CheckKernelArgAbi(/*expected_schema=*/124,
                                             /*expected_version=*/456),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("schema mismatch")));
}

TEST(GpuDeviceCommunicatorKernelArgAbiTest, RejectsVersionMismatch) {
  FakeGpuDeviceCommunicator communicator(/*device_abi_schema=*/123,
                                         /*device_abi_version=*/456);

  EXPECT_THAT(communicator.CheckKernelArgAbi(/*expected_schema=*/123,
                                             /*expected_version=*/457),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("version mismatch")));
}

}  // namespace
}  // namespace xla::gpu
