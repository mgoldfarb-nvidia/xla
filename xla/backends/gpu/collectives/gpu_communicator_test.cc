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

#include <string>

#include <gtest/gtest.h>
#include "absl/container/btree_set.h"
#include "absl/strings/str_cat.h"

namespace xla::gpu {
namespace {

using Requirements = GpuDeviceCommunicator::Requirements;

TEST(GpuDeviceCommunicatorRequirementsTest,
     PreservesLegacyAggregateInitialization) {
  Requirements requirements{8};

  EXPECT_EQ(requirements.lsa_barrier_count, 8);
}

TEST(GpuDeviceCommunicatorRequirementsTest, EqualityUsesLsaBarrierCount) {
  EXPECT_EQ(Requirements{8}, Requirements{8});
  EXPECT_FALSE(Requirements{8} == Requirements{7});
}

TEST(GpuDeviceCommunicatorRequirementsTest, OrdersLargerLsaBarrierCountsFirst) {
  absl::btree_set<Requirements> requirements = {
      Requirements{1}, Requirements{3}, Requirements{2}};

  ASSERT_EQ(requirements.size(), 3);
  auto it = requirements.begin();
  EXPECT_EQ((it++)->lsa_barrier_count, 3);
  EXPECT_EQ((it++)->lsa_barrier_count, 2);
  EXPECT_EQ((it++)->lsa_barrier_count, 1);
}

TEST(GpuDeviceCommunicatorRequirementsTest, StringifiesLsaBarrierCount) {
  EXPECT_EQ(absl::StrCat(Requirements{8}), "{lsa_barrier_count: 8}");
}

}  // namespace
}  // namespace xla::gpu
