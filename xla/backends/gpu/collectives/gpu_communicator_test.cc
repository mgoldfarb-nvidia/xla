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
#include <memory>
#include <string>
#include <vector>

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
using ::testing::AllOf;
using ::testing::HasSubstr;
using Requirements = GpuDeviceCommunicator::Requirements;

class FakeGpuDeviceCommunicator final : public GpuDeviceCommunicator {
 public:
  FakeGpuDeviceCommunicator(uint64_t device_abi_schema,
                            uint64_t device_abi_version)
      : GpuDeviceCommunicator(device_abi_schema, device_abi_version) {}

  int64_t lsa_size() const final { return 0; }
  const Info& info() const final { return info_; }
  std::string ToString() const final { return "FakeGpuDeviceCommunicator"; }
  se::PackedKernelArg PackKernelArg() const final {
    return se::PackedKernelArg(/*size_bytes=*/0, [](absl::Span<char>) {});
  }

 private:
  Info info_;
};

TEST(GpuDeviceCommunicatorRequirementsTest, DefaultsToLocalWithNoResources) {
  Requirements requirements;

  EXPECT_EQ(requirements.peer_access,
            GpuDeviceCommunicator::PeerAccess::kLocalDomain);
  EXPECT_EQ(requirements.required_features, 0);
  EXPECT_EQ(requirements.preferred_features, 0);
  EXPECT_EQ(requirements.local_barrier_count, 0);
  EXPECT_EQ(requirements.team_barrier_count, 0);
  EXPECT_EQ(requirements.notification_slot_count, 0);
  EXPECT_EQ(requirements.completion_slot_count, 0);
}

TEST(GpuDeviceCommunicatorRequirementsTest, EqualityUsesAllFields) {
  Requirements lhs;
  lhs.peer_access = GpuDeviceCommunicator::PeerAccess::kHierarchical;
  lhs.required_features = GpuDeviceCommunicator::kNetworkDeviceOperations;
  lhs.preferred_features = GpuDeviceCommunicator::kLocalMulticast;
  lhs.local_barrier_count = 1;
  lhs.team_barrier_count = 2;
  lhs.notification_slot_count = 3;
  lhs.completion_slot_count = 4;
  Requirements rhs = lhs;

  EXPECT_EQ(lhs, rhs);
  ++rhs.completion_slot_count;
  EXPECT_FALSE(lhs == rhs);
}

TEST(GpuDeviceCommunicatorRequirementsTest, OrdersAllFieldsDeterministically) {
  Requirements local;
  Requirements hierarchical;
  hierarchical.peer_access = GpuDeviceCommunicator::PeerAccess::kHierarchical;
  Requirements with_feature = local;
  with_feature.required_features =
      GpuDeviceCommunicator::kNetworkDeviceOperations;
  Requirements with_barrier = local;
  with_barrier.team_barrier_count = 1;

  absl::btree_set<Requirements> requirements = {hierarchical, with_feature,
                                                with_barrier, local};

  ASSERT_EQ(requirements.size(), 4);
  auto it = requirements.begin();
  EXPECT_EQ(*it++, hierarchical);
  EXPECT_EQ(*it++, with_feature);
  EXPECT_EQ(*it++, with_barrier);
  EXPECT_EQ(*it++, local);
}

TEST(GpuDeviceCommunicatorRequirementsTest, StringifiesAllFields) {
  Requirements requirements;
  requirements.peer_access = GpuDeviceCommunicator::PeerAccess::kHierarchical;
  requirements.required_features =
      GpuDeviceCommunicator::kNetworkDeviceOperations;
  requirements.preferred_features = GpuDeviceCommunicator::kLocalMulticast;
  requirements.local_barrier_count = 1;
  requirements.team_barrier_count = 2;
  requirements.notification_slot_count = 3;
  requirements.completion_slot_count = 4;

  EXPECT_EQ(absl::StrCat(requirements),
            "{peer_access: 1, required_features: 0x2, preferred_features: 0x1, "
            "local_barrier_count: 1, team_barrier_count: 2, "
            "notification_slot_count: 3, completion_slot_count: 4}");
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

TEST(GpuCliqueBarrierTokenTest, AcceptsMatchingNonzeroTokens) {
  GpuCliqueBarrierToken token{0x1234, 0x5678};
  std::vector<GpuCliqueBarrierToken> gathered(4, token);

  EXPECT_THAT(ValidateGpuCliqueBarrierTokens(token, gathered), IsOk());
}

TEST(GpuCliqueBarrierTokenTest, RejectsZeroToken) {
  std::vector<GpuCliqueBarrierToken> gathered(2, GpuCliqueBarrierToken{});

  EXPECT_THAT(
      ValidateGpuCliqueBarrierTokens({}, gathered),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("nonzero")));
}

TEST(GpuCliqueBarrierTokenTest, RejectsEmptyGather) {
  EXPECT_THAT(
      ValidateGpuCliqueBarrierTokens({1, 2}, {}),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("at least one")));
}

TEST(GpuCliqueBarrierTokenTest, ReportsFirstMismatchingRank) {
  GpuCliqueBarrierToken token{0x1234, 0x5678};
  std::vector<GpuCliqueBarrierToken> gathered = {
      token, token, GpuCliqueBarrierToken{0x1234, 0x9abc}, token};

  EXPECT_THAT(ValidateGpuCliqueBarrierTokens(token, gathered),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       AllOf(HasSubstr("rank 2"), HasSubstr("mismatch"))));
}

TEST(GpuCliqueBarrierTokenTest, MismatchFailsFromEveryRanksPerspective) {
  std::vector<GpuCliqueBarrierToken> gathered = {{0x1, 0x2}, {0x3, 0x4}};

  for (GpuCliqueBarrierToken local_token : gathered) {
    EXPECT_THAT(
        ValidateGpuCliqueBarrierTokens(local_token, gathered),
        StatusIs(absl::StatusCode::kFailedPrecondition, HasSubstr("mismatch")));
  }
}

TEST(GpuProviderHostStorageTest, RetainsUntilProviderTeardownCompletes) {
  GpuProviderHostStorage storage;
  auto value = std::make_shared<int>(42);
  std::weak_ptr<int> weak = value;

  storage.RetainUntilProviderTeardown(value);
  value.reset();
  EXPECT_FALSE(weak.expired());
  EXPECT_EQ(storage.retained_count_for_test(), 1);

  storage.ProviderTeardownComplete();
  EXPECT_TRUE(weak.expired());
  EXPECT_EQ(storage.retained_count_for_test(), 0);
}

TEST(GpuProviderHostStorageTest, DoesNotRetainAfterProviderTeardown) {
  GpuProviderHostStorage storage;
  storage.ProviderTeardownComplete();
  auto value = std::make_shared<int>(42);
  std::weak_ptr<int> weak = value;

  storage.RetainUntilProviderTeardown(value);
  value.reset();

  EXPECT_TRUE(weak.expired());
  EXPECT_EQ(storage.retained_count_for_test(), 0);
}

TEST(GpuProviderHostStorageTest, RetainsOnlyFailedProviderOutputs) {
  GpuProviderHostStorage storage;
  auto completed = std::make_shared<int>(1);
  auto failed = std::make_shared<int>(2);
  std::weak_ptr<int> completed_weak = completed;
  std::weak_ptr<int> failed_weak = failed;

  storage.RetainOnFailure(absl::OkStatus(), completed);
  storage.RetainOnFailure(absl::InternalError("failed"), failed);
  completed.reset();
  failed.reset();

  EXPECT_TRUE(completed_weak.expired());
  EXPECT_FALSE(failed_weak.expired());
  EXPECT_EQ(storage.retained_count_for_test(), 1);

  storage.ProviderTeardownComplete();
  EXPECT_TRUE(failed_weak.expired());
}

TEST(GpuProviderHostStorageTest, QuarantinesAfterProviderTeardownFailure) {
  GpuProviderHostStorage storage;
  auto retained_before_failure = std::make_shared<int>(1);
  std::weak_ptr<int> before_weak = retained_before_failure;
  storage.RetainUntilProviderTeardown(retained_before_failure);

  storage.ProviderTeardownFailed();
  retained_before_failure.reset();
  EXPECT_FALSE(before_weak.expired());
  EXPECT_EQ(storage.retained_count_for_test(), 0);

  // A later completion signal cannot retroactively establish safety for
  // pointers already exposed during a failed teardown.
  storage.ProviderTeardownComplete();
  auto retained_after_failure = std::make_shared<int>(2);
  std::weak_ptr<int> after_weak = retained_after_failure;
  storage.RetainUntilProviderTeardown(retained_after_failure);
  retained_after_failure.reset();
  EXPECT_FALSE(after_weak.expired());
}

TEST(ScopedGpuCommGroupLockOwnershipTest, TracksNestedSameThreadOwnership) {
  int first_state;
  int second_state;
  EXPECT_FALSE(internal::IsGpuCommGroupLockOwnedByCurrentThread(&first_state));

  {
    internal::ScopedGpuCommGroupLockOwnership first(&first_state);
    EXPECT_TRUE(internal::IsGpuCommGroupLockOwnedByCurrentThread(&first_state));
    EXPECT_FALSE(
        internal::IsGpuCommGroupLockOwnedByCurrentThread(&second_state));

    {
      internal::ScopedGpuCommGroupLockOwnership nested(&first_state);
      internal::ScopedGpuCommGroupLockOwnership second(&second_state);
      EXPECT_TRUE(
          internal::IsGpuCommGroupLockOwnedByCurrentThread(&first_state));
      EXPECT_TRUE(
          internal::IsGpuCommGroupLockOwnedByCurrentThread(&second_state));
    }

    EXPECT_TRUE(internal::IsGpuCommGroupLockOwnedByCurrentThread(&first_state));
    EXPECT_FALSE(
        internal::IsGpuCommGroupLockOwnedByCurrentThread(&second_state));
  }

  EXPECT_FALSE(internal::IsGpuCommGroupLockOwnedByCurrentThread(&first_state));
}

TEST(RunGpuCommGroupWithLockTest, MarksDeduplicatedStatesAndSupportsNesting) {
  struct FakeCancel {
    bool IsCancelled() const { return cancelled; }
    bool cancelled = false;
  };
  struct FakeCommState {
    std::shared_ptr<FakeCancel> cancel = std::make_shared<FakeCancel>();
    absl::Mutex mutex;
    bool aborted = false;
    bool destroyed = false;
  };

  auto first = std::make_shared<FakeCommState>();
  auto second = std::make_shared<FakeCommState>();
  auto launched = internal::RunGpuCommGroupWithLock(
      std::vector<std::shared_ptr<FakeCommState>>{second, first, second}, [&] {
        EXPECT_TRUE(
            internal::IsGpuCommGroupLockOwnedByCurrentThread(first.get()));
        EXPECT_TRUE(
            internal::IsGpuCommGroupLockOwnedByCurrentThread(second.get()));

        auto nested = internal::RunGpuCommGroupWithLock(
            std::vector<std::shared_ptr<FakeCommState>>{first, second},
            [] { return absl::StatusOr<bool>(false); });
        EXPECT_TRUE(nested.ok()) << nested.status();
        return absl::StatusOr<bool>(true);
      });

  ASSERT_TRUE(launched.ok()) << launched.status();
  EXPECT_TRUE(*launched);
  EXPECT_FALSE(internal::IsGpuCommGroupLockOwnedByCurrentThread(first.get()));
  EXPECT_FALSE(internal::IsGpuCommGroupLockOwnedByCurrentThread(second.get()));
}

TEST(RunGpuCommGroupWithLockTest, RejectsAddingStateToNestedGroup) {
  struct FakeCancel {
    bool IsCancelled() const { return false; }
  };
  struct FakeCommState {
    std::shared_ptr<FakeCancel> cancel = std::make_shared<FakeCancel>();
    absl::Mutex mutex;
    bool aborted = false;
    bool destroyed = false;
  };

  auto first = std::make_shared<FakeCommState>();
  auto second = std::make_shared<FakeCommState>();
  auto outer = internal::RunGpuCommGroupWithLock(
      std::vector<std::shared_ptr<FakeCommState>>{first}, [&] {
        auto nested = internal::RunGpuCommGroupWithLock(
            std::vector<std::shared_ptr<FakeCommState>>{first, second},
            [] { return absl::StatusOr<bool>(true); });
        EXPECT_THAT(nested.status(),
                    StatusIs(absl::StatusCode::kFailedPrecondition,
                             HasSubstr("cannot add")));
        return absl::StatusOr<bool>(false);
      });

  ASSERT_TRUE(outer.ok()) << outer.status();
}

}  // namespace
}  // namespace xla::gpu
