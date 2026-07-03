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

#include "xla/backends/gpu/collectives/gpu_clique.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/container/btree_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/cancellation_token.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/future.h"
#include "xla/runtime/device_id.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

class TestCommunicator final : public Communicator {
 public:
  explicit TestCommunicator(int* aborts) : aborts_(aborts) {}

  absl::Status Abort() override {
    ++*aborts_;
    return absl::OkStatus();
  }

  Future<> AllReduce(stream_executor::DeviceAddressBase,
                     stream_executor::DeviceAddressBase, PrimitiveType, size_t,
                     ReductionKind, const Executor&) override {
    return absl::UnimplementedError("unused");
  }
  Future<> Broadcast(stream_executor::DeviceAddressBase,
                     stream_executor::DeviceAddressBase, PrimitiveType, size_t,
                     RankId, const Executor&) override {
    return absl::UnimplementedError("unused");
  }
  Future<> ReduceScatter(stream_executor::DeviceAddressBase,
                         stream_executor::DeviceAddressBase, PrimitiveType,
                         size_t, ReductionKind, const Executor&) override {
    return absl::UnimplementedError("unused");
  }
  Future<> AllGather(stream_executor::DeviceAddressBase,
                     stream_executor::DeviceAddressBase, PrimitiveType, size_t,
                     const Executor&) override {
    return absl::UnimplementedError("unused");
  }
  Future<> CollectivePermute(stream_executor::DeviceAddressBase,
                             stream_executor::DeviceAddressBase, PrimitiveType,
                             size_t, std::optional<RankId>,
                             absl::Span<const RankId>,
                             const Executor&) override {
    return absl::UnimplementedError("unused");
  }
  Future<> AllToAll(absl::InlinedVector<stream_executor::DeviceAddressBase, 4>,
                    absl::InlinedVector<stream_executor::DeviceAddressBase, 4>,
                    PrimitiveType, size_t, const Executor&) override {
    return absl::UnimplementedError("unused");
  }
  Future<> Send(stream_executor::DeviceAddressBase, PrimitiveType, size_t,
                RankId, const Executor&) override {
    return absl::UnimplementedError("unused");
  }
  Future<> Recv(stream_executor::DeviceAddressBase, PrimitiveType, size_t,
                RankId, const Executor&) override {
    return absl::UnimplementedError("unused");
  }
  absl::StatusOr<size_t> NumRanks() const override { return 1; }
  std::string ToString() const override { return "test-communicator"; }

 private:
  int* aborts_;
};

TEST(GpuCliqueTest, AgreementFailureIsStickyAndCancelsClique) {
  int aborts = 0;
  absl::btree_map<RankId, std::unique_ptr<Communicator>> communicators;
  communicators.emplace(RankId(0), std::make_unique<TestCommunicator>(&aborts));
  GpuClique clique(
      GpuCliqueKey({GlobalDeviceId(0)}, /*num_local_participants=*/1),
      /*ids=*/std::nullopt, std::move(communicators),
      /*peer_access_enabled=*/true, std::make_shared<CancellationToken>());
  absl::Status first =
      clique.PoisonAgreement(absl::DeadlineExceededError("kv timeout"));
  EXPECT_TRUE(absl::IsDeadlineExceeded(first));
  EXPECT_TRUE(clique.IsCancelled());
  EXPECT_EQ(aborts, 1);

  absl::Status second =
      clique.PoisonAgreement(absl::InternalError("later failure"));
  EXPECT_EQ(second, first);
  EXPECT_EQ(clique.agreement_status(), first);
  EXPECT_EQ(aborts, 1);
}

TEST(GpuCliqueTest, RecordsAllFailuresBeforeStartingProviderAbort) {
  int aborts = 0;
  absl::btree_map<RankId, std::unique_ptr<Communicator>> communicators;
  communicators.emplace(RankId(0), std::make_unique<TestCommunicator>(&aborts));
  GpuClique clique(
      GpuCliqueKey({GlobalDeviceId(0)}, /*num_local_participants=*/1),
      /*ids=*/std::nullopt, std::move(communicators),
      /*peer_access_enabled=*/true, std::make_shared<CancellationToken>());

  absl::Status failure =
      clique.RecordAgreementFailure(absl::UnavailableError("peer failed"));
  EXPECT_TRUE(absl::IsUnavailable(failure));
  EXPECT_TRUE(clique.IsCancelled());
  EXPECT_EQ(aborts, 0);

  EXPECT_TRUE(clique.AbortAfterAgreementFailure().ok());
  EXPECT_EQ(aborts, 1);
  EXPECT_TRUE(clique.AbortAfterAgreementFailure().ok());
  EXPECT_EQ(aborts, 1);
}

TEST(GpuCliqueTest, RejectsAbortBeforeAgreementFailure) {
  int aborts = 0;
  absl::btree_map<RankId, std::unique_ptr<Communicator>> communicators;
  communicators.emplace(RankId(0), std::make_unique<TestCommunicator>(&aborts));
  GpuClique clique(
      GpuCliqueKey({GlobalDeviceId(0)}, /*num_local_participants=*/1),
      /*ids=*/std::nullopt, std::move(communicators),
      /*peer_access_enabled=*/true, std::make_shared<CancellationToken>());

  EXPECT_TRUE(absl::IsFailedPrecondition(clique.AbortAfterAgreementFailure()));
  EXPECT_EQ(aborts, 0);
}

}  // namespace
}  // namespace xla::gpu
