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

#include "xla/service/gpu/gpu_executable_completion_barrier.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "xla/runtime/device_id.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/tsl/platform/status_matchers.h"

namespace xla::gpu {
namespace {

TEST(GpuExecutableCompletionBarrierTest, SelectsSafeErrorCleanupDisposition) {
  using BarrierState = GpuExecutableCompletionBarrierState;
  using Disposition = GpuExecutableErrorCleanupDisposition;

  EXPECT_EQ(GetGpuExecutableErrorCleanupDisposition(
                BarrierState::kNotRequired,
                /*deferred_controller_available=*/true,
                /*tail_cleanup_enabled=*/true),
            Disposition::kDeferred);
  EXPECT_EQ(GetGpuExecutableErrorCleanupDisposition(
                BarrierState::kNotRequired,
                /*deferred_controller_available=*/false,
                /*tail_cleanup_enabled=*/true),
            Disposition::kSynchronous);
  EXPECT_EQ(GetGpuExecutableErrorCleanupDisposition(
                BarrierState::kNotRequired,
                /*deferred_controller_available=*/true,
                /*tail_cleanup_enabled=*/false),
            Disposition::kSynchronous);
  EXPECT_EQ(GetGpuExecutableErrorCleanupDisposition(
                BarrierState::kSucceeded,
                /*deferred_controller_available=*/true,
                /*tail_cleanup_enabled=*/true),
            Disposition::kSynchronous);
  EXPECT_EQ(GetGpuExecutableErrorCleanupDisposition(
                BarrierState::kFailed,
                /*deferred_controller_available=*/true,
                /*tail_cleanup_enabled=*/true),
            Disposition::kQuarantine);
  EXPECT_EQ(GetGpuExecutableErrorCleanupDisposition(
                BarrierState::kFailed,
                /*deferred_controller_available=*/false,
                /*tail_cleanup_enabled=*/true),
            Disposition::kFatal);
  EXPECT_EQ(GetGpuExecutableErrorCleanupDisposition(
                BarrierState::kRemote,
                /*deferred_controller_available=*/true,
                /*tail_cleanup_enabled=*/false),
            Disposition::kQuarantine);
  EXPECT_EQ(GetGpuExecutableErrorCleanupDisposition(
                BarrierState::kRemote,
                /*deferred_controller_available=*/false,
                /*tail_cleanup_enabled=*/false),
            Disposition::kFatal);
}

TEST(GpuExecutableCompletionBarrierTest, AggregatesLocalRankCompletionFailure) {
  absl::Status first = absl::OkStatus();
  absl::Status second = absl::InternalError("stream synchronization failed");
  absl::Status third = absl::OkStatus();
  std::vector<absl::Status*> statuses = {&first, &second, &third};

  absl::Status result = AggregateGpuExecutableCompletionStatuses(statuses);
  EXPECT_EQ(result, second);
}

TEST(GpuExecutableCompletionBarrierTest, RejectsNullLocalRankCompletionStatus) {
  std::vector<absl::Status*> statuses = {nullptr};
  EXPECT_EQ(AggregateGpuExecutableCompletionStatuses(statuses).code(),
            absl::StatusCode::kInternal);
}

TEST(GpuExecutableCompletionBarrierTest,
     ResourceInitializationBroadcastsSiblingFailure) {
  const std::array<GlobalDeviceId, 2> participants = {GlobalDeviceId(10),
                                                      GlobalDeviceId(11)};
  std::array<absl::Status, 2> results;

  std::thread first([&] {
    results[0] = RendezvousGpuExecutableStatuses(
        "resource initialization failure", /*run_id=*/1001,
        GpuExecutableRendezvousPhase::kCollectiveResourceInitialization,
        participants, absl::OkStatus(), /*num_local_participants=*/2,
        absl::Seconds(1), absl::Seconds(5));
  });
  std::thread second([&] {
    // Reverse the input order to verify that participant identity is
    // canonicalized before it is used as a rendezvous key.
    const std::array<GlobalDeviceId, 2> reversed = {GlobalDeviceId(11),
                                                    GlobalDeviceId(10)};
    results[1] = RendezvousGpuExecutableStatuses(
        "resource initialization failure", /*run_id=*/1001,
        GpuExecutableRendezvousPhase::kCollectiveResourceInitialization,
        reversed, absl::InternalError("rank 1 failed to acquire clique"),
        /*num_local_participants=*/2, absl::Seconds(1), absl::Seconds(5));
  });
  first.join();
  second.join();

  EXPECT_EQ(results[0], absl::InternalError("rank 1 failed to acquire clique"));
  EXPECT_EQ(results[1], results[0]);
}

TEST(GpuExecutableCompletionBarrierTest,
     InitializationBroadcastsSiblingFailure) {
  std::array<absl::Status, 2> results;

  std::thread first([&] {
    results[0] = RendezvousGpuExecutableStatuses(
        "thunk initialization failure", /*run_id=*/1002,
        GpuExecutableRendezvousPhase::kInitialization,
        /*participant_devices=*/{}, absl::OkStatus(),
        /*num_local_participants=*/2, absl::Seconds(1), absl::Seconds(5));
  });
  std::thread second([&] {
    results[1] = RendezvousGpuExecutableStatuses(
        "thunk initialization failure", /*run_id=*/1002,
        GpuExecutableRendezvousPhase::kInitialization,
        /*participant_devices=*/{},
        absl::FailedPreconditionError("rank 1 Initialize failed"),
        /*num_local_participants=*/2, absl::Seconds(1), absl::Seconds(5));
  });
  first.join();
  second.join();

  EXPECT_EQ(results[0],
            absl::FailedPreconditionError("rank 1 Initialize failed"));
  EXPECT_EQ(results[1], results[0]);
}

TEST(GpuExecutableCompletionBarrierTest, PrepareBroadcastsSiblingFailure) {
  std::array<absl::Status, 2> results;

  std::thread first([&] {
    results[0] = RendezvousGpuExecutableStatuses(
        "thunk preparation failure", /*run_id=*/1003,
        GpuExecutableRendezvousPhase::kPrepare,
        /*participant_devices=*/{}, absl::OkStatus(),
        /*num_local_participants=*/2, absl::Seconds(1), absl::Seconds(5));
  });
  std::thread second([&] {
    results[1] = RendezvousGpuExecutableStatuses(
        "thunk preparation failure", /*run_id=*/1003,
        GpuExecutableRendezvousPhase::kPrepare,
        /*participant_devices=*/{},
        absl::FailedPreconditionError("rank 1 Prepare failed"),
        /*num_local_participants=*/2, absl::Seconds(1), absl::Seconds(5));
  });
  first.join();
  second.join();

  EXPECT_EQ(results[0], absl::FailedPreconditionError("rank 1 Prepare failed"));
  EXPECT_EQ(results[1], results[0]);
}

TEST(GpuExecutableCompletionBarrierTest,
     SameRunDisjointAndOverlappingTeamsDoNotCrossPair) {
  auto run_scoped_rounds = [](int64_t run_id,
                              std::array<GlobalDeviceId, 2> first_team,
                              std::array<GlobalDeviceId, 2> second_team) {
    std::array<absl::Status, 4> results;
    const absl::Status first_team_failure =
        absl::UnavailableError("first team failed");

    std::array<std::thread, 4> threads = {
        std::thread([&] {
          results[0] = RendezvousGpuExecutableStatuses(
              "first scoped team", run_id,
              GpuExecutableRendezvousPhase::kCompletion, first_team,
              absl::OkStatus(), /*num_local_participants=*/2, absl::Seconds(1),
              absl::Seconds(5));
        }),
        std::thread([&] {
          results[2] = RendezvousGpuExecutableStatuses(
              "second scoped team", run_id,
              GpuExecutableRendezvousPhase::kCompletion, second_team,
              absl::OkStatus(), /*num_local_participants=*/2, absl::Seconds(1),
              absl::Seconds(5));
        }),
        std::thread([&] {
          results[1] = RendezvousGpuExecutableStatuses(
              "first scoped team", run_id,
              GpuExecutableRendezvousPhase::kCompletion, first_team,
              first_team_failure, /*num_local_participants=*/2,
              absl::Seconds(1), absl::Seconds(5));
        }),
        std::thread([&] {
          results[3] = RendezvousGpuExecutableStatuses(
              "second scoped team", run_id,
              GpuExecutableRendezvousPhase::kCompletion, second_team,
              absl::OkStatus(), /*num_local_participants=*/2, absl::Seconds(1),
              absl::Seconds(5));
        })};
    for (std::thread& thread : threads) thread.join();

    EXPECT_EQ(results[0], first_team_failure);
    EXPECT_EQ(results[1], first_team_failure);
    EXPECT_OK(results[2]);
    EXPECT_OK(results[3]);
  };

  run_scoped_rounds(/*run_id=*/1003, {GlobalDeviceId(0), GlobalDeviceId(1)},
                    {GlobalDeviceId(2), GlobalDeviceId(3)});
  run_scoped_rounds(/*run_id=*/1004, {GlobalDeviceId(0), GlobalDeviceId(1)},
                    {GlobalDeviceId(1), GlobalDeviceId(2)});
}

TEST(GpuExecutableCompletionBarrierTest,
     CountsAllRequestedDevicesWithoutLocalMap) {
  const absl::flat_hash_set<GlobalDeviceId> requested_devices = {
      GlobalDeviceId(10), GlobalDeviceId(11), GlobalDeviceId(12),
      GlobalDeviceId(13)};
  size_t observed_participants = 0;
  int barrier_calls = 0;

  EXPECT_OK(MaybeRunGpuExecutableCompletionBarrier(
      requested_devices, GlobalDeviceId(10),
      /*local_to_global_device_map=*/nullptr, absl::OkStatus(),
      [&](size_t num_participants) {
        ++barrier_calls;
        observed_participants = num_participants;
        return absl::OkStatus();
      }));

  EXPECT_EQ(barrier_calls, 1);
  EXPECT_EQ(observed_participants, 4);
}

TEST(GpuExecutableCompletionBarrierTest,
     CountsOnlyRequestedDevicesInLocalDeviceMap) {
  const absl::flat_hash_set<GlobalDeviceId> requested_devices = {
      GlobalDeviceId(10), GlobalDeviceId(11), GlobalDeviceId(12),
      GlobalDeviceId(13)};
  const GpuExecutableRunOptions::DeviceIdMap local_to_global_device_map = {
      {LocalDeviceId(0), GlobalDeviceId(10)},
      {LocalDeviceId(1), GlobalDeviceId(12)},
      {LocalDeviceId(2), GlobalDeviceId(99)}};
  size_t observed_participants = 0;

  EXPECT_OK(MaybeRunGpuExecutableCompletionBarrier(
      requested_devices, GlobalDeviceId(10), &local_to_global_device_map,
      absl::OkStatus(), [&](size_t num_participants) {
        observed_participants = num_participants;
        return absl::OkStatus();
      }));

  EXPECT_EQ(observed_participants, 2);
}

TEST(GpuExecutableCompletionBarrierTest, NonParticipantSkipsBarrier) {
  const absl::flat_hash_set<GlobalDeviceId> requested_devices = {
      GlobalDeviceId(10), GlobalDeviceId(11)};
  int barrier_calls = 0;

  EXPECT_OK(MaybeRunGpuExecutableCompletionBarrier(
      requested_devices, GlobalDeviceId(12),
      /*local_to_global_device_map=*/nullptr, absl::OkStatus(), [&](size_t) {
        ++barrier_calls;
        return absl::OkStatus();
      }));

  EXPECT_EQ(barrier_calls, 0);
}

TEST(GpuExecutableCompletionBarrierTest,
     ExecutionErrorStillRunsBarrierAndRemainsPrimary) {
  const absl::flat_hash_set<GlobalDeviceId> requested_devices = {
      GlobalDeviceId(10)};
  int barrier_calls = 0;

  absl::Status status = MaybeRunGpuExecutableCompletionBarrier(
      requested_devices, GlobalDeviceId(10),
      /*local_to_global_device_map=*/nullptr,
      absl::InternalError("thunk execution failed"), [&](size_t) {
        ++barrier_calls;
        return absl::UnavailableError("completion barrier failed");
      });

  EXPECT_EQ(barrier_calls, 1);
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_EQ(status.message(), "thunk execution failed");
}

TEST(GpuExecutableCompletionBarrierTest,
     BarrierErrorReturnedWhenExecutionSucceeded) {
  const absl::flat_hash_set<GlobalDeviceId> requested_devices = {
      GlobalDeviceId(10)};

  absl::Status status = MaybeRunGpuExecutableCompletionBarrier(
      requested_devices, GlobalDeviceId(10),
      /*local_to_global_device_map=*/nullptr, absl::OkStatus(), [](size_t) {
        return absl::UnavailableError("completion barrier failed");
      });

  EXPECT_EQ(status.code(), absl::StatusCode::kUnavailable);
  EXPECT_EQ(status.message(), "completion barrier failed");
}

TEST(GpuExecutableCompletionBarrierTest, RejectsEmptyLocalIntersection) {
  const absl::flat_hash_set<GlobalDeviceId> requested_devices = {
      GlobalDeviceId(10)};
  const GpuExecutableRunOptions::DeviceIdMap local_to_global_device_map = {
      {LocalDeviceId(0), GlobalDeviceId(99)}};
  int barrier_calls = 0;

  absl::Status status = MaybeRunGpuExecutableCompletionBarrier(
      requested_devices, GlobalDeviceId(10), &local_to_global_device_map,
      absl::OkStatus(), [&](size_t) {
        ++barrier_calls;
        return absl::OkStatus();
      });

  EXPECT_EQ(barrier_calls, 0);
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_EQ(status.message(),
            "No local participants found for barrier after executable");
}

TEST(GpuExecutableCompletionBarrierTest,
     SynchronizesEachDistinctNonNullStreamOnce) {
  auto* first = reinterpret_cast<stream_executor::Stream*>(uintptr_t{1});
  auto* second = reinterpret_cast<stream_executor::Stream*>(uintptr_t{2});
  std::vector<stream_executor::Stream*> streams = {first, nullptr, second,
                                                   first, second};
  std::vector<stream_executor::Stream*> synchronized;

  EXPECT_OK(SynchronizeGpuExecutableStreams(
      streams, [&](stream_executor::Stream* stream) {
        synchronized.push_back(stream);
        return absl::OkStatus();
      }));

  EXPECT_EQ(synchronized,
            (std::vector<stream_executor::Stream*>{first, second}));
}

TEST(GpuExecutableCompletionBarrierTest,
     StreamSynchronizationContinuesAfterFailure) {
  auto* first = reinterpret_cast<stream_executor::Stream*>(uintptr_t{1});
  auto* second = reinterpret_cast<stream_executor::Stream*>(uintptr_t{2});
  std::vector<stream_executor::Stream*> streams = {first, second};
  int calls = 0;

  absl::Status status = SynchronizeGpuExecutableStreams(
      streams, [&](stream_executor::Stream* stream) {
        ++calls;
        return stream == first ? absl::InternalError("first stream failed")
                               : absl::OkStatus();
      });

  EXPECT_EQ(calls, 2);
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_EQ(status.message(), "first stream failed");
}

}  // namespace
}  // namespace xla::gpu
