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

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "xla/runtime/device_id.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/tsl/platform/status_matchers.h"

namespace xla::gpu {
namespace {

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
