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

#include "xla/backends/gpu/collectives/gpu_clique_agreement.h"

#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/executable_run_options.h"
#include "xla/pjrt/distributed/in_memory_key_value_store.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/runtime/device_id.h"
#include "xla/runtime/process_id.h"
#include "xla/tsl/distributed_runtime/call_options.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_service_agent.h"
#include "xla/tsl/platform/test.h"

namespace xla::gpu {
namespace {

using DeviceToProcess = absl::flat_hash_map<GlobalDeviceId, ProcessId>;

class RecordingKeyValueStore : public KeyValueStoreInterface {
 public:
  RecordingKeyValueStore() : store_(/*allow_overwrite=*/false) {}

  absl::StatusOr<std::string> Get(absl::string_view key,
                                  absl::Duration timeout) override {
    return store_.Get(key, timeout);
  }

  absl::StatusOr<std::string> TryGet(absl::string_view key) override {
    return store_.TryGet(key);
  }

  absl::Status Set(absl::string_view key, absl::string_view value) override {
    {
      absl::MutexLock lock(mutex_);
      set_keys_.emplace_back(key);
    }
    return store_.Set(key, value);
  }

  std::shared_ptr<tsl::CallOptions> AsyncGet(
      absl::string_view key,
      tsl::CoordinationServiceAgent::StatusOrValueCallback done) override {
    return store_.AsyncGet(key, std::move(done));
  }

  std::vector<std::string> SetKeys() const {
    absl::MutexLock lock(mutex_);
    return set_keys_;
  }

 private:
  InMemoryKeyValueStore store_;
  mutable absl::Mutex mutex_;
  std::vector<std::string> set_keys_ ABSL_GUARDED_BY(mutex_);
};

GpuCliqueAgreementRequest Request(
    int64_t run_id, int32_t launch_id, std::string session_id,
    std::vector<GlobalDeviceId> devices, int64_t num_local_participants,
    std::string phase, int64_t logical_slot, GlobalDeviceId device_id,
    absl::Status status = absl::OkStatus(), std::string payload = "plan-v1") {
  return GpuCliqueAgreementRequest(
      RunId(run_id), launch_id, std::move(session_id),
      GpuCliqueKey(std::move(devices), num_local_participants),
      std::move(phase), logical_slot, device_id, std::move(status),
      std::move(payload));
}

struct AgreementCall {
  const GpuCliqueAgreement* agreement;
  GpuCliqueAgreementRequest request;
};

std::vector<absl::Status> RunCalls(std::vector<AgreementCall> calls) {
  std::vector<absl::Status> statuses(calls.size());
  std::vector<std::thread> threads;
  threads.reserve(calls.size());
  for (size_t i = 0; i < calls.size(); ++i) {
    threads.emplace_back([&, i] {
      statuses[i] = AgreeGpuClique(calls[i].agreement, calls[i].request);
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }
  return statuses;
}

void ExpectAllOk(const std::vector<absl::Status>& statuses) {
  for (const absl::Status& status : statuses) {
    EXPECT_TRUE(status.ok()) << status;
  }
}

TEST(GpuCliqueAgreementTest, SupportsOneProcessWithManyGpus) {
  std::vector<GlobalDeviceId> devices = {GlobalDeviceId(0), GlobalDeviceId(1),
                                         GlobalDeviceId(2)};
  ExpectAllOk(RunCalls({
      {nullptr,
       Request(11, 0, "local_many_gpus", devices, 3, "prepare", 0, devices[0])},
      {nullptr,
       Request(11, 0, "local_many_gpus", devices, 3, "prepare", 0, devices[1])},
      {nullptr,
       Request(11, 0, "local_many_gpus", devices, 3, "prepare", 0, devices[2])},
  }));
}

TEST(GpuCliqueAgreementTest, NullAgreementRejectsNonLocalClique) {
  std::vector<GlobalDeviceId> devices = {GlobalDeviceId(5), GlobalDeviceId(6)};
  absl::Status status = AgreeGpuClique(
      /*agreement=*/nullptr,
      Request(12, 15, "null_non_local", devices,
              /*num_local_participants=*/1, "prepare", 0, devices[0]));
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_TRUE(absl::StrContains(status.message(), "configured"));
}

TEST(GpuCliqueAgreementTest, MissingLocalRankTimesOut) {
  std::vector<GlobalDeviceId> devices = {GlobalDeviceId(7), GlobalDeviceId(8)};
  DeviceToProcess topology = {{devices[0], ProcessId(0)},
                              {devices[1], ProcessId(0)}};
  GpuCliqueAgreement agreement(
      ProcessId(0), topology, /*kv_store=*/nullptr,
      GpuCliqueAgreementOptions{absl::Milliseconds(50)});

  absl::Status status = agreement.Agree(Request(
      13, 0, "missing_local_rank", devices, 2, "prepare", 0, devices[0]));
  EXPECT_EQ(status.code(), absl::StatusCode::kDeadlineExceeded);
  EXPECT_TRUE(absl::StrContains(status.message(), "2 local GPU ranks"));
}

TEST(GpuCliqueAgreementTest, RejectsDifferentCanonicalPayloads) {
  std::vector<GlobalDeviceId> devices = {GlobalDeviceId(9), GlobalDeviceId(10)};
  std::vector<absl::Status> statuses = RunCalls({
      {nullptr, Request(14, 0, "different_payloads", devices, 2, "prepare", 0,
                        devices[0], absl::OkStatus(), "plan-a")},
      {nullptr, Request(14, 0, "different_payloads", devices, 2, "prepare", 0,
                        devices[1], absl::OkStatus(), "plan-b")},
  });

  ASSERT_EQ(statuses.size(), 2);
  EXPECT_EQ(statuses[0].code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(statuses[0].message(), statuses[1].message());
  EXPECT_TRUE(absl::StrContains(statuses[0].message(), "canonical payload"));
}

TEST(GpuCliqueAgreementTest, SupportsOneProcessPerGpu) {
  std::vector<GlobalDeviceId> devices = {GlobalDeviceId(10),
                                         GlobalDeviceId(11)};
  DeviceToProcess topology = {{devices[0], ProcessId(0)},
                              {devices[1], ProcessId(1)}};
  auto kv_store = std::make_shared<RecordingKeyValueStore>();
  GpuCliqueAgreement process0(ProcessId(0), topology, kv_store);
  GpuCliqueAgreement process1(ProcessId(1), topology, kv_store);

  // RunId is intentionally different across processes: it is local-only.
  ExpectAllOk(RunCalls({
      {&process0, Request(101, 7, "one_process_per_gpu", devices, 1, "prepare",
                          0, devices[0])},
      {&process1, Request(202, 7, "one_process_per_gpu", devices, 1, "prepare",
                          0, devices[1])},
  }));

  std::vector<std::string> keys = kv_store->SetKeys();
  EXPECT_EQ(keys.size(), 4);
  EXPECT_TRUE(absl::c_any_of(keys, [](const std::string& key) {
    return absl::StrContains(key, "/proposal/");
  }));
  EXPECT_TRUE(absl::c_any_of(keys, [](const std::string& key) {
    return absl::StrContains(key, "/vote/");
  }));
}

TEST(GpuCliqueAgreementTest, SupportsUnevenGpuCountsAcrossProcesses) {
  std::vector<GlobalDeviceId> devices = {GlobalDeviceId(20), GlobalDeviceId(21),
                                         GlobalDeviceId(22)};
  DeviceToProcess topology = {{devices[0], ProcessId(0)},
                              {devices[1], ProcessId(0)},
                              {devices[2], ProcessId(1)}};
  auto kv_store = std::make_shared<RecordingKeyValueStore>();
  GpuCliqueAgreement process0(ProcessId(0), topology, kv_store);
  GpuCliqueAgreement process1(ProcessId(1), topology, kv_store);

  ExpectAllOk(RunCalls({
      {&process0, Request(301, 8, "uneven_gpu_counts", devices, 2, "prepare", 4,
                          devices[0])},
      {&process0, Request(301, 8, "uneven_gpu_counts", devices, 2, "prepare", 4,
                          devices[1])},
      {&process1, Request(401, 8, "uneven_gpu_counts", devices, 1, "prepare", 4,
                          devices[2])},
  }));

  // Each process publishes exactly one proposal and one vote regardless of its
  // local GPU count.
  EXPECT_EQ(kv_store->SetKeys().size(), 4);
}

TEST(GpuCliqueAgreementTest, OrderingMismatchCollidesAtSameGeneration) {
  std::vector<GlobalDeviceId> devices = {GlobalDeviceId(30),
                                         GlobalDeviceId(31)};
  DeviceToProcess topology = {{devices[0], ProcessId(0)},
                              {devices[1], ProcessId(1)}};
  auto kv_store =
      std::make_shared<InMemoryKeyValueStore>(/*allow_overwrite=*/false);
  GpuCliqueAgreement process0(ProcessId(0), topology, kv_store);
  GpuCliqueAgreement process1(ProcessId(1), topology, kv_store);

  std::vector<absl::Status> statuses = RunCalls({
      {&process0, Request(501, 9, "ordering_mismatch", devices, 1, "prepare", 0,
                          devices[0])},
      {&process1, Request(601, 10, "ordering_mismatch", devices, 1, "register",
                          1, devices[1])},
  });

  ASSERT_EQ(statuses.size(), 2);
  EXPECT_EQ(statuses[0].code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(statuses[0].code(), statuses[1].code());
  EXPECT_EQ(statuses[0].message(), statuses[1].message());
  EXPECT_TRUE(absl::StrContains(statuses[0].message(), "differs"));
  EXPECT_FALSE(absl::IsDeadlineExceeded(statuses[0]));
}

TEST(GpuCliqueAgreementTest, SupportsDefaultLaunchIdForInitialization) {
  std::vector<GlobalDeviceId> devices = {GlobalDeviceId(40),
                                         GlobalDeviceId(41)};
  DeviceToProcess topology = {{devices[0], ProcessId(0)},
                              {devices[1], ProcessId(1)}};
  auto kv_store =
      std::make_shared<InMemoryKeyValueStore>(/*allow_overwrite=*/false);
  GpuCliqueAgreement process0(ProcessId(0), topology, kv_store);
  GpuCliqueAgreement process1(ProcessId(1), topology, kv_store);

  std::vector<absl::Status> statuses = RunCalls({
      {&process0,
       Request(701, 0, "zero_launch_id", devices, 1, "prepare", 0, devices[0])},
      {&process1,
       Request(801, 0, "zero_launch_id", devices, 1, "prepare", 0, devices[1])},
  });

  ASSERT_EQ(statuses.size(), 2);
  ExpectAllOk(statuses);
}

TEST(GpuCliqueAgreementTest, AggregatesRankErrorsDeterministically) {
  std::vector<GlobalDeviceId> devices = {GlobalDeviceId(50), GlobalDeviceId(51),
                                         GlobalDeviceId(52)};
  DeviceToProcess topology = {{devices[0], ProcessId(0)},
                              {devices[1], ProcessId(0)},
                              {devices[2], ProcessId(1)}};
  auto kv_store =
      std::make_shared<InMemoryKeyValueStore>(/*allow_overwrite=*/false);
  GpuCliqueAgreement process0(ProcessId(0), topology, kv_store);
  GpuCliqueAgreement process1(ProcessId(1), topology, kv_store);

  std::vector<absl::Status> statuses = RunCalls({
      {&process1,
       Request(901, 12, "deterministic_errors", devices, 1, "prepare", 0,
               devices[2], absl::InvalidArgumentError("device-52-error"))},
      {&process0, Request(902, 12, "deterministic_errors", devices, 2,
                          "prepare", 0, devices[1])},
      {&process0,
       Request(902, 12, "deterministic_errors", devices, 2, "prepare", 0,
               devices[0], absl::UnavailableError("device-50-error"))},
  });

  ASSERT_EQ(statuses.size(), 3);
  for (const absl::Status& status : statuses) {
    EXPECT_EQ(status.code(), absl::StatusCode::kUnavailable);
    EXPECT_EQ(status.message(), statuses[0].message());
  }
  size_t device50 = statuses[0].message().find("device-50-error");
  size_t device52 = statuses[0].message().find("device-52-error");
  ASSERT_NE(device50, std::string::npos);
  ASSERT_NE(device52, std::string::npos);
  EXPECT_LT(device50, device52);
}

TEST(GpuCliqueAgreementTest, UsesMonotonicSessionGeneration) {
  std::vector<GlobalDeviceId> devices = {GlobalDeviceId(60),
                                         GlobalDeviceId(61)};
  DeviceToProcess topology = {{devices[0], ProcessId(0)},
                              {devices[1], ProcessId(1)}};
  auto kv_store = std::make_shared<RecordingKeyValueStore>();
  GpuCliqueAgreement process0(ProcessId(0), topology, kv_store);
  GpuCliqueAgreement process1(ProcessId(1), topology, kv_store);

  ExpectAllOk(RunCalls({
      {&process0, Request(1001, 13, "monotonic_generation", devices, 1,
                          "prepare", 0, devices[0])},
      {&process1, Request(1002, 13, "monotonic_generation", devices, 1,
                          "prepare", 0, devices[1])},
  }));
  ExpectAllOk(RunCalls({
      {&process0, Request(1001, 13, "monotonic_generation", devices, 1,
                          "register", 1, devices[0])},
      {&process1, Request(1002, 13, "monotonic_generation", devices, 1,
                          "register", 1, devices[1])},
  }));

  std::vector<std::string> keys = kv_store->SetKeys();
  EXPECT_TRUE(absl::c_any_of(keys, [](const std::string& key) {
    return absl::StrContains(key, "/generation/0/");
  }));
  EXPECT_TRUE(absl::c_any_of(keys, [](const std::string& key) {
    return absl::StrContains(key, "/generation/1/");
  }));
}

TEST(GpuCliqueAgreementTest, MissingProcessTimesOut) {
  std::vector<GlobalDeviceId> devices = {GlobalDeviceId(70),
                                         GlobalDeviceId(71)};
  DeviceToProcess topology = {{devices[0], ProcessId(0)},
                              {devices[1], ProcessId(1)}};
  auto kv_store =
      std::make_shared<InMemoryKeyValueStore>(/*allow_overwrite=*/false);
  GpuCliqueAgreement process0(
      ProcessId(0), topology, kv_store,
      GpuCliqueAgreementOptions{absl::Milliseconds(50)});

  absl::Status status =
      process0.Agree(Request(1101, 14, "missing_process_timeout", devices, 1,
                             "prepare", 0, devices[0]));
  EXPECT_EQ(status.code(), absl::StatusCode::kDeadlineExceeded);
  EXPECT_TRUE(absl::StrContains(status.message(), "proposal/1"));
}

}  // namespace
}  // namespace xla::gpu
