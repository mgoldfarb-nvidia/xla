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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/runtime/device_id.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/rendezvous.h"
#include "xla/stream_executor/stream.h"

namespace xla::gpu {
namespace {

struct GpuExecutableRendezvousKey {
  int64_t run_id;
  GpuExecutableRendezvousPhase phase;
  std::vector<int64_t> participant_devices;

  template <typename H>
  friend H AbslHashValue(H h, const GpuExecutableRendezvousKey& key) {
    return H::combine(std::move(h), key.run_id, key.phase,
                      key.participant_devices);
  }

  friend bool operator==(const GpuExecutableRendezvousKey& lhs,
                         const GpuExecutableRendezvousKey& rhs) {
    return lhs.run_id == rhs.run_id && lhs.phase == rhs.phase &&
           lhs.participant_devices == rhs.participant_devices;
  }
};

}  // namespace

absl::Status AggregateGpuExecutableCompletionStatuses(
    absl::Span<absl::Status* const> statuses) {
  absl::Status result = absl::OkStatus();
  for (const absl::Status* status : statuses) {
    if (status == nullptr) {
      result.Update(absl::InternalError(
          "Completion rendezvous received a null rank status"));
    } else {
      result.Update(*status);
    }
  }
  return result;
}

absl::Status RendezvousGpuExecutableStatuses(
    absl::string_view name, int64_t run_id, GpuExecutableRendezvousPhase phase,
    absl::Span<const GlobalDeviceId> participant_devices,
    absl::Status local_status, size_t num_local_participants,
    absl::Duration warn_stuck_timeout, absl::Duration terminate_timeout) {
  std::vector<int64_t> canonical_participants;
  canonical_participants.reserve(participant_devices.size());
  for (GlobalDeviceId device : participant_devices) {
    canonical_participants.push_back(device.value());
  }
  absl::c_sort(canonical_participants);
  canonical_participants.erase(
      std::unique(canonical_participants.begin(), canonical_participants.end()),
      canonical_participants.end());

  GpuExecutableRendezvousKey key{run_id, phase,
                                 std::move(canonical_participants)};
  absl::StatusOr<std::shared_ptr<absl::Status>> rendezvous_status =
      Rendezvous<absl::Status>(
          name, key, local_status, num_local_participants,
          [](absl::Span<absl::Status*> statuses) {
            return AggregateGpuExecutableCompletionStatuses(statuses);
          },
          warn_stuck_timeout, terminate_timeout);
  if (!rendezvous_status.ok()) {
    local_status.Update(rendezvous_status.status());
    return local_status;
  }
  return **rendezvous_status;
}

absl::Status MaybeRunGpuExecutableCompletionBarrier(
    const absl::flat_hash_set<GlobalDeviceId>& requested_devices,
    GlobalDeviceId current_device,
    const GpuExecutableRunOptions::DeviceIdMap* local_to_global_device_map,
    absl::Status execute_status,
    absl::FunctionRef<absl::Status(size_t)> run_barrier) {
  if (!requested_devices.contains(current_device)) {
    return execute_status;
  }

  size_t num_local_participants = requested_devices.size();
  if (local_to_global_device_map != nullptr) {
    absl::flat_hash_set<GlobalDeviceId> local_devices;
    local_devices.reserve(local_to_global_device_map->size());
    for (const auto& entry : *local_to_global_device_map) {
      local_devices.insert(entry.second);
    }
    num_local_participants = absl::c_count_if(
        requested_devices,
        [&](GlobalDeviceId device) { return local_devices.contains(device); });
  }

  if (num_local_participants == 0) {
    execute_status.Update(absl::InternalError(
        "No local participants found for barrier after executable"));
    return execute_status;
  }

  absl::Status barrier_status = run_barrier(num_local_participants);
  execute_status.Update(std::move(barrier_status));
  return execute_status;
}

absl::Status SynchronizeGpuExecutableStreams(
    absl::Span<stream_executor::Stream* const> streams,
    absl::FunctionRef<absl::Status(stream_executor::Stream*)> synchronize) {
  absl::flat_hash_set<stream_executor::Stream*> synchronized;
  synchronized.reserve(streams.size());

  absl::Status status = absl::OkStatus();
  for (stream_executor::Stream* stream : streams) {
    if (stream == nullptr || !synchronized.insert(stream).second) {
      continue;
    }
    status.Update(synchronize(stream));
  }
  return status;
}

}  // namespace xla::gpu
