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
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/runtime/device_id.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/stream_executor/stream.h"

namespace xla::gpu {

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
