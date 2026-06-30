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

#ifndef XLA_SERVICE_GPU_GPU_EXECUTABLE_COMPLETION_BARRIER_H_
#define XLA_SERVICE_GPU_GPU_EXECUTABLE_COMPLETION_BARRIER_H_

#include <cstddef>

#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/runtime/device_id.h"
#include "xla/service/gpu/gpu_executable_run_options.h"

namespace stream_executor {
class Stream;
}  // namespace stream_executor

namespace xla::gpu {

// Runs an executable completion barrier when `current_device` participates.
//
// If a local-to-global device map is available, only requested devices hosted
// by the current process count as rendezvous participants. `execute_status` is
// the primary status: a barrier failure replaces it only when execution itself
// succeeded.
absl::Status MaybeRunGpuExecutableCompletionBarrier(
    const absl::flat_hash_set<GlobalDeviceId>& requested_devices,
    GlobalDeviceId current_device,
    const GpuExecutableRunOptions::DeviceIdMap* local_to_global_device_map,
    absl::Status execute_status,
    absl::FunctionRef<absl::Status(size_t)> run_barrier);

// Synchronizes each distinct non-null stream exactly once. All streams are
// attempted even if an earlier synchronization fails, so error cleanup does
// not abandon work queued on later streams.
absl::Status SynchronizeGpuExecutableStreams(
    absl::Span<stream_executor::Stream* const> streams,
    absl::FunctionRef<absl::Status(stream_executor::Stream*)> synchronize);

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_GPU_EXECUTABLE_COMPLETION_BARRIER_H_
