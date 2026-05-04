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

// Proposal B1 (docs/lhs_pgle_baseline_improvements.md):
//
// PGLE-informed force-async pre-pass. Scans every async-capable collective
// in the module; when its PGLE-recorded cost exceeds the threshold, flips
// `collective_backend_config.is_sync` from true to false and strips any
// residual `_scheduling_group_id` annotation. This recovers async behavior
// that was incorrectly forced sync (e.g., by broken-singleton residuals
// post-LegalizeSchedulingAnnotations, or by AsyncCollectiveCreator
// heuristics that didn't see real runtime cost).
//
// Pass ordering REQUIREMENT (enforced via DCHECKs at runtime):
//   AsyncCollectiveCreator           // wraps eligible collectives
//   LegalizeSchedulingAnnotations    // strips ineligible anchors (+ A1
//                                    //   drops singleton residuals)
//   ApplyPgleForceAsync (this pass)  // flips is_sync based on PGLE
//   RunLatencyHidingSchedulerPasses  // consumes is_sync
//
// `threshold_us == 0` is a no-op (no collective is forced async). This is
// the default — opt-in via `xla_gpu_pgle_force_async_threshold_us`.

#ifndef XLA_SERVICE_GPU_PGLE_FORCE_ASYNC_H_
#define XLA_SERVICE_GPU_PGLE_FORCE_ASYNC_H_

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_module.h"
#include "tsl/profiler/protobuf/profiled_instructions.pb.h"

namespace xla::gpu {

// Iterates async-start collectives in `module`. For each whose PGLE cost
// (looked up by name in `profile`) exceeds `threshold_us`, sets
// `collective_backend_config.is_sync = false` and clears any residual
// `_scheduling_group_id` frontend attribute.
//
// Returns the number of instructions whose is_sync was flipped from true
// to false (0 if `threshold_us <= 0`).
absl::StatusOr<int> ApplyPgleForceAsync(
    HloModule* module,
    const tensorflow::profiler::ProfiledInstructionsProto& profile,
    float threshold_us);

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_PGLE_FORCE_ASYNC_H_
