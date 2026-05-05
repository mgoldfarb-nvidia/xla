/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
==============================================================================*/

// Proposal C3 (docs/lhs_pgle_baseline_improvements.md):
//
// Auto-window-target pass. Programmatic version of `_xla_window_target`
// rules: scans every async-collective `*-start` whose PGLE-recorded cost
// exceeds a threshold and selects safe-anchor compute ops to pin into its
// window. The selected (collective, anchor) pairings are emitted as an
// in-memory `CollectiveHintsConfig` and fed to a second instance of
// `CollectiveHintsAnnotatorPass`, which inserts the control deps
// `start -> anchor -> done` using its existing cycle-safe machinery.
//
// Anchor selection:
//   - Same computation as the collective.
//   - After the collective in HLO file order (otherwise can't be pinned
//     "into" the window).
//   - Operand-tree independent of the collective (no dataflow chain
//     either direction). Uses HloReachabilityMap.
//   - Eligible opcode (kCustomCall / kFusion-Triton-GEMM / kDot /
//     kConvolution) — same predicate as `keep_sync_annotation` in
//     `gpu_hlo_schedule.cc`.
//   - PGLE compute cost >= `min_compute_us`.
//   - Not already claimed (via `_xla_window_target` from manual hints,
//     or by a previous iteration of this pass).
//   - Closest-in-schedule first; ties broken by HLO instruction `name`.
//
// Greedy fill: pick anchors until cumulative compute cost >= collective
// cost OR `max_per_collective` reached.
//
// Skipped: collectives already paired via `_scheduling_group_id` (user
// supplied a manual pair).
//
// Pass ordering (enforced by placement in `gpu_hlo_schedule.cc`):
//   ... AsyncCollectiveCreator ...
//   CollectiveHintsAnnotatorPass (textproto-driven, if any)
//   LegalizeSchedulingAnnotations  (incl. A1's singleton drop)
//   ApplyPgleForceAsync (B1, if threshold > 0)
//   AutoWindowTargetPass (this; runs only if threshold > 0)
//   RunLatencyHidingSchedulerPasses

#ifndef XLA_SERVICE_GPU_AUTO_WINDOW_TARGET_H_
#define XLA_SERVICE_GPU_AUTO_WINDOW_TARGET_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/gpu/collective_hints.pb.h"
#include "tsl/profiler/protobuf/profiled_instructions.pb.h"

namespace xla::gpu {

// Pure helper: builds the auto-config from the current module state +
// PGLE profile. Exposed for direct unit testing of the selection logic.
// The returned config has zero rules if `threshold_us <= 0` or no
// candidates qualify.
//
// `max_total_rules`: safety cap on total rules emitted (across all
// collectives in the module). Rules are emitted in priority order
// (PGLE-cost-descending across collectives, then within each collective
// the highest-cost anchors first); the lowest-priority rules are dropped
// when the cap is hit. Set to <=0 to disable the cap.
absl::StatusOr<CollectiveHintsConfig> BuildAutoWindowTargetConfig(
    const HloModule& module,
    const tensorflow::profiler::ProfiledInstructionsProto& profile,
    float threshold_us, float min_compute_us, int max_per_collective,
    int max_total_rules);

class AutoWindowTargetPass : public HloModulePass {
 public:
  AutoWindowTargetPass(
      tensorflow::profiler::ProfiledInstructionsProto profile,
      float threshold_us, float min_compute_us, int max_per_collective,
      int max_total_rules)
      : profile_(std::move(profile)),
        threshold_us_(threshold_us),
        min_compute_us_(min_compute_us),
        max_per_collective_(max_per_collective),
        max_total_rules_(max_total_rules) {}

  absl::string_view name() const override { return "auto-window-target"; }

  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads)
      override;

 private:
  tensorflow::profiler::ProfiledInstructionsProto profile_;
  float threshold_us_;
  float min_compute_us_;
  int max_per_collective_;
  int max_total_rules_;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_AUTO_WINDOW_TARGET_H_
