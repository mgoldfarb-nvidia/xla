/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_COLLECTIVE_HINTS_ANNOTATOR_H_
#define XLA_SERVICE_GPU_COLLECTIVE_HINTS_ANNOTATOR_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/gpu/collective_hints.pb.h"

namespace xla {
namespace gpu {

// HLO pass that reads a CollectiveHintsConfig and injects hints onto matching
// collective async-start instructions: latency_metadata, _scheduling_group_id,
// and _xla_stream_annotation as frontend_attributes; force_earliest_schedule
// via GpuBackendConfig (used by the LHS GPU scheduler).
//
// All matching rules are merged. When multiple rules match, the rule with the
// highest `priority` field wins for each hint attribute. Rules with equal
// priority are resolved by their order in the list (later entry wins).
// Only instructions that are async collective starts are candidates.
//
// This pass runs before LegalizeSchedulingAnnotations in
// RunLatencyHidingSchedulerPasses.
class CollectiveHintsAnnotatorPass : public HloModulePass {
 public:
  explicit CollectiveHintsAnnotatorPass(CollectiveHintsConfig config);

  // Creates a pass by loading a CollectiveHintsConfig proto text file from
  // `path`. Returns an error if the file cannot be read or parsed.
  static absl::StatusOr<CollectiveHintsAnnotatorPass> Create(
      absl::string_view path);

  absl::string_view name() const override {
    return "collective-hints-annotator";
  }

  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  CollectiveHintsConfig config_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_COLLECTIVE_HINTS_ANNOTATOR_H_
