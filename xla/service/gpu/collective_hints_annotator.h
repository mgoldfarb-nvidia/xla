// Copyright 2026 The OpenXLA Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef XLA_SERVICE_GPU_COLLECTIVE_HINTS_ANNOTATOR_H_
#define XLA_SERVICE_GPU_COLLECTIVE_HINTS_ANNOTATOR_H_

#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/gpu/collective_hints.pb.h"

namespace xla::gpu {

inline constexpr absl::string_view kCollectiveHintsReceiptAttr =
    "_xla_collective_hints_receipt";
inline constexpr absl::string_view kCollectiveHintsFingerprintAttr =
    "_xla_collective_hints_fingerprint";
inline constexpr absl::string_view kCollectiveHintRuleIdsAttr =
    "_xla_collective_hint_rule_ids";
inline constexpr absl::string_view kCollectiveHintWindowTargetAttr =
    "_xla_window_target";

// Loads a strict textproto. Missing files, parse errors, and unknown fields
// return errors; legacy configs containing stream_id therefore fail closed.
absl::StatusOr<CollectiveHintsConfig> LoadCollectiveHintsConfig(
    absl::string_view path);

// Applies fingerprint-bound, exact-match scheduling annotations before
// LegalizeSchedulingAnnotations and LHS.
class CollectiveHintsAnnotatorPass : public HloModulePass {
 public:
  CollectiveHintsAnnotatorPass(CollectiveHintsConfig config,
                               std::string module_fingerprint);

  absl::string_view name() const override {
    return "collective-hints-annotator";
  }

  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  CollectiveHintsConfig config_;
  std::string module_fingerprint_;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_COLLECTIVE_HINTS_ANNOTATOR_H_
