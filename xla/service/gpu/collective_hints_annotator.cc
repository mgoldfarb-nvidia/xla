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

#include "xla/service/gpu/collective_hints_annotator.h"

#include <algorithm>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/collective_hints.pb.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace gpu {

namespace {

// Returns true if `instr` is an async collective start operation.
bool IsAsyncCollectiveStart(const HloInstruction* instr) {
  return hlo_query::IsAsyncCollectiveStartOp(instr, /*include_send_recv=*/true);
}

// Returns true if `op_name` matches the pass filter specified in `hint`.
// An empty pass_filter matches anything.
bool MatchesPassFilter(absl::string_view op_name,
                       absl::string_view pass_filter) {
  if (pass_filter.empty()) return true;
  if (pass_filter == "forward") {
    return absl::StrContains(op_name, "/jvp(") &&
           !absl::StrContains(op_name, "transpose(");
  }
  if (pass_filter == "backward") {
    return absl::StrContains(op_name, "transpose(");
  }
  if (pass_filter == "remat") {
    return absl::StrContains(op_name, "rematted_computation");
  }
  LOG(WARNING) << "CollectiveHintsAnnotator: unknown pass_filter value: "
               << pass_filter;
  return true;
}

// Returns true if `hint` matches `instr`.
bool Matches(const CollectiveHint& hint, const HloInstruction* instr) {
  // Legacy name-based matching.
  if (!hint.name().empty()) {
    absl::string_view rule_name = hint.name();
    absl::string_view instr_name = instr->name();
    // Check for glob pattern (* or ?)
    bool is_glob = absl::StrContains(rule_name, '*') ||
                   absl::StrContains(rule_name, '?');
    if (is_glob) {
      // Simple glob: split on '*' and check prefix/suffix/interior substrings.
      // We support the common patterns: "prefix*", "*suffix", "prefix*suffix".
      // For full glob support this would need fnmatch, but these cover ~all
      // real-world cases.
      std::string pattern(rule_name);
      if (pattern == "*") return true;
      // Find the first '*' and split.
      auto star_pos = pattern.find('*');
      absl::string_view prefix = absl::string_view(pattern).substr(0, star_pos);
      absl::string_view suffix =
          absl::string_view(pattern).substr(star_pos + 1);
      if (!prefix.empty() && !absl::StartsWith(instr_name, prefix)) {
        VLOG(3) << "CollectiveHintsAnnotator: rule name glob '" << rule_name
                << "' prefix mismatch for '" << instr_name << "'";
        return false;
      }
      if (!suffix.empty() && !absl::EndsWith(instr_name, suffix)) {
        VLOG(3) << "CollectiveHintsAnnotator: rule name glob '" << rule_name
                << "' suffix mismatch for '" << instr_name << "'";
        return false;
      }
      return true;
    }
    if (instr_name != rule_name) {
      VLOG(3) << "CollectiveHintsAnnotator: rule name '" << rule_name
              << "' != '" << instr_name << "'";
      return false;
    }
    return true;
  }

  // Metadata-based matching.
  const CollectiveHintMatch& m = hint.match();
  bool has_op_name_filter = !m.op_name().empty() ||
                            !m.op_name_suffix().empty() ||
                            !m.op_name_contains().empty();

  // Collective type filter (opcode substring match).
  // For new-style async wrappers (kAsyncStart wrapping a collective), derive
  // the effective opcode string from the wrapped opcode (e.g. "reduce-scatter"
  // → "reduce-scatter-start") so that collective_type="reduce-scatter" matches
  // both legacy and async-wrapped reduce-scatter-start instructions.
  if (!m.collective_type().empty()) {
    std::string opcode_str;
    if (const auto* async = DynCast<HloAsyncStartInstruction>(instr)) {
      opcode_str =
          std::string(HloOpcodeString(async->async_wrapped_opcode())) + "-start";
    } else {
      opcode_str = std::string(HloOpcodeString(instr->opcode()));
    }
    if (!absl::StrContains(opcode_str, m.collective_type())) {
      VLOG(3) << "CollectiveHintsAnnotator: collective_type '"
              << m.collective_type() << "' does not match opcode '" << opcode_str
              << "' for '" << instr->name() << "'";
      return false;
    }
  }

  // op_name filters.
  // For HloAsyncStartInstruction wrappers, the outer instruction's own
  // metadata.op_name is often empty; fall through to the async-wrapped
  // instruction's op_name so metadata-based rules can target the collective
  // by its JAX source-location path.
  std::string effective_op_name = instr->metadata().op_name();
  if (effective_op_name.empty()) {
    if (const auto* async = DynCast<HloAsyncStartInstruction>(instr);
        async != nullptr && async->async_wrapped_instruction() != nullptr) {
      effective_op_name =
          async->async_wrapped_instruction()->metadata().op_name();
    }
  }
  const std::string& op_name = effective_op_name;
  if (has_op_name_filter) {
    if (op_name.empty()) {
      VLOG(3) << "CollectiveHintsAnnotator: op_name filter requires metadata"
                 " but '" << instr->name() << "' has no op_name";
      return false;
    }
    if (!m.op_name().empty() && op_name != m.op_name()) {
      VLOG(3) << "CollectiveHintsAnnotator: op_name '" << m.op_name()
              << "' != '" << op_name << "' for '" << instr->name() << "'";
      return false;
    }
    if (!m.op_name_contains().empty() &&
        !absl::StrContains(op_name, m.op_name_contains())) {
      VLOG(3) << "CollectiveHintsAnnotator: op_name_contains '"
              << m.op_name_contains() << "' not found in '" << op_name
              << "' for '" << instr->name() << "'";
      return false;
    }
    if (!m.op_name_suffix().empty() &&
        !absl::EndsWith(op_name, m.op_name_suffix())) {
      VLOG(3) << "CollectiveHintsAnnotator: op_name_suffix '"
              << m.op_name_suffix() << "' not a suffix of '" << op_name
              << "' for '" << instr->name() << "'";
      return false;
    }
  }

  // Pass filter.
  if (!m.pass_filter().empty()) {
    if (op_name.empty()) {
      VLOG(3) << "CollectiveHintsAnnotator: pass_filter requires metadata"
                 " but '" << instr->name() << "' has no op_name";
      return false;
    }
    if (!MatchesPassFilter(op_name, m.pass_filter())) {
      VLOG(3) << "CollectiveHintsAnnotator: pass_filter '" << m.pass_filter()
              << "' does not match op_name '" << op_name
              << "' for '" << instr->name() << "'";
      return false;
    }
  }

  // Operand/result shape filter (substring match on any result shape component).
  if (!m.operand_shape().empty()) {
    bool shape_match = false;
    // Check each shape component of the instruction's result shape.
    const Shape& result_shape = instr->shape();
    auto check_shape = [&](const Shape& s) {
      if (absl::StrContains(ShapeUtil::HumanString(s), m.operand_shape())) {
        shape_match = true;
      }
    };
    if (result_shape.IsTuple()) {
      for (const Shape& s : result_shape.tuple_shapes()) {
        check_shape(s);
      }
    } else {
      check_shape(result_shape);
    }
    if (!shape_match) {
      VLOG(3) << "CollectiveHintsAnnotator: operand_shape '"
              << m.operand_shape() << "' not found in shape of '"
              << instr->name() << "'";
      return false;
    }
  }

  return true;
}

// Applies the hints from `hint` onto `instr` by merging into its
// frontend_attributes. Existing attributes are not overwritten.
absl::Status ApplyHints(const CollectiveHint& hint, HloInstruction* instr) {
  if (!hint.latency_metadata().empty()) {
    instr->add_frontend_attribute("latency_metadata", hint.latency_metadata());
  }
  if (hint.force_earliest()) {
    auto gpu_config = instr->backend_config<GpuBackendConfig>();
    if (gpu_config.ok()) {
      gpu_config->set_force_earliest_schedule(true);
      TF_RETURN_IF_ERROR(instr->set_backend_config(*gpu_config));
    }
  }
  if (!hint.scheduling_group_id().empty()) {
    instr->add_frontend_attribute("_scheduling_group_id",
                                  hint.scheduling_group_id());
  }
  return absl::OkStatus();
}

}  // namespace

CollectiveHintsAnnotatorPass::CollectiveHintsAnnotatorPass(
    CollectiveHintsConfig config)
    : config_(std::move(config)) {}

/*static*/ absl::StatusOr<CollectiveHintsAnnotatorPass>
CollectiveHintsAnnotatorPass::Create(absl::string_view path) {
  tsl::Env* env = tsl::Env::Default();
  std::string path_str(path);
  if (!env->FileExists(path_str).ok()) {
    return absl::NotFoundError(
        absl::StrCat("CollectiveHintsConfig file not found: ", path));
  }
  CollectiveHintsConfig config;
  absl::Status s = tsl::ReadTextProto(env, path_str, &config);
  if (!s.ok()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Failed to parse CollectiveHintsConfig proto text from '", path,
        "': ", s.message()));
  }
  LOG(INFO) << "CollectiveHintsAnnotator: loaded " << config.hints_size()
            << " rule(s) from " << path;
  return CollectiveHintsAnnotatorPass(std::move(config));
}

absl::StatusOr<bool> CollectiveHintsAnnotatorPass::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (config_.hints_size() == 0) return false;

  bool changed = false;
  // Opcodes that can legally carry a `_scheduling_group_id` per the fork's
  // `keep_sync_annotation` predicate in `gpu_hlo_schedule.cc`. Used to gate
  // compute-op matches; async collective starts are always eligible.
  auto IsComputeGroupTarget = [](const HloInstruction* i) {
    switch (i->opcode()) {
      case HloOpcode::kCustomCall:
      case HloOpcode::kDot:
      case HloOpcode::kConvolution:
      case HloOpcode::kFusion:
        return true;
      default:
        return false;
    }
  };

  // Per-rule counter of instructions that have satisfied the rule's non-
  // ordinal criteria so far. Used to implement `match_ordinal`.
  std::vector<int32_t> rule_seen(config_.hints_size(), 0);

  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instr : computation->instructions()) {
      const bool is_async_start = IsAsyncCollectiveStart(instr);
      const bool is_compute_target = IsComputeGroupTarget(instr);
      if (!is_async_start && !is_compute_target) continue;

      VLOG(3) << "CollectiveHintsAnnotator: checking '"
              << instr->name() << "' (opcode="
              << HloOpcodeString(instr->opcode()) << ", op_name='"
              << instr->metadata().op_name() << "')";

      // Collect all matching rules, then sort by priority (ascending) so that
      // higher-priority rules are applied last and win. Rules with equal
      // priority retain their original list order (stable_sort).
      //
      // A rule with `match_ordinal` set is applied only when its per-rule
      // seen-counter equals the ordinal (after bumping).  This lets users
      // distinguish otherwise-identical instructions by their position in
      // iteration order — stable across recompiles as long as the traced
      // program order is stable.
      std::vector<const CollectiveHint*> matching;
      for (int r = 0; r < config_.hints_size(); ++r) {
        const CollectiveHint& hint = config_.hints(r);
        if (!Matches(hint, instr)) continue;
        const int32_t seen_before = rule_seen[r]++;
        if (hint.has_match_ordinal() &&
            seen_before != hint.match_ordinal()) {
          VLOG(3) << "CollectiveHintsAnnotator: rule " << r
                  << " skipped '" << instr->name()
                  << "' — ordinal mismatch (seen=" << seen_before
                  << ", want=" << hint.match_ordinal() << ")";
          continue;
        }
        matching.push_back(&hint);
      }
      if (matching.empty()) {
        VLOG(2) << "CollectiveHintsAnnotator: no rules matched '"
                << instr->name() << "' (op_name='"
                << instr->metadata().op_name() << "')";
        continue;
      }
      std::stable_sort(matching.begin(), matching.end(),
                       [](const CollectiveHint* a, const CollectiveHint* b) {
                         return a->priority() < b->priority();
                       });

      struct MergedHints {
        std::string latency_metadata;
        bool force_earliest = false;
        std::string scheduling_group_id;
        int32_t stream_id = 0;
      } merged;

      for (const CollectiveHint* hint : matching) {
        if (!hint->latency_metadata().empty()) {
          merged.latency_metadata = hint->latency_metadata();
        }
        if (hint->force_earliest()) {
          merged.force_earliest = true;
        }
        if (!hint->scheduling_group_id().empty()) {
          merged.scheduling_group_id = hint->scheduling_group_id();
        }
        if (hint->stream_id() != 0) {
          merged.stream_id = hint->stream_id();
        }
      }

      VLOG(2) << "CollectiveHintsAnnotator: matched '" << instr->name()
              << "' with " << matching.size() << " rule(s)"
              << " (op_name='" << instr->metadata().op_name() << "')";

      bool any_attr = false;
      // `latency_metadata` and `force_earliest_schedule` are meaningful only
      // on async collective starts — the GpuLatencyEstimator reads the former
      // and LHS reads ForceDelay on the latter. Skip (with a warning) for
      // compute-op targets so a user-authored rule doesn't silently no-op.
      if (!merged.latency_metadata.empty()) {
        if (is_async_start) {
          instr->add_frontend_attribute("latency_metadata",
                                        merged.latency_metadata);
          VLOG(2) << "  -> latency_metadata=" << merged.latency_metadata;
          any_attr = true;
        } else {
          VLOG(1) << "CollectiveHintsAnnotator: skipping latency_metadata on "
                     "non-collective '" << instr->name() << "' (no effect)";
        }
      }
      if (merged.force_earliest) {
        if (is_async_start) {
          auto gpu_config = instr->backend_config<GpuBackendConfig>();
          if (gpu_config.ok()) {
            gpu_config->set_force_earliest_schedule(true);
            TF_RETURN_IF_ERROR(instr->set_backend_config(*gpu_config));
            VLOG(2) << "  -> force_earliest_schedule=true (backend_config)";
            any_attr = true;
          }
        } else {
          VLOG(1) << "CollectiveHintsAnnotator: skipping force_earliest on "
                     "non-collective '" << instr->name() << "' (no effect)";
        }
      }
      if (!merged.scheduling_group_id.empty()) {
        instr->add_frontend_attribute("_scheduling_group_id",
                                      merged.scheduling_group_id);
        VLOG(2) << "  -> _scheduling_group_id=" << merged.scheduling_group_id;
        any_attr = true;
      }
      if (merged.stream_id != 0) {
        instr->add_frontend_attribute("_xla_stream_annotation",
                                      absl::StrCat(merged.stream_id));
        VLOG(2) << "  -> _xla_stream_annotation=" << merged.stream_id;
        any_attr = true;
      }
      if (any_attr) changed = true;
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
