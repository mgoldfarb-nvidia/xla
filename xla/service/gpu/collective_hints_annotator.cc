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

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include <map>

#include "xla/hlo/analysis/hlo_reachability.h"
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

  // Sequencing bookkeeping: per-computation, per-sequence_id, list of
  // annotated instructions. Consulted after the main loop to add control
  // dependencies between batches (see `sequence_id` in collective_hints.proto).
  std::map<HloComputation*, std::map<int32_t, std::vector<HloInstruction*>>>
      batches_per_comp;

  // window_target bookkeeping: per matched compute instruction, the list of
  // collective names that should "contain" it in their [start, done] window.
  // Consulted after the main loop to add the control deps (see
  // `window_target` in collective_hints.proto).
  std::map<HloInstruction*, std::vector<std::string>> window_target_per_instr;

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
        int32_t sequence_id = 0;   // 0 = not batched
        std::vector<std::string> window_target;  // empty = none
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
        // stream_id (proto field 6) is deprecated and the producer was
        // removed when kGpuAsyncStreamCollectives0/1 was retired. The
        // proto field is kept for back-compat textproto parsing; setting
        // it now is a no-op.
        if (hint->sequence_id() > 0) {
          merged.sequence_id = hint->sequence_id();
        }
        // Higher-priority rule wins for window_target — replace, don't append,
        // matching the merge semantics of the other singleton-valued fields.
        if (!hint->window_target().empty()) {
          merged.window_target.assign(hint->window_target().begin(),
                                      hint->window_target().end());
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
      // window_target: only meaningful on compute ops. The matched async-start
      // case would create a chicken-and-egg dependency (collective C as a
      // window_target for itself or another collective). Reject (log+skip) if
      // misapplied.
      if (!merged.window_target.empty()) {
        if (is_async_start) {
          VLOG(1) << "CollectiveHintsAnnotator: skipping window_target on "
                     "async-collective-start '" << instr->name()
                  << "' — only compute ops can be placed inside a "
                     "collective window";
        } else if (merged.window_target.size() > 1) {
          VLOG(1) << "CollectiveHintsAnnotator: skipping multi-target "
                     "window_target on '" << instr->name() << "' (got "
                  << merged.window_target.size()
                  << " targets, only 1 supported in Phase 1; multi-target "
                     "would require LHS-side state tracking and is not yet "
                     "implemented)";
        } else {
          // Defer the actual control-dep insertion + dataflow validation to
          // the post-loop pass; we need a per-computation reachability map
          // and a name → instruction lookup, which are cheaper to build once.
          window_target_per_instr[instr] = merged.window_target;
          // Visibility attribute (joined with "," for forward compatibility
          // with multi-target).
          instr->add_frontend_attribute(
              "_xla_window_target", absl::StrJoin(merged.window_target, ","));
          VLOG(2) << "  -> _xla_window_target="
                  << absl::StrJoin(merged.window_target, ",");
          any_attr = true;
        }
      }
      if (any_attr) changed = true;

      // Track this instruction's batch membership for post-loop control-dep
      // insertion. Only record when the instruction was actually annotated.
      if (any_attr && merged.sequence_id > 0) {
        batches_per_comp[computation][merged.sequence_id].push_back(instr);
      }
    }
  }

  // ── Sequencing: add HLO control dependencies between batches ────────────
  //
  // For every computation with ≥ 2 sequence_ids, for every ordered pair of
  // sequence_ids (N, M) with N < M, add a control dep from each instruction
  // in batch N (or its paired -done, for async starts) to each instruction
  // in batch M. This forces batch M to schedule strictly after batch N.
  //
  // Before adding an edge, check the computation's reachability to guarantee
  // it would not introduce a cycle with existing dataflow; return an error
  // naming the offending instructions if so.
  // Note: kReduceScatter uses the new-style async wrapper pair
  // (kAsyncStart/kAsyncDone), so there's no kReduceScatterDone opcode.
  auto paired_done = [](HloInstruction* start) -> HloInstruction* {
    for (HloInstruction* u : start->users()) {
      HloOpcode op = u->opcode();
      if (op == HloOpcode::kAsyncDone ||
          op == HloOpcode::kAllGatherDone ||
          op == HloOpcode::kAllReduceDone ||
          op == HloOpcode::kCollectivePermuteDone) {
        return u;
      }
    }
    return nullptr;
  };
  auto is_async_start_h = [](HloInstruction* i) {
    return hlo_query::IsAsyncCollectiveStartOp(i, /*include_send_recv=*/true);
  };
  // True if `i` is an async-collective start or done. Used to gate the
  // compute-anchor→compute-anchor skip below: such deps are redundant
  // (scheduling_group_id already clusters each anchor with its
  // collective, so ordering the collectives orders their anchors
  // implicitly) and have caused post-schedule RET_CHECK failures at
  // hlo_schedule.cc when LHS's group-id placement flipped the anchors'
  // relative order (e.g. X vs X.double_buffer_clone in different
  // sequence_id batches).
  auto is_async_op = [&](HloInstruction* i) {
    if (is_async_start_h(i)) return true;
    HloOpcode op = i->opcode();
    return op == HloOpcode::kAsyncDone ||
           op == HloOpcode::kAllGatherDone ||
           op == HloOpcode::kAllReduceDone ||
           op == HloOpcode::kCollectivePermuteDone;
  };

  for (auto& [computation, batches] : batches_per_comp) {
    if (batches.size() < 2) continue;
    std::unique_ptr<HloReachabilityMap> reach =
        HloReachabilityMap::Build(computation);
    std::vector<int32_t> seq_ids;
    for (auto& [sid, _] : batches) seq_ids.push_back(sid);
    // std::map iterates sorted, so seq_ids is already ascending.
    for (size_t i = 0; i + 1 < seq_ids.size(); ++i) {
      const auto& prev_batch = batches[seq_ids[i]];
      for (size_t j = i + 1; j < seq_ids.size(); ++j) {
        const auto& next_batch = batches[seq_ids[j]];
        for (HloInstruction* prev : prev_batch) {
          // For an async-start, the batch completes when its -done is placed.
          HloInstruction* prev_end =
              is_async_start_h(prev) ? paired_done(prev) : nullptr;
          if (prev_end == nullptr) prev_end = prev;
          for (HloInstruction* next : next_batch) {
            if (prev_end == next) continue;
            // Skip compute-anchor → compute-anchor deps (see is_async_op).
            if (!is_async_op(prev_end) && !is_async_op(next)) continue;
            // Cycle check: if `next` can already reach `prev_end`, the new
            // `prev_end -> next` edge closes a cycle.
            if (reach->IsReachable(next, prev_end)) {
              return absl::FailedPreconditionError(absl::StrCat(
                  "CollectiveHintsAnnotator: sequence_id ordering would "
                  "introduce a cycle in computation '", computation->name(),
                  "'. Adding a control dep from '", prev_end->name(),
                  "' (sequence_id=", seq_ids[i], ") to '", next->name(),
                  "' (sequence_id=", seq_ids[j],
                  ") conflicts with existing dataflow."));
            }
            TF_RETURN_IF_ERROR(prev_end->AddControlDependencyTo(next));
            VLOG(2) << "CollectiveHintsAnnotator: control dep "
                    << prev_end->name() << " -> " << next->name()
                    << " (sequence " << seq_ids[i] << " -> " << seq_ids[j]
                    << ")";
            changed = true;
          }
        }
      }
    }
  }

  // ── window_target: add control deps to pin compute inside collective windows
  //
  // For each (compute_instr, [target_name]) recorded above (Phase 1: exactly
  // one target per compute), find the named target in the same computation,
  // verify it's an async-collective-start, locate its paired -done, validate
  // dataflow legality (compute must not be in target's pred or succ closure;
  // adding the control deps must not introduce a cycle), and add:
  //   target_start  -> compute       (so compute runs AFTER start)
  //   compute       -> target_done   (so compute runs BEFORE done)
  //
  // Any rule that fails validation is logged and skipped (its FE attribute
  // remains for visibility, but no control deps are added — callers can grep
  // for `CollectiveHintsAnnotator window_target: skip` in logs to find them).
  if (!window_target_per_instr.empty()) {
    // One reachability map per computation, cached for repeated lookups.
    absl::flat_hash_map<HloComputation*, std::unique_ptr<HloReachabilityMap>>
        reach_per_comp;
    auto reach_for = [&](HloComputation* c) -> HloReachabilityMap* {
      auto it = reach_per_comp.find(c);
      if (it == reach_per_comp.end()) {
        it = reach_per_comp
                 .emplace(c, HloReachabilityMap::Build(c))
                 .first;
      }
      return it->second.get();
    };

    for (const auto& [compute, targets] : window_target_per_instr) {
      // Phase 1 invariant — already enforced above, but assert defensively.
      if (targets.size() != 1) continue;
      const std::string& target_name = targets.front();
      HloComputation* comp = compute->parent();
      // Linear scan of the comp; the cost is O(comp_instrs) per
      // window_target rule, which is fine for the typical handful of rules
      // a textproto carries. If this ever becomes hot we can pre-build a
      // name → instr map per computation alongside reach_per_comp.
      HloInstruction* target = nullptr;
      for (HloInstruction* i : comp->instructions()) {
        if (i->name() == target_name) {
          target = i;
          break;
        }
      }
      if (target == nullptr) {
        VLOG(1) << "CollectiveHintsAnnotator window_target: skip — target '"
                << target_name << "' not found in computation '"
                << comp->name() << "' (compute='" << compute->name() << "')";
        continue;
      }
      if (!is_async_start_h(target)) {
        VLOG(1) << "CollectiveHintsAnnotator window_target: skip — target '"
                << target_name << "' is not an async-collective-start "
                << "(opcode=" << HloOpcodeString(target->opcode())
                << "; compute='" << compute->name() << "')";
        continue;
      }
      HloInstruction* target_done = paired_done(target);
      if (target_done == nullptr) {
        VLOG(1) << "CollectiveHintsAnnotator window_target: skip — could "
                << "not find paired *-done for target '" << target_name
                << "' (compute='" << compute->name() << "')";
        continue;
      }
      HloReachabilityMap* reach = reach_for(comp);
      // Cycle check 1: compute already reaches target → compute is a
      // dataflow predecessor of target. Adding `target -> compute` would
      // close the cycle. (Equivalent to "target is in compute's succ
      // closure".)
      if (reach->IsReachable(compute, target)) {
        VLOG(1) << "CollectiveHintsAnnotator window_target: skip — compute '"
                << compute->name() << "' is a dataflow PREDECESSOR of "
                << "target '" << target_name << "'; pinning compute into "
                << "target's window would cycle";
        continue;
      }
      // Cycle check 2: target_done already reaches compute → compute is a
      // dataflow successor of the collective's result. It can't run during
      // the collective; adding `compute -> target_done` would cycle.
      if (reach->IsReachable(target_done, compute)) {
        VLOG(1) << "CollectiveHintsAnnotator window_target: skip — compute '"
                << compute->name() << "' is a dataflow SUCCESSOR of "
                << "target_done '" << target_done->name() << "' (compute "
                << "depends on the collective's result, can't run inside its "
                << "window)";
        continue;
      }
      TF_RETURN_IF_ERROR(target->AddControlDependencyTo(compute));
      TF_RETURN_IF_ERROR(compute->AddControlDependencyTo(target_done));
      VLOG(2) << "CollectiveHintsAnnotator window_target: pinned '"
              << compute->name() << "' inside ['" << target->name()
              << "', '" << target_done->name() << "'] (control deps added)";
      // Reachability map is now stale — invalidate the cache for this comp.
      reach_per_comp.erase(comp);
      changed = true;
    }
  }

  // End-of-pass inventory: with VLOG(1) enabled, dump every instruction
  // we annotated so downstream-pass crashes (e.g. absl::flat_hash_map::at
  // failures in legalize_scheduling_annotations or LHS) can be
  // cross-referenced against exactly what this pass wrote.
  if (VLOG_IS_ON(1)) {
    int annotated_count = 0;
    int deps_added_count = 0;
    for (auto& [computation, batches] : batches_per_comp) {
      for (auto& [sid, insts] : batches) {
        for (HloInstruction* instr : insts) {
          ++annotated_count;
          const auto& attrs = instr->frontend_attributes().map();
          auto it = attrs.find("_scheduling_group_id");
          std::string group_id =
              (it != attrs.end()) ? it->second : std::string("<unset>");
          VLOG(1) << "CollectiveHintsAnnotator output: comp='"
                  << computation->name() << "' instr='" << instr->name()
                  << "' opcode=" << HloOpcodeString(instr->opcode())
                  << " sequence_id=" << sid
                  << " group_id='" << group_id << "'"
                  << " n_control_pred="
                  << instr->control_predecessors().size()
                  << " n_control_succ=" << instr->control_successors().size();
          deps_added_count += instr->control_successors().size();
        }
      }
    }
    VLOG(1) << "CollectiveHintsAnnotator output: total annotated="
            << annotated_count
            << " (sum of control_successors across annotated instructions="
            << deps_added_count
            << "; note that count includes deps not added by this pass)";
  }

  return changed;
}

}  // namespace gpu
}  // namespace xla
