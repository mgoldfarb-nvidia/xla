/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
==============================================================================*/

#include "xla/service/gpu/auto_window_target.h"

#include <algorithm>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/hlo_reachability.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/gpu/collective_hints_annotator.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/scheduling_annotations_util.h"
#include "xla/side_effect_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

// Mirrors `keep_sync_annotation` in gpu_hlo_schedule.cc. A compute op is
// eligible to anchor a window only if it would survive
// LegalizeSchedulingAnnotations.
bool IsEligibleAnchor(const HloInstruction& hlo) {
  switch (hlo.opcode()) {
    case HloOpcode::kCustomCall:
    case HloOpcode::kDot:
    case HloOpcode::kConvolution:
      return true;
    case HloOpcode::kFusion:
      return IsGpuFusionKind(hlo, kTritonGemmFusionKind);
    default:
      return false;
  }
}

double LookupPgleCostUs(
    const HloInstruction& instr,
    const absl::flat_hash_map<std::string, double>& cost_by_name) {
  auto it = cost_by_name.find(instr.name());
  if (it != cost_by_name.end()) return it->second;
  if (instr.opcode() == HloOpcode::kAsyncStart ||
      instr.opcode() == HloOpcode::kAsyncDone) {
    absl::string_view wrapped = instr.async_wrapped_instruction()->name();
    auto it2 = cost_by_name.find(wrapped);
    if (it2 != cost_by_name.end()) return it2->second;
  }
  return -1.0;
}

// Returns the `*-done` instruction matching `start`, or nullptr.
// Reduce-scatter has no dedicated kReduceScatterDone opcode — it uses
// the kAsyncDone wrapper.
const HloInstruction* FindMatchingDone(const HloInstruction* start) {
  for (HloInstruction* user : start->users()) {
    HloOpcode op = user->opcode();
    if (op == HloOpcode::kAllGatherDone || op == HloOpcode::kAllReduceDone ||
        op == HloOpcode::kCollectivePermuteDone ||
        op == HloOpcode::kAsyncDone) {
      return user;
    }
  }
  return nullptr;
}

}  // namespace

absl::StatusOr<CollectiveHintsConfig> BuildAutoWindowTargetConfig(
    const HloModule& module,
    const tensorflow::profiler::ProfiledInstructionsProto& profile,
    float threshold_us, float min_compute_us, int max_per_collective) {
  CollectiveHintsConfig config;
  if (threshold_us <= 0.0f) {
    return config;  // Disabled.
  }

  // Build PGLE name -> cost map.
  absl::flat_hash_map<std::string, double> cost_by_name;
  cost_by_name.reserve(profile.costs_size());
  for (const auto& c : profile.costs()) {
    cost_by_name[c.name()] = c.cost_us();
  }

  // Track anchors already claimed (manual hints or earlier picks).
  // The annotator uses the literal "_xla_window_target" frontend attr;
  // mirror that here. (No exported constant; see
  // collective_hints_annotator.cc.)
  static constexpr absl::string_view kWindowTargetFEAttr =
      "_xla_window_target";
  absl::flat_hash_set<const HloInstruction*> claimed;
  for (const HloComputation* comp : module.MakeNonfusionComputations()) {
    for (const HloInstruction* instr : comp->instructions()) {
      if (instr->frontend_attributes().map().contains(
              kWindowTargetFEAttr)) {
        claimed.insert(instr);
      }
    }
  }

  // Walk each computation independently.
  for (HloComputation* comp : module.MakeNonfusionComputations()) {
    // Build reachability map once per computation.
    auto reach = HloReachabilityMap::Build(comp);

    // Cache HLO order index: prefers schedule order when the input module
    // has a schedule (typical in the simulator path on
    // `*after_optimizations.hlo.pb`); falls back to instruction-list
    // order otherwise. Used for closeness-by-distance ranking.
    absl::flat_hash_map<const HloInstruction*, int> file_idx;
    int idx = 0;
    if (module.has_schedule() && module.schedule().is_computation_scheduled(comp)) {
      for (const HloInstruction* instr :
           module.schedule().sequence(comp).instructions()) {
        file_idx[instr] = idx++;
      }
    } else {
      for (const HloInstruction* instr : comp->instructions()) {
        file_idx[instr] = idx++;
      }
    }

    // Collectives in this computation, ordered by descending PGLE cost
    // (so the heaviest gets first pick from the anchor pool).
    std::vector<HloInstruction*> collectives;
    for (HloInstruction* instr : comp->instructions()) {
      if (!hlo_query::IsAsyncCollectiveStartOp(instr,
                                                /*include_send_recv=*/false)) {
        continue;
      }
      // Skip if user already paired this collective via gid.
      if (HasSchedulingAnnotation(instr)) continue;
      double cost = LookupPgleCostUs(*instr, cost_by_name);
      if (cost < threshold_us) continue;
      collectives.push_back(instr);
    }
    std::sort(collectives.begin(), collectives.end(),
              [&](HloInstruction* a, HloInstruction* b) {
                double ca = LookupPgleCostUs(*a, cost_by_name);
                double cb = LookupPgleCostUs(*b, cost_by_name);
                if (ca != cb) return ca > cb;
                return a->name() < b->name();
              });

    VLOG(2) << "Computation '" << comp->name() << "' has "
            << collectives.size() << " collective(s) above threshold; "
            << file_idx.size() << " total instructions";
    for (HloInstruction* coll : collectives) {
      const HloInstruction* done = FindMatchingDone(coll);
      VLOG(2) << "  collective '" << coll->name() << "' at pos="
              << file_idx[coll] << "/" << file_idx.size();
      if (done == nullptr) {
        VLOG(2) << "AutoWindowTarget: no matching *-done for "
                << coll->name() << "; skipping";
        continue;
      }
      double coll_cost = LookupPgleCostUs(*coll, cost_by_name);
      int coll_pos = file_idx[coll];

      // Find candidate anchors in this computation:
      //   - eligible opcode
      //   - after `coll` in HLO file order
      //   - operand-tree independent of `coll` (both directions)
      //   - PGLE cost >= min_compute_us
      //   - not already claimed
      struct Cand {
        HloInstruction* instr;
        double cost_us;
        int distance;  // file_idx - coll_pos
      };
      std::vector<Cand> candidates;
      int reject_claimed = 0, reject_opcode = 0,
          reject_descendant = 0, reject_ancestor = 0, reject_done = 0,
          reject_cost = 0, eligible_total = 0;
      for (HloInstruction* instr : comp->instructions()) {
        if (claimed.contains(instr)) { ++reject_claimed; continue; }
        if (!IsEligibleAnchor(*instr)) { ++reject_opcode; continue; }
        ++eligible_total;
        // Operand-tree independence (cycle-correctness): the anchor must
        // not depend on the collective AND the collective must not depend
        // on the anchor. Note that the manual textproto-authoring flow
        // also filters by "after the collective in schedule order" — that
        // is a heuristic for "doesn't extend critical paths" — but C3
        // runs before LHS so there is no schedule yet to consult. We
        // accept candidates regardless of natural order; the annotator's
        // control deps will force them into the window. Cycle detection
        // is delegated to the annotator's existing logic.
        if (reach->IsReachable(coll, instr)) { ++reject_descendant; continue; }
        if (reach->IsReachable(instr, coll)) { ++reject_ancestor; continue; }
        if (instr == done) { ++reject_done; continue; }
        double cost = LookupPgleCostUs(*instr, cost_by_name);
        if (cost < min_compute_us) { ++reject_cost; continue; }
        int pos = file_idx[instr];
        candidates.push_back({instr, cost, std::abs(pos - coll_pos)});
      }
      VLOG(2) << "  '" << coll->name() << "' candidates=" << candidates.size()
              << " (eligible_opcode=" << eligible_total
              << " reject: opcode=" << reject_opcode
              << " descendant=" << reject_descendant
              << " ancestor=" << reject_ancestor
              << " cost=" << reject_cost
              << " claimed=" << reject_claimed << ")";

      // Sort: highest PGLE cost first (matches what manual textproto
      // authors pick — heavy compute fills more of the collective's
      // window per anchor). Ties broken by HLO name (deterministic
      // across recompiles).
      std::sort(candidates.begin(), candidates.end(),
                [](const Cand& a, const Cand& b) {
                  if (a.cost_us != b.cost_us) return a.cost_us > b.cost_us;
                  return a.instr->name() < b.instr->name();
                });

      // Greedy fill until cumulative >= collective cost OR cap.
      double cumulative = 0.0;
      int picked = 0;
      for (const Cand& c : candidates) {
        if (picked >= max_per_collective) break;
        if (cumulative >= coll_cost) break;
        // Emit a window_target rule into the auto-config.
        CollectiveHint* h_anchor = config.add_hints();
        h_anchor->set_name(c.instr->name());
        h_anchor->add_window_target(coll->name());

        claimed.insert(c.instr);
        cumulative += c.cost_us;
        ++picked;

        VLOG(2) << "AutoWindowTarget: pinned '" << c.instr->name()
                << "' (" << c.cost_us << " us) into '" << coll->name()
                << "'s window (cumulative " << cumulative << "/"
                << coll_cost << " us)";
      }
      if (picked == 0) {
        VLOG(1) << "AutoWindowTarget: no eligible anchors found for '"
                << coll->name() << "' (cost " << coll_cost << " us)";
      }
    }
  }

  return config;
}

absl::StatusOr<bool> AutoWindowTargetPass::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (threshold_us_ <= 0.0f) {
    return false;
  }

  TF_ASSIGN_OR_RETURN(
      CollectiveHintsConfig auto_config,
      BuildAutoWindowTargetConfig(*module, profile_, threshold_us_,
                                   min_compute_us_, max_per_collective_));

  if (auto_config.hints_size() == 0) {
    VLOG(1) << "AutoWindowTargetPass: no auto-pin opportunities found "
               "(threshold=" << threshold_us_ << " us). No-op.";
    return false;
  }

  LOG(INFO) << "AutoWindowTargetPass: emitted "
            << auto_config.hints_size()
            << " window_target rule(s); delegating to "
               "CollectiveHintsAnnotatorPass to insert control deps.";

  // Run the existing annotator pass with our in-memory config. The
  // annotator's window_target machinery handles cycle detection,
  // frontend-attribute setting, and control-dep insertion.
  CollectiveHintsAnnotatorPass annotator(std::move(auto_config));
  return annotator.Run(module, execution_threads);
}

}  // namespace xla::gpu
