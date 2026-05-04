/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
==============================================================================*/

#include "xla/service/gpu/pgle_force_async.h"

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/scheduling_annotations_util.h"
#include "xla/side_effect_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

// Returns the PGLE-recorded cost for `instr` if present. Tries the
// instruction's own name first, then (if it's an async-start wrapper) the
// wrapped instruction's name — matching how PGLE attributes async work.
double LookupPgleCostUs(
    const HloInstruction& instr,
    const absl::flat_hash_map<std::string, double>& cost_by_name) {
  auto it = cost_by_name.find(instr.name());
  if (it != cost_by_name.end()) {
    return it->second;
  }
  if (instr.opcode() == HloOpcode::kAsyncStart ||
      instr.opcode() == HloOpcode::kAsyncDone) {
    absl::string_view wrapped_name =
        instr.async_wrapped_instruction()->name();
    auto it2 = cost_by_name.find(wrapped_name);
    if (it2 != cost_by_name.end()) {
      return it2->second;
    }
  }
  return -1.0;
}

}  // namespace

absl::StatusOr<int> ApplyPgleForceAsync(
    HloModule* module,
    const tensorflow::profiler::ProfiledInstructionsProto& profile,
    float threshold_us) {
  if (threshold_us <= 0.0f) {
    // Pass disabled (default). No-op.
    return 0;
  }

  // Build name → cost_us map from PGLE.
  absl::flat_hash_map<std::string, double> cost_by_name;
  cost_by_name.reserve(profile.costs_size());
  for (const auto& c : profile.costs()) {
    cost_by_name[c.name()] = c.cost_us();
  }

  int n_flipped = 0;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    for (HloInstruction* instr : computation->instructions()) {
      if (!hlo_query::IsAsyncCollectiveStartOp(instr,
                                                /*include_send_recv=*/false)) {
        continue;
      }

      double cost_us = LookupPgleCostUs(*instr, cost_by_name);
      if (cost_us < threshold_us) {
        continue;
      }

      // Read existing backend config; flip is_sync if currently true.
      TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                          instr->backend_config<GpuBackendConfig>());
      auto* coll_config = gpu_config.mutable_collective_backend_config();
      if (!coll_config->is_sync()) {
        // Already async; nothing to do.
        continue;
      }
      coll_config->set_is_sync(false);
      TF_RETURN_IF_ERROR(instr->set_backend_config(gpu_config));

      // Strip any residual _scheduling_group_id — a forced-async collective
      // should not carry a hint that might re-bind it to a (possibly-empty)
      // scheduling group. A1 should already have dropped singletons; this is
      // belt-and-suspenders.
      const auto& attrs = instr->frontend_attributes().map();
      if (attrs.contains(kXlaSchedulingGroupIdAttr)) {
        FrontendAttributes new_attrs = instr->frontend_attributes();
        new_attrs.mutable_map()->erase(kXlaSchedulingGroupIdAttr);
        instr->set_frontend_attributes(new_attrs);
      }

      ++n_flipped;
      VLOG(1) << "ApplyPgleForceAsync: flipped is_sync=false on '"
              << instr->name() << "' (PGLE cost " << cost_us
              << " us > threshold " << threshold_us << " us)";
    }
  }

  if (n_flipped > 0) {
    LOG(INFO) << "ApplyPgleForceAsync: flipped " << n_flipped
              << " collective-start(s) from sync to async (threshold "
              << threshold_us << " us)";
  }
  return n_flipped;
}

}  // namespace xla::gpu
