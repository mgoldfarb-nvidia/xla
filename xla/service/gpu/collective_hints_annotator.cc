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

#include "xla/service/gpu/collective_hints_annotator.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/hlo_reachability.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/shape_util.h"
#include "xla/side_effect_util.h"
#include "xla/tsl/platform/env.h"
#include "tsl/platform/protobuf.h"

namespace xla::gpu {
namespace {

struct ResolvedRule {
  const CollectiveHint* hint;
  std::vector<HloInstruction*> instructions;
};

struct ResolvedWindow {
  HloInstruction* start;
  HloInstruction* done;
  int64_t scheduling_group_id;
};

constexpr absl::string_view kSchedulingGroupOrderAttr =
    "scheduling_group_order";

bool IsLowerHexFingerprint(absl::string_view value) {
  if (value.size() != 32) {
    return false;
  }
  return absl::c_all_of(value, [](unsigned char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
  });
}

bool IsRuleId(absl::string_view value) {
  return !value.empty() && absl::c_all_of(value, [](unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.';
  });
}

absl::Status ValidateSelector(const CollectiveHint& hint) {
  if (!hint.has_match()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "collective hint '", hint.rule_id(), "' requires a selector"));
  }
  const CollectiveHintMatch& match = hint.match();
  const bool by_instruction_name = !match.instruction_name().empty();
  const bool has_any_signature_field = !match.op_name().empty() ||
                                       !match.opcode().empty() ||
                                       !match.shape().empty();
  const bool has_complete_signature = !match.op_name().empty() &&
                                      !match.opcode().empty() &&
                                      !match.shape().empty();
  if (by_instruction_name == has_any_signature_field) {
    return absl::InvalidArgumentError(
        absl::StrCat("collective hint '", hint.rule_id(),
                     "' must select exactly one of instruction_name or "
                     "(op_name, opcode, shape)"));
  }
  if (has_any_signature_field && !has_complete_signature) {
    return absl::InvalidArgumentError(absl::StrCat(
        "collective hint '", hint.rule_id(),
        "' requires the complete (op_name, opcode, shape) signature"));
  }
  if (by_instruction_name && !match.pass_filter().empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("collective hint '", hint.rule_id(),
                     "' cannot combine instruction_name with pass_filter"));
  }
  if (!match.pass_filter().empty() && match.pass_filter() != "forward" &&
      match.pass_filter() != "backward" && match.pass_filter() != "remat") {
    return absl::InvalidArgumentError(absl::StrCat(
        "collective hint '", hint.rule_id(), "' has unsupported pass_filter '",
        match.pass_filter(), "'"));
  }
  if (!hint.has_expected_match_count() || hint.expected_match_count() == 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("collective hint '", hint.rule_id(),
                     "' requires positive expected_match_count"));
  }
  if (by_instruction_name && hint.expected_match_count() != 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "collective hint '", hint.rule_id(),
        "' selecting instruction_name must expect exactly one match"));
  }
  return absl::OkStatus();
}

absl::Status ValidateConfigSchema(const CollectiveHintsConfig& config) {
  if (!IsLowerHexFingerprint(config.module_fingerprint())) {
    return absl::InvalidArgumentError(
        "collective hints require a 32-character lowercase hexadecimal "
        "module_fingerprint");
  }
  if (config.hints().empty()) {
    return absl::InvalidArgumentError(
        "collective hints config requires at least one hint");
  }

  absl::flat_hash_set<std::string> rule_ids;
  for (const CollectiveHint& hint : config.hints()) {
    if (!IsRuleId(hint.rule_id())) {
      return absl::InvalidArgumentError(
          "collective hint rule_id must contain only [A-Za-z0-9_.-]");
    }
    if (!rule_ids.insert(hint.rule_id()).second) {
      return absl::InvalidArgumentError(absl::StrCat(
          "duplicate collective hint rule_id '", hint.rule_id(), "'"));
    }
    ABSL_RETURN_IF_ERROR(ValidateSelector(hint));
    if (hint.has_scheduling_group_id() && hint.scheduling_group_id() < 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("collective hint '", hint.rule_id(),
                       "' scheduling_group_id must be nonnegative"));
    }
    if (!hint.window_target().empty() && !hint.has_scheduling_group_id()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "collective hint '", hint.rule_id(),
          "' window_target requires scheduling_group_id"));
    }
    if (!hint.force_earliest() && !hint.has_scheduling_group_id() &&
        hint.window_target().empty()) {
      return absl::InvalidArgumentError(
          absl::StrCat("collective hint '", hint.rule_id(),
                       "' requires at least one supported action"));
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateConfig(const CollectiveHintsConfig& config,
                            absl::string_view module_fingerprint) {
  ABSL_RETURN_IF_ERROR(ValidateConfigSchema(config));
  if (config.module_fingerprint() != module_fingerprint) {
    return absl::FailedPreconditionError(absl::StrCat(
        "collective hints fingerprint ", config.module_fingerprint(),
        " does not match compiled module fingerprint ", module_fingerprint));
  }
  return absl::OkStatus();
}

bool MatchesPassFilter(absl::string_view op_name,
                       absl::string_view pass_filter) {
  if (pass_filter.empty()) {
    return true;
  }
  if (pass_filter == "forward") {
    return absl::StrContains(op_name, "/jvp(") &&
           !absl::StrContains(op_name, "transpose(");
  }
  if (pass_filter == "backward") {
    return absl::StrContains(op_name, "transpose(");
  }
  return absl::StrContains(op_name, "rematted_computation");
}

bool Matches(const CollectiveHintMatch& match,
             const HloInstruction& instruction) {
  if (!match.instruction_name().empty()) {
    return match.instruction_name() == instruction.name();
  }
  const absl::string_view op_name = instruction.metadata().op_name();
  return match.op_name() == op_name &&
         match.opcode() == HloOpcodeString(instruction.opcode()) &&
         match.shape() == ShapeUtil::HumanString(instruction.shape()) &&
         MatchesPassFilter(op_name, match.pass_filter());
}

std::string QualifiedName(const HloInstruction& instruction) {
  return absl::StrCat(instruction.parent()->name(), "/", instruction.name());
}

bool IsSupportedComputeTarget(const HloInstruction& instruction) {
  if (IsCublasGemm(instruction)) {
    return true;
  }
  return instruction.opcode() == HloOpcode::kFusion &&
         IsGpuFusionKind(instruction, kTritonGemmFusionKind);
}

bool IsSchedulingGroupTarget(const HloInstruction& instruction) {
  return hlo_query::IsAsyncCollectiveStartOp(&instruction,
                                             /*include_send_recv=*/true) ||
         IsSupportedComputeTarget(instruction);
}

absl::StatusOr<HloInstruction*> FindAsyncDone(HloInstruction* start) {
  std::vector<HloInstruction*> done;
  for (HloInstruction* user : start->users()) {
    if (hlo_query::IsAsyncCollectiveDoneOp(user,
                                           /*include_send_recv=*/true) &&
        user->operand_count() > 0 && user->operand(0) == start) {
      done.push_back(user);
    }
  }
  if (done.size() != 1) {
    return absl::FailedPreconditionError(absl::StrCat(
        "window target '", QualifiedName(*start), "' has ", done.size(),
        " matching async done instructions; expected exactly one"));
  }
  return done.front();
}

HloInstruction* FindInstruction(HloComputation* computation,
                                absl::string_view name) {
  for (HloInstruction* instruction : computation->instructions()) {
    if (instruction->name() == name) {
      return instruction;
    }
  }
  return nullptr;
}

absl::Status ValidateWindowIndependence(const HloInstruction& start,
                                        const HloInstruction& compute) {
  if (start.parent() != compute.parent()) {
    return absl::InvalidArgumentError(
        "window_target and compute must be in one computation");
  }
  auto reachability = HloReachabilityMap::Build(start.parent());
  if (reachability->IsReachable(&start, &compute) ||
      reachability->IsReachable(&compute, &start)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "window_target '", QualifiedName(start), "' and compute '",
        QualifiedName(compute), "' must be data/control independent"));
  }
  return absl::OkStatus();
}

absl::Status CheckFrontendAttribute(const HloInstruction& instruction,
                                    absl::string_view key,
                                    absl::string_view expected) {
  const auto& attributes = instruction.frontend_attributes().map();
  auto it = attributes.find(std::string(key));
  if (it == attributes.end() || it->second == expected) {
    return absl::OkStatus();
  }
  return absl::FailedPreconditionError(absl::StrCat(
      "instruction '", QualifiedName(instruction), "' already has ", key, "='",
      it->second, "', refusing to overwrite with '", expected, "'"));
}

}  // namespace

absl::StatusOr<CollectiveHintsConfig> LoadCollectiveHintsConfig(
    absl::string_view path) {
  if (path.empty()) {
    return absl::InvalidArgumentError(
        "collective hints path must not be empty");
  }
  std::string contents;
  absl::Status read_status =
      tsl::ReadFileToString(tsl::Env::Default(), std::string(path), &contents);
  if (!read_status.ok()) {
    return absl::Status(read_status.code(),
                        absl::StrCat("failed to read collective hints file '",
                                     path, "': ", read_status.message()));
  }

  CollectiveHintsConfig config;
  tsl::protobuf::TextFormat::Parser parser;
  parser.AllowUnknownField(false);
  if (!parser.ParseFromString(contents, &config)) {
    return absl::InvalidArgumentError(
        absl::StrCat("failed to parse collective hints textproto '", path,
                     "'; unknown and deprecated fields are rejected"));
  }
  ABSL_RETURN_IF_ERROR(ValidateConfigSchema(config));
  return config;
}

CollectiveHintsAnnotatorPass::CollectiveHintsAnnotatorPass(
    CollectiveHintsConfig config, std::string module_fingerprint)
    : config_(std::move(config)),
      module_fingerprint_(std::move(module_fingerprint)) {}

absl::StatusOr<bool> CollectiveHintsAnnotatorPass::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  ABSL_RETURN_IF_ERROR(ValidateConfig(config_, module_fingerprint_));
  const auto& module_attributes = module->frontend_attributes().map();
  auto fingerprint = module_attributes.find(
      std::string(kCollectiveHintsFingerprintAttr));
  if (fingerprint == module_attributes.end() ||
      fingerprint->second != module_fingerprint_) {
    return absl::FailedPreconditionError(absl::StrCat(
        "module frontend attribute ", kCollectiveHintsFingerprintAttr,
        " must be '", module_fingerprint_,
        "' before applying collective hints"));
  }
  if (module_attributes.contains(std::string(kCollectiveHintsReceiptAttr))) {
    return absl::FailedPreconditionError(
        "collective hints were already applied to this module");
  }

  std::vector<HloInstruction*> candidates;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instruction : computation->instructions()) {
      candidates.push_back(instruction);
    }
  }
  absl::c_sort(candidates,
               [](const HloInstruction* lhs, const HloInstruction* rhs) {
                 return QualifiedName(*lhs) < QualifiedName(*rhs);
               });

  std::vector<const CollectiveHint*> hints;
  hints.reserve(config_.hints_size());
  for (const CollectiveHint& hint : config_.hints()) {
    hints.push_back(&hint);
  }
  absl::c_sort(hints, [](const CollectiveHint* lhs, const CollectiveHint* rhs) {
    return lhs->rule_id() < rhs->rule_id();
  });

  std::vector<ResolvedRule> resolved;
  resolved.reserve(hints.size());
  for (const CollectiveHint* hint : hints) {
    std::vector<HloInstruction*> matches;
    for (HloInstruction* instruction : candidates) {
      if (Matches(hint->match(), *instruction)) {
        matches.push_back(instruction);
      }
    }
    if (matches.size() != hint->expected_match_count()) {
      std::vector<std::string> names;
      names.reserve(matches.size());
      for (const HloInstruction* instruction : matches) {
        names.push_back(QualifiedName(*instruction));
      }
      return absl::FailedPreconditionError(
          absl::StrCat("collective hint '", hint->rule_id(), "' expected ",
                       hint->expected_match_count(), " match(es), found ",
                       matches.size(), ": [", absl::StrJoin(names, ","), "]"));
    }
    resolved.push_back({hint, std::move(matches)});
  }

  absl::flat_hash_map<HloInstruction*, int64_t> scheduling_groups;
  absl::flat_hash_map<HloInstruction*, ResolvedWindow> window_targets;
  absl::flat_hash_map<HloInstruction*, std::vector<std::string>> rule_ids;
  for (const ResolvedRule& rule : resolved) {
    for (HloInstruction* instruction : rule.instructions) {
      rule_ids[instruction].push_back(rule.hint->rule_id());
      if (rule.hint->force_earliest()) {
        if (!hlo_query::IsAsyncCollectiveStartOp(instruction,
                                                 /*include_send_recv=*/true)) {
          return absl::InvalidArgumentError(absl::StrCat(
              "collective hint '", rule.hint->rule_id(),
              "' force_earliest requires async collective starts, matched '",
              QualifiedName(*instruction), "'"));
        }
        absl::StatusOr<GpuBackendConfig> gpu_config =
            instruction->backend_config<GpuBackendConfig>();
        if (!gpu_config.ok()) {
          return absl::Status(
              gpu_config.status().code(),
              absl::StrCat("collective hint '", rule.hint->rule_id(),
                           "' cannot parse GPU backend config for '",
                           QualifiedName(*instruction),
                           "': ", gpu_config.status().message()));
        }
      }
      if (rule.hint->has_scheduling_group_id()) {
        if (!IsSchedulingGroupTarget(*instruction)) {
          return absl::InvalidArgumentError(absl::StrCat(
              "collective hint '", rule.hint->rule_id(),
              "' scheduling_group_id matched unsupported instruction '",
              QualifiedName(*instruction), "'"));
        }
        const int64_t group_id = rule.hint->scheduling_group_id();
        auto [it, inserted] = scheduling_groups.emplace(instruction, group_id);
        if (!inserted && it->second != group_id) {
          return absl::FailedPreconditionError(
              absl::StrCat("conflicting scheduling_group_id values for '",
                           QualifiedName(*instruction), "'"));
        }
        ABSL_RETURN_IF_ERROR(CheckFrontendAttribute(
            *instruction, kXlaSchedulingGroupIdAttr, absl::StrCat(group_id)));
      }
      if (!rule.hint->window_target().empty()) {
        if (!IsSupportedComputeTarget(*instruction)) {
          return absl::InvalidArgumentError(absl::StrCat(
              "collective hint '", rule.hint->rule_id(),
              "' window_target requires a current-main supported GEMM, "
              "matched '",
              QualifiedName(*instruction), "'"));
        }
        HloInstruction* start =
            FindInstruction(instruction->parent(), rule.hint->window_target());
        if (start == nullptr || !hlo_query::IsAsyncCollectiveStartOp(
                                    start, /*include_send_recv=*/true)) {
          return absl::FailedPreconditionError(absl::StrCat(
              "collective hint '", rule.hint->rule_id(), "' window_target '",
              rule.hint->window_target(),
              "' is not an async collective start in computation '",
              instruction->parent()->name(), "'"));
        }
        ABSL_ASSIGN_OR_RETURN(HloInstruction* done, FindAsyncDone(start));
        const ResolvedWindow window{
            start, done, rule.hint->scheduling_group_id()};
        if (!window_targets.emplace(instruction, window).second) {
          return absl::FailedPreconditionError(
              absl::StrCat("multiple window_target actions matched '",
                           QualifiedName(*instruction), "'"));
        }
        ABSL_RETURN_IF_ERROR(
            ValidateWindowIndependence(*start, *instruction));
        const std::string group =
            absl::StrCat(rule.hint->scheduling_group_id());
        ABSL_RETURN_IF_ERROR(CheckFrontendAttribute(
            *start, kXlaSchedulingGroupIdAttr, group));
        ABSL_RETURN_IF_ERROR(CheckFrontendAttribute(
            *done, kXlaSchedulingGroupIdAttr, group));
        ABSL_RETURN_IF_ERROR(CheckFrontendAttribute(
            *instruction, kSchedulingGroupOrderAttr, "1"));
        ABSL_RETURN_IF_ERROR(
            CheckFrontendAttribute(*start, kSchedulingGroupOrderAttr, "0"));
        ABSL_RETURN_IF_ERROR(
            CheckFrontendAttribute(*done, kSchedulingGroupOrderAttr, "2"));
        ABSL_RETURN_IF_ERROR(CheckFrontendAttribute(
            *instruction, kCollectiveHintWindowTargetAttr, start->name()));
      }
    }
  }
  for (const auto& [instruction, ids] : rule_ids) {
    std::vector<std::string> sorted_ids = ids;
    absl::c_sort(sorted_ids);
    ABSL_RETURN_IF_ERROR(
        CheckFrontendAttribute(*instruction, kCollectiveHintRuleIdsAttr,
                               absl::StrJoin(sorted_ids, ",")));
  }
  for (HloInstruction* instruction : candidates) {
    if (auto group = scheduling_groups.find(instruction);
        group != scheduling_groups.end()) {
      instruction->add_frontend_attribute(
          std::string(kXlaSchedulingGroupIdAttr), absl::StrCat(group->second));
    }
    if (auto window = window_targets.find(instruction);
        window != window_targets.end()) {
      instruction->add_frontend_attribute(
          std::string(kCollectiveHintWindowTargetAttr),
          std::string(window->second.start->name()));
      for (HloInstruction* window_instruction :
           {instruction, window->second.start, window->second.done}) {
        window_instruction->add_frontend_attribute(
            std::string(kXlaSchedulingGroupIdAttr),
            absl::StrCat(window->second.scheduling_group_id));
      }
      instruction->add_frontend_attribute(std::string(kSchedulingGroupOrderAttr),
                                          "1");
      window->second.start->add_frontend_attribute(
          std::string(kSchedulingGroupOrderAttr), "0");
      window->second.done->add_frontend_attribute(
          std::string(kSchedulingGroupOrderAttr), "2");
    }
  }
  for (const ResolvedRule& rule : resolved) {
    if (!rule.hint->force_earliest()) {
      continue;
    }
    for (HloInstruction* instruction : rule.instructions) {
      ABSL_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                            instruction->backend_config<GpuBackendConfig>());
      gpu_config.set_force_earliest_schedule(true);
      ABSL_RETURN_IF_ERROR(instruction->set_backend_config(gpu_config));
    }
  }

  for (HloInstruction* instruction : candidates) {
    auto ids = rule_ids.find(instruction);
    if (ids == rule_ids.end()) {
      continue;
    }
    absl::c_sort(ids->second);
    instruction->add_frontend_attribute(std::string(kCollectiveHintRuleIdsAttr),
                                        absl::StrJoin(ids->second, ","));
  }

  std::vector<std::string> receipt_entries;
  receipt_entries.reserve(resolved.size());
  for (const ResolvedRule& rule : resolved) {
    std::vector<std::string> names;
    names.reserve(rule.instructions.size());
    for (const HloInstruction* instruction : rule.instructions) {
      names.push_back(QualifiedName(*instruction));
    }
    receipt_entries.push_back(absl::StrCat(rule.hint->rule_id(), "=[",
                                           absl::StrJoin(names, ","), "]"));
    LOG(INFO) << "Applied XLA collective hint rule '" << rule.hint->rule_id()
              << "' to exact matches [" << absl::StrJoin(names, ",")
              << "] for module fingerprint " << module_fingerprint_;
  }
  const std::string receipt =
      absl::StrCat("fingerprint=", module_fingerprint_, ";",
                   absl::StrJoin(receipt_entries, ";"));
  module->add_frontend_attribute(std::string(kCollectiveHintsReceiptAttr),
                                 receipt);
  return true;
}

}  // namespace xla::gpu
