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

#include "xla/service/gpu/collective_hints_annotator.h"

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/collective_hints.pb.h"
#include "xla/shape_util.h"
#include "xla/side_effect_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

using ::testing::HasSubstr;

constexpr absl::string_view kFingerprint = "0123456789abcdef0123456789abcdef";

constexpr absl::string_view kCollectiveAndGemmHlo = R"hlo(
  HloModule m

  ENTRY main {
    p0 = f32[4] parameter(0)
    lhs = f32[4,4] parameter(1)
    rhs = f32[4,4] parameter(2)
    ag-start = (f32[4], f32[8]) all-gather-start(p0), dimensions={0},
        replica_groups={}
    gemm = f32[4,4] custom-call(lhs, rhs),
        custom_call_target="__cublas$lt$matmul",
        metadata={op_name="layer/transpose(gemm)"}
    ag-done = f32[8] all-gather-done(ag-start)
    ROOT result = (f32[8], f32[4,4]) tuple(ag-done, gemm)
  }
)hlo";

constexpr absl::string_view kCyclicWindowHlo = R"hlo(
  HloModule m

  ENTRY main {
    lhs = f32[4,4] parameter(0)
    rhs = f32[4,4] parameter(1)
    gemm = f32[4,4] custom-call(lhs, rhs),
        custom_call_target="__cublas$gemm"
    flattened = f32[16] reshape(gemm)
    ag-start = (f32[16], f32[32]) all-gather-start(flattened), dimensions={0},
        replica_groups={}
    ag-done = f32[32] all-gather-done(ag-start)
    ROOT result = f32[32] copy(ag-done)
  }
)hlo";

std::string FrontendAttribute(const HloInstruction& instruction,
                              absl::string_view key) {
  const auto& attributes = instruction.frontend_attributes().map();
  auto it = attributes.find(std::string(key));
  return it == attributes.end() ? "" : it->second;
}

CollectiveHintsConfig Config() {
  CollectiveHintsConfig config;
  config.set_module_fingerprint(kFingerprint);
  return config;
}

CollectiveHint* AddInstructionHint(CollectiveHintsConfig* config,
                                   absl::string_view rule_id,
                                   absl::string_view instruction_name) {
  CollectiveHint* hint = config->add_hints();
  hint->set_rule_id(rule_id);
  hint->mutable_match()->set_instruction_name(instruction_name);
  hint->set_expected_match_count(1);
  return hint;
}

void TagModule(HloModule* module) {
  module->add_frontend_attribute(std::string(kCollectiveHintsFingerprintAttr),
                                 std::string(kFingerprint));
}

class CollectiveHintsAnnotatorTest : public HloHardwareIndependentTestBase {};

TEST_F(CollectiveHintsAnnotatorTest, AppliesExactHintsAndWritesReceipt) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kCollectiveAndGemmHlo,
                                                       /*replica_count=*/2));
  TagModule(module.get());

  CollectiveHintsConfig config = Config();
  CollectiveHint* gemm_hint = config.add_hints();
  gemm_hint->set_rule_id("b-gemm-window");
  gemm_hint->mutable_match()->set_op_name("layer/transpose(gemm)");
  gemm_hint->mutable_match()->set_opcode("custom-call");
  gemm_hint->mutable_match()->set_shape("f32[4,4]");
  gemm_hint->mutable_match()->set_pass_filter("backward");
  gemm_hint->set_expected_match_count(1);
  gemm_hint->set_scheduling_group_id(7);
  gemm_hint->set_window_target("ag-start");
  CollectiveHint* collective_hint =
      AddInstructionHint(&config, "a-collective", "ag-start");
  collective_hint->set_force_earliest(true);
  collective_hint->set_scheduling_group_id(7);

  HloInstruction* gemm =
      module->entry_computation()->GetInstructionWithName("gemm");
  HloInstruction* start =
      module->entry_computation()->GetInstructionWithName("ag-start");
  HloInstruction* done =
      module->entry_computation()->GetInstructionWithName("ag-done");
  ASSERT_NE(gemm, nullptr);
  ASSERT_NE(start, nullptr);
  ASSERT_NE(done, nullptr);
  EXPECT_EQ(ShapeUtil::HumanString(gemm->shape()), "f32[4,4]");

  CollectiveHintsAnnotatorPass pass(std::move(config),
                                    std::string(kFingerprint));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig backend_config,
                          start->backend_config<GpuBackendConfig>());
  EXPECT_TRUE(backend_config.force_earliest_schedule());
  EXPECT_EQ(FrontendAttribute(*start, kXlaSchedulingGroupIdAttr), "7");
  EXPECT_EQ(FrontendAttribute(*gemm, kXlaSchedulingGroupIdAttr), "7");
  EXPECT_EQ(FrontendAttribute(*done, kXlaSchedulingGroupIdAttr), "7");
  EXPECT_EQ(FrontendAttribute(*gemm, kCollectiveHintWindowTargetAttr),
            "ag-start");
  EXPECT_EQ(FrontendAttribute(*start, kCollectiveHintRuleIdsAttr),
            "a-collective");
  EXPECT_EQ(FrontendAttribute(*gemm, kCollectiveHintRuleIdsAttr),
            "b-gemm-window");
  EXPECT_TRUE(gemm->control_predecessors().empty());
  EXPECT_TRUE(done->control_predecessors().empty());
  EXPECT_EQ(FrontendAttribute(*gemm, "scheduling_group_order"), "1");
  EXPECT_EQ(FrontendAttribute(*start, "scheduling_group_order"), "0");
  EXPECT_EQ(FrontendAttribute(*done, "scheduling_group_order"), "2");

  const auto& module_attributes = module->frontend_attributes().map();
  EXPECT_EQ(module_attributes.at(std::string(kCollectiveHintsFingerprintAttr)),
            kFingerprint);
  EXPECT_EQ(module_attributes.at(std::string(kCollectiveHintsReceiptAttr)),
            "fingerprint=0123456789abcdef0123456789abcdef;"
            "a-collective=[main/ag-start];b-gemm-window=[main/gemm]");
  const std::string module_dump = module->ToString();
  EXPECT_THAT(module_dump, HasSubstr(kCollectiveHintsFingerprintAttr));
  EXPECT_THAT(module_dump, HasSubstr("_xla_collective_hints_receipt"));
  EXPECT_THAT(module_dump, HasSubstr("a-collective=[main/ag-start]"));
}

TEST_F(CollectiveHintsAnnotatorTest, RejectsPartialSelector) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kCollectiveAndGemmHlo,
                                                       /*replica_count=*/2));
  TagModule(module.get());
  CollectiveHintsConfig config = Config();
  CollectiveHint* hint = config.add_hints();
  hint->set_rule_id("partial");
  hint->mutable_match()->set_op_name("layer/transpose(gemm)");
  hint->set_force_earliest(true);

  CollectiveHintsAnnotatorPass pass(std::move(config),
                                    std::string(kFingerprint));
  absl::Status status = pass.Run(module.get()).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), HasSubstr("complete (op_name, opcode, shape)"));
}

TEST_F(CollectiveHintsAnnotatorTest, RequiresExpectedMatchCount) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kCollectiveAndGemmHlo,
                                                       /*replica_count=*/2));
  TagModule(module.get());
  CollectiveHintsConfig config = Config();
  CollectiveHint* hint = config.add_hints();
  hint->set_rule_id("missing-count");
  hint->mutable_match()->set_instruction_name("ag-start");
  hint->set_force_earliest(true);

  CollectiveHintsAnnotatorPass pass(std::move(config),
                                    std::string(kFingerprint));
  absl::Status status = pass.Run(module.get()).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(),
              HasSubstr("requires positive expected_match_count"));
}

TEST_F(CollectiveHintsAnnotatorTest, RejectsChangedMatchCountWithoutMutation) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kCollectiveAndGemmHlo,
                                                       /*replica_count=*/2));
  TagModule(module.get());
  CollectiveHintsConfig config = Config();
  CollectiveHint* hint = AddInstructionHint(&config, "wrong-count", "ag-start");
  hint->set_expected_match_count(2);
  hint->set_force_earliest(true);

  CollectiveHintsAnnotatorPass pass(std::move(config),
                                    std::string(kFingerprint));
  absl::Status status = pass.Run(module.get()).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), HasSubstr("must expect exactly one match"));
  EXPECT_FALSE(module->frontend_attributes().map().contains(
      std::string(kCollectiveHintsReceiptAttr)));
}

TEST_F(CollectiveHintsAnnotatorTest, RejectsUnsupportedPassFilter) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kCollectiveAndGemmHlo,
                                                       /*replica_count=*/2));
  TagModule(module.get());
  CollectiveHintsConfig config = Config();
  CollectiveHint* hint = config.add_hints();
  hint->set_rule_id("bad-pass");
  hint->mutable_match()->set_op_name("layer/transpose(gemm)");
  hint->mutable_match()->set_opcode("custom-call");
  hint->mutable_match()->set_shape("f32[4,4]");
  hint->mutable_match()->set_pass_filter("any");
  hint->set_expected_match_count(1);
  hint->set_scheduling_group_id(7);

  CollectiveHintsAnnotatorPass pass(std::move(config),
                                    std::string(kFingerprint));
  absl::Status status = pass.Run(module.get()).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), HasSubstr("unsupported pass_filter 'any'"));
}

TEST_F(CollectiveHintsAnnotatorTest, RejectsFingerprintMismatch) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kCollectiveAndGemmHlo,
                                                       /*replica_count=*/2));
  TagModule(module.get());
  CollectiveHintsConfig config = Config();
  config.set_module_fingerprint("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
  AddInstructionHint(&config, "fingerprint", "ag-start")
      ->set_force_earliest(true);

  CollectiveHintsAnnotatorPass pass(std::move(config),
                                    std::string(kFingerprint));
  absl::Status status = pass.Run(module.get()).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(status.message(), HasSubstr("does not match compiled module"));
}

TEST_F(CollectiveHintsAnnotatorTest,
       RejectsDataDependentWindowWithoutMutation) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kCyclicWindowHlo,
                                                       /*replica_count=*/2));
  TagModule(module.get());
  CollectiveHintsConfig config = Config();
  CollectiveHint* hint = AddInstructionHint(&config, "cycle", "gemm");
  hint->set_scheduling_group_id(7);
  hint->set_window_target("ag-start");

  HloInstruction* gemm =
      module->entry_computation()->GetInstructionWithName("gemm");
  ASSERT_NE(gemm, nullptr);
  CollectiveHintsAnnotatorPass pass(std::move(config),
                                    std::string(kFingerprint));
  absl::Status status = pass.Run(module.get()).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(status.message(), HasSubstr("must be data/control independent"));
  EXPECT_TRUE(gemm->control_predecessors().empty());
  EXPECT_TRUE(gemm->control_successors().empty());
  EXPECT_FALSE(module->frontend_attributes().map().contains(
      std::string(kCollectiveHintsReceiptAttr)));
}

TEST_F(CollectiveHintsAnnotatorTest, StrictLoaderRejectsLegacyAndWrongTypes) {
  std::string path;
  ASSERT_TRUE(tsl::Env::Default()->LocalTempFilename(&path));
  EXPECT_FALSE(LoadCollectiveHintsConfig(path).ok());

  ASSERT_TRUE(tsl::WriteStringToFile(tsl::Env::Default(), path,
                                     R"pb(hints { stream_id: 1 })pb")
                  .ok());
  absl::Status legacy_status = LoadCollectiveHintsConfig(path).status();
  EXPECT_EQ(legacy_status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(legacy_status.message(), HasSubstr("deprecated fields"));

  ASSERT_TRUE(
      tsl::WriteStringToFile(tsl::Env::Default(), path,
                             R"pb(hints { scheduling_group_id: "group-A" })pb")
          .ok());
  absl::Status type_status = LoadCollectiveHintsConfig(path).status();
  EXPECT_EQ(type_status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(tsl::Env::Default()->DeleteFile(path).ok());
}

}  // namespace
}  // namespace xla::gpu
