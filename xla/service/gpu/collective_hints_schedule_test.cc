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

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/alias_info.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/collective_hints_annotator.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/gpu_hlo_schedule.h"
#include "xla/service/hlo_module_config.h"
#include "xla/side_effect_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

using ::testing::HasSubstr;

constexpr absl::string_view kHlo = R"hlo(
  HloModule m

  ENTRY main {
    p0 = f32[4] parameter(0)
    lhs = f32[4,4] parameter(1)
    rhs = f32[4,4] parameter(2)
    gemm = f32[4,4] custom-call(lhs, rhs),
        custom_call_target="__cublas$lt$matmul"
    ag-start = (f32[4], f32[8]) all-gather-start(p0), dimensions={0},
        replica_groups={}
    ag-done = f32[8] all-gather-done(ag-start)
    ROOT result = (f32[8], f32[4,4]) tuple(ag-done, gemm)
  }
)hlo";

constexpr absl::string_view kLegacyGroupedCublasLtHlo = R"hlo(
  HloModule m

  ENTRY main {
    lhs = f32[4,4] parameter(0)
    rhs = f32[4,4] parameter(1)
    ag-start = (f32[4,4], f32[8,4]) all-gather-start(lhs), dimensions={0},
        replica_groups={}, frontend_attributes={_scheduling_group_id="7"}
    ag-done = f32[8,4] all-gather-done(ag-start)
    gathered-lhs = f32[4,4] slice(ag-done), slice={[0:4], [0:4]}
    gemm = f32[4,4] custom-call(gathered-lhs, rhs),
        custom_call_target="__cublas$lt$matmul",
        frontend_attributes={_scheduling_group_id="7"}
    ROOT result = (f32[8,4], f32[4,4]) tuple(ag-done, gemm)
  }
)hlo";

constexpr absl::string_view kFusionHlo = R"hlo(
  HloModule m

  fused_computation {
    p0 = f32[4] parameter(0)
    ROOT negate = f32[4] negate(p0)
  }

  ENTRY main {
    p0 = f32[4] parameter(0)
    ROOT fusion = f32[4] fusion(p0), kind=kInput, calls=fused_computation
  }
)hlo";

class CollectiveHintsScheduleTest : public HloHardwareIndependentTestBase {
 protected:
  ~CollectiveHintsScheduleTest() override {
    for (const std::string& path : temp_files_) {
      (void)tsl::Env::Default()->DeleteFile(path);
    }
  }

  HloModuleConfig ModuleConfig(absl::string_view hints_path = {}) {
    HloModuleConfig config;
    DebugOptions options = GetDebugOptionsForTest();
    options.set_xla_gpu_enable_latency_hiding_scheduler(false);
    options.set_xla_gpu_collective_hints_file(std::string(hints_path));
    config.set_debug_options(options);
    config.set_replica_count(2);
    return config;
  }

  absl::Status Schedule(HloModule* module) {
    se::DeviceDescription device_info =
        TestGpuDeviceInfo::CudaOrRocmDeviceInfo();
    GpuAliasInfo alias_info(device_info);
    return ScheduleGpuModule(module, /*pointer_size=*/8, device_info,
                             &mlir_context_, &alias_info)
        .status();
  }

  absl::StatusOr<std::string> WriteTempFile(absl::string_view contents) {
    std::string path;
    if (!tsl::Env::Default()->LocalTempFilename(&path)) {
      return absl::InternalError("failed to allocate temporary filename");
    }
    ABSL_RETURN_IF_ERROR(tsl::WriteStringToFile(tsl::Env::Default(), path,
                                                std::string(contents)));
    temp_files_.push_back(path);
    return path;
  }

  static std::string Fingerprint(const HloModule& module) {
    return CollectiveHintsFingerprint(module);
  }

  static std::vector<std::string> ScheduledNames(const HloModule& module) {
    std::vector<std::string> names;
    for (const HloInstruction* instruction :
         module.schedule()
             .sequence(module.entry_computation())
             .instructions()) {
      names.push_back(std::string(instruction->name()));
    }
    return names;
  }

  mlir::MLIRContext mlir_context_;
  std::vector<std::string> temp_files_;
};

TEST_F(CollectiveHintsScheduleTest, EmptyPathIsBehaviorNeutral) {
  HloModuleConfig absent_config = ModuleConfig();
  absent_config.mutable_debug_options().clear_xla_gpu_collective_hints_file();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> absent_module,
                          ParseAndReturnVerifiedModule(kHlo, absent_config));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> empty_module,
      ParseAndReturnVerifiedModule(kHlo, ModuleConfig(/*hints_path=*/"")));
  TF_ASSERT_OK(Schedule(absent_module.get()));
  TF_ASSERT_OK(Schedule(empty_module.get()));

  EXPECT_EQ(ScheduledNames(*empty_module), ScheduledNames(*absent_module));
  EXPECT_EQ(empty_module->frontend_attributes().map().at(
                std::string(kFingerprintBeforeLHS)),
            absent_module->frontend_attributes().map().at(
                std::string(kFingerprintBeforeLHS)));

  for (const HloModule* module : {absent_module.get(), empty_module.get()}) {
    EXPECT_FALSE(module->frontend_attributes().map().contains(
        std::string(kCollectiveHintsReceiptAttr)));
    for (const HloInstruction* instruction :
         module->entry_computation()->instructions()) {
      const auto& attributes = instruction->frontend_attributes().map();
      EXPECT_FALSE(
          attributes.contains(std::string(kCollectiveHintRuleIdsAttr)));
      EXPECT_FALSE(
          attributes.contains(std::string(kCollectiveHintWindowTargetAttr)));
    }
  }
}

TEST_F(CollectiveHintsScheduleTest,
       FingerprintIgnoresAutotuningBackendConfiguration) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo, ModuleConfig()));
  const std::string fingerprint = Fingerprint(*module);
  HloInstruction* gemm =
      module->entry_computation()->GetInstructionWithName("gemm");
  ASSERT_NE(gemm, nullptr);
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig config,
                          gemm->backend_config<GpuBackendConfig>());
  config.mutable_gemm_backend_config()->set_selected_algorithm(7);
  TF_ASSERT_OK(gemm->set_backend_config(config));

  EXPECT_EQ(Fingerprint(*module), fingerprint);
}

TEST_F(CollectiveHintsScheduleTest, FingerprintIgnoresFusionKind) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kFusionHlo));
  const std::string fingerprint = Fingerprint(*module);
  module->entry_computation()->root_instruction()->set_fusion_kind(
      HloInstruction::FusionKind::kCustom);

  EXPECT_EQ(Fingerprint(*module), fingerprint);
}

TEST_F(CollectiveHintsScheduleTest,
       UnhintedCublasLtPreservesLegacyAnnotationBehavior) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kLegacyGroupedCublasLtHlo, ModuleConfig()));

  TF_ASSERT_OK(Schedule(module.get()));
}

TEST_F(CollectiveHintsScheduleTest, NonTargetConfigIsBehaviorNeutral) {
  TF_ASSERT_OK_AND_ASSIGN(std::string path, WriteTempFile(absl::StrCat(
                                                "module_fingerprint: \"",
                                                std::string(32, 'f'), "\"\n",
                                                "hints { rule_id: \"rule\" "
                                                "match { instruction_name: "
                                                "\"ag-start\" } "
                                                "expected_match_count: 1 "
                                                "force_earliest: true }\n")));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> baseline,
                          ParseAndReturnVerifiedModule(kHlo, ModuleConfig()));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> non_target,
      ParseAndReturnVerifiedModule(kHlo, ModuleConfig(path)));
  ASSERT_NE(Fingerprint(*non_target), std::string(32, 'f'));

  TF_ASSERT_OK(Schedule(baseline.get()));
  TF_ASSERT_OK(Schedule(non_target.get()));

  EXPECT_EQ(ScheduledNames(*non_target), ScheduledNames(*baseline));
  EXPECT_FALSE(non_target->frontend_attributes().map().contains(
      std::string(kCollectiveHintsReceiptAttr)));
  for (const HloInstruction* instruction :
       non_target->entry_computation()->instructions()) {
    EXPECT_FALSE(instruction->frontend_attributes().map().contains(
        std::string(kCollectiveHintRuleIdsAttr)));
  }
}

TEST_F(CollectiveHintsScheduleTest, LhsScopeLogFormatIsMachineParseable) {
  EXPECT_EQ(FormatLhsScopeBeginLog("module\"\n", 17, "0123456789abcdef"),
            "XSCHED_LHS_SCOPE_BEGIN module_name=\"module\\\"\\n\" "
            "module_id=17 fingerprint_before_lhs=0123456789abcdef");
  EXPECT_EQ(
      FormatLhsScopeEndLog("module\"\n", 17, "0123456789abcdef",
                           absl::InvalidArgumentError("bad\nvalue")),
      "XSCHED_LHS_SCOPE_END module_name=\"module\\\"\\n\" module_id=17 "
      "fingerprint_before_lhs=0123456789abcdef status_code=INVALID_ARGUMENT "
      "status_message=\"bad\\nvalue\"");
  EXPECT_EQ(
      FormatCollectiveHintsBindingLog("module\"\n", 17, "current", "target",
                                      /*selected=*/false),
      "XSCHED_COLLECTIVE_HINTS_BINDING module_name=\"module\\\"\\n\" "
      "module_id=17 collective_hints_fingerprint=current "
      "target_fingerprint=target "
      "selected=false");
}

TEST_F(CollectiveHintsScheduleTest, HintsSurviveLegalizerAndConstrainSchedule) {
  TF_ASSERT_OK_AND_ASSIGN(std::string path, WriteTempFile("placeholder"));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kHlo, ModuleConfig(path)));
  const std::string fingerprint = Fingerprint(*module);
  const std::string hints = absl::StrCat(
      "module_fingerprint: \"", fingerprint, "\"\n",
      "hints { rule_id: \"collective\" match { instruction_name: "
      "\"ag-start\" } expected_match_count: 1 force_earliest: true "
      "scheduling_group_id: 7 }\n",
      "hints { rule_id: \"gemm\" match { instruction_name: \"gemm\" } "
      "expected_match_count: 1 scheduling_group_id: 7 "
      "window_target: \"ag-start\" }\n");
  TF_ASSERT_OK(tsl::WriteStringToFile(tsl::Env::Default(), path, hints));

  TF_ASSERT_OK(Schedule(module.get()));
  HloInstruction* start =
      module->entry_computation()->GetInstructionWithName("ag-start");
  HloInstruction* gemm =
      module->entry_computation()->GetInstructionWithName("gemm");
  HloInstruction* done =
      module->entry_computation()->GetInstructionWithName("ag-done");
  ASSERT_NE(start, nullptr);
  ASSERT_NE(gemm, nullptr);
  ASSERT_NE(done, nullptr);

  EXPECT_EQ(start->frontend_attributes().map().at(kXlaSchedulingGroupIdAttr),
            "7");
  EXPECT_EQ(gemm->frontend_attributes().map().at(kXlaSchedulingGroupIdAttr),
            "7");
  EXPECT_EQ(gemm->frontend_attributes().map().at(
                std::string(kCollectiveHintWindowTargetAttr)),
            "ag-start");
  EXPECT_EQ(gemm->frontend_attributes().map().at("scheduling_group_order"),
            "1");
  EXPECT_EQ(start->frontend_attributes().map().at("scheduling_group_order"),
            "0");
  EXPECT_EQ(done->frontend_attributes().map().at("scheduling_group_order"),
            "2");
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig backend_config,
                          start->backend_config<GpuBackendConfig>());
  EXPECT_TRUE(backend_config.force_earliest_schedule());

  const std::vector<HloInstruction*>& sequence =
      module->schedule().sequence(module->entry_computation()).instructions();
  auto position = [&](const HloInstruction* instruction) {
    return std::find(sequence.begin(), sequence.end(), instruction) -
           sequence.begin();
  };
  EXPECT_LT(position(start), position(gemm));
  EXPECT_LT(position(gemm), position(done));
  EXPECT_THAT(module->frontend_attributes().map().at(
                  std::string(kCollectiveHintsReceiptAttr)),
              HasSubstr("collective=[main/ag-start];gemm=[main/gemm]"));
}

TEST_F(CollectiveHintsScheduleTest, ReadParseAndMatchFailuresPropagate) {
  std::string missing_path;
  ASSERT_TRUE(tsl::Env::Default()->LocalTempFilename(&missing_path));
  missing_path.append(".missing");
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> missing_module,
      ParseAndReturnVerifiedModule(kHlo, ModuleConfig(missing_path)));
  absl::Status read_status = Schedule(missing_module.get());
  EXPECT_FALSE(read_status.ok());
  EXPECT_THAT(read_status.message(), HasSubstr("failed to read collective"));

  TF_ASSERT_OK_AND_ASSIGN(std::string parse_path,
                          WriteTempFile("hints { stream_id: 1 }\n"));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> parse_module,
      ParseAndReturnVerifiedModule(kHlo, ModuleConfig(parse_path)));
  absl::Status parse_status = Schedule(parse_module.get());
  EXPECT_EQ(parse_status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(parse_status.message(), HasSubstr("failed to parse collective"));

  TF_ASSERT_OK_AND_ASSIGN(
      std::string schema_path,
      WriteTempFile(absl::StrCat(
          "module_fingerprint: \"", std::string(32, 'f'), "\"\n",
          "hints { rule_id: \"no-action\" match { instruction_name: "
          "\"ag-start\" } expected_match_count: 1 }\n")));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> schema_module,
      ParseAndReturnVerifiedModule(kHlo, ModuleConfig(schema_path)));
  absl::Status schema_status = Schedule(schema_module.get());
  EXPECT_EQ(schema_status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(schema_status.message(),
              HasSubstr("requires at least one supported action"));

  TF_ASSERT_OK_AND_ASSIGN(std::string match_path, WriteTempFile("placeholder"));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> match_module,
      ParseAndReturnVerifiedModule(kHlo, ModuleConfig(match_path)));
  const std::string match_hints = absl::StrCat(
      "module_fingerprint: \"", Fingerprint(*match_module), "\"\n",
      "hints { rule_id: \"missing\" match { instruction_name: "
      "\"does-not-exist\" } expected_match_count: 1 force_earliest: true "
      "}\n");
  TF_ASSERT_OK(
      tsl::WriteStringToFile(tsl::Env::Default(), match_path, match_hints));
  absl::Status match_status = Schedule(match_module.get());
  EXPECT_EQ(match_status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(match_status.message(),
              HasSubstr("expected 1 match(es), found 0"));
}

}  // namespace
}  // namespace xla::gpu
