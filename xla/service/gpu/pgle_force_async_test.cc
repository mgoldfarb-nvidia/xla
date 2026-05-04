/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
==============================================================================*/

#include "xla/service/gpu/pgle_force_async.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/profiler/protobuf/profiled_instructions.pb.h"

namespace xla::gpu {
namespace {

class PgleForceAsyncTest : public HloHardwareIndependentTestBase {};

tensorflow::profiler::ProfiledInstructionsProto MakeProfile(
    std::initializer_list<std::pair<absl::string_view, double>> entries) {
  tensorflow::profiler::ProfiledInstructionsProto profile;
  for (const auto& [name, cost_us] : entries) {
    auto* c = profile.add_costs();
    c->set_name(std::string(name));
    c->set_cost_us(cost_us);
  }
  return profile;
}

// HLO with a sync collective (`is_sync: true` in backend_config) whose PGLE
// cost is above the threshold. After ApplyPgleForceAsync, is_sync should be
// false and any residual scheduling-group annotation should be cleared.
TEST_F(PgleForceAsyncTest, B1_FlipsSyncToAsyncAboveThreshold) {
  constexpr absl::string_view hlo_string = R"(
    HloModule test
    ENTRY entry {
      p0 = f32[1024,1024]{1,0} parameter(0)
      ags0 = (f32[1024,1024]{1,0}, f32[2048,1024]{1,0}) all-gather-start(p0),
        replica_groups={{0,1}}, dimensions={0},
        frontend_attributes={_scheduling_group_id="42"},
        backend_config={"collective_backend_config":{"is_sync":true}}
      ROOT agd0 = f32[2048,1024]{1,0} all-gather-done(ags0)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto profile = MakeProfile({{"ags0", 500.0}});  // 500 us > threshold
  TF_ASSERT_OK_AND_ASSIGN(
      int n_flipped, ApplyPgleForceAsync(hlo_module.get(), profile,
                                          /*threshold_us=*/100.0f));
  EXPECT_EQ(n_flipped, 1);

  HloInstruction* ags = nullptr;
  for (HloInstruction* instr :
       hlo_module->entry_computation()->instructions()) {
    if (instr->name() == "ags0") ags = instr;
  }
  ASSERT_NE(ags, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          ags->backend_config<GpuBackendConfig>());
  EXPECT_FALSE(gpu_config.collective_backend_config().is_sync());
  EXPECT_FALSE(ags->frontend_attributes().map().contains(
      "_scheduling_group_id"));
}

// PGLE cost below threshold: pass leaves the collective alone.
TEST_F(PgleForceAsyncTest, B1_LeavesAloneBelowThreshold) {
  constexpr absl::string_view hlo_string = R"(
    HloModule test
    ENTRY entry {
      p0 = f32[1024,1024]{1,0} parameter(0)
      ags0 = (f32[1024,1024]{1,0}, f32[2048,1024]{1,0}) all-gather-start(p0),
        replica_groups={{0,1}}, dimensions={0},
        backend_config={"collective_backend_config":{"is_sync":true}}
      ROOT agd0 = f32[2048,1024]{1,0} all-gather-done(ags0)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto profile = MakeProfile({{"ags0", 50.0}});  // 50 us < threshold
  TF_ASSERT_OK_AND_ASSIGN(
      int n_flipped, ApplyPgleForceAsync(hlo_module.get(), profile,
                                          /*threshold_us=*/100.0f));
  EXPECT_EQ(n_flipped, 0);

  HloInstruction* ags = nullptr;
  for (HloInstruction* instr :
       hlo_module->entry_computation()->instructions()) {
    if (instr->name() == "ags0") ags = instr;
  }
  ASSERT_NE(ags, nullptr);
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          ags->backend_config<GpuBackendConfig>());
  EXPECT_TRUE(gpu_config.collective_backend_config().is_sync());
}

// Threshold of 0 disables the pass — no-op even for slow collectives.
TEST_F(PgleForceAsyncTest, B1_ThresholdZeroIsNoOp) {
  constexpr absl::string_view hlo_string = R"(
    HloModule test
    ENTRY entry {
      p0 = f32[1024,1024]{1,0} parameter(0)
      ags0 = (f32[1024,1024]{1,0}, f32[2048,1024]{1,0}) all-gather-start(p0),
        replica_groups={{0,1}}, dimensions={0},
        backend_config={"collective_backend_config":{"is_sync":true}}
      ROOT agd0 = f32[2048,1024]{1,0} all-gather-done(ags0)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto profile = MakeProfile({{"ags0", 1.0e9}});  // huge cost
  TF_ASSERT_OK_AND_ASSIGN(
      int n_flipped, ApplyPgleForceAsync(hlo_module.get(), profile,
                                          /*threshold_us=*/0.0f));
  EXPECT_EQ(n_flipped, 0);
}

// Collective already async (is_sync: false): pass does not double-flip and
// returns 0 for it.
TEST_F(PgleForceAsyncTest, B1_AlreadyAsyncIsNoOp) {
  constexpr absl::string_view hlo_string = R"(
    HloModule test
    ENTRY entry {
      p0 = f32[1024,1024]{1,0} parameter(0)
      ags0 = (f32[1024,1024]{1,0}, f32[2048,1024]{1,0}) all-gather-start(p0),
        replica_groups={{0,1}}, dimensions={0}
      ROOT agd0 = f32[2048,1024]{1,0} all-gather-done(ags0)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto profile = MakeProfile({{"ags0", 500.0}});
  TF_ASSERT_OK_AND_ASSIGN(
      int n_flipped, ApplyPgleForceAsync(hlo_module.get(), profile,
                                          /*threshold_us=*/100.0f));
  EXPECT_EQ(n_flipped, 0);
}

// Collective whose name is NOT in the PGLE profile: pass skips (no
// information to act on).
TEST_F(PgleForceAsyncTest, B1_MissingFromProfileIsNoOp) {
  constexpr absl::string_view hlo_string = R"(
    HloModule test
    ENTRY entry {
      p0 = f32[1024,1024]{1,0} parameter(0)
      ags0 = (f32[1024,1024]{1,0}, f32[2048,1024]{1,0}) all-gather-start(p0),
        replica_groups={{0,1}}, dimensions={0},
        backend_config={"collective_backend_config":{"is_sync":true}}
      ROOT agd0 = f32[2048,1024]{1,0} all-gather-done(ags0)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto profile = MakeProfile({{"some_other_name", 500.0}});
  TF_ASSERT_OK_AND_ASSIGN(
      int n_flipped, ApplyPgleForceAsync(hlo_module.get(), profile,
                                          /*threshold_us=*/100.0f));
  EXPECT_EQ(n_flipped, 0);
}

}  // namespace
}  // namespace xla::gpu
