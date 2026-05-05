/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
==============================================================================*/

#include "xla/service/gpu/auto_window_target.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/collective_hints.pb.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/profiler/protobuf/profiled_instructions.pb.h"

namespace xla::gpu {
namespace {

class AutoWindowTargetTest : public HloHardwareIndependentTestBase {};

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

// Threshold = 0 disables the pass: no rules emitted, even on a module
// with eligible (collective, anchor) pairs.
TEST_F(AutoWindowTargetTest, ThresholdZeroIsNoOp) {
  constexpr absl::string_view hlo_string = R"(
    HloModule test
    ENTRY entry {
      p0 = f32[1024,1024]{1,0} parameter(0)
      p1 = f32[2048]{0} parameter(1)
      ag = (f32[1024,1024]{1,0}, f32[2048,1024]{1,0}) all-gather-start(p0),
          replica_groups={{0,1}}, dimensions={0}
      cc = f32[2048]{0} custom-call(p1), custom_call_target="my_op"
      ROOT done = f32[2048,1024]{1,0} all-gather-done(ag)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto profile = MakeProfile({{"ag", 500.0}, {"cc", 100.0}});

  TF_ASSERT_OK_AND_ASSIGN(
      CollectiveHintsConfig cfg,
      BuildAutoWindowTargetConfig(*hlo_module, profile,
                                   /*threshold_us=*/0.0f,
                                   /*min_compute_us=*/50.0f,
                                   /*max_per_collective=*/4,
                                   /*max_total_rules=*/0));
  EXPECT_EQ(cfg.hints_size(), 0);
}

// Heavy collective + one eligible custom-call after it -> one rule.
TEST_F(AutoWindowTargetTest, EmitsRuleForEligibleAnchor) {
  constexpr absl::string_view hlo_string = R"(
    HloModule test
    ENTRY entry {
      p0 = f32[1024,1024]{1,0} parameter(0)
      p1 = f32[2048]{0} parameter(1)
      ag = (f32[1024,1024]{1,0}, f32[2048,1024]{1,0}) all-gather-start(p0),
          replica_groups={{0,1}}, dimensions={0}
      cc = f32[2048]{0} custom-call(p1), custom_call_target="my_op"
      done = f32[2048,1024]{1,0} all-gather-done(ag)
      ROOT t = (f32[2048,1024]{1,0}, f32[2048]{0}) tuple(done, cc)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto profile = MakeProfile({{"ag", 500.0}, {"cc", 100.0}});

  TF_ASSERT_OK_AND_ASSIGN(
      CollectiveHintsConfig cfg,
      BuildAutoWindowTargetConfig(*hlo_module, profile,
                                   /*threshold_us=*/100.0f,
                                   /*min_compute_us=*/50.0f,
                                   /*max_per_collective=*/4,
                                   /*max_total_rules=*/0));
  ASSERT_EQ(cfg.hints_size(), 1);
  EXPECT_EQ(cfg.hints(0).name(), "cc");
  ASSERT_EQ(cfg.hints(0).window_target_size(), 1);
  EXPECT_EQ(cfg.hints(0).window_target(0), "ag");
}

// Compute candidate that is operand-tree independent of the collective
// is eligible regardless of natural-schedule position. (C3 runs before
// LHS; we don't have schedule order to consult, so we accept any
// independent compute. Control deps `start -> anchor -> done` will pull
// the anchor into the window even if it would have run earlier.)
TEST_F(AutoWindowTargetTest, AcceptsIndependentAnchorRegardlessOfPosition) {
  constexpr absl::string_view hlo_string = R"(
    HloModule test
    ENTRY entry {
      p0 = f32[1024,1024]{1,0} parameter(0)
      p1 = f32[2048]{0} parameter(1)
      cc = f32[2048]{0} custom-call(p1), custom_call_target="my_op"
      ag = (f32[1024,1024]{1,0}, f32[2048,1024]{1,0}) all-gather-start(p0),
          replica_groups={{0,1}}, dimensions={0}
      done = f32[2048,1024]{1,0} all-gather-done(ag)
      ROOT t = (f32[2048,1024]{1,0}, f32[2048]{0}) tuple(done, cc)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto profile = MakeProfile({{"ag", 500.0}, {"cc", 100.0}});

  TF_ASSERT_OK_AND_ASSIGN(
      CollectiveHintsConfig cfg,
      BuildAutoWindowTargetConfig(*hlo_module, profile,
                                   /*threshold_us=*/100.0f,
                                   /*min_compute_us=*/50.0f,
                                   /*max_per_collective=*/4,
                                   /*max_total_rules=*/0));
  ASSERT_EQ(cfg.hints_size(), 1);
  EXPECT_EQ(cfg.hints(0).name(), "cc");
  EXPECT_EQ(cfg.hints(0).window_target(0), "ag");
}

// Compute candidate that depends on the collective's done is rejected
// (would be a dataflow descendant — pinning it into the window cycles).
TEST_F(AutoWindowTargetTest, RejectsDataflowDescendant) {
  constexpr absl::string_view hlo_string = R"(
    HloModule test
    ENTRY entry {
      p0 = f32[1024,1024]{1,0} parameter(0)
      ag = (f32[1024,1024]{1,0}, f32[2048,1024]{1,0}) all-gather-start(p0),
          replica_groups={{0,1}}, dimensions={0}
      done = f32[2048,1024]{1,0} all-gather-done(ag)
      cc = f32[2048,1024]{1,0} custom-call(done), custom_call_target="my_op"
      ROOT t = f32[2048,1024]{1,0} add(cc, cc)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto profile = MakeProfile({{"ag", 500.0}, {"cc", 100.0}});

  TF_ASSERT_OK_AND_ASSIGN(
      CollectiveHintsConfig cfg,
      BuildAutoWindowTargetConfig(*hlo_module, profile,
                                   /*threshold_us=*/100.0f,
                                   /*min_compute_us=*/50.0f,
                                   /*max_per_collective=*/4,
                                   /*max_total_rules=*/0));
  EXPECT_EQ(cfg.hints_size(), 0);
}

// Ineligible opcode (e.g., a kLoop fusion) is rejected.
TEST_F(AutoWindowTargetTest, RejectsIneligibleOpcode) {
  constexpr absl::string_view hlo_string = R"(
    HloModule test
    add_fn {
      a = f32[2048]{0} parameter(0)
      b = f32[2048]{0} parameter(1)
      ROOT add = f32[2048]{0} add(a, b)
    }
    ENTRY entry {
      p0 = f32[1024,1024]{1,0} parameter(0)
      p1 = f32[2048]{0} parameter(1)
      ag = (f32[1024,1024]{1,0}, f32[2048,1024]{1,0}) all-gather-start(p0),
          replica_groups={{0,1}}, dimensions={0}
      loopf = f32[2048]{0} fusion(p1, p1), kind=kLoop, calls=add_fn
      done = f32[2048,1024]{1,0} all-gather-done(ag)
      ROOT t = (f32[2048,1024]{1,0}, f32[2048]{0}) tuple(done, loopf)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto profile = MakeProfile({{"ag", 500.0}, {"loopf", 100.0}});

  TF_ASSERT_OK_AND_ASSIGN(
      CollectiveHintsConfig cfg,
      BuildAutoWindowTargetConfig(*hlo_module, profile,
                                   /*threshold_us=*/100.0f,
                                   /*min_compute_us=*/50.0f,
                                   /*max_per_collective=*/4,
                                   /*max_total_rules=*/0));
  EXPECT_EQ(cfg.hints_size(), 0);
}

// Compute below `min_compute_us` is rejected.
TEST_F(AutoWindowTargetTest, RejectsTooSmallAnchor) {
  constexpr absl::string_view hlo_string = R"(
    HloModule test
    ENTRY entry {
      p0 = f32[1024,1024]{1,0} parameter(0)
      p1 = f32[2048]{0} parameter(1)
      ag = (f32[1024,1024]{1,0}, f32[2048,1024]{1,0}) all-gather-start(p0),
          replica_groups={{0,1}}, dimensions={0}
      cc = f32[2048]{0} custom-call(p1), custom_call_target="my_op"
      done = f32[2048,1024]{1,0} all-gather-done(ag)
      ROOT t = (f32[2048,1024]{1,0}, f32[2048]{0}) tuple(done, cc)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto profile = MakeProfile({{"ag", 500.0}, {"cc", 10.0}});  // 10 < 50

  TF_ASSERT_OK_AND_ASSIGN(
      CollectiveHintsConfig cfg,
      BuildAutoWindowTargetConfig(*hlo_module, profile,
                                   /*threshold_us=*/100.0f,
                                   /*min_compute_us=*/50.0f,
                                   /*max_per_collective=*/4,
                                   /*max_total_rules=*/0));
  EXPECT_EQ(cfg.hints_size(), 0);
}

// Greedy fill stops when cumulative compute cost meets/exceeds the
// collective cost.
TEST_F(AutoWindowTargetTest, GreedyFillStopsAtCumulativeCost) {
  // Five custom-calls of 30us each, after a 50us collective. The pass
  // should pick 2 (cumulative 60 > 50), not all 5.
  constexpr absl::string_view hlo_string = R"(
    HloModule test
    ENTRY entry {
      p0 = f32[1024,1024]{1,0} parameter(0)
      p1 = f32[2048]{0} parameter(1)
      ag = (f32[1024,1024]{1,0}, f32[2048,1024]{1,0}) all-gather-start(p0),
          replica_groups={{0,1}}, dimensions={0}
      cc1 = f32[2048]{0} custom-call(p1), custom_call_target="op1"
      cc2 = f32[2048]{0} custom-call(p1), custom_call_target="op2"
      cc3 = f32[2048]{0} custom-call(p1), custom_call_target="op3"
      cc4 = f32[2048]{0} custom-call(p1), custom_call_target="op4"
      cc5 = f32[2048]{0} custom-call(p1), custom_call_target="op5"
      done = f32[2048,1024]{1,0} all-gather-done(ag)
      ROOT t = (f32[2048,1024]{1,0}, f32[2048]{0}, f32[2048]{0},
                f32[2048]{0}, f32[2048]{0}, f32[2048]{0})
          tuple(done, cc1, cc2, cc3, cc4, cc5)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto profile = MakeProfile({
      {"ag", 50.0},
      {"cc1", 30.0}, {"cc2", 30.0}, {"cc3", 30.0},
      {"cc4", 30.0}, {"cc5", 30.0},
  });

  TF_ASSERT_OK_AND_ASSIGN(
      CollectiveHintsConfig cfg,
      BuildAutoWindowTargetConfig(*hlo_module, profile,
                                   /*threshold_us=*/40.0f,
                                   /*min_compute_us=*/10.0f,
                                   /*max_per_collective=*/10,
                                   /*max_total_rules=*/0));
  // Picks 2 because 30 + 30 = 60 >= 50 (cumulative >= collective).
  EXPECT_EQ(cfg.hints_size(), 2);
}

// max_per_collective caps the number of anchors even if cumulative
// compute < collective cost.
TEST_F(AutoWindowTargetTest, MaxPerCollectiveCaps) {
  constexpr absl::string_view hlo_string = R"(
    HloModule test
    ENTRY entry {
      p0 = f32[1024,1024]{1,0} parameter(0)
      p1 = f32[2048]{0} parameter(1)
      ag = (f32[1024,1024]{1,0}, f32[2048,1024]{1,0}) all-gather-start(p0),
          replica_groups={{0,1}}, dimensions={0}
      cc1 = f32[2048]{0} custom-call(p1), custom_call_target="op1"
      cc2 = f32[2048]{0} custom-call(p1), custom_call_target="op2"
      cc3 = f32[2048]{0} custom-call(p1), custom_call_target="op3"
      cc4 = f32[2048]{0} custom-call(p1), custom_call_target="op4"
      cc5 = f32[2048]{0} custom-call(p1), custom_call_target="op5"
      done = f32[2048,1024]{1,0} all-gather-done(ag)
      ROOT t = (f32[2048,1024]{1,0}, f32[2048]{0}, f32[2048]{0},
                f32[2048]{0}, f32[2048]{0}, f32[2048]{0})
          tuple(done, cc1, cc2, cc3, cc4, cc5)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  // Collective is 10000us; each compute is 50us. Cumulative would never
  // hit the collective cost; the cap should fire.
  auto profile = MakeProfile({
      {"ag", 10000.0},
      {"cc1", 50.0}, {"cc2", 50.0}, {"cc3", 50.0},
      {"cc4", 50.0}, {"cc5", 50.0},
  });

  TF_ASSERT_OK_AND_ASSIGN(
      CollectiveHintsConfig cfg,
      BuildAutoWindowTargetConfig(*hlo_module, profile,
                                   /*threshold_us=*/100.0f,
                                   /*min_compute_us=*/40.0f,
                                   /*max_per_collective=*/3,
                                   /*max_total_rules=*/0));
  EXPECT_EQ(cfg.hints_size(), 3);
}

// Already-claimed anchor (has _xla_window_target frontend attr) is
// skipped — emulates manual hints having pinned the anchor first.
TEST_F(AutoWindowTargetTest, SkipsAlreadyClaimedAnchor) {
  constexpr absl::string_view hlo_string = R"(
    HloModule test
    ENTRY entry {
      p0 = f32[1024,1024]{1,0} parameter(0)
      p1 = f32[2048]{0} parameter(1)
      ag = (f32[1024,1024]{1,0}, f32[2048,1024]{1,0}) all-gather-start(p0),
          replica_groups={{0,1}}, dimensions={0}
      cc = f32[2048]{0} custom-call(p1), custom_call_target="my_op",
          frontend_attributes={_xla_window_target="some_other_collective"}
      done = f32[2048,1024]{1,0} all-gather-done(ag)
      ROOT t = (f32[2048,1024]{1,0}, f32[2048]{0}) tuple(done, cc)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto profile = MakeProfile({{"ag", 500.0}, {"cc", 100.0}});

  TF_ASSERT_OK_AND_ASSIGN(
      CollectiveHintsConfig cfg,
      BuildAutoWindowTargetConfig(*hlo_module, profile,
                                   /*threshold_us=*/100.0f,
                                   /*min_compute_us=*/50.0f,
                                   /*max_per_collective=*/4,
                                   /*max_total_rules=*/0));
  EXPECT_EQ(cfg.hints_size(), 0);
}

// Already-paired collective (has _scheduling_group_id) is skipped —
// emulates manual hints having paired the collective.
TEST_F(AutoWindowTargetTest, SkipsAlreadyPairedCollective) {
  constexpr absl::string_view hlo_string = R"(
    HloModule test
    ENTRY entry {
      p0 = f32[1024,1024]{1,0} parameter(0)
      p1 = f32[2048]{0} parameter(1)
      ag = (f32[1024,1024]{1,0}, f32[2048,1024]{1,0}) all-gather-start(p0),
          replica_groups={{0,1}}, dimensions={0},
          frontend_attributes={_scheduling_group_id="42"}
      cc = f32[2048]{0} custom-call(p1), custom_call_target="my_op"
      done = f32[2048,1024]{1,0} all-gather-done(ag)
      ROOT t = (f32[2048,1024]{1,0}, f32[2048]{0}) tuple(done, cc)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto profile = MakeProfile({{"ag", 500.0}, {"cc", 100.0}});

  TF_ASSERT_OK_AND_ASSIGN(
      CollectiveHintsConfig cfg,
      BuildAutoWindowTargetConfig(*hlo_module, profile,
                                   /*threshold_us=*/100.0f,
                                   /*min_compute_us=*/50.0f,
                                   /*max_per_collective=*/4,
                                   /*max_total_rules=*/0));
  EXPECT_EQ(cfg.hints_size(), 0);
}

// Determinism: same module + profile -> same anchor selection across
// invocations. (Tie-break by HLO instruction name.)
TEST_F(AutoWindowTargetTest, DeterministicTieBreak) {
  constexpr absl::string_view hlo_string = R"(
    HloModule test
    ENTRY entry {
      p0 = f32[1024,1024]{1,0} parameter(0)
      p1 = f32[2048]{0} parameter(1)
      ag = (f32[1024,1024]{1,0}, f32[2048,1024]{1,0}) all-gather-start(p0),
          replica_groups={{0,1}}, dimensions={0}
      cc1 = f32[2048]{0} custom-call(p1), custom_call_target="op1"
      cc2 = f32[2048]{0} custom-call(p1), custom_call_target="op2"
      done = f32[2048,1024]{1,0} all-gather-done(ag)
      ROOT t = (f32[2048,1024]{1,0}, f32[2048]{0}, f32[2048]{0})
          tuple(done, cc1, cc2)
    }
  )";
  // cc1 is closer to ag in file order than cc2; closer wins.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m1,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m2,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto profile = MakeProfile({
      {"ag", 1000.0}, {"cc1", 100.0}, {"cc2", 100.0},
  });

  TF_ASSERT_OK_AND_ASSIGN(
      CollectiveHintsConfig c1,
      BuildAutoWindowTargetConfig(*m1, profile, 100.0f, 50.0f, 1,
                                   /*max_total_rules=*/0));
  TF_ASSERT_OK_AND_ASSIGN(
      CollectiveHintsConfig c2,
      BuildAutoWindowTargetConfig(*m2, profile, 100.0f, 50.0f, 1,
                                   /*max_total_rules=*/0));
  ASSERT_EQ(c1.hints_size(), 1);
  ASSERT_EQ(c2.hints_size(), 1);
  EXPECT_EQ(c1.hints(0).name(), "cc1");
  EXPECT_EQ(c2.hints(0).name(), "cc1");
}

// Option 1 safety cap: max_total_rules limits the total number of
// window_target rules emitted across all collectives. Highest-priority
// (cost-descending) collectives keep their rules; lower-priority ones
// get dropped.
TEST_F(AutoWindowTargetTest, MaxTotalRulesCapsAcrossCollectives) {
  constexpr absl::string_view hlo_string = R"(
    HloModule test
    ENTRY entry {
      p0 = f32[1024,1024]{1,0} parameter(0)
      p1 = f32[2048]{0} parameter(1)
      // Three independent collectives. ag_big (1000us) is the priority
      // pick because it has the largest PGLE cost.
      ag_big = (f32[1024,1024]{1,0}, f32[2048,1024]{1,0})
          all-gather-start(p0), replica_groups={{0,1}}, dimensions={0}
      ag_mid = (f32[1024,1024]{1,0}, f32[2048,1024]{1,0})
          all-gather-start(p0), replica_groups={{0,1}}, dimensions={0}
      ag_small = (f32[1024,1024]{1,0}, f32[2048,1024]{1,0})
          all-gather-start(p0), replica_groups={{0,1}}, dimensions={0}
      cc1 = f32[2048]{0} custom-call(p1), custom_call_target="op1"
      cc2 = f32[2048]{0} custom-call(p1), custom_call_target="op2"
      cc3 = f32[2048]{0} custom-call(p1), custom_call_target="op3"
      d_big = f32[2048,1024]{1,0} all-gather-done(ag_big)
      d_mid = f32[2048,1024]{1,0} all-gather-done(ag_mid)
      d_small = f32[2048,1024]{1,0} all-gather-done(ag_small)
      ROOT t = (f32[2048,1024]{1,0}, f32[2048,1024]{1,0},
                f32[2048,1024]{1,0}, f32[2048]{0}, f32[2048]{0},
                f32[2048]{0})
          tuple(d_big, d_mid, d_small, cc1, cc2, cc3)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto profile = MakeProfile({
      {"ag_big", 1000.0}, {"ag_mid", 800.0}, {"ag_small", 600.0},
      {"cc1", 100.0}, {"cc2", 100.0}, {"cc3", 100.0},
  });

  // Without cap: each collective gets 1 anchor (max_per_collective=1) =>
  // 3 rules total.
  TF_ASSERT_OK_AND_ASSIGN(
      CollectiveHintsConfig uncapped,
      BuildAutoWindowTargetConfig(*hlo_module, profile,
                                   /*threshold_us=*/100.0f,
                                   /*min_compute_us=*/50.0f,
                                   /*max_per_collective=*/1,
                                   /*max_total_rules=*/0));
  EXPECT_EQ(uncapped.hints_size(), 3);

  // With cap=2: only the two heaviest collectives (ag_big, ag_mid) get
  // rules; ag_small (lowest priority) is dropped.
  TF_ASSERT_OK_AND_ASSIGN(
      CollectiveHintsConfig capped,
      BuildAutoWindowTargetConfig(*hlo_module, profile,
                                   /*threshold_us=*/100.0f,
                                   /*min_compute_us=*/50.0f,
                                   /*max_per_collective=*/1,
                                   /*max_total_rules=*/2));
  EXPECT_EQ(capped.hints_size(), 2);

  // Negative cap disables it (same as 0).
  TF_ASSERT_OK_AND_ASSIGN(
      CollectiveHintsConfig disabled_cap,
      BuildAutoWindowTargetConfig(*hlo_module, profile,
                                   /*threshold_us=*/100.0f,
                                   /*min_compute_us=*/50.0f,
                                   /*max_per_collective=*/1,
                                   /*max_total_rules=*/-1));
  EXPECT_EQ(disabled_cap.hints_size(), 3);
}

}  // namespace
}  // namespace xla::gpu
