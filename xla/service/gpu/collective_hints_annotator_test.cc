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

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/collective_hints.pb.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

// Returns the value of `attr` in `instr`'s frontend_attributes, or "" if not
// present.
std::string GetFrontendAttr(const HloInstruction* instr,
                            absl::string_view attr) {
  auto& map = instr->frontend_attributes().map();
  auto it = map.find(std::string(attr));
  return it != map.end() ? it->second : "";
}

class CollectiveHintsAnnotatorTest : public HloHardwareIndependentTestBase {};

// HLO with an all-gather-start and a reduce-scatter-start.
constexpr absl::string_view kTwoCollectivesHlo = R"(
  HloModule m, is_scheduled=true

  reduce_add {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT _ = f32[] add(x, y)
  }

  ENTRY main {
    p0 = f32[4] parameter(0)
    p1 = f32[4] parameter(1)
    ag-start = (f32[4], f32[8]) all-gather-start(p0), dimensions={0},
        replica_groups={}
    rs-start = ((f32[4]), f32[2]) reduce-scatter-start(p1),
        to_apply=reduce_add, dimensions={0}
    add0 = f32[4] add(p0, p1)
    ag-done = f32[8] all-gather-done(ag-start)
    rs-done = f32[2] reduce-scatter-done(rs-start)
    ROOT _ = (f32[8], f32[2], f32[4]) tuple(ag-done, rs-done, add0)
  }
)";

// ---------------------------------------------------------------------------
// latency_metadata
// ---------------------------------------------------------------------------

TEST_F(CollectiveHintsAnnotatorTest, LatencyMetadataByName) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kTwoCollectivesHlo));

  CollectiveHintsConfig config;
  auto* hint = config.add_hints();
  hint->set_name("ag-start");
  hint->set_latency_metadata("12345");

  CollectiveHintsAnnotatorPass pass(config);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* ag =
      FindInstruction(module.get(), "ag-start");
  ASSERT_NE(ag, nullptr);
  EXPECT_EQ(GetFrontendAttr(ag, "latency_metadata"), "12345");

  // rs-start should be unmodified.
  const HloInstruction* rs =
      FindInstruction(module.get(), "rs-start");
  ASSERT_NE(rs, nullptr);
  EXPECT_EQ(GetFrontendAttr(rs, "latency_metadata"), "");
}

TEST_F(CollectiveHintsAnnotatorTest, LatencyMetadataByCollectiveType) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kTwoCollectivesHlo));

  CollectiveHintsConfig config;
  auto* hint = config.add_hints();
  hint->mutable_match()->set_collective_type("all-gather");
  hint->set_latency_metadata("9999");

  CollectiveHintsAnnotatorPass pass(config);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  EXPECT_EQ(GetFrontendAttr(FindInstruction(module.get(), "ag-start"),
                            "latency_metadata"),
            "9999");
  EXPECT_EQ(GetFrontendAttr(FindInstruction(module.get(), "rs-start"),
                            "latency_metadata"),
            "");
}

// ---------------------------------------------------------------------------
// force_earliest
// ---------------------------------------------------------------------------

TEST_F(CollectiveHintsAnnotatorTest, ForceEarliest) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kTwoCollectivesHlo));

  CollectiveHintsConfig config;
  auto* hint = config.add_hints();
  hint->set_name("ag-start");
  hint->set_force_earliest(true);

  CollectiveHintsAnnotatorPass pass(config);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* ag = FindInstruction(module.get(), "ag-start");
  const HloInstruction* rs = FindInstruction(module.get(), "rs-start");
  auto ag_cfg = ag->backend_config<GpuBackendConfig>();
  auto rs_cfg = rs->backend_config<GpuBackendConfig>();
  ASSERT_TRUE(ag_cfg.ok());
  ASSERT_TRUE(rs_cfg.ok());
  EXPECT_TRUE(ag_cfg->force_earliest_schedule());
  EXPECT_FALSE(rs_cfg->force_earliest_schedule());
}

// ---------------------------------------------------------------------------
// scheduling_group_id
// ---------------------------------------------------------------------------

TEST_F(CollectiveHintsAnnotatorTest, SchedulingGroupId) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kTwoCollectivesHlo));

  CollectiveHintsConfig config;
  auto* hint = config.add_hints();
  hint->set_name("rs-start");
  hint->set_scheduling_group_id("42");

  CollectiveHintsAnnotatorPass pass(config);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  EXPECT_EQ(GetFrontendAttr(FindInstruction(module.get(), "rs-start"),
                            "_scheduling_group_id"),
            "42");
  EXPECT_EQ(GetFrontendAttr(FindInstruction(module.get(), "ag-start"),
                            "_scheduling_group_id"),
            "");
}

// HLO with an all-gather-start and a dot that can be paired into a group.
constexpr absl::string_view kAgAndDotHlo = R"(
  HloModule m, is_scheduled=true

  ENTRY main {
    p0 = f32[4] parameter(0)
    p1 = f32[4,4] parameter(1)
    p2 = f32[4,4] parameter(2)
    ag-start = (f32[4], f32[8]) all-gather-start(p0), dimensions={0},
        replica_groups={}
    dot0 = f32[4,4] dot(p1, p2), lhs_contracting_dims={1},
                                 rhs_contracting_dims={0}
    ag-done = f32[8] all-gather-done(ag-start)
    ROOT _ = (f32[8], f32[4,4]) tuple(ag-done, dot0)
  }
)";

TEST_F(CollectiveHintsAnnotatorTest, SchedulingGroupIdPairsComputeOp) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kAgAndDotHlo));

  // Two rules with the same group id: one targets the async start, the other
  // the compute (dot) op. Both should receive _scheduling_group_id.
  CollectiveHintsConfig config;
  auto* ag_hint = config.add_hints();
  ag_hint->set_name("ag-start");
  ag_hint->set_scheduling_group_id("77");
  auto* dot_hint = config.add_hints();
  dot_hint->set_name("dot0");
  dot_hint->set_scheduling_group_id("77");

  CollectiveHintsAnnotatorPass pass(config);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  EXPECT_EQ(GetFrontendAttr(FindInstruction(module.get(), "ag-start"),
                            "_scheduling_group_id"),
            "77");
  EXPECT_EQ(GetFrontendAttr(FindInstruction(module.get(), "dot0"),
                            "_scheduling_group_id"),
            "77");
}

TEST_F(CollectiveHintsAnnotatorTest, LatencyMetadataSkippedOnComputeOp) {
  // A rule targeting a compute op with latency_metadata should *not* write
  // the attribute — the field is only meaningful on async collective starts.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kAgAndDotHlo));

  CollectiveHintsConfig config;
  auto* hint = config.add_hints();
  hint->set_name("dot0");
  hint->set_latency_metadata("5000000");
  hint->set_scheduling_group_id("99");

  CollectiveHintsAnnotatorPass pass(config);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  // Only the group id should land; latency_metadata is silently dropped.
  EXPECT_EQ(GetFrontendAttr(FindInstruction(module.get(), "dot0"),
                            "_scheduling_group_id"),
            "99");
  EXPECT_EQ(GetFrontendAttr(FindInstruction(module.get(), "dot0"),
                            "latency_metadata"),
            "");
}

// stream_id (proto field 6) tests removed when kGpuAsyncStreamCollectives0/1
// were retired. The proto field is retained for back-compat textproto
// parsing but is now a no-op.

// ---------------------------------------------------------------------------
// Priority: higher priority wins regardless of list position.
// ---------------------------------------------------------------------------

// An earlier rule with priority=1 must beat a later rule with priority=0,
// even though list order alone would make the later rule win.
TEST_F(CollectiveHintsAnnotatorTest, HigherPriorityEarlierRuleWins) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kTwoCollectivesHlo));

  CollectiveHintsConfig config;
  // Rule 0 (listed first): catch-all for all-gather, high priority.
  auto* high_prio = config.add_hints();
  high_prio->mutable_match()->set_collective_type("all-gather");
  high_prio->set_latency_metadata("5000");
  high_prio->set_priority(1);

  // Rule 1 (listed second): specific name match, lower priority.
  auto* low_prio = config.add_hints();
  low_prio->set_name("ag-start");
  low_prio->set_latency_metadata("6000");
  low_prio->set_priority(0);

  CollectiveHintsAnnotatorPass pass(config);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  // Rule 0 has higher priority so 5000 wins, despite being listed first.
  EXPECT_EQ(GetFrontendAttr(FindInstruction(module.get(), "ag-start"),
                            "latency_metadata"),
            "5000");
}

// A later rule with priority=1 must beat an earlier rule with priority=0 —
// confirms that explicit priority overrides list-position tie-breaking.
TEST_F(CollectiveHintsAnnotatorTest, HigherPriorityLaterRuleWins) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kTwoCollectivesHlo));

  CollectiveHintsConfig config;
  // Rule 0: catch-all, low priority.
  auto* low_prio = config.add_hints();
  low_prio->mutable_match()->set_collective_type("all-gather");
  low_prio->set_latency_metadata("5000");
  low_prio->set_priority(0);

  // Rule 1: same instruction, higher priority.
  auto* high_prio = config.add_hints();
  high_prio->set_name("ag-start");
  high_prio->set_latency_metadata("6000");
  high_prio->set_priority(1);

  CollectiveHintsAnnotatorPass pass(config);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  // Rule 1 has higher priority so 6000 wins.
  EXPECT_EQ(GetFrontendAttr(FindInstruction(module.get(), "ag-start"),
                            "latency_metadata"),
            "6000");
}

// Priority applies independently per-field: one rule can win one field
// while another wins a different field.
TEST_F(CollectiveHintsAnnotatorTest, HigherPriorityWinsPerFieldIndependently) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kTwoCollectivesHlo));

  CollectiveHintsConfig config;
  // Rule 0: high priority for scheduling_group_id only.
  auto* group_rule = config.add_hints();
  group_rule->mutable_match()->set_collective_type("all-gather");
  group_rule->set_scheduling_group_id("group-A");
  group_rule->set_priority(2);

  // Rule 1: high priority for latency_metadata only.
  auto* latency_rule = config.add_hints();
  latency_rule->mutable_match()->set_collective_type("all-gather");
  latency_rule->set_latency_metadata("9999");
  latency_rule->set_priority(2);

  // Rule 2: low priority, would be overridden by both of the above.
  auto* low = config.add_hints();
  low->set_name("ag-start");
  low->set_scheduling_group_id("group-LOW");
  low->set_latency_metadata("1111");
  low->set_priority(0);

  CollectiveHintsAnnotatorPass pass(config);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* ag = FindInstruction(module.get(), "ag-start");
  // Rules 0 and 1 each win their respective field (priority 2 > priority 0).
  EXPECT_EQ(GetFrontendAttr(ag, "_scheduling_group_id"), "group-A");
  EXPECT_EQ(GetFrontendAttr(ag, "latency_metadata"), "9999");
}

// ---------------------------------------------------------------------------
// Rule merging: later rule overrides earlier for the same field.
// ---------------------------------------------------------------------------

TEST_F(CollectiveHintsAnnotatorTest, LaterRuleOverridesLatencyMetadata) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kTwoCollectivesHlo));

  CollectiveHintsConfig config;
  auto* first = config.add_hints();
  first->mutable_match()->set_collective_type("all-gather");
  first->set_latency_metadata("1000");

  auto* second = config.add_hints();
  second->set_name("ag-start");
  second->set_latency_metadata("2000");

  CollectiveHintsAnnotatorPass pass(config);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  EXPECT_EQ(GetFrontendAttr(FindInstruction(module.get(), "ag-start"),
                            "latency_metadata"),
            "2000");
}

// ---------------------------------------------------------------------------
// No-match: pass returns false and leaves module unchanged.
// ---------------------------------------------------------------------------

TEST_F(CollectiveHintsAnnotatorTest, NoMatchReturnsUnchanged) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kTwoCollectivesHlo));

  CollectiveHintsConfig config;
  auto* hint = config.add_hints();
  hint->set_name("nonexistent-op");
  hint->set_latency_metadata("100");

  CollectiveHintsAnnotatorPass pass(config);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_FALSE(changed);
}

// ---------------------------------------------------------------------------
// Empty config: pass is a no-op.
// ---------------------------------------------------------------------------

TEST_F(CollectiveHintsAnnotatorTest, EmptyConfigIsNoop) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kTwoCollectivesHlo));

  CollectiveHintsAnnotatorPass pass(CollectiveHintsConfig{});
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_FALSE(changed);
}

// ---------------------------------------------------------------------------
// Glob name matching.
// ---------------------------------------------------------------------------

TEST_F(CollectiveHintsAnnotatorTest, GlobPrefixMatchesAgStart) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kTwoCollectivesHlo));

  CollectiveHintsConfig config;
  auto* hint = config.add_hints();
  hint->set_name("ag-*");
  hint->set_latency_metadata("5000");

  CollectiveHintsAnnotatorPass pass(config);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  EXPECT_EQ(GetFrontendAttr(FindInstruction(module.get(), "ag-start"),
                            "latency_metadata"),
            "5000");
  EXPECT_EQ(GetFrontendAttr(FindInstruction(module.get(), "rs-start"),
                            "latency_metadata"),
            "");
}

// ---------------------------------------------------------------------------
// Async-wrapper op_name fall-through
// ---------------------------------------------------------------------------

// HLO with an async-start wrapper whose OUTER instruction has no metadata,
// but whose wrapped reduce-scatter carries an op_name. Exercises the
// annotator's fall-through that lets metadata-based rules target wrapped
// collectives via the inner instruction's op_name.
constexpr absl::string_view kAsyncWrappedRsHlo = R"(
  HloModule m, is_scheduled=true

  add_fn {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT _ = f32[] add(x, y)
  }

  async_computation {
    p = f32[4] parameter(0)
    ROOT inner_rs = f32[2] reduce-scatter(p), dimensions={0},
        to_apply=add_fn, replica_groups={},
        metadata={op_name="my/rs/inner/scope"}
  }

  ENTRY main {
    p0 = f32[4] parameter(0)
    async_rs_start = ((f32[4]), f32[2]) async-start(p0),
        calls=async_computation
    ROOT async_rs_done = f32[2] async-done(async_rs_start)
  }
)";

TEST_F(CollectiveHintsAnnotatorTest, AsyncWrapperInheritsWrappedOpName) {
  // The outer async-start has no op_name of its own, but the wrapped
  // reduce-scatter carries one. A rule with op_name_contains targeting the
  // inner op_name must match the outer wrapper via the fall-through.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kAsyncWrappedRsHlo));

  CollectiveHintsConfig config;
  auto* hint = config.add_hints();
  hint->mutable_match()->set_op_name_contains("my/rs/inner");
  hint->mutable_match()->set_collective_type("reduce-scatter-start");
  hint->set_scheduling_group_id("inherit-ok");

  CollectiveHintsAnnotatorPass pass(config);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  // Rule landed on the outer async-start (what LHS sees), not on the inner
  // reduce-scatter inside the callee computation.
  EXPECT_EQ(GetFrontendAttr(FindInstruction(module.get(), "async_rs_start"),
                            "_scheduling_group_id"),
            "inherit-ok");
}

TEST_F(CollectiveHintsAnnotatorTest, AsyncWrapperOpNameMismatchSkips) {
  // Same HLO, but a rule whose op_name_contains does NOT match the wrapped
  // instruction's op_name. The fall-through must still respect the filter.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kAsyncWrappedRsHlo));

  CollectiveHintsConfig config;
  auto* hint = config.add_hints();
  hint->mutable_match()->set_op_name_contains("some/unrelated/scope");
  hint->mutable_match()->set_collective_type("reduce-scatter-start");
  hint->set_scheduling_group_id("should-not-apply");

  CollectiveHintsAnnotatorPass pass(config);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_FALSE(changed);
  EXPECT_EQ(GetFrontendAttr(FindInstruction(module.get(), "async_rs_start"),
                            "_scheduling_group_id"),
            "");
}

// ---------------------------------------------------------------------------
// match_ordinal
// ---------------------------------------------------------------------------

// HLO with three same-shape all-gather-starts. Used to verify ordinal
// targeting distinguishes otherwise-identical instructions.
constexpr absl::string_view kThreeAgsHlo = R"(
  HloModule m, is_scheduled=true

  ENTRY main {
    p0 = f32[4] parameter(0)
    p1 = f32[4] parameter(1)
    p2 = f32[4] parameter(2)
    ag0-start = (f32[4], f32[8]) all-gather-start(p0), dimensions={0},
        replica_groups={}
    ag1-start = (f32[4], f32[8]) all-gather-start(p1), dimensions={0},
        replica_groups={}
    ag2-start = (f32[4], f32[8]) all-gather-start(p2), dimensions={0},
        replica_groups={}
    ag0-done = f32[8] all-gather-done(ag0-start)
    ag1-done = f32[8] all-gather-done(ag1-start)
    ag2-done = f32[8] all-gather-done(ag2-start)
    ROOT _ = (f32[8], f32[8], f32[8])
        tuple(ag0-done, ag1-done, ag2-done)
  }
)";

TEST_F(CollectiveHintsAnnotatorTest, MatchOrdinalPicksNthMatch) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kThreeAgsHlo));

  // Target the 2nd all-gather-start (ordinal 1) by glob name that matches
  // all three; ordinal filter narrows to exactly one.
  CollectiveHintsConfig config;
  auto* hint = config.add_hints();
  hint->set_name("ag*-start");
  hint->set_match_ordinal(1);
  hint->set_scheduling_group_id("ord1");

  CollectiveHintsAnnotatorPass pass(config);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  EXPECT_EQ(GetFrontendAttr(FindInstruction(module.get(), "ag0-start"),
                            "_scheduling_group_id"),
            "");
  EXPECT_EQ(GetFrontendAttr(FindInstruction(module.get(), "ag1-start"),
                            "_scheduling_group_id"),
            "ord1");
  EXPECT_EQ(GetFrontendAttr(FindInstruction(module.get(), "ag2-start"),
                            "_scheduling_group_id"),
            "");
}

TEST_F(CollectiveHintsAnnotatorTest, MatchOrdinalZeroMatchesFirstOnly) {
  // Sanity: ordinal 0 must pick only the first match (proto3 optional makes
  // 0 a legal value distinct from "unset").
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kThreeAgsHlo));

  CollectiveHintsConfig config;
  auto* hint = config.add_hints();
  hint->set_name("ag*-start");
  hint->set_match_ordinal(0);
  hint->set_scheduling_group_id("ord0");

  CollectiveHintsAnnotatorPass pass(config);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  EXPECT_EQ(GetFrontendAttr(FindInstruction(module.get(), "ag0-start"),
                            "_scheduling_group_id"),
            "ord0");
  EXPECT_EQ(GetFrontendAttr(FindInstruction(module.get(), "ag1-start"),
                            "_scheduling_group_id"),
            "");
  EXPECT_EQ(GetFrontendAttr(FindInstruction(module.get(), "ag2-start"),
                            "_scheduling_group_id"),
            "");
}

TEST_F(CollectiveHintsAnnotatorTest, MatchOrdinalUnsetMatchesAll) {
  // Baseline: without ordinal, every match gets the hint.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kThreeAgsHlo));

  CollectiveHintsConfig config;
  auto* hint = config.add_hints();
  hint->set_name("ag*-start");
  // match_ordinal intentionally left unset.
  hint->set_scheduling_group_id("all");

  CollectiveHintsAnnotatorPass pass(config);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  for (absl::string_view n : {"ag0-start", "ag1-start", "ag2-start"}) {
    EXPECT_EQ(GetFrontendAttr(FindInstruction(module.get(), std::string(n)),
                              "_scheduling_group_id"),
              "all")
        << "on " << n;
  }
}

TEST_F(CollectiveHintsAnnotatorTest, MatchOrdinalCountsPerRule) {
  // Two rules matching the same three instructions; each with a different
  // ordinal targeting a different element. Counters are per-rule.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kThreeAgsHlo));

  CollectiveHintsConfig config;
  auto* r0 = config.add_hints();
  r0->set_name("ag*-start");
  r0->set_match_ordinal(0);
  r0->set_scheduling_group_id("rule0-hit0");
  auto* r2 = config.add_hints();
  r2->set_name("ag*-start");
  r2->set_match_ordinal(2);
  r2->set_scheduling_group_id("rule1-hit2");

  CollectiveHintsAnnotatorPass pass(config);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  EXPECT_EQ(GetFrontendAttr(FindInstruction(module.get(), "ag0-start"),
                            "_scheduling_group_id"),
            "rule0-hit0");
  EXPECT_EQ(GetFrontendAttr(FindInstruction(module.get(), "ag1-start"),
                            "_scheduling_group_id"),
            "");
  EXPECT_EQ(GetFrontendAttr(FindInstruction(module.get(), "ag2-start"),
                            "_scheduling_group_id"),
            "rule1-hit2");
}

// ---------------------------------------------------------------------------
// sequence_id (cross-batch control dependencies)
// ---------------------------------------------------------------------------

TEST_F(CollectiveHintsAnnotatorTest, SequenceIdAddsCrossBatchControlDeps) {
  // Two independent collectives in the same computation, one in batch 1 and
  // one in batch 2. After the pass, the later batch's instruction must have
  // the earlier batch's -done (or the instruction itself) as a control
  // predecessor.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kThreeAgsHlo));
  // kThreeAgsHlo declares `is_scheduled=true` with a specific order that
  // places ag0-done after ag2-start.  In production the annotator runs before
  // LHS produces a schedule; in this test we clear the parsed schedule so
  // adding a control dep ag0-done → ag2-start doesn't fail a schedule
  // consistency check inside the pass.
  module->clear_schedule();

  CollectiveHintsConfig config;
  auto* h0 = config.add_hints();
  h0->set_name("ag0-start");
  h0->set_sequence_id(1);
  h0->set_scheduling_group_id("b1");
  auto* h1 = config.add_hints();
  h1->set_name("ag2-start");
  h1->set_sequence_id(2);
  h1->set_scheduling_group_id("b2");

  CollectiveHintsAnnotatorPass pass(config);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* ag2_start =
      FindInstruction(module.get(), "ag2-start");
  const HloInstruction* ag0_done =
      FindInstruction(module.get(), "ag0-done");
  // ag0-done should be in ag2-start's control predecessors because batch 1's
  // async completes at its -done, and batch 2 must wait for batch 1.
  bool found = false;
  for (const HloInstruction* p : ag2_start->control_predecessors()) {
    if (p == ag0_done) { found = true; break; }
  }
  EXPECT_TRUE(found)
      << "ag2-start should have ag0-done as a control predecessor";
}

TEST_F(CollectiveHintsAnnotatorTest, SameSequenceIdNoIntraBatchControlDep) {
  // Two collectives in the same batch. After the pass, neither should have
  // acquired a control dep on the other — batches run concurrently.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kThreeAgsHlo));

  CollectiveHintsConfig config;
  auto* h0 = config.add_hints();
  h0->set_name("ag0-start");
  h0->set_sequence_id(1);
  h0->set_scheduling_group_id("b1a");
  auto* h1 = config.add_hints();
  h1->set_name("ag1-start");
  h1->set_sequence_id(1);   // same batch
  h1->set_scheduling_group_id("b1b");

  CollectiveHintsAnnotatorPass pass(config);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* ag0_start = FindInstruction(module.get(), "ag0-start");
  const HloInstruction* ag1_start = FindInstruction(module.get(), "ag1-start");
  // Neither should have the other in its control predecessors.
  for (const HloInstruction* p : ag1_start->control_predecessors()) {
    EXPECT_NE(p, ag0_start) << "unexpected intra-batch control dep";
  }
  for (const HloInstruction* p : ag0_start->control_predecessors()) {
    EXPECT_NE(p, ag1_start) << "unexpected intra-batch control dep";
  }
}

TEST_F(CollectiveHintsAnnotatorTest, SequenceIdSkipsComputeOnlyDeps) {
  // Two collectives in different sequence_id batches, each paired with a
  // compute-op anchor (a dot). After the pass, async→async, async→compute,
  // and compute→async cross-batch control deps should all be added, but the
  // compute→compute dep (dot0 → dot1) must be skipped — those anchors are
  // already tethered to their collectives via scheduling_group_id, so their
  // order is implied by the async ordering, and adding the explicit
  // compute→compute dep has caused post-schedule RET_CHECK failures in
  // production when LHS's group-id pull flipped the anchors' relative order.
  constexpr absl::string_view kAgAndTwoDotsHlo = R"(
    HloModule m
    ENTRY main {
      p0 = f32[4] parameter(0)
      p1 = f32[4,4] parameter(1)
      p2 = f32[4,4] parameter(2)
      p3 = f32[4,4] parameter(3)
      p4 = f32[4,4] parameter(4)
      ag0-start = (f32[4], f32[8]) all-gather-start(p0), dimensions={0},
          replica_groups={}
      dot0 = f32[4,4] dot(p1, p2), lhs_contracting_dims={1},
                                   rhs_contracting_dims={0}
      ag0-done = f32[8] all-gather-done(ag0-start)
      ag1-start = (f32[8], f32[16]) all-gather-start(ag0-done), dimensions={0},
          replica_groups={}
      dot1 = f32[4,4] dot(p3, p4), lhs_contracting_dims={1},
                                   rhs_contracting_dims={0}
      ag1-done = f32[16] all-gather-done(ag1-start)
      ROOT _ = (f32[16], f32[4,4], f32[4,4]) tuple(ag1-done, dot0, dot1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kAgAndTwoDotsHlo));

  CollectiveHintsConfig config;
  // batch 1: ag0-start + dot0
  auto* h0_ag = config.add_hints();
  h0_ag->set_name("ag0-start");
  h0_ag->set_sequence_id(1);
  h0_ag->set_scheduling_group_id("b1");
  auto* h0_dot = config.add_hints();
  h0_dot->set_name("dot0");
  h0_dot->set_sequence_id(1);
  h0_dot->set_scheduling_group_id("b1");
  // batch 2: ag1-start + dot1
  auto* h1_ag = config.add_hints();
  h1_ag->set_name("ag1-start");
  h1_ag->set_sequence_id(2);
  h1_ag->set_scheduling_group_id("b2");
  auto* h1_dot = config.add_hints();
  h1_dot->set_name("dot1");
  h1_dot->set_sequence_id(2);
  h1_dot->set_scheduling_group_id("b2");

  CollectiveHintsAnnotatorPass pass(config);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* dot0 = FindInstruction(module.get(), "dot0");
  const HloInstruction* dot1 = FindInstruction(module.get(), "dot1");
  const HloInstruction* ag0_done = FindInstruction(module.get(), "ag0-done");
  const HloInstruction* ag1_start = FindInstruction(module.get(), "ag1-start");

  // compute → compute must NOT be added.
  for (const HloInstruction* p : dot1->control_predecessors()) {
    EXPECT_NE(p, dot0)
        << "compute-anchor → compute-anchor dep should have been skipped";
  }

  // async → compute should still be added.
  bool async_to_compute = false;
  for (const HloInstruction* p : dot1->control_predecessors()) {
    if (p == ag0_done) { async_to_compute = true; break; }
  }
  EXPECT_TRUE(async_to_compute)
      << "async-done → compute-anchor dep should still be added";

  // compute → async should still be added.
  bool compute_to_async = false;
  for (const HloInstruction* p : ag1_start->control_predecessors()) {
    if (p == dot0) { compute_to_async = true; break; }
  }
  EXPECT_TRUE(compute_to_async)
      << "compute-anchor → async-start dep should still be added";

  // async → async baseline (already exercised by SequenceIdAddsCrossBatchControlDeps,
  // but re-assert here so the test documents the full behavior matrix).
  bool async_to_async = false;
  for (const HloInstruction* p : ag1_start->control_predecessors()) {
    if (p == ag0_done) { async_to_async = true; break; }
  }
  EXPECT_TRUE(async_to_async)
      << "async-done → async-start dep should still be added";
}

TEST_F(CollectiveHintsAnnotatorTest, SequenceIdDetectsCycle) {
  // Construct a module where the "later" batch's target is a dataflow
  // *predecessor* of the "earlier" batch's target. The control dep the pass
  // wants to add (later-batch becomes successor of earlier-batch) closes a
  // cycle and must be rejected with an error.
  //
  // Module: ag1-start consumes ag0-done as its operand (via a bitcast).
  // So ag0-done ← (bitcast) ← ag1-start is the data flow; ag0 comes before
  // ag1 naturally. If we assign ag1 to batch 1 (earlier) and ag0 to batch 2
  // (later), the pass would want to add a control dep from ag1 to ag0
  // (ag1.done -> ag0.start), which reverses the dataflow and creates a
  // cycle.
  constexpr absl::string_view kChainedHlo = R"(
    HloModule m, is_scheduled=true
    ENTRY main {
      p0 = f32[4] parameter(0)
      ag0-start = (f32[4], f32[8]) all-gather-start(p0), dimensions={0},
          replica_groups={}
      ag0-done = f32[8] all-gather-done(ag0-start)
      reshape = f32[4,2] reshape(ag0-done)
      ag1-start = (f32[4,2], f32[8,2]) all-gather-start(reshape),
          dimensions={0}, replica_groups={}
      ROOT ag1-done = f32[8,2] all-gather-done(ag1-start)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kChainedHlo));

  CollectiveHintsConfig config;
  auto* h1 = config.add_hints();
  h1->set_name("ag1-start");
  h1->set_sequence_id(1);   // earlier batch
  h1->set_scheduling_group_id("earlier");
  auto* h0 = config.add_hints();
  h0->set_name("ag0-start");
  h0->set_sequence_id(2);   // later batch, but this is the dataflow predecessor
  h0->set_scheduling_group_id("later");

  CollectiveHintsAnnotatorPass pass(config);
  auto status_or = pass.Run(module.get());
  ASSERT_FALSE(status_or.ok());
  EXPECT_EQ(status_or.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(status_or.status().message(),
              ::testing::HasSubstr("sequence_id ordering would introduce a "
                                   "cycle"));
}

// ---------------------------------------------------------------------------
// All hints applied together on one instruction.
// ---------------------------------------------------------------------------

TEST_F(CollectiveHintsAnnotatorTest, AllHintsAppliedTogether) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kTwoCollectivesHlo));

  CollectiveHintsConfig config;
  auto* hint = config.add_hints();
  hint->set_name("ag-start");
  hint->set_latency_metadata("5000");
  hint->set_force_earliest(true);
  hint->set_scheduling_group_id("7");

  CollectiveHintsAnnotatorPass pass(config);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* ag = FindInstruction(module.get(), "ag-start");
  EXPECT_EQ(GetFrontendAttr(ag, "latency_metadata"), "5000");
  auto ag_cfg = ag->backend_config<GpuBackendConfig>();
  ASSERT_TRUE(ag_cfg.ok());
  EXPECT_TRUE(ag_cfg->force_earliest_schedule());
  EXPECT_EQ(GetFrontendAttr(ag, "_scheduling_group_id"), "7");
}

// ---------------------------------------------------------------------------
// window_target — pin compute inside collective [start, done] window via
// control deps. See comment on `window_target` in collective_hints.proto.
// ---------------------------------------------------------------------------

namespace {

// Returns true iff `pred` is one of `succ`'s control predecessors.
bool HasControlPred(const HloInstruction* succ, const HloInstruction* pred) {
  for (const HloInstruction* p : succ->control_predecessors()) {
    if (p == pred) return true;
  }
  return false;
}

}  // namespace

// HLO with one collective (ag-start) and a clearly-independent compute op
// (the dot, which doesn't read or feed the gather). Used as the canonical
// "compute belongs in window" target for the window_target tests below.
constexpr absl::string_view kAgIndependentDotHlo = R"(
  HloModule m, is_scheduled=true

  ENTRY main {
    p0 = f32[4] parameter(0)
    p1 = f32[4,4] parameter(1)
    p2 = f32[4,4] parameter(2)
    ag-start = (f32[4], f32[8]) all-gather-start(p0), dimensions={0},
        replica_groups={}
    dot0 = f32[4,4] dot(p1, p2), lhs_contracting_dims={1},
                                 rhs_contracting_dims={0}
    ag-done = f32[8] all-gather-done(ag-start)
    ROOT _ = (f32[8], f32[4,4]) tuple(ag-done, dot0)
  }
)";

// Happy path: compute is dataflow-independent of target; control deps get
// added to pin it inside [ag-start, ag-done]; visibility FE attribute is set.
TEST_F(CollectiveHintsAnnotatorTest, WindowTargetSingleHappyPath) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kAgIndependentDotHlo));

  CollectiveHintsConfig config;
  auto* hint = config.add_hints();
  hint->set_name("dot0");
  hint->add_window_target("ag-start");

  CollectiveHintsAnnotatorPass pass(config);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* dot = FindInstruction(module.get(), "dot0");
  const HloInstruction* ag_start = FindInstruction(module.get(), "ag-start");
  const HloInstruction* ag_done = FindInstruction(module.get(), "ag-done");
  ASSERT_NE(dot, nullptr);
  ASSERT_NE(ag_start, nullptr);
  ASSERT_NE(ag_done, nullptr);

  // ag-start -> dot0 (dot must run AFTER start).
  EXPECT_TRUE(HasControlPred(dot, ag_start));
  // dot0 -> ag-done (dot must run BEFORE done).
  EXPECT_TRUE(HasControlPred(ag_done, dot));
  // Visibility FE attribute set with the (single) target name.
  EXPECT_EQ(GetFrontendAttr(dot, "_xla_window_target"), "ag-start");
}

// Multi-target: rejected with log+skip in Phase 1. No control deps added,
// no FE attribute set, pass returns changed=false (no other annotations).
TEST_F(CollectiveHintsAnnotatorTest, WindowTargetMultipleRejectedInPhase1) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kTwoCollectivesHlo));

  CollectiveHintsConfig config;
  auto* hint = config.add_hints();
  hint->set_name("add0");
  hint->add_window_target("ag-start");
  hint->add_window_target("rs-start");

  CollectiveHintsAnnotatorPass pass(config);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_FALSE(changed);  // multi-target ignored, no other hint set

  const HloInstruction* add = FindInstruction(module.get(), "add0");
  ASSERT_NE(add, nullptr);
  EXPECT_EQ(GetFrontendAttr(add, "_xla_window_target"), "");
  EXPECT_EQ(add->control_predecessors().size(), 0);
  EXPECT_EQ(add->control_successors().size(), 0);
}

// window_target on an async-collective-start itself — meaningless. Skipped
// with a log; no FE attr, no control deps.
TEST_F(CollectiveHintsAnnotatorTest, WindowTargetOnAsyncStartSkipped) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kTwoCollectivesHlo));

  CollectiveHintsConfig config;
  auto* hint = config.add_hints();
  hint->set_name("ag-start");
  hint->add_window_target("rs-start");

  CollectiveHintsAnnotatorPass pass(config);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_FALSE(changed);

  const HloInstruction* ag = FindInstruction(module.get(), "ag-start");
  EXPECT_EQ(GetFrontendAttr(ag, "_xla_window_target"), "");
  EXPECT_EQ(ag->control_predecessors().size(), 0);
  EXPECT_EQ(ag->control_successors().size(), 0);
}

// Target doesn't exist — control deps NOT added, but FE attribute IS set
// (visibility for downstream debug). Pass returns changed=true because the
// FE attribute is a real change.
TEST_F(CollectiveHintsAnnotatorTest, WindowTargetMissingTargetSkipsControlDeps) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kAgIndependentDotHlo));

  CollectiveHintsConfig config;
  auto* hint = config.add_hints();
  hint->set_name("dot0");
  hint->add_window_target("does-not-exist");

  CollectiveHintsAnnotatorPass pass(config);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);  // FE attribute was set, even though deps were skipped

  const HloInstruction* dot = FindInstruction(module.get(), "dot0");
  EXPECT_EQ(GetFrontendAttr(dot, "_xla_window_target"), "does-not-exist");
  EXPECT_EQ(dot->control_predecessors().size(), 0);
  EXPECT_EQ(dot->control_successors().size(), 0);
}

// Target is a non-async instruction — skipped (control deps not added).
TEST_F(CollectiveHintsAnnotatorTest, WindowTargetNonAsyncTargetSkipped) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kAgIndependentDotHlo));

  CollectiveHintsConfig config;
  auto* hint = config.add_hints();
  hint->set_name("dot0");
  hint->add_window_target("ag-done");  // -done isn't async-start

  CollectiveHintsAnnotatorPass pass(config);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);  // FE attr set; control deps skipped

  const HloInstruction* dot = FindInstruction(module.get(), "dot0");
  EXPECT_EQ(GetFrontendAttr(dot, "_xla_window_target"), "ag-done");
  EXPECT_EQ(dot->control_predecessors().size(), 0);
  EXPECT_EQ(dot->control_successors().size(), 0);
}

// Compute is in the target's PRED closure — the dot's output feeds (via
// reshape) the all-gather's input. So pre_dot is a dataflow PREDECESSOR
// of ag-start. Adding `ag-start -> pre_dot` would close a cycle.
// Annotator must skip+log.
constexpr absl::string_view kAgComputeIsPredHlo = R"(
  HloModule m, is_scheduled=true

  ENTRY main {
    p0 = f32[4,4] parameter(0)
    p1 = f32[4,4] parameter(1)
    pre_dot = f32[4,4] dot(p0, p1), lhs_contracting_dims={1},
                                    rhs_contracting_dims={0}
    pre_flat = f32[16] reshape(pre_dot)
    ag-start = (f32[16], f32[32]) all-gather-start(pre_flat), dimensions={0},
        replica_groups={}
    ag-done = f32[32] all-gather-done(ag-start)
    ROOT _ = f32[32] negate(ag-done)
  }
)";

TEST_F(CollectiveHintsAnnotatorTest, WindowTargetCycleViaPredClosureSkipped) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kAgComputeIsPredHlo));

  CollectiveHintsConfig config;
  auto* hint = config.add_hints();
  hint->set_name("pre_dot");
  hint->add_window_target("ag-start");

  CollectiveHintsAnnotatorPass pass(config);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);  // FE attr set; control deps skipped

  const HloInstruction* pre_dot = FindInstruction(module.get(), "pre_dot");
  EXPECT_EQ(GetFrontendAttr(pre_dot, "_xla_window_target"), "ag-start");
  // No control deps added — would have cycled (pre_dot is upstream of
  // ag-start through reshape).
  EXPECT_EQ(pre_dot->control_predecessors().size(), 0);
  EXPECT_EQ(pre_dot->control_successors().size(), 0);
}

// Compute is in the target's SUCC closure — a dot consumes the gather's
// result. Annotator must skip+log because adding `compute -> ag-done`
// would close a cycle (ag-done -> dot via dataflow already).
constexpr absl::string_view kAgComputeIsSuccHlo = R"(
  HloModule m, is_scheduled=true

  ENTRY main {
    p0 = f32[4] parameter(0)
    p1 = f32[8,4] parameter(1)
    ag-start = (f32[4], f32[8]) all-gather-start(p0), dimensions={0},
        replica_groups={}
    ag-done = f32[8] all-gather-done(ag-start)
    post_dot = f32[4] dot(ag-done, p1), lhs_contracting_dims={0},
                                        rhs_contracting_dims={0}
    ROOT _ = f32[4] add(post_dot, p0)
  }
)";

TEST_F(CollectiveHintsAnnotatorTest, WindowTargetCycleViaSuccClosureSkipped) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kAgComputeIsSuccHlo));

  CollectiveHintsConfig config;
  auto* hint = config.add_hints();
  hint->set_name("post_dot");
  hint->add_window_target("ag-start");

  CollectiveHintsAnnotatorPass pass(config);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);  // FE attr set; control deps skipped

  const HloInstruction* post = FindInstruction(module.get(), "post_dot");
  EXPECT_EQ(GetFrontendAttr(post, "_xla_window_target"), "ag-start");
  EXPECT_EQ(post->control_predecessors().size(), 0);
  EXPECT_EQ(post->control_successors().size(), 0);
}

// Cross-computation: target lives in a different HloComputation than the
// matched compute. Treated as "target not found" within the compute's
// computation (linear scan limited to compute->parent()), so skip+log.
constexpr absl::string_view kCrossCompHlo = R"(
  HloModule m, is_scheduled=true

  body {
    bp0 = f32[4,4] parameter(0)
    bp1 = f32[4,4] parameter(1)
    ROOT body_dot = f32[4,4] dot(bp0, bp1), lhs_contracting_dims={1},
                                            rhs_contracting_dims={0}
  }

  ENTRY main {
    p0 = f32[4] parameter(0)
    p1 = f32[4,4] parameter(1)
    p2 = f32[4,4] parameter(2)
    ag-start = (f32[4], f32[8]) all-gather-start(p0), dimensions={0},
        replica_groups={}
    main_dot = f32[4,4] call(p1, p2), to_apply=body
    ag-done = f32[8] all-gather-done(ag-start)
    ROOT _ = (f32[8], f32[4,4]) tuple(ag-done, main_dot)
  }
)";

TEST_F(CollectiveHintsAnnotatorTest, WindowTargetCrossCompSkipped) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kCrossCompHlo));

  // Try to pin body_dot (in the `body` computation) into ag-start's window
  // (in `main`). Different computations → annotator can't find the target
  // in body_dot's parent, skip.
  CollectiveHintsConfig config;
  auto* hint = config.add_hints();
  hint->set_name("body_dot");
  hint->add_window_target("ag-start");

  CollectiveHintsAnnotatorPass pass(config);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);  // FE attr set; control deps skipped

  const HloInstruction* body_dot = FindInstruction(module.get(), "body_dot");
  EXPECT_EQ(GetFrontendAttr(body_dot, "_xla_window_target"), "ag-start");
  EXPECT_EQ(body_dot->control_predecessors().size(), 0);
  EXPECT_EQ(body_dot->control_successors().size(), 0);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
