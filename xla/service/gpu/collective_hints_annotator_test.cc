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

// ---------------------------------------------------------------------------
// stream_id
// ---------------------------------------------------------------------------

TEST_F(CollectiveHintsAnnotatorTest, StreamIdByName) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kTwoCollectivesHlo));

  CollectiveHintsConfig config;
  auto* hint = config.add_hints();
  hint->set_name("ag-start");
  hint->set_stream_id(5);

  CollectiveHintsAnnotatorPass pass(config);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  EXPECT_EQ(GetFrontendAttr(FindInstruction(module.get(), "ag-start"),
                            "_xla_stream_annotation"),
            "5");
  // rs-start should not have stream annotation set.
  EXPECT_EQ(GetFrontendAttr(FindInstruction(module.get(), "rs-start"),
                            "_xla_stream_annotation"),
            "");
}

TEST_F(CollectiveHintsAnnotatorTest, StreamIdByCollectiveType) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kTwoCollectivesHlo));

  CollectiveHintsConfig config;
  auto* ag_hint = config.add_hints();
  ag_hint->mutable_match()->set_collective_type("all-gather");
  ag_hint->set_stream_id(5);

  auto* rs_hint = config.add_hints();
  rs_hint->mutable_match()->set_collective_type("reduce-scatter");
  rs_hint->set_stream_id(6);

  CollectiveHintsAnnotatorPass pass(config);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  EXPECT_EQ(GetFrontendAttr(FindInstruction(module.get(), "ag-start"),
                            "_xla_stream_annotation"),
            "5");
  EXPECT_EQ(GetFrontendAttr(FindInstruction(module.get(), "rs-start"),
                            "_xla_stream_annotation"),
            "6");
}

TEST_F(CollectiveHintsAnnotatorTest, StreamIdZeroIsIgnored) {
  // stream_id=0 is the proto default (unset) and must not inject the attribute.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kTwoCollectivesHlo));

  CollectiveHintsConfig config;
  auto* hint = config.add_hints();
  hint->set_name("ag-start");
  hint->set_stream_id(0);

  CollectiveHintsAnnotatorPass pass(config);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  // No attribute was set, so the pass should report no change.
  EXPECT_FALSE(changed);

  EXPECT_EQ(GetFrontendAttr(FindInstruction(module.get(), "ag-start"),
                            "_xla_stream_annotation"),
            "");
}

// ---------------------------------------------------------------------------
// Priority: higher priority wins regardless of list position.
// ---------------------------------------------------------------------------

// An earlier rule with priority=1 must beat a later rule with priority=0,
// even though list order alone would make the later rule win.
TEST_F(CollectiveHintsAnnotatorTest, HigherPriorityEarlierRuleWinsStreamId) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kTwoCollectivesHlo));

  CollectiveHintsConfig config;
  // Rule 0 (listed first): catch-all for all-gather, high priority.
  auto* high_prio = config.add_hints();
  high_prio->mutable_match()->set_collective_type("all-gather");
  high_prio->set_stream_id(5);
  high_prio->set_priority(1);

  // Rule 1 (listed second): specific name match, lower priority.
  auto* low_prio = config.add_hints();
  low_prio->set_name("ag-start");
  low_prio->set_stream_id(6);
  low_prio->set_priority(0);

  CollectiveHintsAnnotatorPass pass(config);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  // Rule 0 has higher priority so stream_id=5 wins, despite being listed first.
  EXPECT_EQ(GetFrontendAttr(FindInstruction(module.get(), "ag-start"),
                            "_xla_stream_annotation"),
            "5");
}

// A later rule with priority=1 must beat an earlier rule with priority=0 —
// confirms that explicit priority overrides list-position tie-breaking.
TEST_F(CollectiveHintsAnnotatorTest, HigherPriorityLaterRuleWinsStreamId) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kTwoCollectivesHlo));

  CollectiveHintsConfig config;
  // Rule 0: catch-all, low priority.
  auto* low_prio = config.add_hints();
  low_prio->mutable_match()->set_collective_type("all-gather");
  low_prio->set_stream_id(5);
  low_prio->set_priority(0);

  // Rule 1: same instruction, higher priority.
  auto* high_prio = config.add_hints();
  high_prio->set_name("ag-start");
  high_prio->set_stream_id(6);
  high_prio->set_priority(1);

  CollectiveHintsAnnotatorPass pass(config);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  // Rule 1 has higher priority so stream_id=6 wins.
  EXPECT_EQ(GetFrontendAttr(FindInstruction(module.get(), "ag-start"),
                            "_xla_stream_annotation"),
            "6");
}

// Priority applies independently per-field: one rule can win stream_id while
// another wins latency_metadata.
TEST_F(CollectiveHintsAnnotatorTest, HigherPriorityWinsPerFieldIndependently) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kTwoCollectivesHlo));

  CollectiveHintsConfig config;
  // Rule 0: high priority for stream_id only.
  auto* stream_rule = config.add_hints();
  stream_rule->mutable_match()->set_collective_type("all-gather");
  stream_rule->set_stream_id(5);
  stream_rule->set_priority(2);

  // Rule 1: high priority for latency_metadata only.
  auto* latency_rule = config.add_hints();
  latency_rule->mutable_match()->set_collective_type("all-gather");
  latency_rule->set_latency_metadata("9999");
  latency_rule->set_priority(2);

  // Rule 2: low priority, would be overridden by both of the above.
  auto* low = config.add_hints();
  low->set_name("ag-start");
  low->set_stream_id(7);
  low->set_latency_metadata("1111");
  low->set_priority(0);

  CollectiveHintsAnnotatorPass pass(config);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* ag = FindInstruction(module.get(), "ag-start");
  // Rules 0 and 1 each win their respective field (priority 2 > priority 0).
  // Between rules 0 and 1 (equal priority, listed in that order), rule 1 wins
  // latency_metadata because it comes later; stream_id is only set by rule 0.
  EXPECT_EQ(GetFrontendAttr(ag, "_xla_stream_annotation"), "5");
  EXPECT_EQ(GetFrontendAttr(ag, "latency_metadata"), "9999");
}

// ---------------------------------------------------------------------------
// Rule merging: later rule overrides earlier for the same field.
// ---------------------------------------------------------------------------

TEST_F(CollectiveHintsAnnotatorTest, LaterRuleOverridesStreamId) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kTwoCollectivesHlo));

  CollectiveHintsConfig config;
  // First rule sets stream_id=5 for all collectives.
  auto* base = config.add_hints();
  base->mutable_match()->set_collective_type("all-gather");
  base->set_stream_id(5);

  // Second rule overrides to stream_id=6 for the same instruction.
  auto* override_hint = config.add_hints();
  override_hint->set_name("ag-start");
  override_hint->set_stream_id(6);

  CollectiveHintsAnnotatorPass pass(config);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  // Second rule wins: stream 6.
  EXPECT_EQ(GetFrontendAttr(FindInstruction(module.get(), "ag-start"),
                            "_xla_stream_annotation"),
            "6");
}

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
  hint->set_stream_id(5);

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
  hint->set_stream_id(5);

  CollectiveHintsAnnotatorPass pass(config);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  EXPECT_EQ(GetFrontendAttr(FindInstruction(module.get(), "ag-start"),
                            "_xla_stream_annotation"),
            "5");
  EXPECT_EQ(GetFrontendAttr(FindInstruction(module.get(), "rs-start"),
                            "_xla_stream_annotation"),
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
  hint->set_stream_id(5);

  CollectiveHintsAnnotatorPass pass(config);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* ag = FindInstruction(module.get(), "ag-start");
  EXPECT_EQ(GetFrontendAttr(ag, "latency_metadata"), "5000");
  auto ag_cfg = ag->backend_config<GpuBackendConfig>();
  ASSERT_TRUE(ag_cfg.ok());
  EXPECT_TRUE(ag_cfg->force_earliest_schedule());
  EXPECT_EQ(GetFrontendAttr(ag, "_scheduling_group_id"), "7");
  EXPECT_EQ(GetFrontendAttr(ag, "_xla_stream_annotation"), "5");
}

}  // namespace
}  // namespace gpu
}  // namespace xla
