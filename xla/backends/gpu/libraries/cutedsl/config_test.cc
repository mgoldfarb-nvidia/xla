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

#include "xla/backends/gpu/libraries/cutedsl/config.h"

#include <cstdint>
#include <limits>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/libraries/cutedsl/config.pb.h"
#include "xla/tsl/platform/status_matchers.h"

namespace xla::gpu::cutedsl {
namespace wire = ::xla::gpu::cutedsl::proto;

namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::tsl::testing::IsOk;
using ::tsl::testing::StatusIs;

wire::CollectiveCallConfigV3Wire TestWireConfig() {
  wire::CollectiveCallConfigV3Wire config;
  config.set_abi_clique_size(2);
  config.set_barrier_before_launch(true);

  wire::CollectiveGroupWire* group = config.mutable_group();
  group->set_mode(wire::CollectiveGroupWire::MODE_CROSS_REPLICA);
  group->set_communication_id(17);
  group->add_replica_groups()->add_members(0);
  group->mutable_replica_groups(0)->add_members(1);

  wire::PeerRegionWire* region = config.add_peer_regions();
  region->set_endpoint(wire::PeerRegionWire::ENDPOINT_ARGUMENT);
  region->set_buffer_index(2);
  region->set_byte_offset(16);
  region->set_byte_size(64);
  region->set_required_alignment(16);
  region->set_memory_kind(wire::PeerRegionWire::MEMORY_KIND_SYMMETRIC);
  return config;
}

absl::StatusOr<wire::CollectiveCallConfigV3> Parse(
    const wire::CollectiveCallConfigV3Wire& config) {
  return ParseAndValidateCollectiveCallConfig(config.SerializeAsString());
}

TEST(CollectiveConfigTest, ParsesPublicProtobufWireFormat) {
  absl::StatusOr<wire::CollectiveCallConfigV3> parsed = Parse(TestWireConfig());
  ASSERT_THAT(parsed, IsOk());

  EXPECT_EQ(parsed->abi_clique_size(), 2);
  EXPECT_EQ(parsed->group_mode(),
            CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA);
  EXPECT_EQ(parsed->communication_id(), 17);
  ASSERT_EQ(parsed->replica_groups_size(), 1);
  EXPECT_THAT(parsed->replica_groups(0).replica_ids(), ElementsAre(0, 1));
  ASSERT_EQ(parsed->peer_regions_size(), 1);
  EXPECT_EQ(parsed->peer_regions(0).endpoint(),
            wire::PEER_REGION_ENDPOINT_PROTO_ARGUMENT);
  EXPECT_EQ(parsed->peer_regions(0).memory_kind(),
            wire::PEER_MEMORY_KIND_PROTO_SYMMETRIC);
  EXPECT_TRUE(parsed->barrier_before_launch());
}

TEST(CollectiveConfigTest, ParsesEmbeddedNulBytes) {
  std::string serialized = TestWireConfig().SerializeAsString();
  ASSERT_NE(serialized.find('\0'), std::string::npos);
  EXPECT_THAT(ParseAndValidateCollectiveCallConfig(serialized), IsOk());
}

TEST(CollectiveConfigTest, RejectsMalformedProtobufAndMissingFields) {
  EXPECT_THAT(ParseAndValidateCollectiveCallConfig("\xff"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Failed to parse")));

  wire::CollectiveCallConfigV3Wire config = TestWireConfig();
  config.clear_abi_clique_size();
  EXPECT_THAT(Parse(config), StatusIs(absl::StatusCode::kInvalidArgument,
                                      HasSubstr("abi_clique_size")));

  config = TestWireConfig();
  config.mutable_peer_regions(0)->clear_required_alignment();
  EXPECT_THAT(Parse(config), StatusIs(absl::StatusCode::kInvalidArgument,
                                      HasSubstr("missing a required field")));
}

TEST(CollectiveConfigTest, ValidatesGroupFields) {
  wire::CollectiveCallConfigV3Wire config = TestWireConfig();
  config.mutable_group()->set_mode(wire::CollectiveGroupWire::MODE_UNSPECIFIED);
  EXPECT_THAT(Parse(config), StatusIs(absl::StatusCode::kInvalidArgument,
                                      HasSubstr("Unsupported collective")));

  config = TestWireConfig();
  config.mutable_group()->set_communication_id(
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) + 1);
  EXPECT_THAT(Parse(config), StatusIs(absl::StatusCode::kInvalidArgument,
                                      HasSubstr("communication_id")));

  config = TestWireConfig();
  config.mutable_group()->mutable_replica_groups(0)->clear_members();
  EXPECT_THAT(Parse(config), StatusIs(absl::StatusCode::kInvalidArgument,
                                      HasSubstr("must not be empty")));
}

TEST(CollectiveConfigTest, ValidatesPeerRegions) {
  wire::CollectiveCallConfigV3Wire config = TestWireConfig();
  config.mutable_peer_regions(0)->set_byte_size(0);
  EXPECT_THAT(Parse(config), StatusIs(absl::StatusCode::kInvalidArgument,
                                      HasSubstr("Invalid or overflowing")));

  config = TestWireConfig();
  config.mutable_peer_regions(0)->set_required_alignment(3);
  EXPECT_THAT(Parse(config), StatusIs(absl::StatusCode::kInvalidArgument,
                                      HasSubstr("positive power of two")));

  config = TestWireConfig();
  config.mutable_peer_regions(0)->set_memory_kind(
      wire::PeerRegionWire::MEMORY_KIND_MULTIMEM);
  absl::StatusOr<wire::CollectiveCallConfigV3> parsed = Parse(config);
  ASSERT_THAT(parsed, IsOk());
  EXPECT_EQ(parsed->peer_regions(0).memory_kind(),
            wire::PEER_MEMORY_KIND_PROTO_MULTIMEM);
}

}  // namespace
}  // namespace xla::gpu::cutedsl
