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

#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <tuple>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/libraries/cutedsl/config.pb.h"

namespace xla::gpu::cutedsl {

using ::xla::CollectiveOpGroupMode;
using ::xla::ReplicaGroup;
namespace wire = ::xla::gpu::cutedsl::proto;

namespace {

absl::Status MissingField(absl::string_view field) {
  return absl::InvalidArgumentError(absl::StrFormat(
      "Missing CuTeDSL collective v3 protobuf field `%s`", field));
}

absl::Status MissingRepeatedField(absl::string_view repeated_field,
                                  size_t index, absl::string_view field) {
  return absl::InvalidArgumentError(absl::StrFormat(
      "Missing CuTeDSL collective v3 protobuf field `%s[%d].%s`",
      repeated_field, index, field));
}

absl::StatusOr<CollectiveOpGroupMode> DecodeGroupMode(
    wire::CollectiveGroupWire::Mode mode) {
  switch (mode) {
    case wire::CollectiveGroupWire::MODE_CROSS_REPLICA:
      return CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA;
    case wire::CollectiveGroupWire::MODE_CROSS_PARTITION:
      return CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_PARTITION;
    case wire::CollectiveGroupWire::MODE_CROSS_REPLICA_AND_PARTITION:
      return CollectiveOpGroupMode::
          COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA_AND_PARTITION;
    case wire::CollectiveGroupWire::MODE_FLATTENED_ID:
      return CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_FLATTENED_ID;
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "Unsupported collective group mode %d", static_cast<int>(mode)));
  }
}

absl::StatusOr<wire::PeerRegionEndpointProto> DecodeEndpoint(
    wire::PeerRegionWire::Endpoint endpoint) {
  switch (endpoint) {
    case wire::PeerRegionWire::ENDPOINT_ARGUMENT:
      return wire::PEER_REGION_ENDPOINT_PROTO_ARGUMENT;
    case wire::PeerRegionWire::ENDPOINT_RESULT:
      return wire::PEER_REGION_ENDPOINT_PROTO_RESULT;
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "Unsupported peer-region endpoint %d", static_cast<int>(endpoint)));
  }
}

absl::StatusOr<wire::PeerMemoryKindProto> DecodeMemoryKind(
    wire::PeerRegionWire::MemoryKind memory_kind) {
  switch (memory_kind) {
    case wire::PeerRegionWire::MEMORY_KIND_SYMMETRIC:
      return wire::PEER_MEMORY_KIND_PROTO_SYMMETRIC;
    case wire::PeerRegionWire::MEMORY_KIND_MULTIMEM:
      return wire::PEER_MEMORY_KIND_PROTO_MULTIMEM;
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "Unsupported peer-memory kind %d", static_cast<int>(memory_kind)));
  }
}

absl::StatusOr<int64_t> DecodeInt64(uint64_t value, absl::string_view field) {
  if (value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return absl::InvalidArgumentError(
        absl::StrFormat("`%s` exceeds the signed 64-bit range", field));
  }
  return static_cast<int64_t>(value);
}

absl::StatusOr<wire::CollectiveCallConfigV3> DecodeWireConfig(
    const wire::CollectiveCallConfigV3Wire& wire_config) {
  if (!wire_config.has_abi_clique_size())
    return MissingField("abi_clique_size");
  if (!wire_config.has_barrier_before_launch()) {
    return MissingField("barrier_before_launch");
  }
  if (!wire_config.has_group()) return MissingField("group");

  const wire::CollectiveGroupWire& wire_group = wire_config.group();
  if (!wire_group.has_mode()) return MissingField("group.mode");
  if (!wire_group.has_communication_id()) {
    return MissingField("group.communication_id");
  }

  wire::CollectiveCallConfigV3 config;
  config.set_abi_clique_size(wire_config.abi_clique_size());
  config.set_barrier_before_launch(wire_config.barrier_before_launch());
  ABSL_ASSIGN_OR_RETURN(CollectiveOpGroupMode group_mode,
                        DecodeGroupMode(wire_group.mode()));
  config.set_group_mode(group_mode);
  ABSL_ASSIGN_OR_RETURN(
      int64_t communication_id,
      DecodeInt64(wire_group.communication_id(), "group.communication_id"));
  config.set_communication_id(communication_id);

  for (const wire::ReplicaGroupWire& wire_replica_group :
       wire_group.replica_groups()) {
    ReplicaGroup* replica_group = config.add_replica_groups();
    for (int64_t member : wire_replica_group.members()) {
      replica_group->add_replica_ids(member);
    }
  }

  for (int region_index = 0; region_index < wire_config.peer_regions_size();
       ++region_index) {
    const wire::PeerRegionWire& wire_region =
        wire_config.peer_regions(region_index);
    if (!wire_region.has_endpoint() || !wire_region.has_buffer_index() ||
        !wire_region.has_byte_offset() || !wire_region.has_byte_size() ||
        !wire_region.has_required_alignment() ||
        !wire_region.has_memory_kind()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "CuTeDSL collective v3 protobuf field `peer_regions[%d]` is "
          "missing a required field",
          region_index));
    }

    wire::PeerRegionProto* region = config.add_peer_regions();
    ABSL_ASSIGN_OR_RETURN(wire::PeerRegionEndpointProto endpoint,
                          DecodeEndpoint(wire_region.endpoint()));
    region->set_endpoint(endpoint);
    ABSL_ASSIGN_OR_RETURN(
        int64_t buffer_index,
        DecodeInt64(wire_region.buffer_index(), "buffer_index"));
    ABSL_ASSIGN_OR_RETURN(
        int64_t byte_offset,
        DecodeInt64(wire_region.byte_offset(), "byte_offset"));
    ABSL_ASSIGN_OR_RETURN(int64_t byte_size,
                          DecodeInt64(wire_region.byte_size(), "byte_size"));
    ABSL_ASSIGN_OR_RETURN(
        int64_t required_alignment,
        DecodeInt64(wire_region.required_alignment(), "required_alignment"));
    ABSL_ASSIGN_OR_RETURN(wire::PeerMemoryKindProto memory_kind,
                          DecodeMemoryKind(wire_region.memory_kind()));
    region->set_buffer_index(buffer_index);
    region->set_byte_offset(byte_offset);
    region->set_byte_size(byte_size);
    region->set_required_alignment(required_alignment);
    region->set_memory_kind(memory_kind);
  }
  return config;
}

absl::Status ValidateGroupMode(int64_t value) {
  switch (value) {
    case static_cast<int64_t>(
        CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA):
    case static_cast<int64_t>(
        CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_PARTITION):
    case static_cast<int64_t>(
        CollectiveOpGroupMode::
            COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA_AND_PARTITION):
    case static_cast<int64_t>(
        CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_FLATTENED_ID):
      return absl::OkStatus();
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("Unsupported collective group mode %d", value));
  }
}

absl::Status ValidateReplicaGroups(const wire::CollectiveCallConfigV3& proto) {
  if (proto.replica_groups().empty()) {
    return absl::InvalidArgumentError(
        "`replica_groups` must contain at least one group");
  }

  std::set<int64_t> unique_members;
  int64_t group_size = -1;

  for (int group_index = 0; group_index < proto.replica_groups_size();
       ++group_index) {
    const ReplicaGroup& group = proto.replica_groups(group_index);
    if (group.replica_ids().empty()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Replica group %d must not be empty", group_index));
    }
    if (group_size == -1) {
      group_size = group.replica_ids_size();
    } else if (group.replica_ids_size() != group_size) {
      return absl::InvalidArgumentError(
          "All replica groups must have equal cardinality");
    }

    for (int64_t member : group.replica_ids()) {
      if (member < 0) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Replica-group member IDs must be nonnegative; got %d", member));
      }
      if (!unique_members.insert(member).second) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Replica-group member ID %d appears more than once", member));
      }
    }
  }

  return absl::OkStatus();
}

absl::Status ValidatePeerRegionFields(const wire::PeerRegionProto& region,
                                      size_t region_index) {
  if (!region.has_endpoint()) {
    return MissingRepeatedField("peer_regions", region_index, "endpoint");
  }
  if (!region.has_buffer_index()) {
    return MissingRepeatedField("peer_regions", region_index, "buffer_index");
  }
  if (!region.has_byte_offset()) {
    return MissingRepeatedField("peer_regions", region_index, "byte_offset");
  }
  if (!region.has_byte_size()) {
    return MissingRepeatedField("peer_regions", region_index, "byte_size");
  }
  if (!region.has_required_alignment()) {
    return MissingRepeatedField("peer_regions", region_index,
                                "required_alignment");
  }
  if (!region.has_memory_kind()) {
    return MissingRepeatedField("peer_regions", region_index, "memory_kind");
  }
  return absl::OkStatus();
}

absl::Status ValidatePeerRegions(const wire::CollectiveCallConfigV3& proto) {
  using PeerRegionKey =
      std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
  std::set<PeerRegionKey> unique_regions;
  for (int region_index = 0; region_index < proto.peer_regions_size();
       ++region_index) {
    const wire::PeerRegionProto& region = proto.peer_regions(region_index);
    ABSL_RETURN_IF_ERROR(ValidatePeerRegionFields(region, region_index));

    switch (region.endpoint()) {
      case wire::PEER_REGION_ENDPOINT_PROTO_ARGUMENT:
      case wire::PEER_REGION_ENDPOINT_PROTO_RESULT:
        break;
      default:
        return absl::InvalidArgumentError(
            absl::StrFormat("Unsupported endpoint %d for peer region %d",
                            region.endpoint(), region_index));
    }
    if (region.buffer_index() < 0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Buffer index for peer region %d must be nonnegative", region_index));
    }
    if (region.byte_offset() < 0 || region.byte_size() <= 0 ||
        region.byte_offset() >
            std::numeric_limits<int64_t>::max() - region.byte_size()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Invalid or overflowing byte range for peer region %d",
          region_index));
    }
    if (region.required_alignment() <= 0 ||
        (region.required_alignment() & (region.required_alignment() - 1)) !=
            0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Required alignment for peer region %d must be a positive power of "
          "two",
          region_index));
    }
    switch (region.memory_kind()) {
      case wire::PEER_MEMORY_KIND_PROTO_SYMMETRIC:
      case wire::PEER_MEMORY_KIND_PROTO_MULTIMEM:
        break;
      default:
        return absl::InvalidArgumentError(
            absl::StrFormat("Unsupported memory kind %d for peer region %d",
                            region.memory_kind(), region_index));
    }

    PeerRegionKey key = {static_cast<int64_t>(region.endpoint()),
                         region.buffer_index(),
                         region.byte_offset(),
                         region.byte_size(),
                         region.required_alignment(),
                         static_cast<int64_t>(region.memory_kind())};
    if (!unique_regions.insert(key).second) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Peer region record %d duplicates an earlier record", region_index));
    }
  }

  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<wire::CollectiveCallConfigV3>
ParseAndValidateCollectiveCallConfig(absl::string_view serialized_config) {
  if (serialized_config.size() >
      static_cast<size_t>(std::numeric_limits<int>::max())) {
    return absl::InvalidArgumentError(
        "CuTeDSL collective v3 protobuf configuration is too large");
  }
  wire::CollectiveCallConfigV3Wire wire_config;
  if (!wire_config.ParseFromArray(serialized_config.data(),
                                  static_cast<int>(serialized_config.size()))) {
    return absl::InvalidArgumentError(
        "Failed to parse CuTeDSL collective v3 protobuf configuration");
  }
  ABSL_ASSIGN_OR_RETURN(wire::CollectiveCallConfigV3 proto,
                        DecodeWireConfig(wire_config));

  if (!proto.has_abi_clique_size()) return MissingField("abi_clique_size");
  if (proto.abi_clique_size() <= 0 ||
      proto.abi_clique_size() > std::numeric_limits<int32_t>::max()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "`abi_clique_size` must be in [1, %d]; got %d",
        std::numeric_limits<int32_t>::max(), proto.abi_clique_size()));
  }

  if (!proto.has_group_mode()) return MissingField("group_mode");
  ABSL_RETURN_IF_ERROR(
      ValidateGroupMode(static_cast<int64_t>(proto.group_mode())));

  if (!proto.has_communication_id()) return MissingField("communication_id");
  if (proto.communication_id() < 0) {
    return absl::InvalidArgumentError("`communication_id` must be nonnegative");
  }

  ABSL_RETURN_IF_ERROR(ValidateReplicaGroups(proto));
  ABSL_RETURN_IF_ERROR(ValidatePeerRegions(proto));
  return proto;
}

}  // namespace xla::gpu::cutedsl
