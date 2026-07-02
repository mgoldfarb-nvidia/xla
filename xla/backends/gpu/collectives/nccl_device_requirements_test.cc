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

#include <cstdint>
#include <limits>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/collectives/nccl_communicator.h"
#include "xla/backends/gpu/collectives/nccl_symmetric_memory.h"
#include "xla/ffi/api/c_api_gpu_collectives_nccl.h"
#include "xla/tsl/lib/core/status_test_util.h"

namespace xla::gpu {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using Requirements = GpuDeviceCommunicator::Requirements;
using PeerAccess = GpuDeviceCommunicator::PeerAccess;
using Topology = GpuDeviceCommunicator::Topology;

Requirements MakeRequirements(
    PeerAccess peer_access = PeerAccess::kLocalDomain,
    GpuDeviceCommunicator::Features required_features = 0,
    GpuDeviceCommunicator::Features preferred_features = 0,
    int32_t local_barrier_count = 0, int32_t team_barrier_count = 0,
    int32_t notification_slot_count = 0, int32_t completion_slot_count = 0) {
  Requirements requirements;
  requirements.peer_access = peer_access;
  requirements.required_features = required_features;
  requirements.preferred_features = preferred_features;
  requirements.local_barrier_count = local_barrier_count;
  requirements.team_barrier_count = team_barrier_count;
  requirements.notification_slot_count = notification_slot_count;
  requirements.completion_slot_count = completion_slot_count;
  return requirements;
}

NcclCapabilities Capabilities(int lsa_team_count,
                              bool supports_full_gin = false,
                              bool supports_rail_gin = false,
                              bool supports_multimem = false, int team_size = 8,
                              int rank = 0) {
  NcclCapabilities capabilities;
  capabilities.supports_device_comm = true;
  capabilities.supports_multimem = supports_multimem;
  capabilities.supports_full_gin = supports_full_gin;
  capabilities.supports_rail_gin = supports_rail_gin;
  capabilities.rank = rank;
  capabilities.team_size = team_size;
  capabilities.lsa_team_count = lsa_team_count;
  return capabilities;
}

static_assert(kNcclDeviceCommAbiSchema ==
              XLA_FFI_GpuCollective_NCCL_DEVICE_COMM_ABI_SCHEMA);
static_assert(kNcclWindowAbiSchema ==
              XLA_FFI_GpuCollective_NCCL_SYMMETRIC_MEMORY_ABI_SCHEMA);

TEST(NcclDeviceRequirementsTest, MapsLsaBarrierCount) {
  ASSERT_OK_AND_ASSIGN(
      NcclDeviceCommPlan plan,
      BuildNcclDeviceCommPlan(
          MakeRequirements(PeerAccess::kLocalDomain, /*required_features=*/0,
                           /*preferred_features=*/0,
                           /*local_barrier_count=*/2),
          Capabilities(/*lsa_team_count=*/1)));

  EXPECT_EQ(plan.requirements.lsaBarrierCount, 2);
#if NCCL_VERSION_CODE >= 22907
  EXPECT_EQ(plan.requirements.barrierCount, 0);
  EXPECT_EQ(plan.requirements.ginConnectionType, NCCL_GIN_CONNECTION_NONE);
#endif
}

TEST(NcclDeviceRequirementsTest, RejectsNegativeLsaBarrierCount) {
  EXPECT_THAT(
      BuildNcclDeviceCommPlan(
          MakeRequirements(PeerAccess::kLocalDomain, /*required_features=*/0,
                           /*preferred_features=*/0,
                           /*local_barrier_count=*/-1),
          Capabilities(/*lsa_team_count=*/1)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("local_barrier_count must be non-negative")));
}

TEST(NcclDeviceRequirementsTest, RejectsNegativeTeamBarrierCount) {
  EXPECT_THAT(
      BuildNcclDeviceCommPlan(
          MakeRequirements(PeerAccess::kLocalDomain, /*required_features=*/0,
                           /*preferred_features=*/0,
                           /*local_barrier_count=*/0,
                           /*team_barrier_count=*/-1),
          Capabilities(/*lsa_team_count=*/1)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("team_barrier_count must be non-negative")));
}

TEST(NcclDeviceRequirementsTest, RejectsUnknownRequiredFeature) {
  EXPECT_THAT(BuildNcclDeviceCommPlan(
                  MakeRequirements(PeerAccess::kLocalDomain,
                                   /*required_features=*/uint64_t{1} << 63),
                  Capabilities(/*lsa_team_count=*/1)),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("Unsupported required")));
}

TEST(NcclDeviceRequirementsTest, IgnoresUnknownPreferredFeature) {
  ASSERT_OK_AND_ASSIGN(
      NcclDeviceCommPlan plan,
      BuildNcclDeviceCommPlan(
          MakeRequirements(PeerAccess::kLocalDomain,
                           /*required_features=*/0,
                           /*preferred_features=*/uint64_t{1} << 63),
          Capabilities(/*lsa_team_count=*/1)));
  EXPECT_EQ(plan.enabled_features, 0);
}

TEST(NcclDeviceRequirementsTest, IgnoresUnavailablePreferredNetworkFeature) {
  ASSERT_OK_AND_ASSIGN(
      NcclDeviceCommPlan plan,
      BuildNcclDeviceCommPlan(
          MakeRequirements(PeerAccess::kLocalDomain, /*required_features=*/0,
                           GpuDeviceCommunicator::kNetworkDeviceOperations),
          Capabilities(/*lsa_team_count=*/0)));
  EXPECT_EQ(plan.enabled_features, 0);
  EXPECT_EQ(plan.topology, Topology::kLocalDomain);
}

TEST(NcclDeviceRequirementsTest, RejectsBarrierCountAdditionOverflow) {
  EXPECT_THAT(BuildNcclDeviceCommPlan(
                  MakeRequirements(PeerAccess::kLocalDomain,
                                   /*required_features=*/0,
                                   /*preferred_features=*/0,
                                   /*local_barrier_count=*/
                                   std::numeric_limits<int32_t>::max(),
                                   /*team_barrier_count=*/1),
                  Capabilities(/*lsa_team_count=*/1)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("overflows NCCL's signed count type")));
}

TEST(NcclDeviceRequirementsTest, RejectsLsaBarrierStorageOverflow) {
  EXPECT_THAT(BuildNcclDeviceCommPlan(
                  MakeRequirements(PeerAccess::kLocalDomain,
                                   /*required_features=*/0,
                                   /*preferred_features=*/0,
                                   /*local_barrier_count=*/
                                   std::numeric_limits<int32_t>::max() / 2),
                  Capabilities(/*lsa_team_count=*/0,
                               /*supports_full_gin=*/false,
                               /*supports_rail_gin=*/false,
                               /*supports_multimem=*/false,
                               /*team_size=*/8)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("LSA barrier storage arithmetic")));
}

#if NCCL_VERSION_CODE >= 22907
TEST(NcclDeviceRequirementsTest, GivesLocalAndTeamBarriersDisjointLsaSlots) {
  ASSERT_OK_AND_ASSIGN(
      NcclDeviceCommPlan plan,
      BuildNcclDeviceCommPlan(
          MakeRequirements(PeerAccess::kLocalDomain, /*required_features=*/0,
                           /*preferred_features=*/0,
                           /*local_barrier_count=*/2,
                           /*team_barrier_count=*/3),
          Capabilities(/*lsa_team_count=*/1)));

  EXPECT_EQ(plan.requirements.lsaBarrierCount, 5);
  EXPECT_EQ(plan.requirements.barrierCount, 0);
  EXPECT_EQ(plan.requirements.ginConnectionType, NCCL_GIN_CONNECTION_NONE);
  EXPECT_EQ(plan.topology, Topology::kLocalDomain);
}

TEST(NcclDeviceRequirementsTest, UsesFullFirstForHiddenTeamBarrier) {
  ASSERT_OK_AND_ASSIGN(
      NcclDeviceCommPlan plan,
      BuildNcclDeviceCommPlan(
          MakeRequirements(PeerAccess::kLocalDomain, /*required_features=*/0,
                           /*preferred_features=*/0,
                           /*local_barrier_count=*/1,
                           /*team_barrier_count=*/2),
          Capabilities(/*lsa_team_count=*/2,
                       /*supports_full_gin=*/true,
                       /*supports_rail_gin=*/true)));

  EXPECT_EQ(plan.requirements.lsaBarrierCount, 3);
  EXPECT_EQ(plan.requirements.barrierCount, 2);
  EXPECT_EQ(plan.requirements.ginConnectionType, NCCL_GIN_CONNECTION_FULL);
  EXPECT_EQ(plan.requirements.ginSignalCount, 0);
  EXPECT_EQ(plan.requirements.ginCounterCount, 0);
  EXPECT_EQ(plan.topology, Topology::kLocalDomain);
  EXPECT_EQ(plan.enabled_features, 0);
}

TEST(NcclDeviceRequirementsTest, FallsBackToRailForHierarchicalAccess) {
  ASSERT_OK_AND_ASSIGN(
      NcclDeviceCommPlan plan,
      BuildNcclDeviceCommPlan(MakeRequirements(PeerAccess::kHierarchical),
                              Capabilities(/*lsa_team_count=*/2,
                                           /*supports_full_gin=*/false,
                                           /*supports_rail_gin=*/true)));

  EXPECT_EQ(plan.requirements.ginConnectionType, NCCL_GIN_CONNECTION_RAIL);
  EXPECT_EQ(plan.topology, Topology::kHierarchical);
}

TEST(NcclDeviceRequirementsTest, PrefersFullForHierarchicalAccess) {
  ASSERT_OK_AND_ASSIGN(
      NcclDeviceCommPlan plan,
      BuildNcclDeviceCommPlan(MakeRequirements(PeerAccess::kHierarchical),
                              Capabilities(/*lsa_team_count=*/2,
                                           /*supports_full_gin=*/true,
                                           /*supports_rail_gin=*/true)));

  EXPECT_EQ(plan.requirements.ginConnectionType, NCCL_GIN_CONNECTION_FULL);
  EXPECT_EQ(plan.topology, Topology::kAllPeers);
}

TEST(NcclDeviceRequirementsTest, DirectAnyPeerRequiresFullAcrossDomains) {
  EXPECT_THAT(
      BuildNcclDeviceCommPlan(MakeRequirements(PeerAccess::kDirectAnyPeer),
                              Capabilities(/*lsa_team_count=*/2,
                                           /*supports_full_gin=*/false,
                                           /*supports_rail_gin=*/true)),
      StatusIs(absl::StatusCode::kUnimplemented, HasSubstr("cannot satisfy")));
}

TEST(NcclDeviceRequirementsTest, DirectAnyPeerNeedsNoGinWithinOneDomain) {
  ASSERT_OK_AND_ASSIGN(
      NcclDeviceCommPlan plan,
      BuildNcclDeviceCommPlan(MakeRequirements(PeerAccess::kDirectAnyPeer),
                              Capabilities(/*lsa_team_count=*/1)));

  EXPECT_EQ(plan.requirements.ginConnectionType, NCCL_GIN_CONNECTION_NONE);
  EXPECT_EQ(plan.topology, Topology::kLocalDomain);
}

TEST(NcclDeviceRequirementsTest, NetworkOperationsRequireFullInOneDomain) {
  ASSERT_OK_AND_ASSIGN(
      NcclDeviceCommPlan plan,
      BuildNcclDeviceCommPlan(
          MakeRequirements(PeerAccess::kLocalDomain,
                           GpuDeviceCommunicator::kNetworkDeviceOperations),
          Capabilities(/*lsa_team_count=*/1,
                       /*supports_full_gin=*/true)));

  EXPECT_EQ(plan.requirements.ginConnectionType, NCCL_GIN_CONNECTION_FULL);
  EXPECT_EQ(plan.topology, Topology::kAllPeers);
  EXPECT_NE(
      plan.enabled_features & GpuDeviceCommunicator::kNetworkDeviceOperations,
      0);
}

TEST(NcclDeviceRequirementsTest, RailCannotProvideLocalNetworkOperations) {
  EXPECT_THAT(
      BuildNcclDeviceCommPlan(
          MakeRequirements(PeerAccess::kLocalDomain,
                           GpuDeviceCommunicator::kNetworkDeviceOperations),
          Capabilities(/*lsa_team_count=*/2,
                       /*supports_full_gin=*/false,
                       /*supports_rail_gin=*/true)),
      StatusIs(absl::StatusCode::kUnimplemented, HasSubstr("cannot satisfy")));
}

TEST(NcclDeviceRequirementsTest, MapsSignalsCountersAndImpliedNetworkFeature) {
  ASSERT_OK_AND_ASSIGN(
      NcclDeviceCommPlan plan,
      BuildNcclDeviceCommPlan(MakeRequirements(PeerAccess::kLocalDomain,
                                               /*required_features=*/0,
                                               /*preferred_features=*/0,
                                               /*local_barrier_count=*/0,
                                               /*team_barrier_count=*/0,
                                               /*notification_slot_count=*/8,
                                               /*completion_slot_count=*/4),
                              Capabilities(/*lsa_team_count=*/2,
                                           /*supports_full_gin=*/true)));

  EXPECT_EQ(plan.requirements.ginSignalCount, 8);
  EXPECT_EQ(plan.requirements.ginCounterCount, 4);
  EXPECT_EQ(plan.requirements.ginConnectionType, NCCL_GIN_CONNECTION_FULL);
  EXPECT_NE(
      plan.enabled_features & GpuDeviceCommunicator::kNetworkDeviceOperations,
      0);
}

TEST(NcclDeviceRequirementsTest, ResolvesPreferredMultimem) {
  ASSERT_OK_AND_ASSIGN(
      NcclDeviceCommPlan supported,
      BuildNcclDeviceCommPlan(
          MakeRequirements(PeerAccess::kLocalDomain,
                           /*required_features=*/0,
                           GpuDeviceCommunicator::kLocalMulticast),
          Capabilities(/*lsa_team_count=*/1,
                       /*supports_full_gin=*/false,
                       /*supports_rail_gin=*/false,
                       /*supports_multimem=*/true)));
  EXPECT_TRUE(supported.requirements.lsaMultimem);
  EXPECT_NE(supported.enabled_features & GpuDeviceCommunicator::kLocalMulticast,
            0);

  ASSERT_OK_AND_ASSIGN(
      NcclDeviceCommPlan unsupported,
      BuildNcclDeviceCommPlan(
          MakeRequirements(PeerAccess::kLocalDomain,
                           /*required_features=*/0,
                           GpuDeviceCommunicator::kLocalMulticast),
          Capabilities(/*lsa_team_count=*/1)));
  EXPECT_FALSE(unsupported.requirements.lsaMultimem);
  EXPECT_EQ(unsupported.enabled_features, 0);
}

TEST(NcclDeviceRequirementsTest, RejectsRequiredMultimemWhenUnsupported) {
  EXPECT_THAT(BuildNcclDeviceCommPlan(
                  MakeRequirements(PeerAccess::kLocalDomain,
                                   GpuDeviceCommunicator::kLocalMulticast),
                  Capabilities(/*lsa_team_count=*/1)),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("no Multimem support")));
}

TEST(NcclDeviceRequirementsTest, RejectsTeamBarrierWithoutLsaTopology) {
  EXPECT_THAT(
      BuildNcclDeviceCommPlan(MakeRequirements(PeerAccess::kLocalDomain,
                                               /*required_features=*/0,
                                               /*preferred_features=*/0,
                                               /*local_barrier_count=*/0,
                                               /*team_barrier_count=*/1),
                              Capabilities(/*lsa_team_count=*/0)),
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("does not report its LSA topology")));
}

TEST(NcclDeviceRequirementsTest, RejectsMultiLsaTeamBarrierWithoutGin) {
  EXPECT_THAT(
      BuildNcclDeviceCommPlan(MakeRequirements(PeerAccess::kLocalDomain,
                                               /*required_features=*/0,
                                               /*preferred_features=*/0,
                                               /*local_barrier_count=*/0,
                                               /*team_barrier_count=*/1),
                              Capabilities(/*lsa_team_count=*/2)),
      StatusIs(absl::StatusCode::kUnimplemented, HasSubstr("cannot satisfy")));
}

TEST(NcclDeviceRequirementsTest, RejectsGinSignalArithmeticOverflow) {
  EXPECT_THAT(
      BuildNcclDeviceCommPlan(MakeRequirements(PeerAccess::kLocalDomain,
                                               /*required_features=*/0,
                                               /*preferred_features=*/0,
                                               /*local_barrier_count=*/0,
                                               /*team_barrier_count=*/1 << 23,
                                               /*notification_slot_count=*/1),
                              Capabilities(/*lsa_team_count=*/2,
                                           /*supports_full_gin=*/true)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("exceeds NCCL's representable maximum")));
}

TEST(NcclDeviceRequirementsTest, BuildsResolvedInfo) {
  Requirements requirements = MakeRequirements(
      PeerAccess::kHierarchical,
      /*required_features=*/0,
      /*preferred_features=*/GpuDeviceCommunicator::kLocalMulticast,
      /*local_barrier_count=*/1,
      /*team_barrier_count=*/1,
      /*notification_slot_count=*/2,
      /*completion_slot_count=*/1);
  NcclCapabilities capabilities =
      Capabilities(/*lsa_team_count=*/2, /*supports_full_gin=*/true,
                   /*supports_rail_gin=*/true,
                   /*supports_multimem=*/true,
                   /*team_size=*/8, /*rank=*/1);
  ASSERT_OK_AND_ASSIGN(NcclDeviceCommPlan plan,
                       BuildNcclDeviceCommPlan(requirements, capabilities));

  ncclDevComm dev_comm{};
  dev_comm.rank = 1;
  dev_comm.nRanks = 8;
  dev_comm.lsaRank = 1;
  dev_comm.lsaSize = 4;
  dev_comm.lsaMultimem.mcBasePtr = reinterpret_cast<void*>(uintptr_t{1});
  dev_comm.lsaBarrier.nBarriers = 2;
  dev_comm.ginConnectionCount = 1;
  dev_comm.ginContextCount = 4;
  dev_comm.ginSignalCount = 4;
  dev_comm.ginCounterCount = 1;
  dev_comm.ginIsRailed = false;

  ASSERT_OK_AND_ASSIGN(
      GpuDeviceCommunicator::Info info,
      BuildNcclDeviceCommInfo(requirements, plan, capabilities, dev_comm));
  EXPECT_EQ(info.rank, 1);
  EXPECT_EQ(info.team_size, 8);
  EXPECT_EQ(info.local_rank, 1);
  EXPECT_EQ(info.local_domain_size, 4);
  EXPECT_EQ(info.local_domain_count, 2);
  EXPECT_EQ(info.topology, Topology::kAllPeers);
  EXPECT_EQ(info.team_barrier_count, 1);
  EXPECT_EQ(info.local_barrier_count, 1);
  EXPECT_EQ(info.notification_slot_count, 2);
  EXPECT_EQ(info.completion_slot_count, 1);
}

TEST(NcclDeviceRequirementsTest, RejectsMissingEnabledMultimemHandle) {
  Requirements requirements = MakeRequirements(
      PeerAccess::kLocalDomain, /*required_features=*/0,
      /*preferred_features=*/GpuDeviceCommunicator::kLocalMulticast);
  NcclCapabilities capabilities =
      Capabilities(/*lsa_team_count=*/1, /*supports_full_gin=*/false,
                   /*supports_rail_gin=*/false,
                   /*supports_multimem=*/true,
                   /*team_size=*/8, /*rank=*/0);
  ASSERT_OK_AND_ASSIGN(NcclDeviceCommPlan plan,
                       BuildNcclDeviceCommPlan(requirements, capabilities));

  ncclDevComm dev_comm{};
  dev_comm.rank = 0;
  dev_comm.nRanks = 8;
  dev_comm.lsaRank = 0;
  dev_comm.lsaSize = 8;

  EXPECT_THAT(
      BuildNcclDeviceCommInfo(requirements, plan, capabilities, dev_comm),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("did not realize the requested LSA Multimem")));
}

TEST(NcclDeviceRequirementsTest, RejectsStickyGinModeMismatch) {
  Requirements requirements = MakeRequirements(PeerAccess::kDirectAnyPeer);
  NcclCapabilities capabilities =
      Capabilities(/*lsa_team_count=*/2, /*supports_full_gin=*/true,
                   /*supports_rail_gin=*/true,
                   /*supports_multimem=*/false,
                   /*team_size=*/8, /*rank=*/0);
  ASSERT_OK_AND_ASSIGN(NcclDeviceCommPlan plan,
                       BuildNcclDeviceCommPlan(requirements, capabilities));

  ncclDevComm dev_comm{};
  dev_comm.rank = 0;
  dev_comm.nRanks = 8;
  dev_comm.lsaRank = 0;
  dev_comm.lsaSize = 4;
  dev_comm.ginConnectionCount = 1;
  dev_comm.ginContextCount = 4;
  dev_comm.ginIsRailed = true;

  EXPECT_THAT(
      BuildNcclDeviceCommInfo(requirements, plan, capabilities, dev_comm),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("bootstrap state is incompatible")));
}

TEST(NcclDeviceRequirementsTest, RejectsInvalidGinResourceCounts) {
  Requirements requirements = MakeRequirements(PeerAccess::kDirectAnyPeer);
  NcclCapabilities capabilities =
      Capabilities(/*lsa_team_count=*/2, /*supports_full_gin=*/true,
                   /*supports_rail_gin=*/true,
                   /*supports_multimem=*/false,
                   /*team_size=*/8, /*rank=*/0);
  ASSERT_OK_AND_ASSIGN(NcclDeviceCommPlan plan,
                       BuildNcclDeviceCommPlan(requirements, capabilities));

  ncclDevComm dev_comm{};
  dev_comm.rank = 0;
  dev_comm.nRanks = 8;
  dev_comm.lsaRank = 0;
  dev_comm.lsaSize = 4;
  dev_comm.ginConnectionCount = 0;
  dev_comm.ginContextCount = 4;
  dev_comm.ginIsRailed = false;

  EXPECT_THAT(
      BuildNcclDeviceCommInfo(requirements, plan, capabilities, dev_comm),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("without usable connections or contexts")));
}

TEST(NcclDeviceRequirementsTest, RejectsChangedCommunicatorIdentity) {
  Requirements requirements = MakeRequirements();
  NcclCapabilities capabilities =
      Capabilities(/*lsa_team_count=*/1, /*supports_full_gin=*/false,
                   /*supports_rail_gin=*/false,
                   /*supports_multimem=*/false,
                   /*team_size=*/8, /*rank=*/0);
  ASSERT_OK_AND_ASSIGN(NcclDeviceCommPlan plan,
                       BuildNcclDeviceCommPlan(requirements, capabilities));

  ncclDevComm dev_comm{};
  dev_comm.rank = 1;
  dev_comm.nRanks = 8;
  dev_comm.lsaRank = 1;
  dev_comm.lsaSize = 8;

  EXPECT_THAT(
      BuildNcclDeviceCommInfo(requirements, plan, capabilities, dev_comm),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("identity changed after capability query")));
}

TEST(NcclDeviceRequirementsTest, RejectsInsufficientLsaBarriers) {
  Requirements requirements =
      MakeRequirements(PeerAccess::kLocalDomain, /*required_features=*/0,
                       /*preferred_features=*/0,
                       /*local_barrier_count=*/1);
  NcclCapabilities capabilities = Capabilities(/*lsa_team_count=*/1);
  ASSERT_OK_AND_ASSIGN(NcclDeviceCommPlan plan,
                       BuildNcclDeviceCommPlan(requirements, capabilities));

  ncclDevComm dev_comm{};
  dev_comm.rank = 0;
  dev_comm.nRanks = 8;
  dev_comm.lsaRank = 0;
  dev_comm.lsaSize = 8;
  dev_comm.lsaBarrier.nBarriers = 0;

  EXPECT_THAT(
      BuildNcclDeviceCommInfo(requirements, plan, capabilities, dev_comm),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("provides 0 LSA barriers, but 1 are required")));
}

TEST(NcclDeviceRequirementsTest, RejectsInsufficientGinResources) {
  Requirements requirements =
      MakeRequirements(PeerAccess::kDirectAnyPeer, /*required_features=*/0,
                       /*preferred_features=*/0,
                       /*local_barrier_count=*/0,
                       /*team_barrier_count=*/0,
                       /*notification_slot_count=*/1,
                       /*completion_slot_count=*/1);
  NcclCapabilities capabilities =
      Capabilities(/*lsa_team_count=*/2, /*supports_full_gin=*/true,
                   /*supports_rail_gin=*/true,
                   /*supports_multimem=*/false,
                   /*team_size=*/8, /*rank=*/0);
  ASSERT_OK_AND_ASSIGN(NcclDeviceCommPlan plan,
                       BuildNcclDeviceCommPlan(requirements, capabilities));

  ncclDevComm dev_comm{};
  dev_comm.rank = 0;
  dev_comm.nRanks = 8;
  dev_comm.lsaRank = 0;
  dev_comm.lsaSize = 4;
  dev_comm.ginConnectionCount = 1;
  dev_comm.ginContextCount = 4;
  dev_comm.ginSignalCount = 1;
  dev_comm.ginCounterCount = 0;
  dev_comm.ginIsRailed = false;

  EXPECT_THAT(
      BuildNcclDeviceCommInfo(requirements, plan, capabilities, dev_comm),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("insufficient GIN resources")));
}
#endif

TEST(NcclDeviceAbiTest, AcceptsExactVersion) {
  EXPECT_OK(ValidateNcclDeviceAbi(/*compile_time_version=*/22907,
                                  /*runtime_version=*/22907));
}

TEST(NcclDeviceRequirementsTest, AgreementPayloadExcludesRankIdentity) {
  Requirements requirements =
      MakeRequirements(PeerAccess::kLocalDomain, /*required_features=*/0,
                       /*preferred_features=*/0, /*local_barrier_count=*/2);
  NcclCapabilities rank0 = Capabilities(/*lsa_team_count=*/1,
                                        /*supports_full_gin=*/false,
                                        /*supports_rail_gin=*/false,
                                        /*supports_multimem=*/false,
                                        /*team_size=*/8, /*rank=*/0);
  NcclCapabilities rank7 = rank0;
  rank7.rank = 7;
  ASSERT_OK_AND_ASSIGN(NcclDeviceCommPlan plan,
                       BuildNcclDeviceCommPlan(requirements, rank0));

  EXPECT_EQ(NcclDeviceCommPlanAgreementPayload(
                plan, rank0, /*runtime_version=*/NCCL_VERSION_CODE),
            NcclDeviceCommPlanAgreementPayload(
                plan, rank7, /*runtime_version=*/NCCL_VERSION_CODE));
}

TEST(NcclDeviceRequirementsTest, AgreementPayloadIncludesResolvedAbiAndPlan) {
  NcclCapabilities capabilities = Capabilities(/*lsa_team_count=*/1);
  ASSERT_OK_AND_ASSIGN(
      NcclDeviceCommPlan one_barrier,
      BuildNcclDeviceCommPlan(
          MakeRequirements(PeerAccess::kLocalDomain, /*required_features=*/0,
                           /*preferred_features=*/0,
                           /*local_barrier_count=*/1),
          capabilities));
  ASSERT_OK_AND_ASSIGN(
      NcclDeviceCommPlan two_barriers,
      BuildNcclDeviceCommPlan(
          MakeRequirements(PeerAccess::kLocalDomain, /*required_features=*/0,
                           /*preferred_features=*/0,
                           /*local_barrier_count=*/2),
          capabilities));

  std::string payload = NcclDeviceCommPlanAgreementPayload(
      one_barrier, capabilities, /*runtime_version=*/NCCL_VERSION_CODE);
  EXPECT_NE(payload, NcclDeviceCommPlanAgreementPayload(
                         two_barriers, capabilities,
                         /*runtime_version=*/NCCL_VERSION_CODE));
  EXPECT_NE(payload, NcclDeviceCommPlanAgreementPayload(
                         one_barrier, capabilities,
                         /*runtime_version=*/NCCL_VERSION_CODE + 1));
}

TEST(NcclDeviceAbiTest, RejectsVersionMismatch) {
  EXPECT_THAT(
      ValidateNcclDeviceAbi(/*compile_time_version=*/22907,
                            /*runtime_version=*/23000),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("XLA compile-time version=22907, loaded runtime "
                         "version=23000")));
}

TEST(NcclWindowDeviceAbiTest, RequiresExactRuntimeVersion) {
  EXPECT_OK(ValidateNcclWindowDeviceAbi(/*compile_time_version=*/22907,
                                        /*runtime_version=*/22907));
  EXPECT_THAT(
      ValidateNcclWindowDeviceAbi(/*compile_time_version=*/22907,
                                  /*runtime_version=*/23000),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("XLA compile-time version=22907, loaded runtime "
                         "version=23000")));
}

TEST(NcclWindowDeviceAbiTest, AgreementPayloadIncludesSizeAndRuntimeAbi) {
  std::string payload = NcclSymmetricMemoryPlanAgreementPayload(
      /*size=*/1024, /*runtime_version=*/NCCL_VERSION_CODE);
  EXPECT_NE(payload, NcclSymmetricMemoryPlanAgreementPayload(
                         /*size=*/2048,
                         /*runtime_version=*/NCCL_VERSION_CODE));
  EXPECT_NE(payload, NcclSymmetricMemoryPlanAgreementPayload(
                         /*size=*/1024,
                         /*runtime_version=*/NCCL_VERSION_CODE + 1));
}

}  // namespace
}  // namespace xla::gpu
