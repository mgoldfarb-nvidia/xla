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

#include "xla/backends/gpu/runtime/ffi_collective_resources.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/loopback_collectives.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_memory.h"
#include "xla/backends/gpu/runtime/collective_memory_requests.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/thunk_id.h"
#include "xla/executable_run_options.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/c_api_gpu_collectives.h"
#include "xla/runtime/device_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_slice.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

class FfiCollectiveResourcesTestPeer {
 public:
  static absl::StatusOr<
      std::tuple<BufferAllocation::Index, uint64_t, uint64_t, uint64_t>>
  FindBufferView(const FfiCollectiveResources& resources,
                 const XLA_FFI_Buffer& buffer) {
    auto view = resources.FindBufferView(buffer);
    if (!view.ok()) return view.status();
    return std::make_tuple(view->allocation, view->offset, view->size,
                           view->allocation_size);
  }

  static absl::Status ValidateKernelArgDestination(void* destination,
                                                   size_t destination_size,
                                                   size_t packed_size) {
    return FfiCollectiveResources::ValidateKernelArgDestination(
        destination, destination_size, packed_size);
  }

  static size_t TaggedBufferCount(const FfiCollectiveResources& resources) {
    size_t count = 0;
    for (const auto& buffer : resources.static_buffers_) {
      if (buffer.memory_space == 1) ++count;
    }
    return count;
  }

  static absl::StatusOr<GpuDeviceCommunicator::Requirements>
  NormalizeRequirements(
      const XLA_FFI_GpuDeviceCommunication_Requirements& requirements) {
    return FfiCollectiveResources::NormalizeRequirements(requirements);
  }

  static std::optional<GpuDeviceCommunicator::Requirements>
  RequestedRequirements(const FfiCollectiveResources& resources) {
    return resources.requested_requirements_;
  }

  static absl::Status ValidateResolvedInfo(
      const GpuDeviceCommunicator::Requirements& requirements,
      const GpuDeviceCommunicator::Info& info) {
    return FfiCollectiveResources::ValidateResolvedInfo(requirements, info);
  }
};

namespace {

using absl_testing::IsOk;
using absl_testing::StatusIs;
using ::testing::HasSubstr;

FfiCollectiveResources MakeResources(
    ThunkId thunk_id, absl::Span<const NullableShapedSlice> operands = {},
    absl::Span<const NullableShapedSlice> results = {}) {
  return FfiCollectiveResources("target", "profile", thunk_id, operands,
                                results,
                                /*uses_device_communication=*/true);
}

Shape MakeU8Shape(int64_t size, int64_t memory_space) {
  Shape shape = ShapeUtil::MakeShape(U8, {size});
  shape.mutable_layout()->set_memory_space(memory_space);
  return shape;
}

class TestSymmetricMemory final : public SymmetricMemory {
 public:
  static constexpr uint64_t kAbiSchema = 0x746573745f6d656d;
  static constexpr uint64_t kAbiVersion = 7;
  using Handle = std::array<uint64_t, 5>;

  TestSymmetricMemory(se::DeviceAddressBase address, Handle handle)
      : SymmetricMemory(kAbiSchema, kAbiVersion),
        address_(address),
        handle_(handle) {}

  se::DeviceAddressBase addr() const override { return address_; }

  std::string ToString() const override { return "test symmetric memory"; }

  PackedKernelArg PackKernelArg() const override {
    return PackedKernelArg(sizeof(handle_), [&](absl::Span<char> packed) {
      std::memcpy(packed.data(), handle_.data(), sizeof(handle_));
    });
  }

 private:
  se::DeviceAddressBase address_;
  Handle handle_;
};

TEST(FfiCollectiveResourcesTest, CallsiteDomainIsStableAndIsolated) {
  FfiCollectiveResources first = MakeResources(ThunkId(7));
  FfiCollectiveResources same_callsite = MakeResources(ThunkId(7));
  FfiCollectiveResources other_callsite = MakeResources(ThunkId(8));
  FfiCollectiveResources other_target("other", "profile", ThunkId(7), {}, {},
                                      /*uses_device_communication=*/true);

  EXPECT_EQ(first.resource_domain(), same_callsite.resource_domain());
  EXPECT_NE(first.resource_domain(), other_callsite.resource_domain());
  EXPECT_NE(first.resource_domain(), other_target.resource_domain());
  EXPECT_NE(first.resource_domain() & (uint64_t{1} << 63), 0);
}

TEST(FfiCollectiveResourcesTest, DiscoversCollectiveMemoryTags) {
  // Logical S(0) buffers may reuse holes in a physical S(1) allocation. Tag
  // discovery must therefore use each operand/result shape, not allocation
  // color.
  BufferAllocation shared(/*index=*/0, /*size=*/64, /*color=*/1);
  std::vector<NullableShapedSlice> operands{
      ShapedSlice{BufferAllocation::Slice(&shared, 0, 16),
                  MakeU8Shape(16, /*memory_space=*/1)},
      ShapedSlice{BufferAllocation::Slice(&shared, 32, 16),
                  MakeU8Shape(16, /*memory_space=*/0)}};
  std::vector<NullableShapedSlice> results{
      ShapedSlice{BufferAllocation::Slice(&shared, 0, 16),
                  MakeU8Shape(16, /*memory_space=*/1)}};

  FfiCollectiveResources resources =
      MakeResources(ThunkId(1), operands, results);
  EXPECT_EQ(FfiCollectiveResourcesTestPeer::TaggedBufferCount(resources), 2);
}

TEST(FfiCollectiveResourcesTest, NormalizesExplicitRequirements) {
  XLA_FFI_GpuDeviceCommunication_Requirements requirements = {};
  requirements.struct_size =
      XLA_FFI_GpuDeviceCommunication_Requirements_STRUCT_SIZE;
  requirements.peer_access = XLA_FFI_GPU_PEER_ACCESS_HIERARCHICAL;
  requirements.required_features =
      XLA_FFI_GPU_DEVICE_COMM_FEATURE_LOCAL_MULTICAST;
  requirements.preferred_features =
      XLA_FFI_GPU_DEVICE_COMM_FEATURE_LOCAL_MULTICAST |
      XLA_FFI_GPU_DEVICE_COMM_FEATURE_NETWORK_DEVICE_OPERATIONS |
      (uint64_t{1} << 63);
  requirements.local_barrier_count = 2;
  requirements.team_barrier_count = 3;
  requirements.notification_slot_count = 4;
  requirements.completion_slot_count = 5;

  ASSERT_OK_AND_ASSIGN(
      GpuDeviceCommunicator::Requirements normalized,
      FfiCollectiveResourcesTestPeer::NormalizeRequirements(requirements));
  EXPECT_EQ(normalized.peer_access,
            GpuDeviceCommunicator::PeerAccess::kHierarchical);
  EXPECT_EQ(normalized.required_features,
            GpuDeviceCommunicator::kLocalMulticast);
  EXPECT_EQ(normalized.preferred_features,
            GpuDeviceCommunicator::kNetworkDeviceOperations);
  EXPECT_EQ(normalized.local_barrier_count, 2);
  EXPECT_EQ(normalized.team_barrier_count, 3);
  EXPECT_EQ(normalized.notification_slot_count, 4);
  EXPECT_EQ(normalized.completion_slot_count, 5);
}

TEST(FfiCollectiveResourcesTest, RejectsInvalidRequirements) {
  XLA_FFI_GpuDeviceCommunication_Requirements requirements = {};
  requirements.struct_size =
      XLA_FFI_GpuDeviceCommunication_Requirements_STRUCT_SIZE;
  requirements.peer_access = XLA_FFI_GPU_PEER_ACCESS_LOCAL_DOMAIN;
  requirements.required_features = uint64_t{1} << 63;
  EXPECT_THAT(
      FfiCollectiveResourcesTestPeer::NormalizeRequirements(requirements),
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("Unknown required")));

  requirements.required_features = 0;
  requirements.peer_access = static_cast<XLA_FFI_GpuPeerAccess>(99);
  EXPECT_THAT(
      FfiCollectiveResourcesTestPeer::NormalizeRequirements(requirements),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Unknown GPU peer access")));

  requirements.peer_access = XLA_FFI_GPU_PEER_ACCESS_LOCAL_DOMAIN;
  requirements.team_barrier_count =
      static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) + 1;
  EXPECT_THAT(
      FfiCollectiveResourcesTestPeer::NormalizeRequirements(requirements),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("team_barrier_count")));
}

TEST(FfiCollectiveResourcesTest,
     RequestIsIdempotentAfterNormalizationAndRejectsConflict) {
  FfiCollectiveResources resources = MakeResources(ThunkId(1));
  ASSERT_OK(resources.BeginInvocation(XLA_FFI_ExecutionStage_PREPARE, nullptr,
                                      nullptr, nullptr, nullptr, nullptr,
                                      nullptr));

  XLA_FFI_GpuDeviceCommunication_Requirements requirements = {};
  requirements.struct_size =
      XLA_FFI_GpuDeviceCommunication_Requirements_STRUCT_SIZE;
  requirements.peer_access = XLA_FFI_GPU_PEER_ACCESS_DIRECT_ANY_PEER;
  requirements.required_features =
      XLA_FFI_GPU_DEVICE_COMM_FEATURE_NETWORK_DEVICE_OPERATIONS;
  requirements.preferred_features =
      XLA_FFI_GPU_DEVICE_COMM_FEATURE_NETWORK_DEVICE_OPERATIONS |
      (uint64_t{1} << 63);
  requirements.team_barrier_count = 2;
  XLA_FFI_GpuCollectives_RequestDeviceCommunication_Args request = {};
  request.struct_size =
      XLA_FFI_GpuCollectives_RequestDeviceCommunication_Args_STRUCT_SIZE;
  request.requirements = &requirements;

  ASSERT_OK(resources.RequestDeviceCommunication(&request));
  ASSERT_OK(resources.RequestDeviceCommunication(&request));
  ASSERT_TRUE(FfiCollectiveResourcesTestPeer::RequestedRequirements(resources)
                  .has_value());
  EXPECT_EQ(FfiCollectiveResourcesTestPeer::RequestedRequirements(resources)
                ->preferred_features,
            0);

  requirements.team_barrier_count = 3;
  EXPECT_THAT(
      resources.RequestDeviceCommunication(&request),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("Conflicting")));
}

TEST(FfiCollectiveResourcesTest, RequestRequiresHandlerTrait) {
  FfiCollectiveResources resources("target", "profile", ThunkId(1), {}, {},
                                   /*uses_device_communication=*/false);
  ASSERT_OK(resources.BeginInvocation(XLA_FFI_ExecutionStage_PREPARE, nullptr,
                                      nullptr, nullptr, nullptr, nullptr,
                                      nullptr));

  XLA_FFI_GpuDeviceCommunication_Requirements requirements = {};
  requirements.struct_size =
      XLA_FFI_GpuDeviceCommunication_Requirements_STRUCT_SIZE;
  XLA_FFI_GpuCollectives_RequestDeviceCommunication_Args request = {};
  request.struct_size =
      XLA_FFI_GpuCollectives_RequestDeviceCommunication_Args_STRUCT_SIZE;
  request.requirements = &requirements;
  EXPECT_THAT(resources.RequestDeviceCommunication(&request),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("USES_DEVICE_COMMUNICATION")));
}

TEST(FfiCollectiveResourcesTest, FreezesNormalizedProfileAcrossInvocations) {
  FfiDeviceCommunicationProfile profile;
  GpuDeviceCommunicator::Requirements first;
  first.team_barrier_count = 1;
  EXPECT_THAT(profile.Freeze(first), IsOk());
  EXPECT_THAT(profile.Freeze(first), IsOk());

  GpuDeviceCommunicator::Requirements changed = first;
  changed.team_barrier_count = 2;
  EXPECT_THAT(profile.Freeze(changed),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("changed across invocations")));
}

TEST(FfiCollectiveResourcesTest, ValidatesResolvedPeerAccess) {
  GpuDeviceCommunicator::Requirements requirements;
  GpuDeviceCommunicator::Info info;
  info.rank = 0;
  info.team_size = 4;
  info.local_rank = 0;
  info.local_domain_size = 2;
  info.local_domain_count = 2;
  info.topology = GpuDeviceCommunicator::Topology::kLocalDomain;

  EXPECT_THAT(
      FfiCollectiveResourcesTestPeer::ValidateResolvedInfo(requirements, info),
      IsOk());

  requirements.peer_access = GpuDeviceCommunicator::PeerAccess::kHierarchical;
  EXPECT_THAT(
      FfiCollectiveResourcesTestPeer::ValidateResolvedInfo(requirements, info),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("hierarchical peer access")));
  info.topology = GpuDeviceCommunicator::Topology::kHierarchical;
  EXPECT_THAT(
      FfiCollectiveResourcesTestPeer::ValidateResolvedInfo(requirements, info),
      IsOk());

  requirements.peer_access = GpuDeviceCommunicator::PeerAccess::kDirectAnyPeer;
  EXPECT_THAT(
      FfiCollectiveResourcesTestPeer::ValidateResolvedInfo(requirements, info),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("direct-any-peer access")));
  info.topology = GpuDeviceCommunicator::Topology::kAllPeers;
  EXPECT_THAT(
      FfiCollectiveResourcesTestPeer::ValidateResolvedInfo(requirements, info),
      IsOk());

  info.local_domain_size = info.team_size;
  EXPECT_THAT(
      FfiCollectiveResourcesTestPeer::ValidateResolvedInfo(requirements, info),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("inconsistent local-domain topology")));

  // A local-domain provider satisfies either wider reachability mode when its
  // single local domain already contains the complete communication team.
  info.local_domain_count = 1;
  info.topology = GpuDeviceCommunicator::Topology::kLocalDomain;
  requirements.peer_access = GpuDeviceCommunicator::PeerAccess::kHierarchical;
  EXPECT_THAT(
      FfiCollectiveResourcesTestPeer::ValidateResolvedInfo(requirements, info),
      IsOk());
  requirements.peer_access = GpuDeviceCommunicator::PeerAccess::kDirectAnyPeer;
  EXPECT_THAT(
      FfiCollectiveResourcesTestPeer::ValidateResolvedInfo(requirements, info),
      IsOk());
}

TEST(FfiCollectiveResourcesTest, ValidatesResolvedFeaturesAndCounts) {
  GpuDeviceCommunicator::Requirements requirements;
  requirements.required_features = GpuDeviceCommunicator::kLocalMulticast;
  requirements.local_barrier_count = 1;
  requirements.team_barrier_count = 2;
  requirements.notification_slot_count = 3;
  requirements.completion_slot_count = 4;

  GpuDeviceCommunicator::Info info;
  info.rank = 0;
  info.team_size = 1;
  info.local_rank = 0;
  info.local_domain_size = 1;
  info.local_domain_count = 1;
  EXPECT_THAT(
      FfiCollectiveResourcesTestPeer::ValidateResolvedInfo(requirements, info),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("omitted required feature")));

  info.enabled_features = GpuDeviceCommunicator::kLocalMulticast;
  EXPECT_THAT(
      FfiCollectiveResourcesTestPeer::ValidateResolvedInfo(requirements, info),
      StatusIs(absl::StatusCode::kInternal, HasSubstr("under-provisioned")));

  info.local_barrier_count = 1;
  info.team_barrier_count = 2;
  info.notification_slot_count = 3;
  info.completion_slot_count = 4;
  EXPECT_THAT(
      FfiCollectiveResourcesTestPeer::ValidateResolvedInfo(requirements, info),
      IsOk());
}

TEST(FfiCollectiveResourcesTest,
     RegistersEveryDistinctCollectiveMemoryAllocation) {
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("Host"));
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor,
                       platform->ExecutorForDevice(0));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                       executor->CreateStream());

  DeviceAssignment device_assignment(/*replica_count=*/1,
                                     /*computation_count=*/1);
  device_assignment(0, 0) = 0;
  LoopbackCollectives collectives;
  GpuExecutableRunOptions gpu_options;
  gpu_options.set_collectives(&collectives);
  ServiceExecutableRunOptions run_options;
  run_options.mutable_run_options()->set_stream(stream.get());
  run_options.mutable_run_options()->set_device_assignment(&device_assignment);
  run_options.mutable_run_options()->set_gpu_executable_run_options(
      &gpu_options);
  run_options.mutable_run_options()->set_local_device_count(1);
  ASSERT_OK_AND_ASSIGN(
      CollectiveParams collective_params,
      CollectiveParams::Create(run_options, /*async_streams=*/{},
                               LocalDeviceId(executor->device_ordinal())));

  std::array<std::byte, 64> first_storage;
  std::array<std::byte, 96> second_storage;
  std::array<std::byte, 32> ordinary_storage;
  BufferAllocation first(/*index=*/0, /*size=*/first_storage.size(),
                         /*color=*/1);
  BufferAllocation second(/*index=*/1, /*size=*/second_storage.size(),
                          /*color=*/1);
  BufferAllocation ordinary(/*index=*/2, /*size=*/ordinary_storage.size(),
                            /*color=*/1);
  std::vector<NullableShapedSlice> operands{
      ShapedSlice{BufferAllocation::Slice(&first, /*offset=*/8, /*size=*/16),
                  MakeU8Shape(16, /*memory_space=*/1)},
      ShapedSlice{BufferAllocation::Slice(&first, /*offset=*/32, /*size=*/16),
                  MakeU8Shape(16, /*memory_space=*/1)},
      ShapedSlice{BufferAllocation::Slice(&ordinary, /*offset=*/0, /*size=*/16),
                  MakeU8Shape(16, /*memory_space=*/0)}};
  std::vector<NullableShapedSlice> results{
      ShapedSlice{BufferAllocation::Slice(&second, /*offset=*/16, /*size=*/32),
                  MakeU8Shape(32, /*memory_space=*/1)}};
  FfiCollectiveResources resources =
      MakeResources(ThunkId(1), operands, results);
  EXPECT_EQ(FfiCollectiveResourcesTestPeer::TaggedBufferCount(resources), 3);

  std::array<se::DeviceAddressBase, 3> addresses{
      se::DeviceAddressBase(first_storage.data(), first_storage.size()),
      se::DeviceAddressBase(second_storage.data(), second_storage.size()),
      se::DeviceAddressBase(ordinary_storage.data(), ordinary_storage.size())};
  BufferAllocations buffers(addresses, executor->device_ordinal(), nullptr);
  CollectiveCliqueRequests clique_requests;
  CollectiveMemoryRequests memory_requests(buffers);
  ASSERT_OK(resources.BeginInvocation(XLA_FFI_ExecutionStage_PREPARE, &buffers,
                                      &collective_params, &clique_requests,
                                      &memory_requests, nullptr, nullptr));
  ASSERT_OK(resources.FinalizeDeviceCommunication());

  ASSERT_EQ(clique_requests.size(), 1);
  std::vector<CollectiveCliqueRequests::CliqueRequest> clique_request =
      clique_requests.OrderedRequestedCliques();
  ASSERT_EQ(clique_request.size(), 1);
  EXPECT_EQ(clique_request[0].dev_comms,
            (absl::btree_set<GpuDeviceCommunicator::Requirements>{
                GpuDeviceCommunicator::Requirements{.team_barrier_count = 1}}));
  std::vector<CollectiveMemoryRequests::CollectiveAllocations> memory_request =
      memory_requests.OrderedSymmetricAllocations();
  ASSERT_EQ(memory_request.size(), 1);
  EXPECT_EQ(memory_request[0].clique, clique_request[0].key);
  EXPECT_EQ(memory_request[0].allocations,
            (absl::btree_set<BufferAllocation::Index>{0, 1}));

  TestSymmetricMemory::Handle expected_handle = {11, 22, 33, 44, 55};
  absl::flat_hash_map<CollectiveMemory::Key,
                      std::shared_ptr<SymmetricMemory>>
      symmetric_memories;
  symmetric_memories.emplace(
      CollectiveMemory::Key{clique_request[0].key, first.index()},
      std::make_shared<TestSymmetricMemory>(addresses[0], expected_handle));
  CollectiveMemory collective_memory(
      buffers, std::move(symmetric_memories),
      /*mcast_memories=*/{}, /*peer_memories=*/{});

  ASSERT_OK(resources.BeginInvocation(XLA_FFI_ExecutionStage_EXECUTE, &buffers,
                                      nullptr, nullptr, nullptr, nullptr,
                                      &collective_memory));
  int64_t dim = 4;
  XLA_FFI_Buffer buffer{XLA_FFI_Buffer_STRUCT_SIZE,
                        /*extension_start=*/nullptr,
                        XLA_FFI_DataType_U8,
                        first_storage.data() + 12,
                        1,
                        &dim};
  ASSERT_OK_AND_ASSIGN(
      auto first_view,
      FfiCollectiveResourcesTestPeer::FindBufferView(resources, buffer));
  EXPECT_EQ(std::get<0>(first_view), 0);
  EXPECT_EQ(std::get<1>(first_view), 12);

  TestSymmetricMemory::Handle actual_handle = {};
  XLA_FFI_GpuCollectives_GetRegisteredMemoryHandle_Args get_memory = {};
  get_memory.struct_size =
      XLA_FFI_GpuCollectives_GetRegisteredMemoryHandle_Args_STRUCT_SIZE;
  get_memory.buffer = &buffer;
  get_memory.expected_abi_schema = TestSymmetricMemory::kAbiSchema;
  get_memory.expected_abi_version = TestSymmetricMemory::kAbiVersion;
  get_memory.destination = actual_handle.data();
  get_memory.destination_size = sizeof(actual_handle);
  ASSERT_OK(resources.GetRegisteredMemoryHandle(&get_memory));
  EXPECT_EQ(actual_handle, expected_handle);
  EXPECT_EQ(get_memory.offset, 12);

  get_memory.destination_size = sizeof(uint64_t);
  EXPECT_THAT(resources.GetRegisteredMemoryHandle(&get_memory),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("destination size mismatch")));

  dim = 8;
  buffer.data = second_storage.data() + 24;
  ASSERT_OK_AND_ASSIGN(
      auto second_view,
      FfiCollectiveResourcesTestPeer::FindBufferView(resources, buffer));
  EXPECT_EQ(std::get<0>(second_view), 1);
  EXPECT_EQ(std::get<1>(second_view), 24);

  dim = 4;
  buffer.data = ordinary_storage.data() + 4;
  EXPECT_THAT(
      FfiCollectiveResourcesTestPeer::FindBufferView(resources, buffer),
      StatusIs(absl::StatusCode::kFailedPrecondition, HasSubstr("not tagged")));

  // A handler that only uses communicator resources must not be forced to add
  // a dummy tagged buffer.
  FfiCollectiveResources communicator_only = MakeResources(ThunkId(2));
  CollectiveCliqueRequests communicator_only_cliques;
  ASSERT_OK(communicator_only.BeginInvocation(
      XLA_FFI_ExecutionStage_PREPARE, &buffers, &collective_params,
      &communicator_only_cliques,
      /*collective_memory_requests=*/nullptr, nullptr, nullptr));
  ASSERT_OK(communicator_only.FinalizeDeviceCommunication());
  ASSERT_EQ(communicator_only_cliques.size(), 1);
  auto communicator_only_requests =
      communicator_only_cliques.OrderedRequestedCliques();
  ASSERT_EQ(communicator_only_requests.size(), 1);
  EXPECT_THAT(communicator_only_requests[0].dev_comms,
              ::testing::ElementsAre(GpuDeviceCommunicator::Requirements{
                  .team_barrier_count = 1}));

  FfiCollectiveResources explicitly_requested = MakeResources(ThunkId(3));
  CollectiveCliqueRequests explicit_cliques;
  CollectiveMemoryRequests explicit_memory(buffers);
  ASSERT_OK(explicitly_requested.BeginInvocation(
      XLA_FFI_ExecutionStage_PREPARE, &buffers, &collective_params,
      &explicit_cliques, &explicit_memory, nullptr, nullptr));
  XLA_FFI_GpuDeviceCommunication_Requirements explicit_requirements = {};
  explicit_requirements.struct_size =
      XLA_FFI_GpuDeviceCommunication_Requirements_STRUCT_SIZE;
  explicit_requirements.peer_access = XLA_FFI_GPU_PEER_ACCESS_HIERARCHICAL;
  explicit_requirements.required_features =
      XLA_FFI_GPU_DEVICE_COMM_FEATURE_NETWORK_DEVICE_OPERATIONS;
  explicit_requirements.local_barrier_count = 2;
  explicit_requirements.team_barrier_count = 3;
  explicit_requirements.notification_slot_count = 4;
  explicit_requirements.completion_slot_count = 5;
  XLA_FFI_GpuCollectives_RequestDeviceCommunication_Args explicit_request = {};
  explicit_request.struct_size =
      XLA_FFI_GpuCollectives_RequestDeviceCommunication_Args_STRUCT_SIZE;
  explicit_request.requirements = &explicit_requirements;
  ASSERT_OK(explicitly_requested.RequestDeviceCommunication(&explicit_request));
  ASSERT_OK(explicitly_requested.FinalizeDeviceCommunication());
  auto explicit_clique_requests = explicit_cliques.OrderedRequestedCliques();
  ASSERT_EQ(explicit_clique_requests.size(), 1);
  GpuDeviceCommunicator::Requirements expected_explicit;
  expected_explicit.peer_access =
      GpuDeviceCommunicator::PeerAccess::kHierarchical;
  expected_explicit.required_features =
      GpuDeviceCommunicator::kNetworkDeviceOperations;
  expected_explicit.local_barrier_count = 2;
  expected_explicit.team_barrier_count = 3;
  expected_explicit.notification_slot_count = 4;
  expected_explicit.completion_slot_count = 5;
  EXPECT_THAT(explicit_clique_requests[0].dev_comms,
              ::testing::ElementsAre(expected_explicit));
}

TEST(FfiCollectiveResourcesTest, SkipsEmptyTaggedBuffersDuringPreparation) {
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("Host"));
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor,
                       platform->ExecutorForDevice(0));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                       executor->CreateStream());

  DeviceAssignment device_assignment(/*replica_count=*/1,
                                     /*computation_count=*/1);
  device_assignment(0, 0) = 0;
  LoopbackCollectives collectives;
  GpuExecutableRunOptions gpu_options;
  gpu_options.set_collectives(&collectives);
  ServiceExecutableRunOptions run_options;
  run_options.mutable_run_options()->set_stream(stream.get());
  run_options.mutable_run_options()->set_device_assignment(&device_assignment);
  run_options.mutable_run_options()->set_gpu_executable_run_options(
      &gpu_options);
  run_options.mutable_run_options()->set_local_device_count(1);
  ASSERT_OK_AND_ASSIGN(
      CollectiveParams collective_params,
      CollectiveParams::Create(run_options, /*async_streams=*/{},
                               LocalDeviceId(executor->device_ordinal())));

  BufferAllocation empty(/*index=*/0, /*size=*/0, /*color=*/1);
  std::vector<NullableShapedSlice> operands{
      ShapedSlice{BufferAllocation::Slice(&empty, /*offset=*/0, /*size=*/0),
                  MakeU8Shape(0, /*memory_space=*/1)}};
  FfiCollectiveResources resources = MakeResources(ThunkId(1), operands);

  std::array<se::DeviceAddressBase, 1> addresses{
      se::DeviceAddressBase(nullptr, 0)};
  BufferAllocations buffers(addresses, executor->device_ordinal(), nullptr);
  CollectiveCliqueRequests clique_requests;
  CollectiveMemoryRequests memory_requests(buffers);
  ASSERT_OK(resources.BeginInvocation(XLA_FFI_ExecutionStage_PREPARE, &buffers,
                                      &collective_params, &clique_requests,
                                      &memory_requests, nullptr, nullptr));
  EXPECT_OK(resources.FinalizeDeviceCommunication());
  EXPECT_EQ(memory_requests.symmetric_size(), 0);
  EXPECT_EQ(clique_requests.size(), 1);

  ASSERT_OK(resources.BeginInvocation(XLA_FFI_ExecutionStage_EXECUTE, &buffers,
                                      nullptr, nullptr, nullptr, nullptr,
                                      nullptr));
  std::byte dummy;
  int64_t dim = 0;
  XLA_FFI_Buffer buffer{XLA_FFI_Buffer_STRUCT_SIZE,
                        /*extension_start=*/nullptr,
                        XLA_FFI_DataType_U8,
                        &dummy,
                        /*rank=*/1,
                        &dim};
  EXPECT_THAT(FfiCollectiveResourcesTestPeer::FindBufferView(resources, buffer),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("must not be empty")));
}

TEST(FfiCollectiveResourcesTest,
     ResolvesTaggedViewsAndRejectsUntaggedOrOutOfBoundsViews) {
  std::array<std::byte, 64> tagged_storage;
  std::array<std::byte, 64> ordinary_storage;
  BufferAllocation tagged(/*index=*/0, /*size=*/tagged_storage.size(),
                          /*color=*/1);
  BufferAllocation ordinary(/*index=*/1, /*size=*/ordinary_storage.size(),
                            /*color=*/1);
  std::vector<NullableShapedSlice> operands{
      ShapedSlice{BufferAllocation::Slice(&tagged, /*offset=*/8, /*size=*/32),
                  MakeU8Shape(32, /*memory_space=*/1)},
      ShapedSlice{BufferAllocation::Slice(&ordinary, /*offset=*/0, /*size=*/32),
                  MakeU8Shape(32, /*memory_space=*/0)}};
  FfiCollectiveResources resources = MakeResources(ThunkId(1), operands);

  std::array<se::DeviceAddressBase, 2> addresses{
      se::DeviceAddressBase(tagged_storage.data(), tagged_storage.size()),
      se::DeviceAddressBase(ordinary_storage.data(), ordinary_storage.size())};
  BufferAllocations buffers(addresses, 0, nullptr);
  ASSERT_OK(resources.BeginInvocation(XLA_FFI_ExecutionStage_EXECUTE, &buffers,
                                      nullptr, nullptr, nullptr, nullptr,
                                      nullptr));

  int64_t dim = 4;
  XLA_FFI_Buffer buffer{XLA_FFI_Buffer_STRUCT_SIZE,
                        /*extension_start=*/nullptr,
                        XLA_FFI_DataType_U8,
                        tagged_storage.data() + 12,
                        1,
                        &dim};
  ASSERT_OK_AND_ASSIGN(
      auto view,
      FfiCollectiveResourcesTestPeer::FindBufferView(resources, buffer));
  EXPECT_EQ(std::get<0>(view), 0);
  EXPECT_EQ(std::get<1>(view), 12);
  EXPECT_EQ(std::get<2>(view), 4);
  EXPECT_EQ(std::get<3>(view), tagged_storage.size());

  buffer.data = ordinary_storage.data() + 4;
  EXPECT_THAT(
      FfiCollectiveResourcesTestPeer::FindBufferView(resources, buffer),
      StatusIs(absl::StatusCode::kFailedPrecondition, HasSubstr("not tagged")));

  dim = 2;
  buffer.data = tagged_storage.data() + 39;
  EXPECT_THAT(
      FfiCollectiveResourcesTestPeer::FindBufferView(resources, buffer),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("exceeds")));

  buffer.data = tagged_storage.data() + 48;
  EXPECT_THAT(FfiCollectiveResourcesTestPeer::FindBufferView(resources, buffer),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("not an FFI operand or result")));
}

TEST(FfiCollectiveResourcesTest, RejectsAmbiguousAliasedAllocations) {
  std::array<std::byte, 64> storage;
  BufferAllocation first(/*index=*/0, /*size=*/storage.size(), /*color=*/1);
  BufferAllocation second(/*index=*/1, /*size=*/storage.size(), /*color=*/1);
  std::vector<NullableShapedSlice> operands{
      ShapedSlice{BufferAllocation::Slice(&first, 8, 32),
                  MakeU8Shape(32, /*memory_space=*/1)},
      ShapedSlice{BufferAllocation::Slice(&second, 8, 32),
                  MakeU8Shape(32, /*memory_space=*/1)}};
  FfiCollectiveResources resources = MakeResources(ThunkId(1), operands);
  std::array<se::DeviceAddressBase, 2> addresses{
      se::DeviceAddressBase(storage.data(), storage.size()),
      se::DeviceAddressBase(storage.data(), storage.size())};
  BufferAllocations buffers(addresses, 0, nullptr);
  ASSERT_OK(resources.BeginInvocation(XLA_FFI_ExecutionStage_EXECUTE, &buffers,
                                      nullptr, nullptr, nullptr, nullptr,
                                      nullptr));

  int64_t dim = 4;
  XLA_FFI_Buffer buffer{XLA_FFI_Buffer_STRUCT_SIZE,
                        /*extension_start=*/nullptr,
                        XLA_FFI_DataType_U8,
                        storage.data() + 12,
                        1,
                        &dim};
  EXPECT_THAT(FfiCollectiveResourcesTestPeer::FindBufferView(resources, buffer),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("aliases multiple allocations")));
}

TEST(FfiCollectiveResourcesTest, ValidatesKernelArgumentDestination) {
  std::array<std::byte, 8> storage;
  EXPECT_THAT(FfiCollectiveResourcesTestPeer::ValidateKernelArgDestination(
                  storage.data(), storage.size(), /*packed_size=*/8),
              IsOk());

  EXPECT_THAT(FfiCollectiveResourcesTestPeer::ValidateKernelArgDestination(
                  /*destination=*/nullptr, storage.size(), /*packed_size=*/8),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("must not be null")));
  EXPECT_THAT(FfiCollectiveResourcesTestPeer::ValidateKernelArgDestination(
                  storage.data(), storage.size(), /*packed_size=*/0),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("empty kernel argument")));
  EXPECT_THAT(FfiCollectiveResourcesTestPeer::ValidateKernelArgDestination(
                  storage.data(), /*destination_size=*/7, /*packed_size=*/8),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("destination size mismatch")));
}

TEST(FfiCollectiveResourcesTest, RequiresDeclarativePreparationBeforeAccess) {
  FfiCollectiveResources resources = MakeResources(ThunkId(1));
  ASSERT_OK(resources.BeginInvocation(XLA_FFI_ExecutionStage_EXECUTE, nullptr,
                                      nullptr, nullptr, nullptr, nullptr,
                                      nullptr));

  XLA_FFI_GpuCollectives_GetDeviceComm_Args get{
      XLA_FFI_GpuCollectives_GetDeviceComm_Args_STRUCT_SIZE,
      /*extension_start=*/nullptr,
      /*ctx=*/nullptr,
      /*expected_abi_schema=*/17,
      /*expected_abi_version=*/18,
      /*destination=*/nullptr,
      /*destination_size=*/0};
  EXPECT_THAT(resources.GetDeviceComm(&get),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("USES_DEVICE_COMMUNICATION")));
  EXPECT_THAT(resources.FinalizeDeviceCommunication(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("only be prepared during FFI Prepare")));
}

TEST(FfiCollectiveResourcesTest, PreparationRequiresCallsiteAndGpuResources) {
  FfiCollectiveResources no_callsite = MakeResources(ThunkId(0));
  ASSERT_OK(no_callsite.BeginInvocation(XLA_FFI_ExecutionStage_PREPARE, nullptr,
                                        nullptr, nullptr, nullptr, nullptr,
                                        nullptr));
  EXPECT_THAT(no_callsite.FinalizeDeviceCommunication(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("non-zero thunk id")));

  FfiCollectiveResources no_resources = MakeResources(ThunkId(1));
  ASSERT_OK(no_resources.BeginInvocation(XLA_FFI_ExecutionStage_PREPARE,
                                         nullptr, nullptr, nullptr, nullptr,
                                         nullptr, nullptr));
  EXPECT_THAT(no_resources.FinalizeDeviceCommunication(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("GPU collective Prepare resources")));
}

}  // namespace
}  // namespace xla::gpu
