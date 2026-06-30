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
#include <tuple>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/thunk_id.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/c_api_gpu_collectives.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
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

  static absl::Status PreparePackedDestination(
      XLA_FFI_GpuCollective_PackedKernelArg* output, uint64_t schema,
      uint64_t abi_version, size_t size, size_t alignment) {
    return FfiCollectiveResources::PreparePackedDestination(
        output, schema, abi_version, size, alignment);
  }

  static size_t TaggedBufferCount(const FfiCollectiveResources& resources) {
    size_t count = 0;
    for (const auto& buffer : resources.static_buffers_) {
      if (buffer.memory_space == 1) ++count;
    }
    return count;
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
                                results);
}

Shape MakeU8Shape(int64_t size, int64_t memory_space) {
  Shape shape = ShapeUtil::MakeShape(U8, {size});
  shape.mutable_layout()->set_memory_space(memory_space);
  return shape;
}

TEST(FfiCollectiveResourcesTest, CallsiteDomainIsStableAndIsolated) {
  FfiCollectiveResources first = MakeResources(ThunkId(7));
  FfiCollectiveResources same_callsite = MakeResources(ThunkId(7));
  FfiCollectiveResources other_callsite = MakeResources(ThunkId(8));
  FfiCollectiveResources other_target("other", "profile", ThunkId(7), {}, {});

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

TEST(FfiCollectiveResourcesTest, PackedArgumentQueryReportsExactRequirements) {
  XLA_FFI_GpuCollective_PackedKernelArg packed{
      XLA_FFI_GpuCollective_PackedKernelArg_STRUCT_SIZE};
  EXPECT_THAT(FfiCollectiveResourcesTestPeer::PreparePackedDestination(
                  &packed, /*schema=*/17, /*abi_version=*/18, /*size=*/8,
                  /*alignment=*/8),
              IsOk());
  EXPECT_EQ(packed.size, 8);
  EXPECT_EQ(packed.alignment, 8);
  EXPECT_EQ(packed.schema, 17);
  EXPECT_EQ(packed.abi_version, 18);

  alignas(8) std::array<std::byte, 16> storage;
  packed.destination = storage.data();
  packed.capacity = 7;
  EXPECT_THAT(FfiCollectiveResourcesTestPeer::PreparePackedDestination(
                  &packed, 17, 18, 8, 8),
              StatusIs(absl::StatusCode::kResourceExhausted));
  packed.capacity = 8;
  EXPECT_THAT(FfiCollectiveResourcesTestPeer::PreparePackedDestination(
                  &packed, 17, 18, 8, 8),
              IsOk());
  packed.destination = storage.data() + 1;
  EXPECT_THAT(
      FfiCollectiveResourcesTestPeer::PreparePackedDestination(&packed, 17, 18,
                                                               8, 8),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("alignment")));

  packed.destination = nullptr;
  packed.capacity = 0;
  EXPECT_THAT(FfiCollectiveResourcesTestPeer::PreparePackedDestination(
                  &packed, /*schema=*/0, /*abi_version=*/18, 8, 8),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("invalid kernel argument metadata")));
}

TEST(FfiCollectiveResourcesTest, RequiresDeclarativePreparationBeforeQueries) {
  FfiCollectiveResources resources = MakeResources(ThunkId(1));
  ASSERT_OK(resources.BeginInvocation(XLA_FFI_ExecutionStage_EXECUTE, nullptr,
                                      nullptr, nullptr, nullptr, nullptr,
                                      nullptr));

  XLA_FFI_GpuCollective_PackedKernelArg packed{
      XLA_FFI_GpuCollective_PackedKernelArg_STRUCT_SIZE};
  XLA_FFI_GpuCollectives_GetDeviceComm_Args get{
      XLA_FFI_GpuCollectives_GetDeviceComm_Args_STRUCT_SIZE,
      /*extension_start=*/nullptr,
      /*ctx=*/nullptr, &packed};
  EXPECT_THAT(resources.GetDeviceComm(&get),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("USES_DEVICE_COMMUNICATION")));
  EXPECT_THAT(resources.PrepareDeviceCommunication(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("only be prepared during FFI Prepare")));
}

TEST(FfiCollectiveResourcesTest, PreparationRequiresCallsiteAndGpuResources) {
  FfiCollectiveResources no_callsite = MakeResources(ThunkId(0));
  ASSERT_OK(no_callsite.BeginInvocation(XLA_FFI_ExecutionStage_PREPARE, nullptr,
                                        nullptr, nullptr, nullptr, nullptr,
                                        nullptr));
  EXPECT_THAT(no_callsite.PrepareDeviceCommunication(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("non-zero thunk id")));

  FfiCollectiveResources no_resources = MakeResources(ThunkId(1));
  ASSERT_OK(no_resources.BeginInvocation(XLA_FFI_ExecutionStage_PREPARE,
                                         nullptr, nullptr, nullptr, nullptr,
                                         nullptr, nullptr));
  EXPECT_THAT(no_resources.PrepareDeviceCommunication(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("GPU collective Prepare resources")));
}

}  // namespace
}  // namespace xla::gpu
