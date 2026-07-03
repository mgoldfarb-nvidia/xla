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

#include "xla/service/gpu/buffer_allocations.h"

#include <optional>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"

namespace xla::gpu {
namespace {

class RecordingAllocator final : public se::DeviceAddressAllocator {
 public:
  RecordingAllocator() : DeviceAddressAllocator(/*platform=*/nullptr) {}

  absl::StatusOr<se::ScopedDeviceAddress<uint8_t>> Allocate(
      int device_ordinal, uint64_t size, bool retry_on_failure,
      int64_t memory_space) override {
    return absl::UnimplementedError("not used");
  }

  absl::StatusOr<se::Stream*> GetStream(int device_ordinal) override {
    return nullptr;
  }

  absl::Status Deallocate(int device_ordinal,
                          se::DeviceAddressBase address) override {
    ordinals.push_back(device_ordinal);
    addresses.push_back(address);
    if (address.opaque() == failing_address) {
      return absl::InternalError("deallocation failed");
    }
    return absl::OkStatus();
  }

  void* failing_address = nullptr;
  std::vector<int> ordinals;
  std::vector<se::DeviceAddressBase> addresses;
};

TEST(BufferAllocationsTest, FindAllocationIndexReturnsCorrectIndex) {
  // Two 64-byte buffers at distinct addresses.
  char buf0[64];
  char buf1[64];
  std::vector<se::DeviceAddressBase> buffers = {
      se::DeviceAddressBase(buf0, 64),
      se::DeviceAddressBase(buf1, 64),
  };

  BufferAllocations allocs(buffers, /*device_ordinal=*/0,
                           /*memory_allocator=*/nullptr);

  // Address at the start of buf0 → index 0.
  EXPECT_EQ(allocs.FindAllocationIndex(se::DeviceAddressBase(buf0, 1)), 0);

  // Address in the middle of buf1 → index 1.
  EXPECT_EQ(allocs.FindAllocationIndex(se::DeviceAddressBase(buf1 + 32, 1)), 1);
}

TEST(BufferAllocationsTest, FindAllocationIndexReturnsNulloptForUnknown) {
  char buf[64];
  char other[64];
  std::vector<se::DeviceAddressBase> buffers = {
      se::DeviceAddressBase(buf, 64),
  };

  BufferAllocations allocs(buffers, /*device_ordinal=*/0,
                           /*memory_allocator=*/nullptr);

  EXPECT_EQ(allocs.FindAllocationIndex(se::DeviceAddressBase(other, 1)),
            std::nullopt);
}

TEST(BufferAllocationsTest, FindAllocationIndexExcludesEndBoundary) {
  char buf[64];
  std::vector<se::DeviceAddressBase> buffers = {
      se::DeviceAddressBase(buf, 64),
  };

  BufferAllocations allocs(buffers, /*device_ordinal=*/0,
                           /*memory_allocator=*/nullptr);

  // Address at exactly buf + size should NOT match (strict less-than).
  EXPECT_EQ(allocs.FindAllocationIndex(se::DeviceAddressBase(buf + 64, 1)),
            std::nullopt);

  // Address at buf + size - 1 should still match.
  EXPECT_EQ(allocs.FindAllocationIndex(se::DeviceAddressBase(buf + 63, 1)), 0);
}

TEST(BufferAllocationsTest, ExtractTearDownAddressesTransfersAndDeduplicates) {
  char storage[64];
  se::DeviceAddressBase address(storage, sizeof(storage));
  std::vector<se::DeviceAddressBase> buffers = {address, address};
  BufferAllocations allocs(buffers, /*device_ordinal=*/0,
                           /*memory_allocator=*/nullptr);

  BufferAllocation allocation0(/*index=*/0, sizeof(storage), /*color=*/0);
  BufferAllocation allocation1(/*index=*/1, sizeof(storage), /*color=*/0);
  std::vector<const BufferAllocation*> allocations = {&allocation0,
                                                      &allocation1};

  std::vector<se::DeviceAddressBase> extracted =
      allocs.ExtractTearDownAddresses(/*live_addresses=*/{}, allocations);
  ASSERT_EQ(extracted.size(), 1);
  EXPECT_EQ(extracted[0].opaque(), storage);
  EXPECT_TRUE(allocs.GetDeviceAddress(0).is_null());
  EXPECT_TRUE(allocs.GetDeviceAddress(1).is_null());
}

TEST(BufferAllocationsTest, DeallocateBufferAddressesTriesEveryAddress) {
  char first;
  char second;
  RecordingAllocator allocator;
  allocator.failing_address = &first;
  std::vector<se::DeviceAddressBase> addresses = {
      se::DeviceAddressBase(&first, 1), se::DeviceAddressBase(),
      se::DeviceAddressBase(&second, 1)};

  absl::Status status =
      DeallocateBufferAddresses(&allocator, /*device_ordinal=*/7, addresses);
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  ASSERT_EQ(allocator.addresses.size(), 2);
  EXPECT_EQ(allocator.addresses[0].opaque(), &first);
  EXPECT_EQ(allocator.addresses[1].opaque(), &second);
  EXPECT_EQ(allocator.ordinals, (std::vector<int>{7, 7}));
}

}  // namespace
}  // namespace xla::gpu
