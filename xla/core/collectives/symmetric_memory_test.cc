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

#include "xla/core/collectives/symmetric_memory.h"

#include <cstdint>
#include <cstring>
#include <string>

#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel_args.h"

namespace xla {
namespace {

namespace se = stream_executor;

struct ProviderMemoryHandle {
  uint64_t words[3];
};

static_assert(sizeof(ProviderMemoryHandle) == 24);

class VariableSizeSymmetricMemory final : public SymmetricMemory {
 public:
  explicit VariableSizeSymmetricMemory(ProviderMemoryHandle handle)
      : handle_(handle) {}

  se::DeviceAddressBase addr() const final { return se::DeviceAddressBase(); }
  std::string ToString() const final { return "VariableSizeSymmetricMemory"; }

  PackedKernelArg PackKernelArg() const final {
    return PackedKernelArg(sizeof(handle_), [&](absl::Span<char> packed) {
      std::memcpy(packed.data(), &handle_, sizeof(handle_));
    });
  }

 private:
  ProviderMemoryHandle handle_;
};

TEST(SymmetricMemoryTest, PacksVariableSizeProviderHandle) {
  ProviderMemoryHandle expected{
      {0x0123456789abcdef, 0xfedcba9876543210, 0x13579bdf2468ace0}};
  VariableSizeSymmetricMemory memory(expected);

  auto packed = se::KernelArgPacking<SymmetricMemory*>::Pack(&memory);
  ASSERT_EQ(packed.size_bytes(), sizeof(expected));

  ProviderMemoryHandle actual;
  std::memcpy(&actual, packed.data(), sizeof(actual));
  EXPECT_EQ(actual.words[0], expected.words[0]);
  EXPECT_EQ(actual.words[1], expected.words[1]);
  EXPECT_EQ(actual.words[2], expected.words[2]);

  auto kernel_args = se::PackKernelArgs(/*shmem_bytes=*/0,
                                        static_cast<SymmetricMemory*>(&memory));
  ASSERT_EQ(kernel_args->number_of_arguments(), 1);
  ASSERT_EQ(kernel_args->argument_addresses().size(), 1);
  ProviderMemoryHandle kernel_arg;
  std::memcpy(&kernel_arg, kernel_args->argument_addresses()[0],
              sizeof(kernel_arg));
  EXPECT_EQ(kernel_arg.words[0], expected.words[0]);
  EXPECT_EQ(kernel_arg.words[1], expected.words[1]);
  EXPECT_EQ(kernel_arg.words[2], expected.words[2]);
}

}  // namespace
}  // namespace xla
