/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_CORE_COLLECTIVES_SYMMETRIC_MEMORY_H_
#define XLA_CORE_COLLECTIVES_SYMMETRIC_MEMORY_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel_args.h"

namespace xla {

// Symmetric memory allows memory allocations from different devices to be
// grouped into a symmetric memory allocation, where each device in a collective
// clique can access peer memory through the symmetric memory handle.
class SymmetricMemory {
 public:
  // Describes the provider ABI for the opaque value returned by
  // PackKernelArg. Schema and version checks are required because device code
  // interprets the concrete provider type hidden behind this interface.
  struct PackedKernelArgMetadata {
    uint64_t device_abi_schema = 0;
    // The provider ABI version after the compile-time and runtime versions have
    // been validated for exact compatibility.
    uint64_t device_abi_version = 0;
    size_t size_bytes = 0;
    size_t alignment = 1;
  };

  virtual ~SymmetricMemory() = default;

  // Device address on the local device backing the symmetric memory.
  virtual stream_executor::DeviceAddressBase addr() const = 0;

  // For platforms that support multimem (i.e. CUDA) returns a multimem address
  // for the LSA (load/store accessible) team associated with the given
  // symmetric memory. Return default-constructed (nullptr) address if multimem
  // is not supported.
  virtual absl::StatusOr<stream_executor::DeviceAddressBase> multimem_addr()
      const {
    return absl::UnimplementedError("Multimem not supported");
  }

  // Returns an address of the symmetric memory on a peer device.
  //
  // The rank argument is in the same rank space as the collective clique that
  // created this memory. Backends are responsible for translating it to any
  // backend-local rank space needed by their memory access API.
  //
  // Useful for clients who need to pass peer addresses directly as a kernel
  // arguments instead of calculating a peer address from the kernel itself.
  virtual absl::StatusOr<stream_executor::DeviceAddressBase> peer_addr(
      RankId rank) const {
    return absl::UnimplementedError("Peer address not supported");
  }

  virtual std::string ToString() const = 0;

  // A packed kernel argument type for passing symmetric memory to device
  // kernels (a platform-specific POD data type, happens to be a pointer).
  using PackedKernelArg = void*;

  // Returns metadata required to safely interpret the opaque value returned by
  // PackKernelArg. Backends that do not expose a stable device ABI return a
  // zero schema and version by default.
  virtual PackedKernelArgMetadata GetKernelArgMetadata() const {
    return {/*device_abi_schema=*/0,
            /*device_abi_version=*/0,
            /*size_bytes=*/sizeof(PackedKernelArg),
            /*alignment=*/alignof(PackedKernelArg)};
  }

  virtual PackedKernelArg PackKernelArg() const = 0;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const SymmetricMemory& mem) {
    absl::Format(&sink, "%s", mem.ToString());
  }
};

}  // namespace xla

namespace stream_executor {
template <>
struct KernelArgPacking<xla::SymmetricMemory*> {
  using Type = xla::SymmetricMemory::PackedKernelArg;
  static Type Pack(xla::SymmetricMemory* mem) { return mem->PackKernelArg(); }
};
}  // namespace stream_executor

#endif  // XLA_CORE_COLLECTIVES_SYMMETRIC_MEMORY_H_
