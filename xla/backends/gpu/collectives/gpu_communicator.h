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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_GPU_COMMUNICATOR_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_GPU_COMMUNICATOR_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/core/collectives/registered_memory.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/future.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace stream_executor {
class StreamExecutor;
}  // namespace stream_executor

namespace xla::gpu {

class GpuSignalDesc : public Communicator::SignalDesc {
 public:
  GpuSignalDesc(int sig_idx, int ctx) : sig_idx_(sig_idx), ctx_(ctx) {}
  int sig_idx() const { return sig_idx_; }
  int ctx() const { return ctx_; }

 private:
  int sig_idx_;
  int ctx_;
};

// Platform-specific handle to the underlying communicator implementation. It
// allows exporting collective communication primitives created and owned by
// the XLA runtime to external libraries, for example via FFI calls.
struct PlatformCommunicatorHandle {
  void* handle = nullptr;  // will be nullptr if not supported
};

// A device communicator that corresponds to the host side GPU communicator
// object (it has same rank in the collective clique and shares underlying
// resources). A host-side GPU communicator object can instantiate multiple
// device-side communicators with different properties.
//
// Device communicator can be passed to GPU kernels to initiate collective
// operations (e.g. Send or Recv) directly from the kernel without having to
// involve host. Memory that can participate in device-initiated collective
// operations typically has to be registered ahead of time (see
// `SymmetricMemory` documentation).
//
// For CUDA this corresponds to:
// https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html
class GpuDeviceCommunicator {
 public:
  virtual ~GpuDeviceCommunicator() = default;

  // Minimum peer reachability required by a device-communication algorithm.
  // Operation semantics are defined by the provider device ABI.
  enum class PeerAccess : uint8_t {
    kLocalDomain = 0,
    kHierarchical = 1,
    kDirectAnyPeer = 2,
  };

  // Handler-visible topology realized by a device communicator.
  enum class Topology : uint8_t {
    kLocalDomain = 0,
    kHierarchical = 1,
    kAllPeers = 2,
  };

  using Features = uint64_t;
  static constexpr Features kLocalMulticast = uint64_t{1} << 0;
  static constexpr Features kNetworkDeviceOperations = uint64_t{1} << 1;
  static constexpr Features kKnownFeatures =
      kLocalMulticast | kNetworkDeviceOperations;

  // Requirements for constructing a device communicator object.
  struct Requirements {
    template <typename Sink>
    friend void AbslStringify(Sink& sink, const Requirements& reqs) {
      absl::Format(&sink,
                   "{peer_access: %d, required_features: 0x%x, "
                   "preferred_features: 0x%x, local_barrier_count: %d, "
                   "team_barrier_count: %d, notification_slot_count: %d, "
                   "completion_slot_count: %d}",
                   static_cast<int>(reqs.peer_access), reqs.required_features,
                   reqs.preferred_features, reqs.local_barrier_count,
                   reqs.team_barrier_count, reqs.notification_slot_count,
                   reqs.completion_slot_count);
    }

    bool operator==(const Requirements& other) const {
      return peer_access == other.peer_access &&
             required_features == other.required_features &&
             preferred_features == other.preferred_features &&
             local_barrier_count == other.local_barrier_count &&
             team_barrier_count == other.team_barrier_count &&
             notification_slot_count == other.notification_slot_count &&
             completion_slot_count == other.completion_slot_count;
    }

    bool operator<(const Requirements& other) const {
      // Preserve the existing strongest/largest-first creation order. Besides
      // minimizing churn, this is the conservative order for providers with
      // communicator-family state fixed by the first device communicator.
      return std::tie(other.peer_access, other.required_features,
                      other.preferred_features, other.local_barrier_count,
                      other.team_barrier_count, other.notification_slot_count,
                      other.completion_slot_count) <
             std::tie(peer_access, required_features, preferred_features,
                      local_barrier_count, team_barrier_count,
                      notification_slot_count, completion_slot_count);
    }

    PeerAccess peer_access = PeerAccess::kLocalDomain;
    Features required_features = 0;
    Features preferred_features = 0;

    // Local and team barriers use independent logical namespaces. Provider
    // adapters are responsible for mapping them to disjoint physical slots.
    int32_t local_barrier_count = 0;

    // The number of barriers to allocate across the communication team.
    // Backends may implement this hierarchically, for example with an LSA
    // barrier inside each local accessibility domain and device networking
    // between domains.
    int32_t team_barrier_count = 0;

    int32_t notification_slot_count = 0;
    int32_t completion_slot_count = 0;
  };

  // Immutable properties and logical resources of a realized device
  // communicator. Topology reports only handler-visible data-plane access;
  // provider networking used solely to implement a team barrier is hidden.
  struct Info {
    int64_t rank = 0;
    int64_t team_size = 0;
    int64_t local_rank = 0;
    int64_t local_domain_size = 0;
    int64_t local_domain_count = 0;
    Topology topology = Topology::kLocalDomain;
    Features enabled_features = 0;
    int32_t team_barrier_count = 0;
    int32_t local_barrier_count = 0;
    int32_t notification_slot_count = 0;
    int32_t completion_slot_count = 0;
  };

  // Returns a platform-specific handle to the underlying communicator object.
  virtual PlatformCommunicatorHandle platform_comm() const {
    return PlatformCommunicatorHandle{nullptr};
  }

  // Returns the size of the load/store accessible communication.
  virtual int64_t lsa_size() const = 0;

  virtual const Info& info() const = 0;

  virtual std::string ToString() const = 0;

  // Checks that device code expecting the given provider ABI can interpret the
  // opaque bytes returned by PackKernelArg.
  absl::Status CheckKernelArgAbi(uint64_t expected_schema,
                                 uint64_t expected_version) const {
    return xla::internal::CheckKernelArgAbi(device_abi_schema_,
                                            device_abi_version_,
                                            expected_schema, expected_version);
  }

  // Packs device communicator as a device kernel argument.
  virtual se::PackedKernelArg PackKernelArg() const = 0;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const GpuDeviceCommunicator& comm) {
    absl::Format(&sink, "%s", comm.ToString());
  }

 protected:
  GpuDeviceCommunicator(uint64_t device_abi_schema, uint64_t device_abi_version)
      : device_abi_schema_(device_abi_schema),
        device_abi_version_(device_abi_version) {}

 private:
  const uint64_t device_abi_schema_;
  const uint64_t device_abi_version_;
};

// GpuCommunicator extends Communicator with synchronous versions of the
// collective methods.
//
// For example, the `Communicator::AllReduce` method (which is asynchronous and
// returns an AsyncValueRef<Event>) has a corresponding synchronous
// `GpuCommunicator::LaunchAllReduce` method which returns an `absl::Status`.
class GpuCommunicator : public Communicator {
 public:
  ~GpuCommunicator() override = default;

  // Returns a platform-specific handle to the underlying communicator object.
  virtual PlatformCommunicatorHandle platform_comm() const {
    return PlatformCommunicatorHandle{nullptr};
  }

  // Returns true iff communicator supports device-initiated communication.
  virtual bool SupportsDeviceComm() const { return false; }

  // Returns the StreamExecutor (and thus the device) this communicator runs on,
  // or nullptr if not backed by a StreamExecutor.
  virtual stream_executor::StreamExecutor* stream_executor() const {
    return nullptr;
  }

  // Creates a new device communicator linked to *this GPU communicator object.
  virtual absl::StatusOr<std::unique_ptr<GpuDeviceCommunicator>>
  CreateDeviceComm(const GpuDeviceCommunicator::Requirements& requirements) {
    return Unimplemented("Device communicator is not implementing");
  }

  // Registers an existing device address range with this communicator for
  // accelerated ("zero-copy") collectives. Unlike CreateSymmetricMemory this
  // makes no symmetry assumption about the buffer's address across ranks and is
  // NOT a collective operation -- each rank may register independently. Returns
  // an RAII handle; destroying it deregisters the range from this communicator.
  virtual absl::StatusOr<std::unique_ptr<RegisteredMemory>>
  CreateRegisteredMemory(se::DeviceAddressBase addr) {
    return Unimplemented("Registered memory is not implemented");
  }

  // Creates a symmetric memory from the existing device address range. This is
  // a collective operation, and all ranks in a clique must call this operation
  // in order to make a progress.
  virtual absl::StatusOr<std::unique_ptr<SymmetricMemory>>
  CreateSymmetricMemory(se::DeviceAddressBase addr) {
    return Unimplemented("Symmetric memory is not implemented");
  }

  //===--------------------------------------------------------------------===//
  // Host-side collective communication APIs
  //===--------------------------------------------------------------------===//

  // Executes a group of collective launches on this communicator. All
  // collective operations in the `group` must use only *this communicator,
  // otherwise behavior is undefined.
  virtual Future<> GroupExecute(absl::AnyInvocable<absl::Status() &&> group) {
    return Future<>(std::move(group)());
  }

  virtual absl::Status LaunchAllReduce(se::DeviceAddressBase send_buffer,
                                       se::DeviceAddressBase recv_buffer,
                                       PrimitiveType dtype, size_t count,
                                       ReductionKind reduction_kind,
                                       const Executor& executor) = 0;

  virtual absl::Status LaunchBroadcast(se::DeviceAddressBase send_buffer,
                                       se::DeviceAddressBase recv_buffer,
                                       PrimitiveType dtype, size_t count,
                                       RankId root,
                                       const Executor& executor) = 0;

  virtual absl::Status LaunchReduceScatter(se::DeviceAddressBase send_buffer,
                                           se::DeviceAddressBase recv_buffer,
                                           PrimitiveType dtype, size_t count,
                                           ReductionKind reduction_kind,
                                           const Executor& executor) = 0;

  virtual absl::Status LaunchAllGather(se::DeviceAddressBase send_buffer,
                                       se::DeviceAddressBase recv_buffer,
                                       PrimitiveType dtype, size_t count,
                                       const Executor& executor) = 0;

  virtual absl::Status LaunchAllToAll(
      absl::InlinedVector<se::DeviceAddressBase, 4> send_buffers,
      absl::InlinedVector<se::DeviceAddressBase, 4> recv_buffers,
      PrimitiveType dtype, size_t count, const Executor& executor) = 0;

  virtual absl::Status LaunchCollectivePermute(
      se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
      PrimitiveType dtype, size_t count, std::optional<RankId> source_rank,
      absl::Span<const RankId> target_ranks, const Executor& executor) = 0;

  virtual absl::Status LaunchSend(se::DeviceAddressBase send_buffer,
                                  PrimitiveType dtype, size_t count,
                                  RankId peer, const Executor& executor) = 0;

  virtual absl::Status LaunchRecv(se::DeviceAddressBase recv_buffer,
                                  PrimitiveType dtype, size_t count,
                                  RankId peer, const Executor& executor) = 0;

  virtual absl::Status LaunchPut(se::DeviceAddressBase send_buffer,
                                 SymmetricMemory* recv_buffer, size_t offset,
                                 size_t count, RankId peer,
                                 const Executor& executor) {
    return Unimplemented("LaunchPut is not implemented");
  }

  virtual absl::Status LaunchSignal(RankId peer, const SignalDesc& signal_desc,
                                    const Executor& executor) {
    return Unimplemented("LaunchSignal is not implemented");
  }

  virtual absl::Status LaunchWaitSignal(RankId peer, int op_cnt,
                                        const SignalDesc& signal_desc,
                                        const Executor& executor) {
    return Unimplemented("LaunchWaitSignal is not implemented");
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const GpuCommunicator& comm) {
    absl::Format(&sink, "%s", comm.ToString());
  }
};

}  // namespace xla::gpu

// Specialize `KernelArgPacking` template to define how to pass device
// communicators to device kernels. We rely on packing device comms to opaque
// packed kernel arguments.
namespace stream_executor {
template <>
struct KernelArgPacking<xla::gpu::GpuDeviceCommunicator*> {
  using Type = PackedKernelArg;
  static Type Pack(xla::gpu::GpuDeviceCommunicator* comm) {
    return comm->PackKernelArg();
  }
};
}  // namespace stream_executor

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_GPU_COMMUNICATOR_H_
