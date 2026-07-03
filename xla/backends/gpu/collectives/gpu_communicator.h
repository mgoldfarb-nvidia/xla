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
#include "absl/strings/string_view.h"
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
class Stream;
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

// Opaque identity of one clique-wide barrier round. Callers must use the same
// token on every rank and must not reuse a token while another round with that
// token can still be in flight on the same communicator.
struct GpuCliqueBarrierToken {
  uint64_t high = 0;
  uint64_t low = 0;

  bool operator==(const GpuCliqueBarrierToken& other) const {
    return high == other.high && low == other.low;
  }
  bool operator!=(const GpuCliqueBarrierToken& other) const {
    return !(*this == other);
  }
};

// Validates the result of a provider all-gather for a clique barrier. Exposed
// for provider-independent tests.
absl::Status ValidateGpuCliqueBarrierTokens(
    GpuCliqueBarrierToken expected,
    absl::Span<const GpuCliqueBarrierToken> gathered);

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
  // Provider-resolved plan for constructing a device communicator. Plans are
  // internal XLA objects: the agreement payload is exchanged between ranks,
  // while provider-specific state stays type-erased and process local.
  class DeviceCommPlan {
   public:
    virtual ~DeviceCommPlan() = default;

    const GpuDeviceCommunicator::Requirements& requirements() const {
      return requirements_;
    }
    absl::string_view provider() const { return provider_; }
    absl::string_view agreement_payload() const { return agreement_payload_; }
    uint64_t creation_priority() const { return creation_priority_; }
    const GpuCommunicator* owner() const { return owner_; }

   protected:
    DeviceCommPlan(const GpuCommunicator* owner,
                   GpuDeviceCommunicator::Requirements requirements,
                   std::string provider, std::string agreement_payload,
                   uint64_t creation_priority)
        : owner_(owner),
          requirements_(requirements),
          provider_(std::move(provider)),
          agreement_payload_(std::move(agreement_payload)),
          creation_priority_(creation_priority) {}

   private:
    const GpuCommunicator* owner_;
    GpuDeviceCommunicator::Requirements requirements_;
    std::string provider_;
    std::string agreement_payload_;
    uint64_t creation_priority_;
  };

  // Provider-resolved plan for collectively registering symmetric memory.
  // The local address remains process-local; agreement_payload contains only
  // cross-rank comparable properties such as size, flags, and provider ABI.
  class SymmetricMemoryPlan {
   public:
    virtual ~SymmetricMemoryPlan() = default;

    se::DeviceAddressBase address() const { return address_; }
    absl::string_view provider() const { return provider_; }
    absl::string_view agreement_payload() const { return agreement_payload_; }
    const GpuCommunicator* owner() const { return owner_; }

   protected:
    SymmetricMemoryPlan(const GpuCommunicator* owner,
                        se::DeviceAddressBase address, std::string provider,
                        std::string agreement_payload)
        : owner_(owner),
          address_(address),
          provider_(std::move(provider)),
          agreement_payload_(std::move(agreement_payload)) {}

   private:
    const GpuCommunicator* owner_;
    se::DeviceAddressBase address_;
    std::string provider_;
    std::string agreement_payload_;
  };

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

  // Returns true iff this provider implements RunCliqueBarrier.
  virtual bool SupportsCliqueBarrier() const { return false; }

  // Synchronously joins a provider-native barrier across all communicator
  // ranks and verifies that every rank supplied exactly `token`. The token
  // must be nonzero, globally coordinated, and unique among barrier rounds
  // that can concurrently reach this communicator. Implementations do not
  // return until device-to-host validation has completed.
  virtual absl::Status RunCliqueBarrier(stream_executor::Stream* stream,
                                        GpuCliqueBarrierToken token) {
    return Unimplemented("Clique barrier is not implemented");
  }

  // Resolves provider-neutral requirements into an immutable provider plan.
  // This must not enter a provider collective operation: the runtime resolves
  // plans on every rank and agrees on their canonical payloads first.
  virtual absl::StatusOr<std::unique_ptr<DeviceCommPlan>> ResolveDeviceCommPlan(
      const GpuDeviceCommunicator::Requirements& requirements) {
    return Unimplemented("Device communicator planning is not implemented");
  }

  // Creates a new device communicator linked to *this GPU communicator from
  // an exact plan previously returned by ResolveDeviceCommPlan.
  virtual absl::StatusOr<std::unique_ptr<GpuDeviceCommunicator>>
  CreateDeviceComm(const DeviceCommPlan& plan) {
    return Unimplemented("Device communicator is not implemented");
  }

  // Legacy convenience wrapper. Runtime collective acquisition uses the
  // explicit resolve/agree/create sequence above; providers that only
  // implement this overload continue to work.
  virtual absl::StatusOr<std::unique_ptr<GpuDeviceCommunicator>>
  CreateDeviceComm(const GpuDeviceCommunicator::Requirements& requirements) {
    auto plan = ResolveDeviceCommPlan(requirements);
    if (!plan.ok()) return plan.status();
    return CreateDeviceComm(**plan);
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

  // Resolves all local, fallible registration checks without entering a
  // provider collective operation.
  virtual absl::StatusOr<std::unique_ptr<SymmetricMemoryPlan>>
  ResolveSymmetricMemoryPlan(se::DeviceAddressBase addr) {
    return Unimplemented("Symmetric memory planning is not implemented");
  }

  // Creates symmetric memory from an exact pre-resolved plan. This is a
  // collective operation, and all ranks in a clique must call it together.
  virtual absl::StatusOr<std::unique_ptr<SymmetricMemory>>
  CreateSymmetricMemory(const SymmetricMemoryPlan& plan) {
    return Unimplemented("Symmetric memory is not implemented");
  }

  // Legacy convenience wrapper. Providers that only implement this overload
  // continue to work.
  virtual absl::StatusOr<std::unique_ptr<SymmetricMemory>>
  CreateSymmetricMemory(se::DeviceAddressBase addr) {
    auto plan = ResolveSymmetricMemoryPlan(addr);
    if (!plan.ok()) return plan.status();
    return CreateSymmetricMemory(**plan);
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
