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

#include "xla/backends/gpu/runtime/collective_memory.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_memory_cache.h"
#include "xla/backends/gpu/runtime/collective_memory_requests.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/rendezvous.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/gpu/multicast_memory.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/util/safe_reinterpret_cast.h"
#include "xla/tsl/util/tied_ref.h"
#include "xla/util.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla::gpu {
namespace {

void AppendAgreementField(absl::string_view name, absl::string_view value,
                          std::string* out) {
  absl::StrAppend(out, name.size(), ":", name, value.size(), ":", value);
}

absl::Status WithAgreementContext(absl::Status status,
                                  absl::string_view context) {
  if (status.ok()) return status;
  return absl::Status(status.code(),
                      absl::StrCat(context, ": ", status.message()));
}

struct SymmetricMemoryState {
  BufferAllocation::Index allocation;
  std::unique_ptr<GpuCommunicator::SymmetricMemoryPlan> plan;
  std::unique_ptr<SymmetricMemory> memory;
};

GpuCliqueBarrierToken SymmetricMemoryBarrierToken(
    int64_t execution_id, absl::string_view execution_id_kind,
    absl::string_view phase, absl::string_view canonical_manifest) {
  std::string input = "xla-symmetric-memory-barrier-v1;";
  AppendAgreementField("execution_id_kind", execution_id_kind, &input);
  AppendAgreementField("execution_id", absl::StrCat(execution_id), &input);
  AppendAgreementField("phase", phase, &input);
  AppendAgreementField("manifest", canonical_manifest, &input);
  tsl::Fprint128 fingerprint = tsl::Fingerprint128(input);
  GpuCliqueBarrierToken token{fingerprint.high64, fingerprint.low64};
  // The all-zero value is reserved for an invalid/uninitialized token.
  if (token == GpuCliqueBarrierToken{}) token.high = 1;
  return token;
}

}  // namespace

CollectiveMemory::CollectiveMemory(
    const BufferAllocations& buffers,
    absl::flat_hash_map<Key, std::shared_ptr<SymmetricMemory>> sym_memories,
    absl::flat_hash_map<Key, MulticastMemory> mcast_memories,
    absl::flat_hash_map<Key, PeerMemory> peer_memories)
    : buffers_(buffers.buffers().begin(), buffers.buffers().end()),
      sym_memories_(std::move(sym_memories)),
      mcast_memories_(std::move(mcast_memories)),
      peer_memories_(std::move(peer_memories)) {}

std::optional<BufferAllocation::Index> CollectiveMemory::FindAllocationIndex(
    se::DeviceAddressBase addr) const {
  for (BufferAllocation::Index i = 0; i < buffers_.size(); ++i) {
    uintptr_t base =
        tsl::safe_reinterpret_cast<uintptr_t>(buffers_[i].opaque());
    uintptr_t ptr = tsl::safe_reinterpret_cast<uintptr_t>(addr.opaque());
    if (base != 0 && ptr >= base && ptr - base < buffers_[i].size()) {
      return i;
    }
  }
  return std::nullopt;
}

se::DeviceAddressBase CollectiveMemory::GetDeviceAddress(
    BufferAllocation::Index allocation) const {
  CHECK_GE(allocation, 0);
  CHECK_LT(allocation, buffers_.size());
  return buffers_[allocation];
}

std::pair<SymmetricMemory*, size_t> CollectiveMemory::FindSymmetricMemory(
    const GpuCliqueKey& clique, BufferAllocation::Index allocation) const {
  auto it = sym_memories_.find(std::make_pair(clique, allocation));
  if (it == sym_memories_.end()) {
    return std::make_pair(nullptr, 0);
  }
  return std::make_pair(it->second.get(), 0);
}

std::pair<SymmetricMemory*, size_t> CollectiveMemory::FindSymmetricMemory(
    const GpuCliqueKey& clique, BufferAllocation::Slice slice) const {
  auto [sym, sym_offset] = FindSymmetricMemory(clique, slice.index());
  return std::make_pair(sym, sym_offset + slice.offset());
}

std::pair<SymmetricMemory*, size_t> CollectiveMemory::FindSymmetricMemory(
    const GpuCliqueKey& clique, se::DeviceAddressBase addr) const {
  auto allocation = FindAllocationIndex(addr);
  if (!allocation.has_value()) {
    return std::make_pair(nullptr, 0);
  }

  // Find offset from the base allocation.
  se::DeviceAddressBase base = GetDeviceAddress(*allocation);
  size_t offset = tsl::safe_reinterpret_cast<uintptr_t>(addr.opaque()) -
                  tsl::safe_reinterpret_cast<uintptr_t>(base.opaque());

  auto [sym, sym_offset] = FindSymmetricMemory(clique, *allocation);
  return std::make_pair(sym, sym_offset + offset);
}

std::pair<void*, size_t> CollectiveMemory::FindMultimemAddress(
    const GpuCliqueKey& clique, BufferAllocation::Index allocation) const {
  auto it = mcast_memories_.find(std::make_pair(clique, allocation));
  if (it == mcast_memories_.end()) {
    return std::make_pair(nullptr, 0);
  }

  return std::make_pair(it->second.multimem_ptr, 0);
}

std::pair<void*, size_t> CollectiveMemory::FindMultimemAddress(
    const GpuCliqueKey& clique, BufferAllocation::Slice slice) const {
  auto [mmem, mmem_offset] = FindMultimemAddress(clique, slice.index());
  return std::make_pair(mmem, mmem_offset + slice.offset());
}

std::pair<void*, size_t> CollectiveMemory::FindMultimemAddress(
    const GpuCliqueKey& clique, se::DeviceAddressBase addr) const {
  auto allocation = FindAllocationIndex(addr);
  if (!allocation.has_value()) {
    return std::make_pair(nullptr, 0);
  }

  // Find offset from the base allocation.
  se::DeviceAddressBase base = GetDeviceAddress(*allocation);
  size_t offset = tsl::safe_reinterpret_cast<uintptr_t>(addr.opaque()) -
                  tsl::safe_reinterpret_cast<uintptr_t>(base.opaque());

  auto [mmem, mmem_offset] = FindMultimemAddress(clique, *allocation);
  return std::make_pair(mmem, mmem_offset + offset);
}

std::optional<se::DeviceAddressBase> CollectiveMemory::FindPeerAddress(
    const GpuCliqueKey& clique, RankId rank,
    BufferAllocation::Index allocation) const {
  auto it = peer_memories_.find(std::make_pair(clique, allocation));
  if (it == peer_memories_.end()) {
    return std::nullopt;
  }

  auto addr = it->second.addrs.find(rank);
  if (addr == it->second.addrs.end()) {
    return std::nullopt;
  }

  return addr->second;
}

std::optional<se::DeviceAddressBase> CollectiveMemory::FindPeerAddress(
    const GpuCliqueKey& clique, RankId rank,
    BufferAllocation::Slice slice) const {
  auto peer_alloc = FindPeerAddress(clique, rank, slice.index());
  if (!peer_alloc.has_value()) {
    return std::nullopt;
  }
  return peer_alloc->GetByteSlice(slice.offset(), slice.size());
}

std::optional<se::DeviceAddressBase> CollectiveMemory::FindPeerAddress(
    const GpuCliqueKey& clique, RankId rank, se::DeviceAddressBase addr) const {
  auto allocation = FindAllocationIndex(addr);
  if (!allocation.has_value()) {
    return std::nullopt;
  }

  // Find offset from the base allocation.
  se::DeviceAddressBase base = GetDeviceAddress(*allocation);
  size_t offset = tsl::safe_reinterpret_cast<uintptr_t>(addr.opaque()) -
                  tsl::safe_reinterpret_cast<uintptr_t>(base.opaque());

  // Find device address for peer allocation.
  auto peer_alloc = FindPeerAddress(clique, rank, *allocation);
  if (!peer_alloc.has_value()) {
    return std::nullopt;
  }

  return peer_alloc->GetByteSlice(offset, addr.size());
}

//===----------------------------------------------------------------------===//
// Local rendezvous parameters.
//===----------------------------------------------------------------------===//

namespace {

// Wrap GpuCliqueKey into a unique struct to guarantee we do not accidentally
// try to run multiple unrelated rendezvous for a same key.
struct RendezvousKey {
  GpuCliqueKey clique_key;

  bool operator==(const RendezvousKey& other) const {
    return clique_key == other.clique_key;
  }

  template <typename H>
  friend H AbslHashValue(H h, const RendezvousKey& key) {
    return H::combine(std::move(h), key.clique_key);
  }
};

// Parameters passed to the rendezvous callback from all ranks.
struct RendezvousParams {
  RankId rank;
  se::StreamExecutor* executor;
  const BufferAllocations& buffers;
};

struct RankCmp {
  bool operator()(const RendezvousParams* a, const RendezvousParams* b) const {
    return a->rank < b->rank;
  }
};

struct DeviceOrdinalFormatter {
  void operator()(std::string* out, const RendezvousParams* param) const {
    absl::StrAppend(out, param->executor->device_ordinal());
  }
};

struct RankFormatter {
  void operator()(std::string* out, const RendezvousParams* param) const {
    absl::StrAppend(out, param->rank.value());
  }
};

}  // namespace

//===----------------------------------------------------------------------===//
// Symmetric memory acquisition.
//===----------------------------------------------------------------------===//

// Acquire symmetric memory for all requested allocation.
static absl::StatusOr<absl::flat_hash_map<CollectiveMemory::Key,
                                          std::shared_ptr<SymmetricMemory>>>
AcquireSymmetricMemory(
    const CollectiveParams& params, CollectiveCliques& cliques,
    const BufferAllocations& buffers,
    absl::Span<const CollectiveMemoryRequests::CollectiveAllocations> allocs) {
  absl::flat_hash_map<CollectiveMemory::Key, std::shared_ptr<SymmetricMemory>>
      sym_memories;

  for (const CollectiveMemoryRequests::CollectiveAllocations& r : allocs) {
    std::optional<RankId> rank = r.clique.rank(params.global_device_id);

    if (!rank.has_value()) {
      return cliques.PoisonAll(
          Internal("Can't find global device id %v in clique key %v",
                   params.global_device_id, r.clique));
    }

    // TODO(ezhulenev): All of the buffer allocations that we make symmetric
    // are created from the same underlying memory allocator. We can
    // significantly improve performance with a few tricks:
    //
    // 1. Coalesce adjacent allocations and create one large symmetric region.
    // 2. Create one big symmetric region from [start, end] addresses, we might
    //    have unused gaps in the middle, but it doesn't matter, we will ignore
    //    them.
    // Currently it's very simple proof of concept.

    absl::Status agreement_status = cliques.agreement_status(r.clique);
    if (!agreement_status.ok()) {
      return cliques.PoisonAll(std::move(agreement_status));
    }
    absl::StatusOr<GpuCommunicator*> comm_or = cliques.GetComm(r.clique, *rank);
    if (!comm_or.ok()) {
      return cliques.PoisonAll(comm_or.status());
    }
    GpuCommunicator* comm = *comm_or;
    if (params.stream == nullptr) {
      return cliques.PoisonAll(
          InvalidArgument("Symmetric memory initialization requires a stream"));
    }
    if (!r.clique.is_local() && params.launch_id == 0) {
      return cliques.PoisonAll(FailedPrecondition(
          "Non-local symmetric memory initialization requires a non-zero, "
          "globally coordinated execution launch id"));
    }

    std::vector<SymmetricMemoryState> states;
    states.reserve(r.allocations.size());
    absl::Status local_status;
    std::string manifest = "xla-symmetric-memory-plan-v1;";
    absl::StrAppend(&manifest, "request_id=", r.id,
                    ";count=", r.allocations.size(), ";");

    for (BufferAllocation::Index i : r.allocations) {
      se::DeviceAddressBase addr = buffers.GetDeviceAddress(i);
      SymmetricMemoryState& state =
          states.emplace_back(SymmetricMemoryState{i, nullptr, nullptr});

      absl::StatusOr<std::unique_ptr<GpuCommunicator::SymmetricMemoryPlan>>
          plan = comm->ResolveSymmetricMemoryPlan(addr);
      if (!plan.ok()) {
        local_status.Update(WithAgreementContext(
            plan.status(),
            absl::StrCat("failed to resolve symmetric memory plan for "
                         "allocation ",
                         i)));
      } else if (*plan == nullptr) {
        local_status.Update(Internal(
            "Provider returned a null symmetric memory plan for allocation %d",
            i));
      } else {
        state.plan = std::move(*plan);
      }

      AppendAgreementField("allocation", absl::StrCat(i), &manifest);
      AppendAgreementField("size", absl::StrCat(addr.size()), &manifest);
      AppendAgreementField("resolved", state.plan == nullptr ? "0" : "1",
                           &manifest);
      AppendAgreementField(
          "provider",
          state.plan == nullptr ? absl::string_view() : state.plan->provider(),
          &manifest);
      AppendAgreementField("plan",
                           state.plan == nullptr
                               ? absl::string_view()
                               : state.plan->agreement_payload(),
                           &manifest);
    }

    // Do not enter any collective registration if local planning failed. Abort
    // the clique so peers already waiting in the manifest barrier are released
    // and no rank can continue into a later registration on its own.
    if (!local_status.ok()) {
      return cliques.PoisonAll(std::move(local_status));
    }

    auto run_barrier_or_poison = [&](absl::string_view phase) {
      // A nonzero launch id is globally coordinated. Local cliques can use the
      // process-local run id because all their ranks share it.
      bool has_global_launch_id = params.launch_id != 0;
      int64_t execution_id =
          has_global_launch_id ? params.launch_id : params.run_id.ToInt();
      absl::string_view execution_id_kind =
          has_global_launch_id ? "launch" : "run";
      absl::Status status = comm->RunCliqueBarrier(
          params.stream, SymmetricMemoryBarrierToken(
                             execution_id, execution_id_kind, phase, manifest));
      if (!status.ok()) {
        return cliques.PoisonAll(WithAgreementContext(
            std::move(status),
            absl::StrCat("symmetric memory ", phase, " barrier failed")));
      }
      return status;
    };

    // This is the only pre-registration provider collective for the clique.
    // Token validation proves every rank resolved the same complete ordered
    // manifest before any rank enters CreateSymmetricMemory.
    RETURN_IF_ERROR(run_barrier_or_poison("resolved-plan"));

    for (SymmetricMemoryState& state : states) {
      absl::StatusOr<std::unique_ptr<SymmetricMemory>> symm =
          comm->CreateSymmetricMemory(*state.plan);
      if (!symm.ok() || *symm == nullptr) {
        absl::Status status =
            symm.ok() ? Internal(
                            "Provider returned a null symmetric memory "
                            "window for allocation %d",
                            state.allocation)
                      : symm.status();
        return cliques.PoisonAll(WithAgreementContext(
            std::move(status),
            absl::StrCat("failed to create symmetric memory for allocation ",
                         state.allocation)));
      }
      // Keep every provider object private until all ranks finish the complete
      // deterministic creation sequence.
      state.memory = std::move(*symm);
    }

    RETURN_IF_ERROR(run_barrier_or_poison("post-create"));

    // The post-create barrier is the publication point. CollectiveMemory is
    // execution-scoped and owns every window over the same execution's backing
    // buffers; no clique- or executable-level cache can outlive those buffers.
    for (SymmetricMemoryState& state : states) {
      CollectiveMemory::Key mem_key =
          std::make_pair(r.clique, state.allocation);
      sym_memories.emplace(std::move(mem_key), std::shared_ptr<SymmetricMemory>(
                                                   std::move(state.memory)));
    }
  }

  return sym_memories;
}

//===----------------------------------------------------------------------===//
// Multicast memory acquisition.
//===----------------------------------------------------------------------===//

namespace {

using MulticastMemoryMap =
    absl::flat_hash_map<CollectiveMemory::Key,
                        CollectiveMemory::MulticastMemory>;

struct MappedPtrFormatter {
  void operator()(std::string* out,
                  const std::pair<RankId, void*>& mapped_ptr) const {
    auto& [rank, ptr] = mapped_ptr;
    absl::StrAppend(out, absl::StrFormat("%d:%p", rank.value(), ptr));
  }
};

// A multicast object and a multimem mapping for all participating ranks.
struct MappedMulticastMemory {
  std::shared_ptr<se::gpu::MulticastMemory> multicast_memory;
  absl::btree_map<RankId, void*> rank_multimem_ptr;
};

using MappedMulticastMemoryMap =
    absl::flat_hash_map<CollectiveMemory::Key, MappedMulticastMemory>;

}  // namespace

// Acquire multicast memory for all requested allocation.
absl::StatusOr<MulticastMemoryMap> AcquireMulticastMemory(
    const CollectiveParams& params, CollectiveCliques& cliques,
    const BufferAllocations& buffers,
    absl::Span<const CollectiveMemoryRequests::CollectiveAllocations> allocs,
    CollectiveMemoryCache& memory_cache) {
  int32_t device_ordinal = params.executor->device_ordinal();

  MulticastMemoryMap mcast_memories;

  for (const CollectiveMemoryRequests::CollectiveAllocations& r : allocs) {
    std::optional<RankId> rank = r.clique.rank(params.global_device_id);

    if (!rank.has_value()) {
      return Internal("[%d] Can't find global device id %v in clique key %v",
                      device_ordinal, params.global_device_id, r.clique);
    }

    // We rely on in-process rendezvous to allocate the multicast memory and set
    // up memory mapping on all ranks, and don't support multi-process mode.
    if (!r.clique.is_local()) {
      return Unimplemented(
          "[%d] Multicast is not supported in multi-process mode in clique %v",
          device_ordinal, r.clique);
    }

    std::string rendezvous_name = absl::StrFormat(
        "[%d] [rank=%v] AcquireMulticastMemory for clique %v: allocs=[%s]",
        device_ordinal, *rank, r.clique, absl::StrJoin(r.allocations, ","));

    // Collect device addresses for mapped allocations.
    std::vector<se::DeviceAddressBase> map_to;
    map_to.reserve(r.allocations.size());
    for (BufferAllocation::Index i : r.allocations) {
      map_to.emplace_back(buffers.GetDeviceAddress(i));
    }

    // A callback for rendezvous to allocate and map the multicast memory. We
    // do one round of rendezvous for each clique.
    auto allocate = [&](absl::Span<RendezvousParams*> params)
        -> absl::StatusOr<MappedMulticastMemoryMap> {
      // Sort all participants by rank to get deterministic execution.
      absl::c_sort(params, RankCmp{});

      VLOG(3) << absl::StrFormat(
          "[%s] [ranks=%s] Allocate collective multimem for clique: %v",
          absl::StrJoin(params, ",", DeviceOrdinalFormatter{}),
          absl::StrJoin(params, ",", RankFormatter{}), r.clique);

      // We deterministically choose the first device to create the
      // multicast memory. We will map the rest of participants to it later.
      auto* gpu_executor = absl::down_cast<stream_executor::gpu::GpuExecutor*>(
          params[0]->executor);
      if (gpu_executor == nullptr) {
        return Unimplemented("Unsupported stream executor type");
      }

      // As a result of rendezvous we collective multicast memory and mapping
      // for all participating ranks to the multimem address.
      MappedMulticastMemoryMap clique_mcast_memories;

      for (BufferAllocation::Index i : r.allocations) {
        // Allocate a multicast object for all participating devices.
        size_t multicast_size;

        multicast_size = params[0]->buffers.GetDeviceAddress(i).size();
        ASSIGN_OR_RETURN(
            std::unique_ptr<se::gpu::MulticastMemory> multicast_memory,
            gpu_executor->CreateMulticastMemory(multicast_size, params.size()));

        // For all participating devices, subscribe to the multicast object.
        for (const auto* param : params) {
          RETURN_IF_ERROR(multicast_memory->SubscribeDevice(
              param->executor->device_ordinal()));
        }

        // For all participating devices, map to the multicast memory.
        absl::btree_map<RankId, void*> mapped_ptrs;
        for (const auto* param : params) {
          ASSIGN_OR_RETURN(
              mapped_ptrs[param->rank],
              multicast_memory->MapMemory(
                  param->buffers.GetDeviceAddress(i),
                  absl::down_cast<stream_executor::gpu::GpuExecutor*>(
                      param->executor)));
        }

        VLOG(3) << absl::StrFormat(
            "[%s] [ranks=%s] Allocated collective multimem for clique: %v; "
            "mapped_ptrs=[%s]",
            absl::StrJoin(params, ",", DeviceOrdinalFormatter{}),
            absl::StrJoin(params, ",", RankFormatter{}), r.clique,
            absl::StrJoin(mapped_ptrs, ", ", MappedPtrFormatter{}));

        // Tie the lifetime of constructed multicast memory to the clique.
        ASSIGN_OR_RETURN(
            tsl::TiedRef<se::gpu::MulticastMemory> tied_multicast_memory,
            cliques.Tie(r.clique, std::move(multicast_memory)));

        // Add to cache inside the rendezvous (single-threaded) to avoid a race
        // where multiple ranks std::move from the same TiedRef.
        se::DeviceAddressBase mcast_addr =
            params[0]->buffers.GetDeviceAddress(i);
        std::shared_ptr<se::gpu::MulticastMemory> cached =
            memory_cache.AddMulticastMemory(r.clique, mcast_addr,
                                            std::move(tied_multicast_memory));

        clique_mcast_memories[std::make_pair(r.clique, i)] =
            MappedMulticastMemory{std::move(cached), std::move(mapped_ptrs)};
      }
      return clique_mcast_memories;
    };

    // We expect that all local participants will collectively allocate the
    // multicast memory. We do one rendezvous for each clique, and from the
    // rendezvous callback allocate multicast memory for all allocations.
    RendezvousKey rendezvous_key = {r.clique};
    RendezvousParams allocate_params = {*rank, params.executor, buffers};

    int64_t num_participants = r.clique.num_local_participants();
    ASSIGN_OR_RETURN(
        std::shared_ptr<MappedMulticastMemoryMap> clique_mcast_memories,
        Rendezvous<MappedMulticastMemoryMap>(rendezvous_name, rendezvous_key,
                                             allocate_params, num_participants,
                                             allocate));

    // Copy clique multicast memory to each participating thread.
    for (auto& [k, v] : *clique_mcast_memories) {
      mcast_memories[k] = {v.multicast_memory, v.rank_multimem_ptr.at(*rank)};
    }
  }

  return mcast_memories;
}

//===----------------------------------------------------------------------===//
// Peer memory acquisition.
//===----------------------------------------------------------------------===//

namespace {

using PeerMemoryMap =
    absl::flat_hash_map<CollectiveMemory::Key, CollectiveMemory::PeerMemory>;

}  // namespace

// Acquire peer memory for all requested allocation.
absl::StatusOr<PeerMemoryMap> AcquirePeerMemory(
    const CollectiveParams& params, const CollectiveCliques& cliques,
    const BufferAllocations& buffers,
    absl::Span<const CollectiveMemoryRequests::CollectiveAllocations> allocs) {
  int32_t device_ordinal = params.executor->device_ordinal();

  PeerMemoryMap peer_memories;

  for (const CollectiveMemoryRequests::CollectiveAllocations& r : allocs) {
    std::optional<RankId> rank = r.clique.rank(params.global_device_id);

    if (!rank.has_value()) {
      return Internal("[%d] Can't find global device id %v in clique key %v",
                      device_ordinal, params.global_device_id, r.clique);
    }

    // We rely on in-process rendezvous to exchange peer memory with all ranks.
    if (!r.clique.is_local()) {
      return Unimplemented(
          "[%d] Peer memory is not supported in multi-process mode in clique "
          "%v",
          device_ordinal, r.clique);
    }

    std::string rendezvous_name = absl::StrFormat(
        "[%d] [rank=%v] AcquirePeerMemory for clique %v: allocs=[%s]",
        device_ordinal, *rank, r.clique, absl::StrJoin(r.allocations, ","));

    // A callback for rendezvous to exchange peer allocation addresses with
    // all participating ranks.
    auto exchange = [&](absl::Span<RendezvousParams*> params)
        -> absl::StatusOr<PeerMemoryMap> {
      // Sort all participants by rank to get deterministic execution.
      absl::c_sort(params, RankCmp{});

      VLOG(3) << absl::StrFormat(
          "[%s] [ranks=%s] Exchange collective peer memory for clique: %v",
          absl::StrJoin(params, ",", DeviceOrdinalFormatter{}),
          absl::StrJoin(params, ",", RankFormatter{}), r.clique);

      PeerMemoryMap clique_peer_memories;
      for (BufferAllocation::Index i : r.allocations) {
        // For all participating devices get allocation device address.
        absl::btree_map<RankId, se::DeviceAddressBase> addrs;
        for (const RendezvousParams* param : params) {
          addrs[param->rank] = param->buffers.GetDeviceAddress(i);
        }
        clique_peer_memories[std::make_pair(r.clique, i)] =
            CollectiveMemory::PeerMemory{std::move(addrs)};
      }
      return clique_peer_memories;
    };

    // We expect that all local participants will collectively allocate the
    // multicast memory. We do one rendezvous for each clique, and from the
    // rendezvous callback allocate multicast memory for all allocations.
    RendezvousKey rendezvous_key = {r.clique};
    RendezvousParams allocate_params = {*rank, params.executor, buffers};

    int64_t num_participants = r.clique.num_local_participants();
    ASSIGN_OR_RETURN(
        std::shared_ptr<PeerMemoryMap> clique_mcast_memories,
        Rendezvous<PeerMemoryMap>(rendezvous_name, rendezvous_key,
                                  allocate_params, num_participants, exchange));

    // Copy clique peer memory to each participating thread.
    for (auto& [k, v] : *clique_mcast_memories) {
      peer_memories[k] = v;
    }
  }

  return peer_memories;
}

//===----------------------------------------------------------------------===//
// Collective memory acquisition.
//===----------------------------------------------------------------------===//

absl::StatusOr<CollectiveMemory> AcquireCollectiveMemory(
    const CollectiveParams& params, CollectiveCliques& cliques,
    const CollectiveMemoryRequests& requests,
    CollectiveMemoryCache& memory_cache) {
  // We rely on deterministic order of memory requests, to guarantee that all
  // ranks create collective memory in identical order, otherwise we can get
  // a deadlock.
  std::vector<CollectiveMemoryRequests::CollectiveAllocations> sym_allocs =
      requests.OrderedSymmetricAllocations();
  std::vector<CollectiveMemoryRequests::CollectiveAllocations> mcast_allocs =
      requests.OrderedMulticastAllocations();
  std::vector<CollectiveMemoryRequests::CollectiveAllocations> peer_allocs =
      requests.OrderedPeerAllocations();

  // Short-circuit if we have nothing to allocate.
  if (sym_allocs.empty() && mcast_allocs.empty() && peer_allocs.empty()) {
    return CollectiveMemory(requests.buffers(), /*sym_memories=*/{},
                            /*mcast_memories=*/{}, /*peer_memories=*/{});
  }

  XLA_VLOG_DEVICE(2, params.executor->device_ordinal()) << absl::StreamFormat(
      " Acquire collective memory for global device id %v: run_id=%v "
      "symmetric=%d multicast=%d peer=%d",
      params.global_device_id, params.run_id, sym_allocs.size(),
      mcast_allocs.size(), peer_allocs.size());
  absl::Time start = absl::Now();

  for (size_t i = 0; i < sym_allocs.size(); ++i) {
    const CollectiveMemoryRequests::CollectiveAllocations& r = sym_allocs[i];
    XLA_VLOG_DEVICE(2, params.executor->device_ordinal()) << absl::StreamFormat(
        "    symmetric memory #%d (global device %v): id=%d; clique=%v; "
        "allocations=[%s]",
        i, params.global_device_id, r.id, r.clique,
        absl::StrJoin(r.allocations, ", "));
  }

  for (size_t i = 0; i < mcast_allocs.size(); ++i) {
    const CollectiveMemoryRequests::CollectiveAllocations& r = mcast_allocs[i];
    XLA_VLOG_DEVICE(2, params.executor->device_ordinal()) << absl::StreamFormat(
        "    multicast memory #%d (global device %v): id=%d; clique=%v; "
        "allocations=[%s]",
        i, params.global_device_id, r.id, r.clique,
        absl::StrJoin(r.allocations, ", "));
  }

  for (size_t i = 0; i < peer_allocs.size(); ++i) {
    const CollectiveMemoryRequests::CollectiveAllocations& r = peer_allocs[i];
    XLA_VLOG_DEVICE(2, params.executor->device_ordinal()) << absl::StreamFormat(
        "    peer memory #%d (global device %v): id=%d; clique=%v; "
        "allocations=[%s]",
        i, params.global_device_id, r.id, r.clique,
        absl::StrJoin(r.allocations, ", "));
  }

  tsl::profiler::TraceMe trace([&] {
    return tsl::profiler::TraceMeEncode("AcquireCollectiveMemory",
                                        {{"sym_allocs", sym_allocs.size()},
                                         {"mcast_allocs", mcast_allocs.size()},
                                         {"peer_allocs", peer_allocs.size()}});
  });

  auto sym_memories_or =
      AcquireSymmetricMemory(params, cliques, requests.buffers(), sym_allocs);
  if (!sym_memories_or.ok()) {
    return cliques.PoisonAll(sym_memories_or.status());
  }
  auto sym_memories = std::move(*sym_memories_or);

  auto mcast_memories_or = AcquireMulticastMemory(
      params, cliques, requests.buffers(), mcast_allocs, memory_cache);
  if (!mcast_memories_or.ok()) {
    return cliques.PoisonAll(mcast_memories_or.status());
  }
  MulticastMemoryMap mcast_memories = std::move(*mcast_memories_or);

  auto peer_memories_or =
      AcquirePeerMemory(params, cliques, requests.buffers(), peer_allocs);
  if (!peer_memories_or.ok()) {
    return cliques.PoisonAll(peer_memories_or.status());
  }
  auto peer_memories = std::move(*peer_memories_or);

  XLA_VLOG_DEVICE(2, params.executor->device_ordinal()) << absl::StreamFormat(
      "Acquired collective memory in %s for global device id %v; "
      "run_id=%v symmetric=%d multicast=%d peer=%d",
      absl::FormatDuration(absl::Now() - start), params.global_device_id,
      params.run_id, sym_allocs.size(), mcast_allocs.size(),
      peer_allocs.size());

  return CollectiveMemory(requests.buffers(), std::move(sym_memories),
                          std::move(mcast_memories), std::move(peer_memories));
}

}  // namespace xla::gpu
