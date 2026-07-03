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

#ifndef XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_CLIQUES_H_
#define XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_CLIQUES_H_

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/collectives/gpu_clique.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_cliques.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/runtime/device_id.h"
#include "xla/tsl/util/tied_ref.h"
#include "xla/util.h"

namespace xla::gpu {

// A collection of collective cliques acquired based on GPU clique requests
// collected from all thunks at prepare stage.
class CollectiveCliques {
 public:
  CollectiveCliques() = default;
  explicit CollectiveCliques(AcquiredCliquesMap cliques_map);

  absl::StatusOr<GpuCommunicator*> GetComm(const GpuCliqueKey& clique_key,
                                           RankId rank) const;

  absl::StatusOr<GpuCommunicator*> GetComm(
      const GpuCliqueKey& clique_key, GlobalDeviceId global_device_id) const;

  absl::StatusOr<GpuDeviceCommunicator*> GetDeviceComm(
      const GpuCliqueKey& clique_key, RankId rank,
      const GpuDeviceCommunicator::Requirements& reqs) const;

  absl::StatusOr<GpuDeviceCommunicator*> GetDeviceComm(
      const GpuCliqueKey& clique_key, GlobalDeviceId global_device_id,
      const GpuDeviceCommunicator::Requirements& reqs) const;

  // Returns whether peer device memory access is possible between all devices
  // in the clique.
  absl::StatusOr<bool> peer_access_enabled(
      const GpuCliqueKey& clique_key) const;

  absl::StatusOr<std::string> agreement_session_id(
      const GpuCliqueKey& clique_key) const;

  absl::Status agreement_status(const GpuCliqueKey& clique_key) const;

  // Records cancellation on every acquired clique, then aborts all providers
  // concurrently. Initialization uses this after a partial multi-clique
  // failure so disjoint peers cannot continue with an earlier clique.
  absl::Status PoisonAll(absl::Status status) const;

  // Establishes clique-wide quiescence after all process-local execution
  // streams have completed. Only non-local device-communication cliques
  // requesting a module execution barrier participate. A failure records
  // cancellation on every requested clique before aborting them concurrently,
  // so a rank waiting in a different overlapping clique is not stranded.
  //
  // The provider barrier exchanges and validates the launch id, making a
  // cross-process launch-order mismatch a terminal error instead of allowing
  // one execution to release another execution's remotely accessible memory.
  // A failed barrier poisons the corresponding clique session.
  absl::Status RunCompletionBarriers(
      const CollectiveParams& params, const CollectiveCliqueRequests& requests,
      stream_executor::Stream* stream,
      const absl::Status& local_completion_status) const;

  // Permanently fails every clique that requested module-completion ordering.
  // This is called before waiting for streams when execution fails before a
  // completion barrier: aborting the provider communicator is what releases a
  // remote peer that may already be blocked in a device-communication kernel.
  absl::Status PoisonCompletionBarriers(
      const CollectiveCliqueRequests& requests,
      const absl::Status& execution_status) const;

  // Ties an object to a clique. Clique takes ownership of the object and will
  // destroy it when the clique is destroyed. When TiedRef is destroyed, the
  // object will be garbage collected.
  template <typename T>
  absl::StatusOr<tsl::TiedRef<T>> Tie(const GpuCliqueKey& clique_key,
                                      std::unique_ptr<T> object);

  bool empty() const { return cliques_map_.empty(); }

 private:
  AcquiredCliquesMap cliques_map_;
};

template <typename T>
absl::StatusOr<tsl::TiedRef<T>> CollectiveCliques::Tie(
    const GpuCliqueKey& clique_key, std::unique_ptr<T> object) {
  // Check that we locked access to a clique for `clique_key`.
  auto clique = cliques_map_.find(clique_key);
  if (clique == cliques_map_.end()) {
    return NotFound("No clique found for clique key: %v", clique_key);
  }
  return (*clique->second)->Tie(std::move(object));
}

// Acquires collective cliques using the given collective parameters for all
// requested GPU cliques.
//
// WARNING: This is a collective operation, that must be called by all
// participating ranks in the requested cliques, otherwise it will lead to a
// deadlock.
absl::StatusOr<CollectiveCliques> AcquireCollectiveCliques(
    const CollectiveParams& params, const CollectiveCliqueRequests& cliques);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_CLIQUES_H_
