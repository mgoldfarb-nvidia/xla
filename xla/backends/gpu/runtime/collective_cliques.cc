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

#include "xla/backends/gpu/runtime/collective_cliques.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/collectives/gpu_clique.h"
#include "xla/backends/gpu/collectives/gpu_clique_agreement.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_cliques.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/runtime/device_id.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla::gpu {
namespace {

void AppendField(absl::string_view name, absl::string_view value,
                 std::string* out) {
  absl::StrAppend(out, name.size(), ":", name, value.size(), ":", value);
}

void AppendRequirements(const GpuDeviceCommunicator::Requirements& reqs,
                        std::string* out) {
  absl::StrAppend(out, "peer_access=", static_cast<int>(reqs.peer_access),
                  ";required_features=", reqs.required_features,
                  ";preferred_features=", reqs.preferred_features,
                  ";local_barriers=", reqs.local_barrier_count,
                  ";team_barriers=", reqs.team_barrier_count,
                  ";notification_slots=", reqs.notification_slot_count,
                  ";completion_slots=", reqs.completion_slot_count, ";");
}

absl::Status WithContext(absl::Status status, absl::string_view context) {
  if (status.ok()) return status;
  return absl::Status(status.code(),
                      absl::StrCat(context, ": ", status.message()));
}

GpuCliqueBarrierToken CompletionBarrierToken(size_t request_id,
                                             int32_t launch_id) {
  tsl::Fprint128 fingerprint = tsl::Fingerprint128(
      absl::StrCat("xla-gpu-module-completion-v1;request=", request_id,
                   ";launch=", launch_id, ";"));
  GpuCliqueBarrierToken token{fingerprint.high64, fingerprint.low64};
  if (token == GpuCliqueBarrierToken{}) token.high = 1;
  return token;
}

struct DeviceCommState {
  GpuDeviceCommunicator::Requirements requirements;
  size_t request_index;
  std::unique_ptr<GpuCommunicator::DeviceCommPlan> plan;
  bool cache_hit = false;
};

absl::Status PoisonGpuCliques(absl::Span<GpuClique* const> cliques,
                              const absl::Status& failure) {
  if (failure.ok()) {
    return InvalidArgument("Cannot poison GPU cliques with an OK status");
  }

  absl::Status result = failure;
  for (GpuClique* clique : cliques) {
    result.Update(clique->RecordAgreementFailure(failure));
  }

  // Record cancellation everywhere before starting any provider abort. An
  // abort can block behind another communicator, so run all of them
  // concurrently and fail fast if the provider cannot reach a terminal state.
  std::vector<absl::Status> abort_statuses(cliques.size());
  {
    std::vector<std::unique_ptr<tsl::Thread>> threads;
    threads.reserve(cliques.size());
    for (size_t i = 0; i < cliques.size(); ++i) {
      threads.emplace_back(tsl::Env::Default()->StartThread(
          tsl::ThreadOptions(), "abort-device-communication-clique", [&, i] {
            abort_statuses[i] = cliques[i]->AbortAfterAgreementFailure();
          }));
    }
  }  // Thread destruction joins every abort.

  for (const absl::Status& abort_status : abort_statuses) {
    if (!abort_status.ok()) {
      LOG(FATAL) << "Failed to abort a GPU clique after a terminal failure: "
                 << abort_status;
    }
  }
  return result;
}

// If acquiring or initializing a later clique fails, ranks participating only
// in an earlier disjoint clique must not continue into execution. This guard
// aborts every clique already acquired by the current rank on any early return.
class AcquiredCliqueFailureGuard {
 public:
  explicit AcquiredCliqueFailureGuard(const AcquiredCliquesMap* cliques)
      : cliques_(cliques) {}

  ~AcquiredCliqueFailureGuard() {
    if (!active_ || cliques_->empty()) return;
    std::vector<GpuClique*> acquired;
    acquired.reserve(cliques_->size());
    for (const auto& entry : *cliques_) {
      acquired.push_back(entry.second->operator->());
    }
    PoisonGpuCliques(
        acquired,
        absl::AbortedError(
            "GPU clique acquisition failed before all requested cliques were "
            "initialized"))
        .IgnoreError();
  }

  void Dismiss() { active_ = false; }

 private:
  const AcquiredCliquesMap* cliques_;
  bool active_ = true;
};

}  // namespace

CollectiveCliques::CollectiveCliques(AcquiredCliquesMap cliques_map)
    : cliques_map_(std::move(cliques_map)) {}

absl::StatusOr<GpuCommunicator*> CollectiveCliques::GetComm(
    const GpuCliqueKey& clique_key, RankId rank) const {
  // Check that we locked access to a clique for `clique_key`.
  auto clique = cliques_map_.find(clique_key);
  if (clique == cliques_map_.end()) {
    return NotFound("No clique found for clique key: %s",
                    clique_key.ToString());
  }

  // Check that clique has a communicator for our rank.
  auto communicator = (*clique->second)->comm(rank);
  if (!communicator.has_value()) {
    return Internal("Communicator for rank %v not found in a NCCL clique %s",
                    rank, clique_key.ToString());
  }

  auto* gpu_communicator = dynamic_cast<GpuCommunicator*>(*communicator);
  if (!gpu_communicator) {
    return Internal("Communicator for rank %v is not a GpuCommunicator", rank);
  }

  return gpu_communicator;
}

absl::StatusOr<GpuCommunicator*> CollectiveCliques::GetComm(
    const GpuCliqueKey& clique_key, GlobalDeviceId global_device_id) const {
  std::optional<RankId> rank = clique_key.rank(global_device_id);
  if (!rank.has_value()) {
    return InvalidArgument("Rank not found for device %v", global_device_id);
  }
  return GetComm(clique_key, *rank);
}

absl::StatusOr<GpuDeviceCommunicator*> CollectiveCliques::GetDeviceComm(
    const GpuCliqueKey& clique_key, RankId rank,
    const GpuDeviceCommunicator::Requirements& reqs) const {
  // Check that we locked access to a clique for `clique_key`.
  auto clique = cliques_map_.find(clique_key);
  if (clique == cliques_map_.end()) {
    return NotFound("No clique found for clique key: %v", clique_key);
  }

  auto device_comm = (*clique->second)->device_comm(rank, reqs);
  if (!device_comm.has_value()) {
    return Internal(
        "Device communicator for rank %v and requirements %v not found in a "
        "NCCL clique %v",
        rank, reqs, clique_key);
  }

  return *device_comm;
}

absl::StatusOr<GpuDeviceCommunicator*> CollectiveCliques::GetDeviceComm(
    const GpuCliqueKey& clique_key, GlobalDeviceId global_device_id,
    const GpuDeviceCommunicator::Requirements& reqs) const {
  std::optional<RankId> rank = clique_key.rank(global_device_id);
  if (!rank.has_value()) {
    return InvalidArgument("Rank not found for device %v", global_device_id);
  }
  return GetDeviceComm(clique_key, *rank, reqs);
}

absl::StatusOr<bool> CollectiveCliques::peer_access_enabled(
    const GpuCliqueKey& clique_key) const {
  // Check that we locked access to a clique for `clique_key`.
  auto clique = cliques_map_.find(clique_key);
  if (clique == cliques_map_.end()) {
    return NotFound("No clique found for clique key: %v", clique_key);
  }

  return (*clique->second)->peer_access_enabled();
}

absl::StatusOr<std::string> CollectiveCliques::agreement_session_id(
    const GpuCliqueKey& clique_key) const {
  auto clique = cliques_map_.find(clique_key);
  if (clique == cliques_map_.end()) {
    return NotFound("No clique found for clique key: %v", clique_key);
  }
  return (*clique->second)->agreement_session_id();
}

absl::Status CollectiveCliques::agreement_status(
    const GpuCliqueKey& clique_key) const {
  auto clique = cliques_map_.find(clique_key);
  if (clique == cliques_map_.end()) {
    return NotFound("No clique found for clique key: %v", clique_key);
  }
  return (*clique->second)->agreement_status();
}

absl::Status CollectiveCliques::PoisonAll(absl::Status status) const {
  std::vector<GpuClique*> cliques;
  cliques.reserve(cliques_map_.size());
  for (const auto& entry : cliques_map_) {
    cliques.push_back(entry.second->operator->());
  }
  return PoisonGpuCliques(cliques, status);
}

absl::Status CollectiveCliques::RunCompletionBarriers(
    const CollectiveParams& params, const CollectiveCliqueRequests& requests,
    stream_executor::Stream* stream,
    const absl::Status& local_completion_status) const {
  if (params.launch_id == 0) {
    return PoisonCompletionBarriers(
        requests,
        FailedPrecondition(
            "A distributed completion barrier requires a non-zero, globally "
            "coordinated execution launch id"));
  }
  if (stream == nullptr) {
    return PoisonCompletionBarriers(
        requests,
        InvalidArgument("A distributed completion barrier requires a stream"));
  }
  if (!local_completion_status.ok()) {
    return PoisonCompletionBarriers(requests, local_completion_status);
  }

  // Use the same deterministic order as clique acquisition. A rank can be a
  // member of multiple overlapping cliques, so an inconsistent order could
  // otherwise create a distributed wait cycle.
  for (const CollectiveCliqueRequests::CliqueRequest& request :
       requests.OrderedRequestedCliques()) {
    if (!request.barrier_after_module_execution_requested ||
        request.key.is_local() || request.dev_comms.empty()) {
      continue;
    }

    // Never retry an agreement for a poisoned session. A previous transport
    // error might have been observed as success by a remote process.
    absl::Status session_status = agreement_status(request.key);
    if (!session_status.ok()) {
      return PoisonCompletionBarriers(requests, session_status);
    }

    absl::StatusOr<GpuCommunicator*> communicator =
        GetComm(request.key, params.global_device_id);
    absl::Status status;
    if (!communicator.ok()) {
      status = communicator.status();
    } else if (!(*communicator)->SupportsCliqueBarrier()) {
      status = Unimplemented(
          "The provider does not support a non-local completion barrier");
    } else {
      status =
          (*communicator)
              ->RunCliqueBarrier(
                  stream, CompletionBarrierToken(request.id, params.launch_id));
    }
    if (!status.ok()) {
      return PoisonCompletionBarriers(requests, status);
    }
  }

  return absl::OkStatus();
}

absl::Status CollectiveCliques::PoisonCompletionBarriers(
    const CollectiveCliqueRequests& requests,
    const absl::Status& execution_status) const {
  if (execution_status.ok()) {
    return InvalidArgument(
        "Cannot poison completion barriers with an OK execution status");
  }

  absl::Status result = execution_status;
  std::vector<GpuClique*> poisoned_cliques;
  for (const CollectiveCliqueRequests::CliqueRequest& request :
       requests.OrderedRequestedCliques()) {
    if (!request.barrier_after_module_execution_requested) continue;

    auto clique = cliques_map_.find(request.key);
    if (clique == cliques_map_.end()) {
      result.Update(
          NotFound("No clique found for clique key: %v", request.key));
      continue;
    }
    GpuClique* gpu_clique = clique->second->operator->();
    poisoned_cliques.push_back(gpu_clique);
  }
  result.Update(PoisonGpuCliques(poisoned_cliques, execution_status));
  return result;
}

absl::StatusOr<CollectiveCliques> AcquireCollectiveCliques(
    const CollectiveParams& params, const CollectiveCliqueRequests& cliques) {
  std::vector<CollectiveCliqueRequests::CliqueRequest> ordered_cliques =
      cliques.OrderedRequestedCliques();
  if (ordered_cliques.empty()) {
    return CollectiveCliques();
  }

  XLA_VLOG_DEVICE(2, params.executor->device_ordinal()) << absl::StreamFormat(
      "[run=%v] Acquire %d collective cliques for global device id %v; "
      "max number of channels for collectives %d; max number of "
      "channels for p2p %d; use_minimal_resource=%v",
      params.run_id, ordered_cliques.size(), params.global_device_id,
      params.collective_max_nchannels, params.p2p_max_nchannels,
      params.collective_use_minimal_resource);

  for (size_t i = 0; i < ordered_cliques.size(); ++i) {
    const CollectiveCliqueRequests::CliqueRequest& r = ordered_cliques[i];
    XLA_VLOG_DEVICE(2, params.executor->device_ordinal()) << absl::StreamFormat(
        "    clique #%d (global device %v): num_local_participants=%d; id=%d; "
        "key=%v; dev_comms=[%s]",
        i, params.global_device_id, r.key.num_local_participants(), r.id, r.key,
        absl::StrJoin(r.dev_comms, ", "));
  }

  tsl::profiler::TraceMe trace([&] {
    return tsl::profiler::TraceMeEncode(
        "AcquireCollectiveCliques", {{"num_cliques", ordered_cliques.size()}});
  });

  AcquiredCliquesMap cliques_map;
  AcquiredCliqueFailureGuard acquisition_failure_guard(&cliques_map);
  auto start_micros = tsl::Env::Default()->NowMicros();

  for (const CollectiveCliqueRequests::CliqueRequest& r : ordered_cliques) {
    std::optional<RankId> rank = r.key.rank(params.global_device_id);

    if (!rank.has_value()) {
      return Internal("Can't find global device id %v in clique key %v",
                      params.global_device_id, r.key);
    }

    // Default clique id callback that generates a unique clique id for the
    // clique key. This callback supports only local cliques (all ranks belong
    // to the same process), as otherwise clique id should be exchanged across
    // multiple processes via an external storage (i.e. builtin KV store).
    //
    // IMPORTANT: This callback is called once for the clique key by the
    // rendezvous leader elected inside the `AcquireGpuClique` implementation.
    CliqueIdCallback default_clique_id_callback =
        [&](const CliqueKey& key) -> absl::StatusOr<CliqueIds> {
      VLOG(4) << absl::StrFormat("Get local NCCL clique ids: clique=%v", key);
      auto& gpu_key = absl::down_cast<const GpuCliqueKey&>(key);
      if (!gpu_key.is_local()) {
        return Internal(
            "For non-local GPU cliques (cliques that span multiple processes) "
            "clique id callback must be passed via execution params");
      }
      ASSIGN_OR_RETURN(CliqueId clique_id,
                       params.collectives->CreateUniqueCliqueId());
      return CliqueIds(clique_id);
    };

    // Communication id 1 is the XLA P2P lane. Other non-zero ids can be
    // backend-owned collective lanes (for example callsite-exclusive public
    // FFI collectives) and must retain the collective channel policy.
    int64_t max_channels = r.key.communication_id() == CommunicationId(1)
                               ? params.p2p_max_nchannels
                               : params.collective_max_nchannels;

    ASSIGN_OR_RETURN(
        std::shared_ptr<LockableGpuClique::Lock> clique,
        AcquireGpuClique(params.collectives, params.executor, params.run_id,
                         r.key, r.device_groups,
                         params.clique_id_callback ? *params.clique_id_callback
                                                   : default_clique_id_callback,
                         *rank, cliques_map, max_channels,
                         params.collective_use_minimal_resource));

    // Agreement failures have an uncertain distributed outcome. Do not let a
    // later execution reuse the same communicator session and accidentally
    // pair with a stale or partially completed round.
    RETURN_IF_ERROR((*clique)->agreement_status());

    cliques_map[r.key] = std::move(clique);
  }

  auto end_micros = tsl::Env::Default()->NowMicros();
  XLA_VLOG_DEVICE(2, params.executor->device_ordinal()) << absl::StreamFormat(
      "[global_device=%v] [run=%v] Acquired %d collective cliques in %s; ",
      params.global_device_id, params.run_id, cliques_map.size(),
      absl::FormatDuration(absl::Microseconds(end_micros - start_micros)));

  // Resolve provider plans and cache state on every rank before entering a
  // provider collective. The initialization protocol makes runtime ABI and
  // provider capability skew deterministic errors instead of hangs, and only
  // publishes cache entries after every rank confirms creation.
  for (const CollectiveCliqueRequests::CliqueRequest& r : ordered_cliques) {
    std::optional<RankId> rank = r.key.rank(params.global_device_id);
    std::shared_ptr<LockableGpuClique::Lock> clique = cliques_map.at(r.key);
    if (r.dev_comms.empty()) continue;

    auto* comm = dynamic_cast<GpuCommunicator*>(*(*clique)->comm(*rank));
    if (comm == nullptr) {
      return Internal(
          "Communicator for rank %v in clique %v is not a GPU "
          "communicator",
          *rank, r.key);
    }

    ASSIGN_OR_RETURN(std::string session_id, (*clique)->agreement_session_id());
    std::vector<DeviceCommState> states;
    states.reserve(r.dev_comms.size());
    absl::Status local_status;
    std::string manifest = "xla-device-comm-plan-v1;";
    absl::StrAppend(&manifest, "request_id=", r.id,
                    ";count=", r.dev_comms.size(), ";");

    size_t request_index = 0;
    for (const GpuDeviceCommunicator::Requirements& reqs : r.dev_comms) {
      DeviceCommState& state = states.emplace_back(
          DeviceCommState{reqs, request_index++, nullptr, false});
      absl::StatusOr<std::unique_ptr<GpuCommunicator::DeviceCommPlan>> plan =
          comm->ResolveDeviceCommPlan(reqs);
      if (!plan.ok()) {
        local_status.Update(WithContext(
            plan.status(), absl::StrCat("failed to resolve device communicator "
                                        "plan #",
                                        state.request_index)));
      } else if (*plan == nullptr) {
        local_status.Update(
            Internal("Provider returned a null device communicator plan #%d",
                     state.request_index));
      } else {
        state.plan = std::move(*plan);
      }

      std::optional<std::string> cached_plan =
          (*clique)->device_comm_plan(*rank, reqs);
      state.cache_hit = cached_plan.has_value();
      if (state.plan != nullptr && cached_plan.has_value() &&
          *cached_plan != state.plan->agreement_payload()) {
        local_status.Update(FailedPrecondition(
            "Cached device communicator plan #%d differs from the newly "
            "resolved provider plan",
            state.request_index));
      }

      std::string requirements;
      AppendRequirements(reqs, &requirements);
      AppendField("requirements", requirements, &manifest);
      AppendField("resolved", state.plan == nullptr ? "0" : "1", &manifest);
      AppendField(
          "provider",
          state.plan == nullptr ? absl::string_view() : state.plan->provider(),
          &manifest);
      AppendField("plan",
                  state.plan == nullptr ? absl::string_view()
                                        : state.plan->agreement_payload(),
                  &manifest);
      AppendField("priority",
                  state.plan == nullptr
                      ? std::string()
                      : absl::StrCat(state.plan->creation_priority()),
                  &manifest);
      AppendField("cache_hit", state.cache_hit ? "1" : "0", &manifest);
      AppendField("cached_plan",
                  cached_plan.has_value() ? absl::string_view(*cached_plan)
                                          : absl::string_view(),
                  &manifest);
    }

    bool all_cache_hits = std::all_of(
        states.begin(), states.end(),
        [](const DeviceCommState& state) { return state.cache_hit; });
    if (all_cache_hits) {
      if (!local_status.ok()) {
        return (*clique)->RecordAgreementFailure(std::move(local_status));
      }
      // Agreement is an initialization protocol. A successful post-create
      // round established that every rank in this clique session owns the
      // same resource; repeating KVS rounds on every execution would leak
      // coordinator keys and add control-plane latency.
      continue;
    }

    auto agree_or_poison = [&](GpuCliqueAgreementRequest request) {
      absl::Status status =
          AgreeGpuClique(params.clique_agreement, std::move(request));
      if (!status.ok()) {
        return (*clique)->RecordAgreementFailure(std::move(status));
      }
      return status;
    };

    RETURN_IF_ERROR(agree_or_poison(GpuCliqueAgreementRequest(
        params.run_id, params.launch_id, session_id, r.key,
        "device-communicator-plan", static_cast<int64_t>(r.id),
        params.global_device_id, local_status, manifest)));

    std::vector<DeviceCommState*> missing;
    missing.reserve(states.size());
    for (DeviceCommState& state : states) {
      if (!state.cache_hit) missing.push_back(&state);
    }
    std::stable_sort(missing.begin(), missing.end(),
                     [](const DeviceCommState* a, const DeviceCommState* b) {
                       return a->plan->creation_priority() <
                              b->plan->creation_priority();
                     });

    for (DeviceCommState* state : missing) {
      XLA_VLOG_DEVICE(2, params.executor->device_ordinal())
          << absl::StreamFormat("Create device communicator: rank=%v clique=%v",
                                *rank, r.key);

      absl::StatusOr<std::unique_ptr<GpuDeviceCommunicator>> dev_comm =
          comm->CreateDeviceComm(*state->plan);
      absl::Status create_status;
      if (!dev_comm.ok()) {
        create_status = WithContext(dev_comm.status(),
                                    "failed to create device communicator");
      } else if (*dev_comm == nullptr) {
        create_status =
            Internal("Provider returned a null device communicator");
      }
      RETURN_IF_ERROR(agree_or_poison(GpuCliqueAgreementRequest(
          params.run_id, params.launch_id, session_id, r.key,
          "device-communicator-create",
          static_cast<int64_t>(state->request_index), params.global_device_id,
          create_status, std::string(state->plan->agreement_payload()))));

      absl::Status add_status = (*clique)->AddDeviceComm(
          *rank, state->requirements,
          std::string(state->plan->agreement_payload()), std::move(*dev_comm));
      if (!add_status.ok()) {
        return (*clique)->RecordAgreementFailure(std::move(add_status));
      }
    }
  }

  acquisition_failure_guard.Dismiss();
  return CollectiveCliques(std::move(cliques_map));
}

}  // namespace xla::gpu
