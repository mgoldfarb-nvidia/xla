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
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/collectives/gpu_clique.h"
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

void AppendManifestField(absl::string_view name, absl::string_view value,
                         std::string* manifest) {
  absl::StrAppend(manifest, name.size(), ":", name, value.size(), ":", value);
}

GpuCliqueBarrierToken DeviceCommBarrierToken(const CollectiveParams& params,
                                             absl::string_view phase,
                                             absl::string_view manifest) {
  std::string input = "xla-device-comm-barrier-v1;";
  AppendManifestField("execution_id_kind",
                      params.launch_id == 0 ? "run" : "launch", &input);
  AppendManifestField("execution_id",
                      absl::StrCat(params.launch_id == 0 ? params.run_id.ToInt()
                                                         : params.launch_id),
                      &input);
  AppendManifestField("phase", phase, &input);
  AppendManifestField("manifest", manifest, &input);
  tsl::Fprint128 fingerprint = tsl::Fingerprint128(input);
  GpuCliqueBarrierToken token{fingerprint.high64, fingerprint.low64};
  if (token == GpuCliqueBarrierToken{}) token.high = 1;
  return token;
}

absl::Status DeviceCommFailure(const GpuCliqueKey& clique,
                               absl::Status status) {
  if (!status.ok() && clique.num_devices() > 1) {
    LOG(FATAL) << "Multi-rank device communicator initialization failed "
                  "before clique quiescence was established: "
               << status;
  }
  return status;
}

struct DeviceCommState {
  size_t request_index = 0;
  GpuDeviceCommunicator::Requirements requirements;
  bool cached = false;
  std::unique_ptr<GpuCommunicator::DeviceCommPlan> plan;
  std::unique_ptr<GpuDeviceCommunicator> device_comm;
};

}  // namespace

std::string GpuCliqueKeyAgreementPayload(const GpuCliqueKey& clique_key) {
  std::string payload = "xla-gpu-clique-key-v1;";
  AppendManifestField("device_count", absl::StrCat(clique_key.devices().size()),
                      &payload);
  for (GlobalDeviceId device : clique_key.devices()) {
    AppendManifestField("device", absl::StrCat(device.value()), &payload);
  }
  AppendManifestField("communication_id",
                      absl::StrCat(clique_key.communication_id().value()),
                      &payload);
  AppendManifestField("incarnation_count",
                      absl::StrCat(clique_key.incarnations().size()), &payload);
  for (IncarnationId incarnation : clique_key.incarnations()) {
    AppendManifestField("incarnation", absl::StrCat(incarnation.value()),
                        &payload);
  }
  return payload;
}

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

absl::Status CollectiveCliques::RunRemoteBarriers(
    const CollectiveParams& params, const CollectiveCliqueRequests& requests,
    stream_executor::Stream* stream, absl::string_view phase,
    const absl::Status& local_status) const {
  if (params.launch_id == 0) {
    return absl::FailedPreconditionError(
        "A distributed device-communication barrier requires a non-zero, "
        "globally coordinated execution launch id");
  }
  if (stream == nullptr) {
    return absl::InvalidArgumentError(
        "A distributed device-communication barrier requires a stream");
  }

  for (const CollectiveCliqueRequests::CliqueRequest& request :
       requests.OrderedRequestedCliques()) {
    if (!request.barrier_after_module_execution_requested ||
        request.key.is_local() || request.dev_comms.empty()) {
      continue;
    }

    ASSIGN_OR_RETURN(GpuCommunicator * communicator,
                     GetComm(request.key, params.global_device_id));
    if (!communicator->SupportsCliqueBarrier()) {
      return absl::UnimplementedError(
          "The provider does not support a non-local completion barrier");
    }

    std::string manifest = "xla-device-communication-runtime-barrier-v1;";
    AppendManifestField("clique", GpuCliqueKeyAgreementPayload(request.key),
                        &manifest);
    AppendManifestField("request_id", absl::StrCat(request.id), &manifest);
    AppendManifestField("phase", phase, &manifest);
    AppendManifestField("status_code",
                        absl::StrCat(static_cast<int>(local_status.code())),
                        &manifest);
    AppendManifestField("status", local_status.message(), &manifest);
    RETURN_IF_ERROR(communicator->RunCliqueBarrier(
        stream, DeviceCommBarrierToken(params, phase, manifest)));
  }
  return absl::OkStatus();
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

    absl::StatusOr<std::shared_ptr<LockableGpuClique::Lock>> clique_or =
        AcquireGpuClique(params.collectives, params.executor, params.run_id,
                         r.key, r.device_groups,
                         params.clique_id_callback ? *params.clique_id_callback
                                                   : default_clique_id_callback,
                         *rank, cliques_map, max_channels,
                         params.collective_use_minimal_resource);
    if (!clique_or.ok()) {
      if (!r.key.is_local() && !r.dev_comms.empty()) {
        LOG(FATAL) << "Non-local device-communication clique acquisition "
                      "failed before peers could converge: "
                   << clique_or.status();
      }
      return clique_or.status();
    }
    std::shared_ptr<LockableGpuClique::Lock> clique = std::move(*clique_or);

    cliques_map[r.key] = std::move(clique);
  }

  auto end_micros = tsl::Env::Default()->NowMicros();
  XLA_VLOG_DEVICE(2, params.executor->device_ordinal()) << absl::StreamFormat(
      "[global_device=%v] [run=%v] Acquired %d collective cliques in %s; ",
      params.global_device_id, params.run_id, cliques_map.size(),
      absl::FormatDuration(absl::Microseconds(end_micros - start_micros)));

  // Resolve and agree on the complete device-communicator plan before any rank
  // enters provider creation. The already-acquired host communicator is the
  // agreement transport, which keeps this first layer NCCL-local and avoids a
  // separate distributed protocol.
  for (const CollectiveCliqueRequests::CliqueRequest& r : ordered_cliques) {
    std::optional<RankId> rank = r.key.rank(params.global_device_id);
    std::shared_ptr<LockableGpuClique::Lock> clique = cliques_map.at(r.key);
    if (r.dev_comms.empty()) continue;

    if (!rank.has_value() || params.stream == nullptr) {
      return DeviceCommFailure(
          r.key, Internal("Device communicator initialization is missing rank "
                          "or execution stream for clique %v",
                          r.key));
    }
    if (!r.key.is_local() && params.launch_id == 0) {
      return FailedPrecondition(
          "Non-local device communication requires a non-zero, globally "
          "coordinated execution launch id");
    }

    auto* comm = dynamic_cast<GpuCommunicator*>(*(*clique)->comm(*rank));
    if (comm == nullptr) {
      return Internal("Communicator for clique %v is not a GpuCommunicator",
                      r.key);
    }
    if (!comm->SupportsCliqueBarrier()) {
      return Unimplemented(
          "Device communicator provider for clique %v does not support "
          "initialization agreement",
          r.key);
    }

    std::vector<DeviceCommState> states;
    states.reserve(r.dev_comms.size());
    absl::Status local_status;
    std::string manifest = "xla-device-comm-plan-v1;";
    AppendManifestField("clique", GpuCliqueKeyAgreementPayload(r.key),
                        &manifest);
    AppendManifestField("count", absl::StrCat(r.dev_comms.size()), &manifest);

    size_t request_index = 0;
    for (const GpuDeviceCommunicator::Requirements& requirements :
         r.dev_comms) {
      DeviceCommState& state = states.emplace_back();
      state.request_index = request_index++;
      state.requirements = requirements;
      state.cached = (*clique)->device_comm(*rank, requirements).has_value();
      if (!state.cached) {
        absl::StatusOr<std::unique_ptr<GpuCommunicator::DeviceCommPlan>> plan =
            comm->ResolveDeviceCommPlan(requirements);
        if (!plan.ok()) {
          local_status.Update(plan.status());
        } else if (*plan == nullptr) {
          local_status.Update(absl::InternalError(
              "Provider returned a null device communicator plan"));
        } else {
          state.plan = std::move(*plan);
        }
      }

      AppendManifestField("requirements", absl::StrCat(requirements),
                          &manifest);
      AppendManifestField("cached", state.cached ? "1" : "0", &manifest);
      AppendManifestField(
          "provider",
          state.plan == nullptr ? absl::string_view() : state.plan->provider(),
          &manifest);
      AppendManifestField("plan",
                          state.plan == nullptr
                              ? absl::string_view()
                              : state.plan->agreement_payload(),
                          &manifest);
    }
    AppendManifestField("status_code",
                        absl::StrCat(static_cast<int>(local_status.code())),
                        &manifest);
    AppendManifestField("status", local_status.message(), &manifest);

    absl::Status preflight = comm->RunCliqueBarrier(
        params.stream,
        DeviceCommBarrierToken(params, "resolved-plan", manifest));
    if (!preflight.ok()) {
      return DeviceCommFailure(r.key, std::move(preflight));
    }
    if (!local_status.ok()) {
      return local_status;
    }

    std::vector<DeviceCommState*> creation_order;
    creation_order.reserve(states.size());
    for (DeviceCommState& state : states) {
      if (!state.cached) creation_order.push_back(&state);
    }
    std::stable_sort(
        creation_order.begin(), creation_order.end(),
        [](const DeviceCommState* lhs, const DeviceCommState* rhs) {
          return lhs->plan->creation_priority() <
                 rhs->plan->creation_priority();
        });

    for (DeviceCommState* state : creation_order) {
      {
        XLA_VLOG_DEVICE(2, params.executor->device_ordinal())
            << absl::StreamFormat(
                   "Create device communicator: rank=%v clique=%v", *rank,
                   r.key);
        absl::StatusOr<std::unique_ptr<GpuDeviceCommunicator>> device_comm =
            comm->CreateDeviceComm(*state->plan);
        if (!device_comm.ok()) {
          if (r.key.num_devices() > 1) {
            LOG(FATAL) << "Device communicator creation failed inside a "
                          "multi-rank provider collective; quiescence is "
                          "unknown: "
                       << device_comm.status();
          }
          return device_comm.status();
        }
        if (*device_comm == nullptr) {
          absl::Status status = absl::InternalError(
              "Provider returned a null device communicator");
          if (r.key.num_devices() > 1) {
            LOG(FATAL) << status;
          }
          return status;
        }
        state->device_comm = std::move(*device_comm);
      }

      std::string outcome = manifest;
      AppendManifestField("created_index", absl::StrCat(state->request_index),
                          &outcome);
      AppendManifestField("created", state->device_comm != nullptr ? "1" : "0",
                          &outcome);
      absl::Status post_create = comm->RunCliqueBarrier(
          params.stream,
          DeviceCommBarrierToken(params, "post-create", outcome));
      if (!post_create.ok()) {
        return DeviceCommFailure(r.key, std::move(post_create));
      }
    }

    // Publish only after every rank has completed the full creation sequence.
    for (DeviceCommState& state : states) {
      if (state.cached) continue;
      absl::Status status = (*clique)->AddDeviceComm(
          *rank, state.requirements, std::move(state.device_comm));
      if (!status.ok()) {
        return DeviceCommFailure(r.key, std::move(status));
      }
    }
  }

  return CollectiveCliques(std::move(cliques_map));
}

}  // namespace xla::gpu
