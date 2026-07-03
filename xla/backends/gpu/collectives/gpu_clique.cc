/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/gpu/collectives/gpu_clique.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/btree_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/collectives/cancellation_token.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/core/collectives/clique.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/service/lockable.h"
#include "xla/tsl/platform/logging.h"
#include "xla/util.h"

namespace xla::gpu {
namespace {

void AppendField(absl::string_view value, std::string* out) {
  absl::StrAppendFormat(out, "%d:", value.size());
  out->append(value.data(), value.size());
}

void AppendCanonicalCliqueKey(const GpuCliqueKey& key, std::string* out) {
  absl::StrAppendFormat(out, "devices=%d;", key.devices().size());
  for (GlobalDeviceId device : key.devices()) {
    absl::StrAppendFormat(out, "%d,", device.value());
  }
  absl::StrAppendFormat(out, ";communication=%d;incarnations=%d;",
                        key.communication_id().value(),
                        key.incarnations().size());
  for (IncarnationId incarnation : key.incarnations()) {
    absl::StrAppendFormat(out, "%d,", incarnation.value());
  }
}

}  // namespace

GpuClique::GpuClique(
    GpuCliqueKey key, std::optional<CliqueIds> ids,
    absl::btree_map<RankId, std::unique_ptr<Communicator>> communicators,
    bool peer_access_enabled, std::shared_ptr<CancellationToken> cancel,
    const GpuClique* parent)
    : Clique(std::move(communicators)),
      key_(key),
      ids_(ids),
      peer_access_enabled_(peer_access_enabled),
      cancel_(std::move(cancel)),
      parent_(parent) {}

std::optional<GpuDeviceCommunicator*> GpuClique::device_comm(
    RankId rank, const GpuDeviceCommunicator::Requirements& reqs) const {
  absl::MutexLock lock(mu_);
  if (auto it = device_communicators_.find(std::make_pair(rank, reqs));
      it != device_communicators_.end()) {
    return it->second.communicator.get();
  }
  return std::nullopt;
}

std::optional<std::string> GpuClique::device_comm_plan(
    RankId rank, const GpuDeviceCommunicator::Requirements& reqs) const {
  absl::MutexLock lock(mu_);
  if (auto it = device_communicators_.find(std::make_pair(rank, reqs));
      it != device_communicators_.end()) {
    return it->second.agreement_payload;
  }
  return std::nullopt;
}

absl::Status GpuClique::AddDeviceComm(
    RankId rank, GpuDeviceCommunicator::Requirements reqs,
    std::string agreement_payload,
    std::unique_ptr<GpuDeviceCommunicator> communicator) {
  if (communicator == nullptr) {
    return InvalidArgument(
        "Cannot add a null device communicator for rank %v and requirements "
        "%v",
        rank, reqs);
  }
  absl::MutexLock lock(mu_);
  auto emplaced = device_communicators_.emplace(
      std::make_pair(rank, reqs),
      DeviceCommunicatorEntry{std::move(agreement_payload),
                              std::move(communicator)});
  if (!emplaced.second) {
    return InvalidArgument(
        "Rank %v and requirements %v already exist in clique", rank, reqs);
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> GpuClique::agreement_session_id() const {
  std::string session = "xla-gpu-clique-session-v1;";
  if (parent_ != nullptr) {
    auto parent_session = parent_->agreement_session_id();
    if (!parent_session.ok()) return parent_session.status();
    absl::StrAppend(&session, "split;");
    AppendField(*parent_session, &session);
  } else {
    absl::StrAppend(&session, "root;");
    if (!ids_.has_value()) {
      if (!key_.is_local()) {
        return FailedPrecondition(
            "Non-local GPU clique %v has no globally shared clique id", key_);
      }
      absl::StrAppend(&session, "local;");
    } else {
      absl::StrAppendFormat(&session, "ids=%d;", ids_->size());
      for (const CliqueId& id : ids_->data()) {
        AppendField(absl::string_view(id.data().data(), id.data().size()),
                    &session);
      }
    }
  }
  AppendCanonicalCliqueKey(key_, &session);
  return session;
}

absl::Status GpuClique::agreement_status() const {
  absl::MutexLock lock(mu_);
  return agreement_status_;
}

absl::Status GpuClique::PoisonAgreement(absl::Status status) {
  status = RecordAgreementFailure(std::move(status));
  if (status.ok()) return status;

  absl::Status abort_status = AbortAfterAgreementFailure();
  if (!abort_status.ok()) {
    LOG(FATAL) << "Failed to abort GPU clique after terminal agreement "
                  "failure: "
               << abort_status;
  }
  return status;
}

absl::Status GpuClique::RecordAgreementFailure(absl::Status status) {
  if (status.ok()) {
    return InvalidArgument("Cannot poison clique agreement with OK status");
  }

  {
    absl::MutexLock lock(mu_);
    if (agreement_status_.ok()) agreement_status_ = std::move(status);
    status = agreement_status_;
  }

  // Cancel immediately so queued communicator work observes the terminal
  // session failure before batch failure handling starts any blocking aborts.
  cancel_->Cancel();
  return status;
}

absl::Status GpuClique::AbortAfterAgreementFailure() {
  {
    absl::MutexLock lock(mu_);
    if (agreement_status_.ok()) {
      return FailedPrecondition(
          "Cannot abort clique before recording an agreement failure");
    }
    if (agreement_abort_started_) return absl::OkStatus();
    agreement_abort_started_ = true;
  }

  // A local cancellation token is not sufficient on its own: after a
  // distributed-control failure a remote rank may already have entered a
  // provider collective. Aborting propagates the failure through the provider
  // and prevents that peer from waiting for a rank that will never arrive.
  return Abort();
}

std::string GpuClique::DebugString() const {
  std::string out = absl::StrFormat(
      "key: %v; fingerprint(id): %d; size: %d; communicators: ", key_,
      ids_.has_value() ? ids_->fingerprint() : 0, num_communicators());
  int32_t cnt = 0;
  ForEachComm([&](RankId rank, Communicator* comm) {
    if (cnt++) {
      absl::StrAppend(&out, ", ");
    }
    absl::StrAppendFormat(&out, "[rank=%d, comm=%p]", rank.value(), comm);
  });
  return out;
}

absl::Status GpuClique::HealthCheck() const {
  absl::Status health_check = absl::OkStatus();
  ForEachComm([&health_check](RankId rank, Communicator* comm) {
    if (auto s = comm->HealthCheck(); !s.ok()) {
      LOG(ERROR) << "GPU communicator error (rank " << rank << "): " << s;
      if (health_check.ok()) {
        health_check = std::move(s);  // return first error
      }
    }
  });
  return health_check;
}

absl::Status GpuClique::Abort() {
  VLOG(1) << "Aborting GpuClique " << key();
  absl::Status result = absl::OkStatus();
  ForEachComm([this, &result](RankId rank, Communicator* comm) {
    if (absl::Status s = comm->Abort(); !s.ok()) {
      LOG(ERROR) << "Error aborting GPU communicator (rank " << rank
                 << ") for clique " << key() << ": " << s;
      result = std::move(s);
    }
  });
  return result;
}

void GpuClique::Cancel() {
  VLOG(1) << "Cancel GpuClique " << key();
  cancel_->Cancel();
}

bool GpuClique::IsCancelled() const { return cancel_->IsCancelled(); }

std::string GpuClique::LockableName::ToString(const GpuClique& clique) {
  return absl::StrFormat("lockable clique %v", clique.key());
}

LockableGpuClique::LockableGpuClique(
    GpuCliqueKey clique_key, std::optional<CliqueIds> clique_ids,
    absl::btree_map<RankId, std::unique_ptr<Communicator>> communicators,
    bool peer_access_enabled, std::shared_ptr<CancellationToken> cancel,
    const GpuClique* parent)
    : Lockable(std::move(clique_key), clique_ids, std::move(communicators),
               peer_access_enabled, std::move(cancel), parent) {}

absl::Status LockableGpuClique::HealthCheck() const {
  return value().HealthCheck();
}

absl::Status LockableGpuClique::Abort() { return mutable_value().Abort(); }

void LockableGpuClique::Cancel() { mutable_value().Cancel(); }

bool LockableGpuClique::IsParentSupersetOf(
    const GpuCliqueKey& clique_key) const {
  const GpuClique* p = this->value().parent();
  if (p == nullptr) return false;
  return clique_key.IsSubsetOf(p->key());
}

std::string LockableGpuClique::DebugString() const {
  return absl::StrFormat("LockableGpuClique: %s", value().DebugString());
}

}  // namespace xla::gpu
