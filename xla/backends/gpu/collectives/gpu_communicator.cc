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

#include "xla/backends/gpu/collectives/gpu_communicator.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"

namespace xla::gpu {
namespace {

thread_local std::vector<const void*> group_locked_comm_states;

void QuarantineProviderHostStorage(std::shared_ptr<void> storage) {
  // A failed communicator teardown gives us no provider lifetime boundary.
  // Intentionally retain the storage until process exit (and leak the
  // container itself so static destruction cannot race provider threads).
  static auto* mutex = new absl::Mutex;
  static auto* quarantine = new std::vector<std::shared_ptr<void>>;
  absl::MutexLock lock(*mutex);
  quarantine->push_back(std::move(storage));
}

}  // namespace

absl::Status ValidateGpuCliqueBarrierTokens(
    GpuCliqueBarrierToken expected,
    absl::Span<const GpuCliqueBarrierToken> gathered) {
  if (expected == GpuCliqueBarrierToken{}) {
    return absl::InvalidArgumentError(
        "A clique barrier requires a nonzero token");
  }
  if (gathered.empty()) {
    return absl::InvalidArgumentError(
        "A clique barrier requires at least one gathered rank token");
  }

  for (size_t rank = 0; rank < gathered.size(); ++rank) {
    if (gathered[rank] != expected) {
      return absl::FailedPreconditionError(absl::StrFormat(
          "Clique barrier token mismatch at rank %d: got (%016x, %016x), "
          "expected (%016x, %016x)",
          rank, gathered[rank].high, gathered[rank].low, expected.high,
          expected.low));
    }
  }
  return absl::OkStatus();
}

void GpuProviderHostStorage::RetainUntilProviderTeardown(
    std::shared_ptr<void> storage) {
  if (storage == nullptr) return;
  absl::MutexLock lock(mutex_);
  if (provider_teardown_complete_) return;
  if (provider_teardown_failed_) {
    QuarantineProviderHostStorage(std::move(storage));
    return;
  }
  retained_.push_back(std::move(storage));
}

void GpuProviderHostStorage::RetainOnFailure(const absl::Status& status,
                                             std::shared_ptr<void> storage) {
  if (!status.ok()) {
    RetainUntilProviderTeardown(std::move(storage));
  }
}

void GpuProviderHostStorage::ProviderTeardownComplete() {
  absl::MutexLock lock(mutex_);
  if (provider_teardown_failed_) return;
  provider_teardown_complete_ = true;
  retained_.clear();
}

void GpuProviderHostStorage::ProviderTeardownFailed() {
  std::vector<std::shared_ptr<void>> retained;
  {
    absl::MutexLock lock(mutex_);
    if (provider_teardown_complete_ || provider_teardown_failed_) return;
    provider_teardown_failed_ = true;
    retained.swap(retained_);
  }
  for (auto& storage : retained) {
    QuarantineProviderHostStorage(std::move(storage));
  }
}

size_t GpuProviderHostStorage::retained_count_for_test() const {
  absl::MutexLock lock(mutex_);
  return retained_.size();
}

namespace internal {

ScopedGpuCommGroupLockOwnership::ScopedGpuCommGroupLockOwnership(
    const void* comm_state)
    : comm_state_(comm_state) {
  group_locked_comm_states.push_back(comm_state_);
}

ScopedGpuCommGroupLockOwnership::~ScopedGpuCommGroupLockOwnership() {
  auto it = std::find(group_locked_comm_states.rbegin(),
                      group_locked_comm_states.rend(), comm_state_);
  if (it != group_locked_comm_states.rend()) {
    group_locked_comm_states.erase(std::next(it).base());
  }
}

bool IsGpuCommGroupLockOwnedByCurrentThread(const void* comm_state) {
  return std::find(group_locked_comm_states.begin(),
                   group_locked_comm_states.end(),
                   comm_state) != group_locked_comm_states.end();
}

}  // namespace internal

}  // namespace xla::gpu
