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

#include "xla/backends/gpu/collectives/nccl_registered_memory.h"

#include <memory>
#include <string>
#include <thread>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/collectives/cancellation_token.h"
#include "xla/backends/gpu/collectives/nccl_errors.h"
#include "xla/backends/gpu/collectives/nccl_types.h"
#include "xla/future.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/executor.h"

// Include NCCL after XLA headers.
#include "third_party/nccl/nccl.h"

namespace xla::gpu {

namespace {

Future<> Execute(absl::AnyInvocable<absl::Status() &&> f,
                 const std::shared_ptr<tsl::Executor>& executor,
                 std::thread::id owner_thread_id) {
  return executor != nullptr && std::this_thread::get_id() != owner_thread_id
             ? MakeFutureOn<void>(*executor, std::move(f))
             : Future<>(std::move(f)());
}

}  // namespace

NcclRegisteredMemory::NcclRegisteredMemory(
    std::shared_ptr<NcclCommState> comm, void* handle,
    stream_executor::DeviceAddressBase addr,
    std::shared_ptr<tsl::Executor> executor,
    stream_executor::StreamExecutor* stream_executor)
    : comm_(std::move(comm)),
      handle_(handle),
      addr_(addr),
      executor_(std::move(executor)),
      stream_executor_(stream_executor),
      owner_thread_id_(std::this_thread::get_id()) {}

absl::StatusOr<std::unique_ptr<NcclRegisteredMemory>>
NcclRegisteredMemory::Create(std::shared_ptr<NcclCommState> comm_state,
                             stream_executor::DeviceAddressBase addr,
                             std::shared_ptr<tsl::Executor> executor,
                             stream_executor::StreamExecutor* stream_executor) {
  if (stream_executor == nullptr) {
    return absl::InvalidArgumentError(
        "StreamExecutor is required to create NCCL registered memory");
  }
  auto activation = stream_executor->Activate();

  auto handle = std::make_shared<void*>(nullptr);
  ncclResult_t nccl_status = ncclSuccess;
  if (comm_state->cancel->IsCancelled()) {
    return absl::FailedPreconditionError("NcclCommunicator aborted");
  }
  {
    absl::MutexLock lock(comm_state->mutex);
    if (comm_state->cancel->IsCancelled() || comm_state->aborted ||
        comm_state->destroyed) {
      return absl::FailedPreconditionError("NcclCommunicator aborted");
    }
    VLOG(3) << absl::StrFormat(
        "Create NCCL registered memory on comm=%p from: ptr=%p; size=%ld",
        comm_state->comm, addr.opaque(), addr.size());
    nccl_status = ncclCommRegister(comm_state->comm, addr.opaque(), addr.size(),
                                   handle.get());
    RETURN_IF_ERROR(XLA_NCCL_STATUS(nccl_status));
  }
  if (nccl_status == ncclInProgress) {
    absl::Status status =
        ::xla::gpu::PollUntilDone(*comm_state, *comm_state->cancel);
    comm_state->host_storage.RetainOnFailure(status, handle);
    RETURN_IF_ERROR(status);
  }
  return absl::WrapUnique(new NcclRegisteredMemory(
      comm_state, *handle, addr, std::move(executor), stream_executor));
}

NcclRegisteredMemory::~NcclRegisteredMemory() {
  absl::Status status =
      Execute(
          [this] {
            auto activation = stream_executor_->Activate();
            VLOG(3) << absl::StrFormat("Destroy %v", *this);
            ncclResult_t nccl_status = ncclSuccess;
            if (comm_->cancel->IsCancelled()) {
              VLOG(1) << "Skipping NCCL registered memory teardown after "
                         "parent cancellation";
              return absl::OkStatus();
            }
            {
              absl::MutexLock lock(comm_->mutex);
              if (comm_->cancel->IsCancelled() || comm_->aborted ||
                  comm_->destroyed) {
                VLOG(1) << "Skipping NCCL registered memory teardown after "
                           "parent abort or destruction";
                return absl::OkStatus();
              }
              nccl_status = ncclCommDeregister(comm_->comm, handle_);
            }
            RETURN_IF_ERROR(XLA_NCCL_STATUS(nccl_status));
            if (nccl_status == ncclInProgress) {
              RETURN_IF_ERROR(
                  ::xla::gpu::PollUntilDone(*comm_, *comm_->cancel));
            }
            return absl::OkStatus();
          },
          executor_, owner_thread_id_)
          .Await();
  if (!status.ok()) {
    LOG(ERROR) << "Failed to destroy NCCL registered memory: " << status;
  }
}

stream_executor::DeviceAddressBase NcclRegisteredMemory::addr() const {
  return addr_;
}

std::string NcclRegisteredMemory::ToString() const {
  absl::MutexLock lock(comm_->mutex);
  return absl::StrFormat(
      "NcclRegisteredMemory(comm=%p, handle=%p, ptr=%p, size=%ld)", comm_->comm,
      handle_, addr_.opaque(), addr_.size());
}

}  // namespace xla::gpu
