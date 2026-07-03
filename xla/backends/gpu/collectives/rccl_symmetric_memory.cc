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

#include "xla/backends/gpu/collectives/rccl_symmetric_memory.h"

#include <cstring>
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
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/cancellation_token.h"
#include "xla/backends/gpu/collectives/rccl_errors.h"
#include "xla/backends/gpu/collectives/rccl_types.h"
#include "xla/future.h"
#include "xla/stream_executor/device_address.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/util.h"

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

RcclSymmetricMemory::RcclSymmetricMemory(
    std::shared_ptr<RcclCommState> comm, ncclWindow_t win,
    stream_executor::DeviceAddressBase addr,
    std::shared_ptr<tsl::Executor> executor,
    stream_executor::StreamExecutor* stream_executor)
    : comm_(std::move(comm)),
      win_(win),
      addr_(addr),
      executor_(std::move(executor)),
      stream_executor_(stream_executor),
      owner_thread_id_(std::this_thread::get_id()) {}

absl::StatusOr<std::unique_ptr<RcclSymmetricMemory>>
RcclSymmetricMemory::Create(std::shared_ptr<RcclCommState> comm,
                            stream_executor::DeviceAddressBase addr,
                            std::shared_ptr<tsl::Executor> executor,
                            stream_executor::StreamExecutor* stream_executor) {
  if (comm == nullptr) {
    return absl::InvalidArgumentError(
        "RCCL communicator state must not be null");
  }
  VLOG(3) << absl::StrFormat(
      "Create RCCL symmetric memory on comm=%p from: ptr=%p; size=%ld",
      comm->comm, addr.opaque(), addr.size());
  std::unique_ptr<stream_executor::ActivateContext> activation =
      stream_executor != nullptr ? stream_executor->Activate() : nullptr;

  auto win = std::make_shared<ncclWindow_t>(nullptr);
  ncclResult_t rccl_status = ncclSuccess;
  if (comm->cancel->IsCancelled()) {
    return absl::FailedPreconditionError("RcclCommunicator aborted");
  }
  {
    absl::MutexLock lock(comm->mutex);
    if (comm->cancel->IsCancelled() || comm->aborted || comm->destroyed) {
      return absl::FailedPreconditionError("RcclCommunicator aborted");
    }
    rccl_status = ncclCommWindowRegister(comm->comm, addr.opaque(), addr.size(),
                                         win.get(), NCCL_WIN_COLL_SYMMETRIC);
    RETURN_IF_ERROR(XLA_RCCL_STATUS(rccl_status));
  }
  if (rccl_status == ncclInProgress) {
    absl::Status status = ::xla::gpu::PollUntilDone(*comm, *comm->cancel);
    comm->host_storage.RetainOnFailure(status, win);
    RETURN_IF_ERROR(status);
  }

  return absl::WrapUnique(new RcclSymmetricMemory(
      std::move(comm), *win, addr, std::move(executor), stream_executor));
}

RcclSymmetricMemory::~RcclSymmetricMemory() {
  VLOG(3) << absl::StrFormat("Destroy %v", *this);
  absl::Status status =
      Execute(
          [this]() -> absl::Status {
            std::unique_ptr<stream_executor::ActivateContext> activation =
                stream_executor_ != nullptr ? stream_executor_->Activate()
                                            : nullptr;
            ncclResult_t rccl_status = ncclSuccess;
            if (comm_->cancel->IsCancelled()) {
              VLOG(1) << "Skipping RCCL symmetric memory teardown after "
                         "parent cancellation";
              return absl::OkStatus();
            }
            {
              absl::MutexLock lock(comm_->mutex);
              if (comm_->cancel->IsCancelled() || comm_->aborted ||
                  comm_->destroyed) {
                VLOG(1) << "Skipping RCCL symmetric memory teardown after "
                           "parent abort or destruction";
                return absl::OkStatus();
              }
              rccl_status = ncclCommWindowDeregister(comm_->comm, win_);
            }
            RETURN_IF_ERROR(XLA_RCCL_STATUS(rccl_status));
            if (rccl_status == ncclInProgress) {
              RETURN_IF_ERROR(
                  ::xla::gpu::PollUntilDone(*comm_, *comm_->cancel));
            }
            return absl::OkStatus();
          },
          executor_, owner_thread_id_)
          .Await();
  if (!status.ok()) {
    LOG(ERROR) << "Failed to destroy RCCL symmetric memory: " << status;
  }
}

stream_executor::DeviceAddressBase RcclSymmetricMemory::addr() const {
  return addr_;
}

std::string RcclSymmetricMemory::ToString() const {
  return absl::StrFormat(
      "RcclSymmetricMemory(comm=%p, win=%p, ptr=%p, size=%ld)", comm_->comm,
      win_, addr_.opaque(), addr_.size());
}

RcclSymmetricMemory::PackedKernelArg RcclSymmetricMemory::PackKernelArg()
    const {
  return PackedKernelArg(sizeof(win_), [&](absl::Span<char> packed) {
    std::memcpy(packed.data(), &win_, sizeof(win_));
  });
}

}  // namespace xla::gpu
