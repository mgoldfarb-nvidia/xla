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

#include "xla/backends/gpu/collectives/nccl_symmetric_memory.h"

#include <cstdint>
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
#include "xla/core/collectives/rank_id.h"
#include "xla/future.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream_executor.h"

// Include NCCL after XLA headers.
#include "third_party/nccl/nccl.h"
#include "third_party/nccl/nccl_device.h"
#include "xla/tsl/concurrency/executor.h"

namespace xla::gpu {

namespace {

Future<> Execute(absl::AnyInvocable<absl::Status() &&> f,
                 const std::shared_ptr<tsl::Executor>& executor,
                 std::thread::id owner_thread_id) {
  return executor != nullptr && std::this_thread::get_id() != owner_thread_id
             ? MakeFutureOn<void>(*executor, std::move(f))
             : Future<>(std::move(f)());
}

template <typename T>
absl::StatusOr<T> ExecuteAwait(absl::AnyInvocable<absl::StatusOr<T>() &&> f,
                               const std::shared_ptr<tsl::Executor>& executor,
                               std::thread::id owner_thread_id) {
  return executor != nullptr && std::this_thread::get_id() != owner_thread_id
             ? MakeFutureOn<T>(*executor, std::move(f)).Await()
             : std::move(f)();
}

void LogInterconnectStatus(stream_executor::StreamExecutor* stream_executor) {
  if (stream_executor == nullptr) {
    LOG(ERROR) << "Interconnect Status: Unknown (StreamExecutor not available)";
    return;
  }

  std::string interconnect_status = "Unknown (Failed to get status)";
  absl::StatusOr<std::string> interconnect_status_or =
      stream_executor->GetInterconnectStatus();
  if (!interconnect_status_or.ok()) {
    LOG(ERROR) << "Failed to get interconnect status: "
               << interconnect_status_or.status();
    return;
  }

  LOG(ERROR) << "Interconnect Status: " << *interconnect_status_or;
}

}  // namespace

absl::Status ValidateNcclWindowDeviceAbi(uint64_t compile_time_version,
                                         uint64_t runtime_version) {
  if (compile_time_version != runtime_version) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "NCCL symmetric memory device ABI mismatch: XLA compile-time "
        "version=%d, loaded runtime version=%d",
        compile_time_version, runtime_version));
  }
  return absl::OkStatus();
}

NcclSymmetricMemory::NcclSymmetricMemory(
    std::shared_ptr<NcclCommState> comm_state, ncclWindow_t win,
    stream_executor::DeviceAddressBase addr,
    std::shared_ptr<tsl::Executor> executor,
    stream_executor::StreamExecutor* stream_executor,
    uint64_t validated_device_abi_version)
    : SymmetricMemory(kNcclWindowAbiSchema, validated_device_abi_version),
      comm_state_(comm_state),
      win_(win),
      addr_(addr),
      executor_(std::move(executor)),
      stream_executor_(stream_executor),
      owner_thread_id_(std::this_thread::get_id()) {}

absl::StatusOr<std::unique_ptr<NcclSymmetricMemory>>
NcclSymmetricMemory::Create(std::shared_ptr<NcclCommState> comm_state,
                            stream_executor::DeviceAddressBase addr,
                            const std::shared_ptr<tsl::Executor> executor,
                            stream_executor::StreamExecutor* stream_executor) {
  if (stream_executor == nullptr) {
    return absl::InvalidArgumentError(
        "StreamExecutor is required to create NCCL symmetric memory");
  }
  auto activation = stream_executor->Activate();

  int runtime_version = 0;
  XLA_NCCL_RETURN_IF_ERROR(ncclGetVersion(&runtime_version));
  if (absl::Status status =
          ValidateNcclWindowDeviceAbi(NCCL_VERSION_CODE, runtime_version);
      !status.ok()) {
    return status;
  }

  return Create(std::move(comm_state), addr, executor, stream_executor,
                runtime_version);
}

absl::StatusOr<std::unique_ptr<NcclSymmetricMemory>>
NcclSymmetricMemory::Create(std::shared_ptr<NcclCommState> comm_state,
                            stream_executor::DeviceAddressBase addr,
                            const std::shared_ptr<tsl::Executor> executor,
                            stream_executor::StreamExecutor* stream_executor,
                            uint64_t validated_device_abi_version) {
  if (stream_executor == nullptr) {
    return absl::InvalidArgumentError(
        "StreamExecutor is required to create NCCL symmetric memory");
  }
  RETURN_IF_ERROR(ValidateNcclWindowDeviceAbi(NCCL_VERSION_CODE,
                                              validated_device_abi_version));
  auto activation = stream_executor->Activate();

  auto win = std::make_shared<ncclWindow_t>(nullptr);
  absl::Status registration_status;
  ncclResult_t nccl_status = ncclSuccess;
  if (comm_state->cancel->IsCancelled()) {
    return absl::FailedPreconditionError("NcclCommunicator aborted");
  }
  {
    VLOG(3) << absl::StrFormat(
        "Create NCCL symmetric memory on comm=%p from: ptr=%p; size=%ld",
        comm_state->comm, addr.opaque(), addr.size());
    absl::MutexLock lock(comm_state->mutex);
    if (comm_state->cancel->IsCancelled() || comm_state->aborted ||
        comm_state->destroyed) {
      return absl::FailedPreconditionError("NcclCommunicator aborted");
    }
    nccl_status =
        ncclCommWindowRegister(comm_state->comm, addr.opaque(), addr.size(),
                               win.get(), NCCL_WIN_COLL_SYMMETRIC);
    registration_status = XLA_NCCL_STATUS(nccl_status);
  }
  if (registration_status.ok() && nccl_status == ncclInProgress) {
    registration_status =
        ::xla::gpu::PollUntilDone(*comm_state, *comm_state->cancel);
    comm_state->host_storage.RetainOnFailure(registration_status, win);
  }

  if (!registration_status.ok()) {
    LogInterconnectStatus(stream_executor);
    return registration_status;
  }

  return absl::WrapUnique(
      new NcclSymmetricMemory(comm_state, *win, addr, executor, stream_executor,
                              validated_device_abi_version));
}

NcclSymmetricMemory::~NcclSymmetricMemory() {
  absl::Status status =
      Execute(
          [&] {
            auto activation = stream_executor_->Activate();
            VLOG(3) << absl::StrFormat(
                "Destroy %v with addr=%p, size=%ld executor=%p", *this,
                addr_.opaque(), addr_.size(), executor_.get());
            ncclResult_t nccl_status = ncclSuccess;
            if (comm_state_->cancel->IsCancelled()) {
              VLOG(1) << "Skipping NCCL symmetric memory teardown after "
                         "parent cancellation";
              return absl::OkStatus();
            }
            {
              absl::MutexLock lock(comm_state_->mutex);
              if (comm_state_->cancel->IsCancelled() || comm_state_->aborted ||
                  comm_state_->destroyed) {
                VLOG(1) << "Skipping NCCL symmetric memory teardown after "
                           "parent abort or destruction";
                return absl::OkStatus();
              }
              nccl_status = ncclCommWindowDeregister(comm_state_->comm, win_);
            }
            RETURN_IF_ERROR(XLA_NCCL_STATUS(nccl_status));
            if (nccl_status == ncclInProgress) {
              RETURN_IF_ERROR(::xla::gpu::PollUntilDone(*comm_state_,
                                                        *comm_state_->cancel));
            }
            return absl::OkStatus();
          },
          executor_, owner_thread_id_)
          .Await();
  if (!status.ok()) {
    LOG(ERROR) << "Failed to destroy NCCL symmetric memory: " << status;
  }
}

stream_executor::DeviceAddressBase NcclSymmetricMemory::addr() const {
  return addr_;
}

absl::StatusOr<stream_executor::DeviceAddressBase>
NcclSymmetricMemory::multimem_addr() const {
  return ExecuteAwait<stream_executor::DeviceAddressBase>(
      [this]() -> absl::StatusOr<stream_executor::DeviceAddressBase> {
        auto activation = stream_executor_->Activate();
#if (NCCL_VERSION_CODE >= 22900) || defined(USE_NCCL_HOST_API)
        auto multimem = std::make_shared<void*>(nullptr);
        ncclResult_t nccl_status = ncclSuccess;
        if (comm_state_->cancel->IsCancelled()) {
          return absl::FailedPreconditionError(
              "NCCL communicator is no longer usable");
        }
        {
          absl::MutexLock lock(comm_state_->mutex);
          if (comm_state_->cancel->IsCancelled() || comm_state_->aborted ||
              comm_state_->destroyed) {
            return absl::FailedPreconditionError(
                "NCCL communicator is no longer usable");
          }
          nccl_status =
              ncclGetLsaMultimemDevicePointer(win_, 0, multimem.get());
          RETURN_IF_ERROR(XLA_NCCL_STATUS(nccl_status));
        }
        if (nccl_status == ncclInProgress) {
          absl::Status status =
              ::xla::gpu::PollUntilDone(*comm_state_, *comm_state_->cancel);
          comm_state_->host_storage.RetainOnFailure(status, multimem);
          RETURN_IF_ERROR(status);
        }
        if (*multimem) {
          return stream_executor::DeviceAddressBase(*multimem, addr_.size());
        }
#endif
        return absl::UnimplementedError(
            "Multimem not supported on this NCCL version or device");
      },
      executor_, owner_thread_id_);
}

absl::StatusOr<stream_executor::DeviceAddressBase>
NcclSymmetricMemory::peer_addr(RankId peer) const {
  return ExecuteAwait<stream_executor::DeviceAddressBase>(
      [this, peer]() -> absl::StatusOr<stream_executor::DeviceAddressBase> {
        auto activation = stream_executor_->Activate();
#if (NCCL_VERSION_CODE >= 22902) || defined(USE_NCCL_HOST_API)
        auto peer_addr = std::make_shared<void*>(nullptr);
        ncclResult_t nccl_status = ncclSuccess;
        if (comm_state_->cancel->IsCancelled()) {
          return absl::FailedPreconditionError(
              "NCCL communicator is no longer usable");
        }
        {
          absl::MutexLock lock(comm_state_->mutex);
          if (comm_state_->cancel->IsCancelled() || comm_state_->aborted ||
              comm_state_->destroyed) {
            return absl::FailedPreconditionError(
                "NCCL communicator is no longer usable");
          }
          nccl_status =
              ncclGetPeerDevicePointer(win_, 0, peer.value(), peer_addr.get());
          RETURN_IF_ERROR(XLA_NCCL_STATUS(nccl_status));
        }
        if (nccl_status == ncclInProgress) {
          absl::Status status =
              ::xla::gpu::PollUntilDone(*comm_state_, *comm_state_->cancel);
          comm_state_->host_storage.RetainOnFailure(status, peer_addr);
          RETURN_IF_ERROR(status);
        }
        if (*peer_addr) {
          return stream_executor::DeviceAddressBase(*peer_addr, addr_.size());
        }
        return absl::FailedPreconditionError(absl::StrFormat(
            "Peer rank %d is not load/store accessible from this rank",
            peer.value()));
#else
        return absl::UnimplementedError(
            "Peer address requires ncclGetPeerDevicePointer (NCCL >= 2.29.2)");
#endif
      },
      executor_, owner_thread_id_);
}

std::string NcclSymmetricMemory::ToString() const {
  absl::MutexLock lock(comm_state_->mutex);
  return absl::StrFormat(
      "NcclSymmetricMemory(comm=%p, win=%p, ptr=%p, size=%ld)",
      comm_state_->comm, win_, addr_.opaque(), addr_.size());
}

NcclSymmetricMemory::PackedKernelArg NcclSymmetricMemory::PackKernelArg()
    const {
  return win_;
}

}  // namespace xla::gpu
