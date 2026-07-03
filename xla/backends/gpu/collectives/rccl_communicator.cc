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

#include "xla/backends/gpu/collectives/rccl_communicator.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/functional/function_ref.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "rocm/include/hip/hip_runtime.h"
#include "rocm/include/rccl/rccl.h"
#include "rocm/rocm_config.h"  // IWYU pragma: keep
#include "xla/backends/gpu/collectives/cancellation_token.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/collectives/rccl_errors.h"
#include "xla/backends/gpu/collectives/rccl_group.h"
#include "xla/backends/gpu/collectives/rccl_symmetric_memory.h"
#include "xla/backends/gpu/collectives/single_threaded_executor.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/future.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

class ResolvedRcclSymmetricMemoryPlan final
    : public GpuCommunicator::SymmetricMemoryPlan {
 public:
  ResolvedRcclSymmetricMemoryPlan(const GpuCommunicator* owner,
                                  se::DeviceAddressBase address,
                                  int runtime_version)
      : SymmetricMemoryPlan(
            owner, address, "rccl-window",
            absl::StrFormat("rccl-symmetric-memory-plan-v1;compile_version=%d;"
                            "runtime_version=%d;size=%d;flags=%d;",
                            NCCL_VERSION_CODE, runtime_version, address.size(),
                            NCCL_WIN_COLL_SYMMETRIC)),
        runtime_version_(runtime_version) {}

  int runtime_version() const { return runtime_version_; }

 private:
  int runtime_version_;
};

hipStream_t AsHipStream(se::Stream* stream) {
  return absl::bit_cast<hipStream_t>(stream->platform_specific_handle().stream);
}

se::Stream* ToStream(const Communicator::Executor& executor) {
  return absl::down_cast<const GpuCollectives::Executor&>(executor).stream();
}

// Serializes one RCCL call that consumes a communicator handle. Cancellation
// is checked before waiting for the mutex and again after acquiring it so an
// operation queued behind another caller cannot overtake Abort and touch a
// communicator after cancellation has been published.
absl::Status WithLiveRcclComm(
    RcclCommState& state,
    absl::FunctionRef<absl::Status(ncclComm_t)> provider_call)
    ABSL_NO_THREAD_SAFETY_ANALYSIS {
  if (state.cancel->IsCancelled()) {
    return FailedPrecondition("RcclCommunicator aborted");
  }

  if (internal::IsGpuCommGroupLockOwnedByCurrentThread(&state)) {
    return provider_call(state.comm);
  }
  if (IsInsideRcclGroupLaunch()) {
    return FailedPrecondition(
        "RCCL group operation used a communicator not locked by the group");
  }

  absl::MutexLock lock(state.mutex);
  if (state.cancel->IsCancelled() || state.aborted || state.destroyed) {
    return FailedPrecondition("RcclCommunicator aborted");
  }
  return provider_call(state.comm);
}

//==-----------------------------------------------------------------------===//
// Conversions between XLA and RCCL data types
//==-----------------------------------------------------------------------===//

static size_t ToNcclCount(PrimitiveType dtype, size_t count) {
  return primitive_util::IsComplexType(dtype) ? count * 2 : count;
}

static absl::StatusOr<ncclDataType_t> ToNcclDataType(
    PrimitiveType dtype, bool is_reduction_op,
    se::RocmComputeCapability rocm_cc) {
  switch (dtype) {
    case F8E5M2:
      return rocm_cc.has_ocp_fp8_support() ? ncclFloat8e5m2 : ncclInt8;
    case F8E4M3FN:
      return rocm_cc.has_ocp_fp8_support() ? ncclFloat8e4m3 : ncclInt8;
    case S8:
    case F8E5M2FNUZ:
    case F8E4M3FNUZ:
    case F8E8M0FNU:
      return ncclInt8;
    case PRED:
    case U8:
      return ncclUint8;
    case S32:
      return ncclInt32;
    case U32:
      return ncclUint32;
    case S64:
      return ncclInt64;
    case U64:
      return ncclUint64;
    case F16:
      return ncclFloat16;
    case F32:
    case C64:
      return ncclFloat32;
    case F64:
    case C128:
      return ncclFloat64;
    case S16:
    case U16:
      // For reductions we expect 16 bit integer types to be promoted to 32-bit.
      if (is_reduction_op) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Unsupported data type for reduction operation: %s",
                            primitive_util::LowercasePrimitiveTypeName(dtype)));
      }
      // For collectives that just move data around, we can use ncclFloat16 for
      // 16-bit integer data types.
      return ncclFloat16;
    case BF16:
      return ncclBfloat16;
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("Unsupported data type: %s",
                          primitive_util::LowercasePrimitiveTypeName(dtype)));
  }
}

static ncclRedOp_t ToNcclReduction(ReductionKind kind) {
  switch (kind) {
    case ReductionKind::SUM:
      return ncclSum;
    case ReductionKind::PRODUCT:
      return ncclProd;
    case ReductionKind::MIN:
      return ncclMin;
    case ReductionKind::MAX:
      return ncclMax;
  }
}

}  // namespace

//==-----------------------------------------------------------------------===//
// RCCL Communicator
//==-----------------------------------------------------------------------===//

absl::StatusOr<std::unique_ptr<RcclCommunicator>> RcclCommunicator::Create(
    se::StreamExecutor* stream_executor,
    absl::AnyInvocable<absl::StatusOr<ncclComm_t>()> make_comm,
    std::shared_ptr<CancellationToken> cancel, bool is_async, tsl::Env& env) {
  if (stream_executor == nullptr) {
    return InvalidArgument("RcclCommunicator requires a StreamExecutor");
  }
  if (cancel == nullptr) {
    cancel = std::make_shared<CancellationToken>();
  }
  auto f = [cancel, &make_comm]() -> absl::StatusOr<ncclComm_t> {
    ASSIGN_OR_RETURN(ncclComm_t comm, make_comm());
    RETURN_IF_ERROR(::xla::gpu::PollUntilDone(comm, *cancel));
    return comm;
  };

  if (!is_async) {
    // If this RcclCommunicator is synchronous, construct ncclComm_t in the
    // calling thread.
    ASSIGN_OR_RETURN(ncclComm_t comm, f());
    auto comm_state = std::make_shared<RcclCommState>(comm, cancel);
    return absl::WrapUnique(new RcclCommunicator(
        stream_executor, std::move(comm_state), nullptr, std::move(cancel)));
  }

  // If this RcclCommunicator is asynchronous, then all operations on the
  // underlying ncclComm_t, including its creation, must take place on the
  // single threaded executor.
  auto executor = std::make_shared<SingleThreadedExecutor>(env);
  ASSIGN_OR_RETURN(ncclComm_t comm,
                   MakeFutureOn<ncclComm_t>(*executor, f).Await());
  auto comm_state = std::make_shared<RcclCommState>(comm, cancel);
  return absl::WrapUnique(
      new RcclCommunicator(stream_executor, std::move(comm_state),
                           std::move(executor), std::move(cancel)));
}

RcclCommunicator::~RcclCommunicator() {
  auto f = [this]() -> absl::Status {
    if (comm_ == nullptr) {
      VLOG(1) << "Skipping destruction; null comm_ " << *this;
      return absl::OkStatus();
    }

    // Note that we intentionally don't call PollUntilDone. Once comm_ has been
    // destroyed, we can no longer safely touch it.
    absl::MutexLock lock(comm_->mutex);
    if (comm_->aborted) {
      VLOG(1) << "Skipping destruction of already-aborted RCCL communicator";
      return absl::OkStatus();
    }
    if (comm_->destroyed) {
      return absl::OkStatus();
    }
    VLOG(1) << "Destroy RCCL communicator " << comm_->comm;
    ncclResult_t rccl_status = ncclCommDestroy(comm_->comm);
    comm_->destroyed = true;
    // RCCL follows NCCL's teardown contract: the communicator is no longer
    // accessible after Destroy returns, including provider references to host
    // output storage retained for cancelled nonblocking operations.
    if (rccl_status == ncclSuccess) {
      comm_->host_storage.ProviderTeardownComplete();
    } else {
      comm_->host_storage.ProviderTeardownFailed();
    }
    return XLA_RCCL_STATUS(rccl_status);
  };

  if (absl::Status s = Execute(f).Await(); !s.ok()) {
    LOG(ERROR) << "RcclCommunicator::~RcclCommunicator: " << s;
  }
}

absl::Status RcclCommunicator::Abort() {
  // By setting the cancellation token all pending collectives scheduled on
  // executor_ will cancel. This will allow the aborting lambda below to run.
  cancel_->Cancel();

  return ExecuteAwait([this]() -> absl::Status {
    VLOG(1) << "Abort RCCL communicator: " << *this;
    absl::MutexLock lock(comm_->mutex);
    if (comm_->aborted || comm_->destroyed) {
      return FailedPrecondition("RcclCommunicator already aborted");
    }
    comm_->aborted = true;
    // Note that we intentionally don't call PollUntilDone. Once comm_
    // has been aborted, we can no longer safely touch it.
    ncclResult_t rccl_status = ncclCommAbort(comm_->comm);
    // Abort destroys the communicator after aborting uncompleted operations,
    // so provider-owned references to retained host output storage are dead.
    if (rccl_status == ncclSuccess) {
      comm_->host_storage.ProviderTeardownComplete();
    } else {
      comm_->host_storage.ProviderTeardownFailed();
    }
    return XLA_RCCL_STATUS(rccl_status);
  });
}

absl::Status RcclCommunicator::HealthCheck() const {
  return ExecuteAwait([this]() -> absl::Status {
    VLOG(5) << "Get last async error for RCCL communicator: " << *this;
    if (cancel_->IsCancelled()) {
      return absl::FailedPreconditionError("RcclCommunicator aborted");
    }

    ncclResult_t async_err;
    std::string last_error;
    RETURN_IF_ERROR(WithLiveRcclComm(*comm_, [&](ncclComm_t comm) {
      XLA_RCCL_RETURN_IF_ERROR(ncclCommGetAsyncError(comm, &async_err));
      if (async_err != ncclSuccess) last_error = ncclGetLastError(comm);
      return absl::OkStatus();
    }));
    if (async_err == ncclSuccess) {
      return absl::OkStatus();
    }

    return Internal("%s. Last RCCL error (maybe unrelated): %s", last_error,
                    ncclGetErrorString(async_err));
  });
}

absl::StatusOr<size_t> RcclCommunicator::NumRanks() const {
  return ExecuteAwait<size_t>([this]() -> absl::StatusOr<size_t> {
    VLOG(5) << "Get the number of ranks in RCCL communicator: " << *this;
    if (cancel_->IsCancelled()) {
      return absl::FailedPreconditionError("RcclCommunicator aborted");
    }

    // We intentionally don't call PollUntilDone. ncclCommCount is
    // blocking.
    int32_t count = 0;
    RETURN_IF_ERROR(WithLiveRcclComm(*comm_, [&](ncclComm_t comm) {
      return XLA_RCCL_STATUS(ncclCommCount(comm, &count));
    }));
    return count;
  });
}

absl::Status RcclCommunicator::RunCliqueBarrier(se::Stream* stream,
                                                GpuCliqueBarrierToken token) {
  if (token == GpuCliqueBarrierToken{}) {
    return InvalidArgument("A clique barrier requires a nonzero token");
  }
  if (stream == nullptr) {
    return InvalidArgument("A clique barrier requires a stream");
  }
  if (cancel_->IsCancelled()) {
    return FailedPrecondition("RcclCommunicator aborted");
  }

  constexpr size_t kWordsPerToken = 2;
  static_assert(sizeof(GpuCliqueBarrierToken) ==
                kWordsPerToken * sizeof(uint64_t));

  if (stream->parent() != stream_executor_) {
    return InvalidArgument(
        "Clique barrier stream belongs to a different StreamExecutor");
  }
  se::StreamExecutor* stream_executor = stream_executor_;
  auto activation = stream_executor->Activate();
  absl::MutexLock scratch_lock(completion_barrier_mutex_);

  if (completion_barrier_team_size_ == 0) {
    ASSIGN_OR_RETURN(
        auto identity,
        ExecuteAwait<std::pair<int, int>>(
            [this]() -> absl::StatusOr<std::pair<int, int>> {
              int rank = -1;
              int team_size = 0;
              RETURN_IF_ERROR(WithLiveRcclComm(*comm_, [&](ncclComm_t comm) {
                XLA_RCCL_RETURN_IF_ERROR(ncclCommUserRank(comm, &rank));
                return XLA_RCCL_STATUS(ncclCommCount(comm, &team_size));
              }));
              return std::make_pair(rank, team_size);
            }));
    completion_barrier_rank_ = identity.first;
    completion_barrier_team_size_ = identity.second;
    if (completion_barrier_team_size_ <= 0 || completion_barrier_rank_ < 0 ||
        completion_barrier_rank_ >= completion_barrier_team_size_) {
      return FailedPrecondition(
          "RCCL communicator has invalid barrier identity: rank %d of %d",
          completion_barrier_rank_, completion_barrier_team_size_);
    }
  }

  if (completion_barrier_device_scratch_.address().is_null()) {
    size_t scratch_bytes = static_cast<size_t>(completion_barrier_team_size_) *
                           kWordsPerToken * sizeof(uint64_t);
    se::DeviceAddress<uint64_t> device_scratch =
        stream_executor->AllocateArray<uint64_t>(
            static_cast<size_t>(completion_barrier_team_size_) *
            kWordsPerToken);
    if (device_scratch.is_null()) {
      return ResourceExhausted(
          "Failed to allocate %d bytes of RCCL clique barrier scratch",
          scratch_bytes);
    }
    se::DeviceAddressHandle device_handle(stream_executor, device_scratch);
    ASSIGN_OR_RETURN(std::unique_ptr<se::MemoryAllocation> host_scratch,
                     stream_executor->HostMemoryAllocate(scratch_bytes));
    if (host_scratch->address().is_null() ||
        host_scratch->address().size() < scratch_bytes) {
      return Internal("Invalid RCCL clique barrier host scratch allocation");
    }
    completion_barrier_device_scratch_ = std::move(device_handle);
    completion_barrier_host_scratch_ = std::move(host_scratch);
  }

  int rank = completion_barrier_rank_;
  int team_size = completion_barrier_team_size_;
  size_t scratch_bytes =
      static_cast<size_t>(team_size) * kWordsPerToken * sizeof(uint64_t);
  auto* host_tokens = static_cast<GpuCliqueBarrierToken*>(
      completion_barrier_host_scratch_->address().opaque());
  host_tokens[rank] = token;

  se::DeviceAddress<uint64_t> device_scratch(
      completion_barrier_device_scratch_.address());
  se::DeviceAddress<uint64_t> rank_slot = device_scratch.GetSlice(
      static_cast<size_t>(rank) * kWordsPerToken, kWordsPerToken);
  RETURN_IF_ERROR(
      stream->Memcpy(&rank_slot, &host_tokens[rank], sizeof(token)));

  RETURN_IF_ERROR(
      ExecuteAwait([this, stream, device_scratch, rank_slot]() -> absl::Status {
        if (cancel_->IsCancelled()) {
          return FailedPrecondition("RcclCommunicator aborted");
        }
        auto activation = stream_executor_->Activate();
        ncclResult_t rccl_status = ncclSuccess;
        RETURN_IF_ERROR(WithLiveRcclComm(*comm_, [&](ncclComm_t comm) {
          rccl_status = ncclAllGather(rank_slot.opaque(),
                                      device_scratch.opaque(), kWordsPerToken,
                                      ncclUint64, comm, AsHipStream(stream));
          return XLA_RCCL_STATUS(rccl_status);
        }));
        if (rccl_status == ncclInProgress) {
          RETURN_IF_ERROR(::xla::gpu::PollUntilDone(*comm_, *comm_->cancel));
        }
        return absl::OkStatus();
      }));

  RETURN_IF_ERROR(stream->Memcpy(host_tokens, device_scratch, scratch_bytes));
  RETURN_IF_ERROR(stream->BlockHostUntilDone());
  if (cancel_->IsCancelled()) {
    return FailedPrecondition("RcclCommunicator aborted");
  }
  return ValidateGpuCliqueBarrierTokens(
      token, absl::Span<const GpuCliqueBarrierToken>(host_tokens, team_size));
}

Future<> RcclCommunicator::GroupExecute(
    absl::AnyInvocable<absl::Status() &&> group) {
  return Execute([group = std::move(group), this]() mutable {
    return GroupLaunch([&] { return std::move(group)(); });
  });
}

absl::Status RcclCommunicator::GroupLaunch(
    absl::FunctionRef<absl::Status()> group) {
  if (cancel_->IsCancelled()) {
    return FailedPrecondition("RcclCommunicator aborted");
  }

  ASSIGN_OR_RETURN(bool did_launch,
                   internal::RunGpuCommGroupWithLock(
                       std::vector<std::shared_ptr<RcclCommState>>{comm_},
                       [&] { return RcclGroupLaunch(group); }));
  if (did_launch) {
    return PollUntilDone();
  }
  return absl::OkStatus();
}

Future<> RcclCommunicator::AllReduce(se::DeviceAddressBase send_buffer,
                                     se::DeviceAddressBase recv_buffer,
                                     PrimitiveType dtype, size_t count,
                                     ReductionKind reduction_kind,
                                     const Communicator::Executor& executor) {
  return Execute([send_buffer, recv_buffer, dtype, count, reduction_kind,
                  &executor, this]() -> absl::Status {
    return LaunchAllReduce(send_buffer, recv_buffer, dtype, count,
                           reduction_kind, executor);
  });
}

Future<> RcclCommunicator::Broadcast(se::DeviceAddressBase send_buffer,
                                     se::DeviceAddressBase recv_buffer,
                                     PrimitiveType dtype, size_t count,
                                     RankId root, const Executor& executor) {
  return Execute(
      [send_buffer, recv_buffer, dtype, count, root, &executor, this]() {
        return LaunchBroadcast(send_buffer, recv_buffer, dtype, count, root,
                               executor);
      });
}

Future<> RcclCommunicator::ReduceScatter(se::DeviceAddressBase send_buffer,
                                         se::DeviceAddressBase recv_buffer,
                                         PrimitiveType dtype, size_t count,
                                         ReductionKind reduction_kind,
                                         const Executor& executor) {
  return Execute([send_buffer, recv_buffer, dtype, count, reduction_kind,
                  &executor, this]() {
    return LaunchReduceScatter(send_buffer, recv_buffer, dtype, count,
                               reduction_kind, executor);
  });
}

Future<> RcclCommunicator::AllGather(se::DeviceAddressBase send_buffer,
                                     se::DeviceAddressBase recv_buffer,
                                     PrimitiveType dtype, size_t count,
                                     const Executor& executor) {
  return Execute([send_buffer, recv_buffer, dtype, count, &executor, this]() {
    return LaunchAllGather(send_buffer, recv_buffer, dtype, count, executor);
  });
}

Future<> RcclCommunicator::AllToAll(
    absl::InlinedVector<se::DeviceAddressBase, 4> send_buffers,
    absl::InlinedVector<se::DeviceAddressBase, 4> recv_buffers,
    PrimitiveType dtype, size_t count, const Executor& executor) {
  return Execute([send_buffers, recv_buffers, dtype, count, &executor, this]() {
    return LaunchAllToAll(send_buffers, recv_buffers, dtype, count, executor);
  });
}

Future<> RcclCommunicator::CollectivePermute(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, std::optional<RankId> source_rank,
    absl::Span<const RankId> target_ranks, const Executor& executor) {
  std::vector<RankId> owned_target_ranks(target_ranks.begin(),
                                         target_ranks.end());
  return Execute([send_buffer, recv_buffer, dtype, count, source_rank,
                  owned_target_ranks = std::move(owned_target_ranks), &executor,
                  this]() {
    return LaunchCollectivePermute(send_buffer, recv_buffer, dtype, count,
                                   source_rank, owned_target_ranks, executor);
  });
}

Future<> RcclCommunicator::Send(se::DeviceAddressBase send_buffer,
                                PrimitiveType dtype, size_t count, RankId peer,
                                const Executor& executor) {
  return Execute([send_buffer, dtype, count, peer, &executor, this]() {
    return LaunchSend(send_buffer, dtype, count, peer, executor);
  });
}

Future<> RcclCommunicator::Recv(se::DeviceAddressBase recv_buffer,
                                PrimitiveType dtype, size_t count, RankId peer,
                                const Executor& executor) {
  return Execute([recv_buffer, dtype, count, peer, &executor, this]() {
    return LaunchRecv(recv_buffer, dtype, count, peer, executor);
  });
}

absl::Status RcclCommunicator::LaunchAllReduce(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, ReductionKind reduction_kind,
    const Communicator::Executor& executor) {
  if (cancel_->IsCancelled()) {
    return FailedPrecondition("RcclCommunicator aborted");
  }
  se::Stream* stream = ToStream(executor);

  VLOG(3) << absl::StreamFormat(
      "[%d] Launch RCCL AllReduce operation; send_buffer=%p; "
      "recv_buffer=%p; dtype=%s; count=%d; reduction_kind=%v; comm=%p; "
      "stream=%p",
      stream->parent()->device_ordinal(), send_buffer.opaque(),
      recv_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
      count, reduction_kind, comm_->comm, stream);

  ASSIGN_OR_RETURN(
      ncclDataType_t nccl_dtype,
      ToNcclDataType(
          dtype, /*is_reduction_op=*/true,
          stream->parent()->GetDeviceDescription().rocm_compute_capability()));

  RETURN_IF_ERROR(WithLiveRcclComm(*comm_, [&](ncclComm_t comm) {
    return XLA_RCCL_STATUS(ncclAllReduce(
        send_buffer.opaque(), recv_buffer.opaque(), ToNcclCount(dtype, count),
        nccl_dtype, ToNcclReduction(reduction_kind), comm,
        AsHipStream(stream)));
  }));
  if (!IsInsideRcclGroupLaunch()) {
    RETURN_IF_ERROR(PollUntilDone());
  }
  return absl::OkStatus();
}

absl::Status RcclCommunicator::LaunchBroadcast(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, RankId root, const Executor& executor) {
  if (cancel_->IsCancelled()) {
    return absl::FailedPreconditionError("RcclCommunicator aborted");
  }
  se::Stream* stream = ToStream(executor);

  VLOG(3) << absl::StreamFormat(
      "[%d] Launch RCCL Broadcast operation; send_buffer=%p; "
      "recv_buffer=%p; dtype=%s; count=%d; root=%d; comm=%p; "
      "stream=%p",
      stream->parent()->device_ordinal(), send_buffer.opaque(),
      recv_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
      count, root.value(), comm_->comm, stream);

  ASSIGN_OR_RETURN(
      ncclDataType_t nccl_dtype,
      ToNcclDataType(
          dtype, /*is_reduction_op=*/false,
          stream->parent()->GetDeviceDescription().rocm_compute_capability()));

  RETURN_IF_ERROR(WithLiveRcclComm(*comm_, [&](ncclComm_t comm) {
    return XLA_RCCL_STATUS(ncclBroadcast(
        send_buffer.opaque(), recv_buffer.opaque(), ToNcclCount(dtype, count),
        nccl_dtype, root.value(), comm, AsHipStream(stream)));
  }));
  if (!IsInsideRcclGroupLaunch()) {
    RETURN_IF_ERROR(PollUntilDone());
  }
  return absl::OkStatus();
}

absl::Status RcclCommunicator::LaunchReduceScatter(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, ReductionKind reduction_kind,
    const Executor& executor) {
  if (cancel_->IsCancelled()) {
    return absl::FailedPreconditionError("RcclCommunicator aborted");
  }
  se::Stream* stream = ToStream(executor);

  VLOG(3) << absl::StreamFormat(
      "[%d] Launch RCCL ReduceScatter operation; send_buffer=%p; "
      "recv_buffer=%p; dtype=%s; count=%d; reduction_kind=%v; comm=%p; "
      "stream=%p",
      stream->parent()->device_ordinal(), send_buffer.opaque(),
      recv_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
      count, reduction_kind, comm_->comm, stream);

  ASSIGN_OR_RETURN(
      ncclDataType_t nccl_dtype,
      ToNcclDataType(
          dtype, /*is_reduction_op=*/true,
          stream->parent()->GetDeviceDescription().rocm_compute_capability()));

  RETURN_IF_ERROR(WithLiveRcclComm(*comm_, [&](ncclComm_t comm) {
    return XLA_RCCL_STATUS(ncclReduceScatter(
        send_buffer.opaque(), recv_buffer.opaque(), ToNcclCount(dtype, count),
        nccl_dtype, ToNcclReduction(reduction_kind), comm,
        AsHipStream(stream)));
  }));
  if (!IsInsideRcclGroupLaunch()) {
    RETURN_IF_ERROR(PollUntilDone());
  }
  return absl::OkStatus();
}

absl::Status RcclCommunicator::LaunchAllGather(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, const Executor& executor) {
  if (cancel_->IsCancelled()) {
    return absl::FailedPreconditionError("RcclCommunicator aborted");
  }
  se::Stream* stream = ToStream(executor);

  VLOG(3) << absl::StreamFormat(
      "[%d] Launch RCCL AllGather operation; send_buffer=%p; "
      "recv_buffer=%p; dtype=%s; count=%d; comm=%p; stream=%p",
      stream->parent()->device_ordinal(), send_buffer.opaque(),
      recv_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
      count, comm_->comm, stream);

  ASSIGN_OR_RETURN(
      ncclDataType_t nccl_dtype,
      ToNcclDataType(
          dtype, /*is_reduction_op=*/false,
          stream->parent()->GetDeviceDescription().rocm_compute_capability()));

  RETURN_IF_ERROR(WithLiveRcclComm(*comm_, [&](ncclComm_t comm) {
    return XLA_RCCL_STATUS(ncclAllGather(
        send_buffer.opaque(), recv_buffer.opaque(), ToNcclCount(dtype, count),
        nccl_dtype, comm, AsHipStream(stream)));
  }));
  if (!IsInsideRcclGroupLaunch()) {
    RETURN_IF_ERROR(PollUntilDone());
  }
  return absl::OkStatus();
}

absl::Status RcclCommunicator::LaunchAllToAll(
    absl::InlinedVector<se::DeviceAddressBase, 4> send_buffers,
    absl::InlinedVector<se::DeviceAddressBase, 4> recv_buffers,
    PrimitiveType dtype, size_t count, const Executor& executor) {
  if (cancel_->IsCancelled()) {
    return absl::FailedPreconditionError("RcclCommunicator aborted");
  }
  se::Stream* stream = ToStream(executor);

  auto buffer_formatter = [](std::string* out, se::DeviceAddressBase buffer) {
    absl::StrAppendFormat(out, "%p", buffer.opaque());
  };

  VLOG(3) << absl::StreamFormat(
      "[%d] Launch RCCL AllToAll operation; send_buffers=[%s]; "
      "recv_buffers=[%s]; dtype=%s; count=%d; comm=%p; stream=%p",
      stream->parent()->device_ordinal(),
      absl::StrJoin(send_buffers, ", ", buffer_formatter),
      absl::StrJoin(recv_buffers, ", ", buffer_formatter),
      primitive_util::LowercasePrimitiveTypeName(dtype), count, comm_->comm,
      stream);

  if (send_buffers.size() != recv_buffers.size()) {
    return InvalidArgument(
        "Number of send buffers must match number of recv buffers: %d != %d",
        send_buffers.size(), recv_buffers.size());
  }

  int32_t num_ranks;
  RETURN_IF_ERROR(WithLiveRcclComm(*comm_, [&](ncclComm_t comm) {
    return XLA_RCCL_STATUS(ncclCommCount(comm, &num_ranks));
  }));

  if (send_buffers.size() != num_ranks) {
    return InvalidArgument(
        "Number of send buffers must match number of ranks: %d != %d",
        send_buffers.size(), num_ranks);
  }

  ASSIGN_OR_RETURN(
      ncclDataType_t nccl_dtype,
      ToNcclDataType(
          dtype, /*is_reduction_op=*/false,
          stream->parent()->GetDeviceDescription().rocm_compute_capability()));

  auto group = [&] {
    for (size_t i = 0; i < send_buffers.size(); ++i) {
      se::DeviceAddressBase send_buffer = send_buffers[i];
      se::DeviceAddressBase recv_buffer = recv_buffers[i];

      RETURN_IF_ERROR(WithLiveRcclComm(*comm_, [&](ncclComm_t comm) {
        return XLA_RCCL_STATUS(ncclSend(send_buffer.opaque(),
                                        ToNcclCount(dtype, count), nccl_dtype,
                                        i, comm, AsHipStream(stream)));
      }));
      RETURN_IF_ERROR(WithLiveRcclComm(*comm_, [&](ncclComm_t comm) {
        return XLA_RCCL_STATUS(ncclRecv(recv_buffer.opaque(),
                                        ToNcclCount(dtype, count), nccl_dtype,
                                        i, comm, AsHipStream(stream)));
      }));
    }
    return absl::OkStatus();
  };
  return GroupLaunch(group);
}

absl::Status RcclCommunicator::LaunchCollectivePermute(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, std::optional<RankId> source_rank,
    absl::Span<const RankId> target_ranks, const Executor& executor) {
  if (cancel_->IsCancelled()) {
    return FailedPrecondition("RcclCommunicator aborted");
  }
  se::Stream* stream = ToStream(executor);

  auto rank_formatter = [](std::string* out, RankId rank) {
    absl::StrAppendFormat(out, "%d", rank.value());
  };

  VLOG(3) << absl::StreamFormat(
      "[%d] Launch RCCL CollectivePermute operation; send_buffer=%p; "
      "recv_buffer=%p; dtype=%s; source_rank=%s; target_[ranks=%s]; count=%d; "
      "comm=%p; stream=%p",
      stream->parent()->device_ordinal(), send_buffer.opaque(),
      recv_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
      source_rank ? absl::StrCat(source_rank->value()) : "<empty>",
      absl::StrJoin(target_ranks, ", ", rank_formatter), count, comm_->comm,
      stream);

  ASSIGN_OR_RETURN(
      ncclDataType_t nccl_dtype,
      ToNcclDataType(
          dtype, /*is_reduction_op=*/false,
          stream->parent()->GetDeviceDescription().rocm_compute_capability()));

  // Short-circuit if there is no source or target rank.
  if (!source_rank && target_ranks.empty()) {
    return absl::OkStatus();
  }

  auto group = [&] {
    if (source_rank) {
      RETURN_IF_ERROR(WithLiveRcclComm(*comm_, [&](ncclComm_t comm) {
        return XLA_RCCL_STATUS(ncclRecv(
            recv_buffer.opaque(), ToNcclCount(dtype, count), nccl_dtype,
            source_rank->value(), comm, AsHipStream(stream)));
      }));
    }

    for (RankId target_rank : target_ranks) {
      RETURN_IF_ERROR(WithLiveRcclComm(*comm_, [&](ncclComm_t comm) {
        return XLA_RCCL_STATUS(ncclSend(
            send_buffer.opaque(), ToNcclCount(dtype, count), nccl_dtype,
            target_rank.value(), comm, AsHipStream(stream)));
      }));
    }

    return absl::OkStatus();
  };

  return GroupLaunch(group);
}

absl::Status RcclCommunicator::LaunchSend(se::DeviceAddressBase send_buffer,
                                          PrimitiveType dtype, size_t count,
                                          RankId peer,
                                          const Executor& executor) {
  if (cancel_->IsCancelled()) {
    return absl::FailedPreconditionError("RcclCommunicator aborted");
  }
  se::Stream* stream = ToStream(executor);

  VLOG(3) << absl::StreamFormat(
      "[%d] Launch RCCL Send operation; send_buffer=%p; dtype=%s; "
      "count=%d; peer=%d; comm=%p; stream=%p",
      stream->parent()->device_ordinal(), send_buffer.opaque(),
      primitive_util::LowercasePrimitiveTypeName(dtype), count, peer.value(),
      comm_->comm, stream);

  ASSIGN_OR_RETURN(
      ncclDataType_t nccl_dtype,
      ToNcclDataType(
          dtype, /*is_reduction_op=*/false,
          stream->parent()->GetDeviceDescription().rocm_compute_capability()));

  RETURN_IF_ERROR(WithLiveRcclComm(*comm_, [&](ncclComm_t comm) {
    return XLA_RCCL_STATUS(ncclSend(send_buffer.opaque(),
                                    ToNcclCount(dtype, count), nccl_dtype,
                                    peer.value(), comm, AsHipStream(stream)));
  }));
  if (!IsInsideRcclGroupLaunch()) {
    RETURN_IF_ERROR(PollUntilDone());
  }
  return absl::OkStatus();
}

absl::Status RcclCommunicator::LaunchRecv(se::DeviceAddressBase recv_buffer,
                                          PrimitiveType dtype, size_t count,
                                          RankId peer,
                                          const Executor& executor) {
  if (cancel_->IsCancelled()) {
    return absl::FailedPreconditionError("RcclCommunicator aborted");
  }
  se::Stream* stream = ToStream(executor);

  VLOG(3) << absl::StreamFormat(
      "[%d] Launch RCCL Recv operation; recv_buffer=%p; dtype=%s; "
      "count=%d; peer=%d; comm=%p; stream=%p",
      stream->parent()->device_ordinal(), recv_buffer.opaque(),
      primitive_util::LowercasePrimitiveTypeName(dtype), count, peer.value(),
      comm_->comm, stream);

  ASSIGN_OR_RETURN(
      ncclDataType_t nccl_dtype,
      ToNcclDataType(
          dtype, /*is_reduction_op=*/false,
          stream->parent()->GetDeviceDescription().rocm_compute_capability()));

  RETURN_IF_ERROR(WithLiveRcclComm(*comm_, [&](ncclComm_t comm) {
    return XLA_RCCL_STATUS(ncclRecv(recv_buffer.opaque(),
                                    ToNcclCount(dtype, count), nccl_dtype,
                                    peer.value(), comm, AsHipStream(stream)));
  }));
  if (!IsInsideRcclGroupLaunch()) {
    RETURN_IF_ERROR(PollUntilDone());
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<SymmetricMemory>>
RcclCommunicator::CreateSymmetricMemory(se::DeviceAddressBase addr) {
  ASSIGN_OR_RETURN(std::unique_ptr<SymmetricMemoryPlan> plan,
                   ResolveSymmetricMemoryPlan(addr));
  return CreateSymmetricMemory(*plan);
}

absl::StatusOr<std::unique_ptr<GpuCommunicator::SymmetricMemoryPlan>>
RcclCommunicator::ResolveSymmetricMemoryPlan(se::DeviceAddressBase addr) {
  return ExecuteAwait<std::unique_ptr<SymmetricMemoryPlan>>(
      [this, addr]() -> absl::StatusOr<std::unique_ptr<SymmetricMemoryPlan>> {
        if (cancel_->IsCancelled()) {
          return FailedPrecondition("RcclCommunicator aborted");
        }
        if (addr.opaque() == nullptr || addr.size() == 0) {
          return InvalidArgument(
              "RCCL symmetric memory requires a non-empty device address");
        }

        int runtime_version = 0;
        RETURN_IF_ERROR(XLA_RCCL_STATUS(ncclGetVersion(&runtime_version)));
        if (runtime_version != NCCL_VERSION_CODE) {
          return FailedPrecondition(
              "RCCL window ABI mismatch: compiled against version %d but "
              "loaded runtime version %d",
              NCCL_VERSION_CODE, runtime_version);
        }
        return std::make_unique<ResolvedRcclSymmetricMemoryPlan>(
            this, addr, runtime_version);
      });
}

absl::StatusOr<std::unique_ptr<SymmetricMemory>>
RcclCommunicator::CreateSymmetricMemory(const SymmetricMemoryPlan& plan) {
  return ExecuteAwait<std::unique_ptr<SymmetricMemory>>(
      [this, &plan]() -> absl::StatusOr<std::unique_ptr<SymmetricMemory>> {
        if (plan.owner() != this) {
          return InvalidArgument(
              "RCCL symmetric memory plan belongs to another communicator");
        }
        auto* rccl_plan =
            dynamic_cast<const ResolvedRcclSymmetricMemoryPlan*>(&plan);
        if (rccl_plan == nullptr) {
          return InvalidArgument("Expected a resolved RCCL window plan");
        }
        if (cancel_->IsCancelled()) {
          return FailedPrecondition("RcclCommunicator aborted");
        }

        int runtime_version = 0;
        RETURN_IF_ERROR(XLA_RCCL_STATUS(ncclGetVersion(&runtime_version)));
        if (runtime_version != rccl_plan->runtime_version()) {
          return FailedPrecondition(
              "RCCL runtime version changed after window plan resolution: "
              "%d vs %d",
              rccl_plan->runtime_version(), runtime_version);
        }
        return RcclSymmetricMemory::Create(comm_, plan.address(), executor_,
                                           stream_executor_);
      });
}

std::string RcclCommunicator::ToString() const {
  // comm_ should not be "touched" outside of executor_, but we are printing the
  // pointer itself and not touching the value, so this is safe.
  return absl::StrFormat("RcclCommunicator(ncclComm_t=%p)", comm_->comm);
}

absl::Status RcclCommunicator::PollUntilDone() const {
  if (cancel_->IsCancelled()) {
    return FailedPrecondition("RcclCommunicator aborted");
  }
  return ::xla::gpu::PollUntilDone(*comm_, *comm_->cancel);
}

Future<> RcclCommunicator::Execute(
    absl::AnyInvocable<absl::Status() &&> f) const {
  return executor_ ? MakeFutureOn<void>(*executor_, std::move(f))
                   : Future<>(std::move(f)());
}

template <typename T>
Future<T> RcclCommunicator::Execute(
    absl::AnyInvocable<absl::StatusOr<T>() &&> f) const {
  return executor_ ? MakeFutureOn<T>(*executor_, std::move(f))
                   : Future<T>(std::move(f)());
}

}  // namespace xla::gpu
