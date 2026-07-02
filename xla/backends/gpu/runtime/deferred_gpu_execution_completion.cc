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

#include "xla/backends/gpu/runtime/deferred_gpu_execution_completion.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/env.h"

namespace xla::gpu {
namespace {

using Cleanup = DeferredGpuExecutionCompletion::Cleanup;

struct CleanupQuarantine {
  absl::Mutex mutex;
  std::vector<Cleanup> cleanups ABSL_GUARDED_BY(mutex);
};

CleanupQuarantine* GetCleanupQuarantine() {
  // Intentionally leaked. Destructing the quarantine during process shutdown
  // could release resources while a GPU runtime is still using them.
  static auto* quarantine = new CleanupQuarantine;
  return quarantine;
}

void QuarantineCleanup(Cleanup cleanup, absl::Status reason) {
  LOG(ERROR) << "Quarantining deferred GPU execution cleanup for the process "
                "lifetime: "
             << reason;
  CleanupQuarantine* quarantine = GetCleanupQuarantine();
  absl::MutexLock lock(&quarantine->mutex);
  quarantine->cleanups.push_back(std::move(cleanup));
}

void DispatchCleanup(Cleanup cleanup) {
  tsl::Env::Default()->SchedClosure(
      [cleanup = std::move(cleanup)]() mutable { std::move(cleanup)(); });
}

}  // namespace

class DeferredGpuExecutionRegistry::State {
 public:
  void Acquire() {
    absl::MutexLock lock(&mutex_);
    ++leases_;
  }

  void Release() {
    absl::MutexLock lock(&mutex_);
    --leases_;
  }

  void WaitForAll() {
    absl::MutexLock lock(&mutex_);
    mutex_.Await(absl::Condition(this, &State::HasNoLeases));
  }

  bool WaitForAll(absl::Duration timeout) {
    absl::MutexLock lock(&mutex_);
    return mutex_.AwaitWithTimeout(absl::Condition(this, &State::HasNoLeases),
                                   timeout);
  }

  size_t outstanding() {
    absl::MutexLock lock(&mutex_);
    return leases_;
  }

 private:
  bool HasNoLeases() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    return leases_ == 0;
  }

  absl::Mutex mutex_;
  int64_t leases_ ABSL_GUARDED_BY(mutex_) = 0;
};

DeferredGpuExecutionRegistry::Lease::Lease(std::shared_ptr<State> state)
    : state_(std::move(state)) {}

DeferredGpuExecutionRegistry::Lease::~Lease() {
  if (state_ != nullptr) state_->Release();
}

DeferredGpuExecutionRegistry::Lease::Lease(Lease&& other) noexcept
    : state_(std::move(other.state_)) {}

DeferredGpuExecutionRegistry::DeferredGpuExecutionRegistry()
    : state_(std::make_shared<State>()) {}

DeferredGpuExecutionRegistry::~DeferredGpuExecutionRegistry() = default;

DeferredGpuExecutionRegistry::Lease DeferredGpuExecutionRegistry::Acquire() {
  state_->Acquire();
  return Lease(state_);
}

void DeferredGpuExecutionRegistry::WaitForAll() { state_->WaitForAll(); }

bool DeferredGpuExecutionRegistry::WaitForAll(absl::Duration timeout) {
  return state_->WaitForAll(timeout);
}

size_t DeferredGpuExecutionRegistry::outstanding() const {
  return state_->outstanding();
}

class DeferredGpuExecutionCompletion::State {
 public:
  enum class Completion { kPending, kSucceeded, kFailed };

  ~State() {
    std::optional<Cleanup> cleanup;
    {
      absl::MutexLock lock(&mutex_);
      cleanup = TakeCleanupLocked();
    }
    if (cleanup.has_value()) {
      QuarantineCleanup(
          std::move(*cleanup),
          absl::FailedPreconditionError(
              "completion state was destroyed before the stream tail ran"));
    }
  }

  absl::Status Arm(Cleanup cleanup) {
    if (!cleanup) {
      return absl::InvalidArgumentError("cleanup payload must not be empty");
    }

    Completion completion;
    absl::Status failure;
    std::optional<Cleanup> ready_cleanup;
    {
      absl::MutexLock lock(&mutex_);
      if (armed_) {
        return absl::FailedPreconditionError(
            "deferred GPU execution completion is already armed");
      }
      armed_ = true;
      cleanup_.emplace(std::move(cleanup));
      completion = completion_;
      failure = failure_;
      if (completion != Completion::kPending) {
        ready_cleanup = TakeCleanupLocked();
      }
    }

    if (ready_cleanup.has_value()) {
      if (completion == Completion::kSucceeded) {
        DispatchCleanup(std::move(*ready_cleanup));
      } else {
        QuarantineCleanup(std::move(*ready_cleanup), std::move(failure));
      }
    }
    return absl::OkStatus();
  }

  absl::Status BeginScheduling() {
    absl::MutexLock lock(&mutex_);
    if (scheduling_started_) {
      return absl::FailedPreconditionError(
          "deferred GPU execution completion is already scheduled");
    }
    scheduling_started_ = true;
    return absl::OkStatus();
  }

  absl::Status Quarantine(absl::Status reason) {
    std::optional<Cleanup> ready_cleanup;
    {
      absl::MutexLock lock(&mutex_);
      if (scheduling_started_ || completion_ != Completion::kPending) {
        return absl::FailedPreconditionError(
            "deferred GPU execution completion is already scheduled");
      }
      scheduling_started_ = true;
      completion_ = Completion::kFailed;
      failure_ = reason;
      if (armed_) ready_cleanup = TakeCleanupLocked();
    }

    if (ready_cleanup.has_value()) {
      QuarantineCleanup(std::move(*ready_cleanup), std::move(reason));
    }
    return absl::OkStatus();
  }

  void Complete(absl::Status status) {
    Completion completion =
        status.ok() ? Completion::kSucceeded : Completion::kFailed;
    std::optional<Cleanup> ready_cleanup;
    {
      absl::MutexLock lock(&mutex_);
      if (completion_ != Completion::kPending) return;
      completion_ = completion;
      failure_ = status;
      if (armed_) ready_cleanup = TakeCleanupLocked();
    }

    if (ready_cleanup.has_value()) {
      if (completion == Completion::kSucceeded) {
        DispatchCleanup(std::move(*ready_cleanup));
      } else {
        QuarantineCleanup(std::move(*ready_cleanup), std::move(status));
      }
    }
  }

 private:
  std::optional<Cleanup> TakeCleanupLocked()
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    if (!cleanup_.has_value()) return std::nullopt;
    std::optional<Cleanup> cleanup(std::move(cleanup_));
    cleanup_.reset();
    return cleanup;
  }

  absl::Mutex mutex_;
  bool armed_ ABSL_GUARDED_BY(mutex_) = false;
  bool scheduling_started_ ABSL_GUARDED_BY(mutex_) = false;
  Completion completion_ ABSL_GUARDED_BY(mutex_) = Completion::kPending;
  absl::Status failure_ ABSL_GUARDED_BY(mutex_);
  std::optional<Cleanup> cleanup_ ABSL_GUARDED_BY(mutex_);
};

DeferredGpuExecutionCompletion::DeferredGpuExecutionCompletion()
    : state_(std::make_shared<State>()) {}

DeferredGpuExecutionCompletion::~DeferredGpuExecutionCompletion() = default;

absl::Status DeferredGpuExecutionCompletion::Arm(Cleanup cleanup) {
  return state_->Arm(std::move(cleanup));
}

absl::Status DeferredGpuExecutionCompletion::Quarantine(absl::Status reason) {
  if (reason.ok()) {
    return absl::InvalidArgumentError(
        "quarantine reason must be a non-OK status");
  }
  return state_->Quarantine(std::move(reason));
}

absl::Status DeferredGpuExecutionCompletion::Schedule(
    stream_executor::Stream* main_stream,
    absl::Span<stream_executor::Stream* const> execution_streams,
    BeforeTail before_tail) {
  std::vector<StreamHandle> stream_handles;
  stream_handles.reserve(execution_streams.size());
  for (stream_executor::Stream* stream : execution_streams) {
    stream_handles.push_back(stream);
  }

  auto wait_for = [](StreamHandle main, StreamHandle other) {
    return static_cast<stream_executor::Stream*>(main)->WaitFor(
        static_cast<stream_executor::Stream*>(other));
  };
  BlockUntilDone block_until_done = [](StreamHandle main) {
    return static_cast<stream_executor::Stream*>(main)->BlockHostUntilDone();
  };
  auto enqueue = [](StreamHandle main, HostCallback callback,
                    ErrorCallback error_callback) {
    return static_cast<stream_executor::Stream*>(main)
        ->DoHostCallbackWithStatus(std::move(callback),
                                   std::move(error_callback));
  };
  return ScheduleImpl(main_stream, stream_handles, std::move(before_tail),
                      wait_for, std::move(block_until_done), enqueue);
}

absl::Status DeferredGpuExecutionCompletion::ScheduleImpl(
    StreamHandle main_stream, absl::Span<StreamHandle const> execution_streams,
    BeforeTail before_tail, WaitFor wait_for, BlockUntilDone block_until_done,
    Enqueue enqueue) {
  std::shared_ptr<State> state = state_;
  absl::Status status = state->BeginScheduling();
  if (!status.ok()) return status;

  if (main_stream == nullptr) {
    status = absl::InvalidArgumentError("main stream must not be null");
    state->Complete(status);
    return status;
  }

  absl::flat_hash_set<StreamHandle> joined_streams;
  for (StreamHandle stream : execution_streams) {
    if (stream == nullptr || stream == main_stream ||
        !joined_streams.insert(stream).second) {
      continue;
    }
    status = wait_for(main_stream, stream);
    if (!status.ok()) {
      state->Complete(status);
      return status;
    }
  }

  if (before_tail) {
    status = std::move(before_tail)();
    if (!status.ok()) {
      state->Complete(status);
      return status;
    }
  }

  HostCallback callback = [state, main_stream,
                           block_until_done =
                               std::move(block_until_done)]() mutable {
    // Do not complete the latch from inside the stream callback. Completion can
    // release the stream owner and its callback registry while the registry is
    // still returning from this callback. A separate worker synchronizes the
    // stream first; that cannot return until this callback has fully exited.
    tsl::Env::Default()->SchedClosure(
        [state, main_stream,
         block_until_done = std::move(block_until_done)]() mutable {
          state->Complete(std::move(block_until_done)(main_stream));
        });
    return absl::OkStatus();
  };
  ErrorCallback error_callback = [state](absl::Status status) mutable {
    if (status.ok()) {
      status = absl::InternalError(
          "stream tail callback reported failure without an error status");
    }
    // Completing inline is safe on the callback-registry error path because a
    // non-OK completion quarantines its cleanup instead of releasing any
    // stream or registry owner while this callback is still returning.
    state->Complete(std::move(status));
  };

  status = enqueue(main_stream, std::move(callback), std::move(error_callback));
  if (!status.ok()) state->Complete(status);
  return status;
}

DeferredGpuExecution::~DeferredGpuExecution() {
  if (!active() || armed_) return;
  absl::Status status = Arm();
  if (!status.ok()) {
    LOG(ERROR) << "Failed to arm deferred GPU execution during destruction: "
               << status;
  }
}

absl::Status DeferredGpuExecution::Defer(
    stream_executor::Stream* main_stream,
    absl::Span<stream_executor::Stream* const> execution_streams,
    Cleanup cleanup) {
  if (active()) {
    return absl::FailedPreconditionError(
        "deferred GPU execution is already active");
  }
  if (main_stream == nullptr) {
    return absl::InvalidArgumentError("main stream must not be null");
  }
  if (!cleanup) {
    return absl::InvalidArgumentError("cleanup payload must not be empty");
  }

  main_stream_ = main_stream;
  execution_streams_.assign(execution_streams.begin(), execution_streams.end());
  cleanups_.push_back(std::move(cleanup));
  completion_ = std::make_shared<DeferredGpuExecutionCompletion>();
  return absl::OkStatus();
}

absl::Status DeferredGpuExecution::AddCleanup(Cleanup cleanup) {
  if (!active()) {
    return absl::FailedPreconditionError(
        "deferred GPU execution is not active");
  }
  if (armed_) {
    return absl::FailedPreconditionError(
        "deferred GPU execution is already armed");
  }
  if (!cleanup) {
    return absl::InvalidArgumentError("cleanup payload must not be empty");
  }

  cleanups_.push_back(std::move(cleanup));
  return absl::OkStatus();
}

absl::Status DeferredGpuExecution::Schedule(BeforeTail before_tail) {
  if (!active()) {
    return absl::FailedPreconditionError(
        "deferred GPU execution is not active");
  }
  if (scheduled_) {
    return absl::FailedPreconditionError(
        "deferred GPU execution is already scheduled");
  }

  scheduled_ = true;
  return completion_->Schedule(main_stream_, execution_streams_,
                               std::move(before_tail));
}

absl::Status DeferredGpuExecution::Quarantine(absl::Status reason) {
  if (!active()) {
    return absl::FailedPreconditionError(
        "deferred GPU execution is not active");
  }
  if (scheduled_) {
    return absl::FailedPreconditionError(
        "deferred GPU execution is already scheduled");
  }
  if (reason.ok()) {
    return absl::InvalidArgumentError(
        "quarantine reason must be a non-OK status");
  }

  absl::Status status = completion_->Quarantine(std::move(reason));
  if (status.ok()) {
    scheduled_ = true;
    quarantined_ = true;
  }
  return status;
}

absl::Status DeferredGpuExecution::Arm() {
  if (!active()) {
    return absl::FailedPreconditionError(
        "deferred GPU execution is not active");
  }
  if (armed_) {
    return absl::FailedPreconditionError(
        "deferred GPU execution is already armed");
  }
  if (cleanups_.empty()) {
    return absl::InternalError(
        "deferred GPU execution has no cleanup payloads");
  }

  Cleanup cleanup = [cleanups = std::move(cleanups_)]() mutable {
    for (Cleanup& cleanup : cleanups) {
      std::move(cleanup)();
      cleanup = nullptr;
    }
    cleanups.clear();
  };
  armed_ = true;
  return completion_->Arm(std::move(cleanup));
}

}  // namespace xla::gpu
