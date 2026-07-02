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

#ifndef XLA_BACKENDS_GPU_RUNTIME_DEFERRED_GPU_EXECUTION_COMPLETION_H_
#define XLA_BACKENDS_GPU_RUNTIME_DEFERRED_GPU_EXECUTION_COMPLETION_H_

#include <cstddef>
#include <memory>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "absl/types/span.h"

namespace stream_executor {
class Stream;
}  // namespace stream_executor

namespace xla::gpu {

// Tracks deferred executions whose cleanup still depends on an owning runtime
// object (for example, a PjRt client that owns stream pools and allocators).
// The owner acquires one lease per deferred execution and waits for all leases
// before tearing down those dependencies.
class DeferredGpuExecutionRegistry {
 private:
  class State;

 public:
  class Lease {
   public:
    ~Lease();

    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    Lease(Lease&& other) noexcept;
    Lease& operator=(Lease&&) = delete;

   private:
    friend class DeferredGpuExecutionRegistry;
    explicit Lease(std::shared_ptr<State> state);

    std::shared_ptr<State> state_;
  };

  DeferredGpuExecutionRegistry();
  ~DeferredGpuExecutionRegistry();

  DeferredGpuExecutionRegistry(const DeferredGpuExecutionRegistry&) = delete;
  DeferredGpuExecutionRegistry& operator=(const DeferredGpuExecutionRegistry&) =
      delete;

  // Acquires a lease that must be retained through deferred cleanup.
  Lease Acquire();

  // Blocks until every acquired lease has been released. Callers must prevent
  // new acquisitions before invoking this during owner teardown.
  void WaitForAll();

  // Waits up to timeout and returns whether all leases were released.
  bool WaitForAll(absl::Duration timeout);

  size_t outstanding() const;

 private:
  std::shared_ptr<State> state_;
};

// Defers cleanup of resources that can still be referenced by asynchronous GPU
// work until every execution stream reaches a common tail.
//
// This class is a two-sided latch. The execution owner calls Arm() exactly once
// with a move-only cleanup payload, while Schedule() arranges for the stream
// side to signal completion. Either side may arrive first. The stream tail
// callback schedules a host worker that blocks until the main stream is fully
// quiescent (including callback exit), and only then signals completion. Once
// both sides arrive, the cleanup runs exactly once on a host worker scheduled
// by tsl::Env. It never runs in the device host callback.
//
// If stream joining, before-tail work, callback enqueueing, or asynchronous
// stream completion fails, the cleanup is intentionally retained for the
// lifetime of the process. Releasing potentially in-use GPU resources is less
// safe than leaking them.
//
// The object may be destroyed after Schedule() returns; callbacks retain the
// shared latch state. Arm() and Schedule() are thread safe, but each may be
// called at most once.
class DeferredGpuExecutionCompletion {
 public:
  using Cleanup = absl::AnyInvocable<void() &&>;
  using BeforeTail = absl::AnyInvocable<absl::Status() &&>;

  DeferredGpuExecutionCompletion();
  ~DeferredGpuExecutionCompletion();

  DeferredGpuExecutionCompletion(const DeferredGpuExecutionCompletion&) =
      delete;
  DeferredGpuExecutionCompletion& operator=(
      const DeferredGpuExecutionCompletion&) = delete;
  DeferredGpuExecutionCompletion(DeferredGpuExecutionCompletion&&) = delete;
  DeferredGpuExecutionCompletion& operator=(DeferredGpuExecutionCompletion&&) =
      delete;

  // Attaches the cleanup payload and arms this completion. Returns an error if
  // called more than once or with an empty payload.
  absl::Status Arm(Cleanup cleanup);

  // Nonblockingly joins every distinct, non-null stream other than main_stream
  // into main_stream, invokes before_tail if present, then appends a tail host
  // callback to main_stream. before_tail runs on the scheduling thread and can
  // enqueue work that must execute after all joins and before the tail
  // callback. main_stream must be non-null, and the cleanup payload must retain
  // its owner until cleanup begins (or forever when quarantined). A join,
  // before_tail, enqueue, or final stream synchronization error permanently
  // fails this completion and causes an attached (or subsequently attached)
  // cleanup to be quarantined.
  absl::Status Schedule(
      stream_executor::Stream* main_stream,
      absl::Span<stream_executor::Stream* const> execution_streams,
      BeforeTail before_tail = nullptr);

  // Permanently fails this completion and quarantines an attached (or later
  // attached) cleanup. This is used when no local stream tail can prove that
  // remote devices have stopped accessing the retained resources.
  absl::Status Quarantine(absl::Status reason);

 private:
  friend class DeferredGpuExecutionCompletionTestPeer;

  using StreamHandle = void*;
  using HostCallback = absl::AnyInvocable<absl::Status() &&>;
  using ErrorCallback = absl::AnyInvocable<void(absl::Status) &&>;
  using WaitFor = absl::FunctionRef<absl::Status(StreamHandle, StreamHandle)>;
  using BlockUntilDone = absl::AnyInvocable<absl::Status(StreamHandle) &&>;
  using Enqueue = absl::FunctionRef<absl::Status(StreamHandle, HostCallback,
                                                 ErrorCallback)>;

  class State;

  // Opaque handles and injected operations keep the scheduling state machine
  // testable without constructing GPU streams. Only the test peer can use this
  // seam; production callers use Schedule().
  absl::Status ScheduleImpl(StreamHandle main_stream,
                            absl::Span<StreamHandle const> execution_streams,
                            BeforeTail before_tail, WaitFor wait_for,
                            BlockUntilDone block_until_done, Enqueue enqueue);

  std::shared_ptr<State> state_;
};

// Execution-scoped orchestration for DeferredGpuExecutionCompletion.
//
// Defer() records raw execution streams and the first cleanup. AddCleanup()
// appends additional cleanups in execution order. Schedule() installs the
// stream tail, and Arm() atomically hands the ordered cleanup sequence to the
// shared completion latch. Schedule() and Arm() may be called in either order.
//
// This object is not thread safe. The stream callback may race with Arm(), but
// that synchronization is handled by DeferredGpuExecutionCompletion. If an
// active object is destroyed before Arm(), its destructor arms it so resources
// are either released after successful completion or quarantined.
class DeferredGpuExecution {
 public:
  using Cleanup = DeferredGpuExecutionCompletion::Cleanup;
  using BeforeTail = DeferredGpuExecutionCompletion::BeforeTail;

  DeferredGpuExecution() = default;
  ~DeferredGpuExecution();

  DeferredGpuExecution(const DeferredGpuExecution&) = delete;
  DeferredGpuExecution& operator=(const DeferredGpuExecution&) = delete;
  DeferredGpuExecution(DeferredGpuExecution&&) = delete;
  DeferredGpuExecution& operator=(DeferredGpuExecution&&) = delete;

  // Starts one deferred execution. Raw stream pointers must remain valid until
  // Schedule() returns. Defer() may be called at most once.
  absl::Status Defer(
      stream_executor::Stream* main_stream,
      absl::Span<stream_executor::Stream* const> execution_streams,
      Cleanup cleanup);

  // Appends a cleanup to the sequence. Cleanups run in insertion order. This
  // must be called after Defer() and before Arm().
  absl::Status AddCleanup(Cleanup cleanup);

  // Schedules the stream tail. May be called at most once.
  absl::Status Schedule(BeforeTail before_tail = nullptr);

  // Marks this execution unsafe to clean up. Its cleanup sequence is retained
  // for process lifetime after Arm().
  absl::Status Quarantine(absl::Status reason);

  // Hands the ordered cleanup sequence to the completion latch. May be called
  // at most once.
  absl::Status Arm();

  // Returns true after a successful Defer(). It remains true for the lifetime
  // of this one-shot orchestration object.
  bool active() const { return completion_ != nullptr; }
  bool quarantined() const { return quarantined_; }

  // Disables ordinary tail-based cleanup while still allowing callers to
  // quarantine a remote-communication failure. This is used when another
  // resource (for example, VMM aliases) requires synchronous teardown.
  void set_completion_cleanup_enabled(bool enabled) {
    completion_cleanup_enabled_ = enabled;
  }
  bool completion_cleanup_enabled() const {
    return completion_cleanup_enabled_;
  }

 private:
  friend class DeferredGpuExecutionCompletionTestPeer;

  stream_executor::Stream* main_stream_ = nullptr;
  std::vector<stream_executor::Stream*> execution_streams_;
  std::vector<Cleanup> cleanups_;
  std::shared_ptr<DeferredGpuExecutionCompletion> completion_;
  bool scheduled_ = false;
  bool armed_ = false;
  bool quarantined_ = false;
  bool completion_cleanup_enabled_ = true;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_DEFERRED_GPU_EXECUTION_COMPLETION_H_
