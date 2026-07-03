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

#include <atomic>
#include <memory>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "absl/types/span.h"

namespace xla::gpu {

TEST(DeferredGpuExecutionRegistryTest, WaitsForAllOutstandingLeases) {
  DeferredGpuExecutionRegistry registry;
  std::optional<DeferredGpuExecutionRegistry::Lease> first(registry.Acquire());
  std::optional<DeferredGpuExecutionRegistry::Lease> second(registry.Acquire());
  absl::Notification wait_started;
  absl::Notification wait_finished;

  std::thread waiter([&] {
    wait_started.Notify();
    registry.WaitForAll();
    wait_finished.Notify();
  });

  wait_started.WaitForNotification();
  first.reset();
  EXPECT_FALSE(
      wait_finished.WaitForNotificationWithTimeout(absl::Milliseconds(50)));
  second.reset();
  EXPECT_TRUE(wait_finished.WaitForNotificationWithTimeout(absl::Seconds(10)));
  waiter.join();
}

class FakeTailScheduler {
 public:
  using HostCallback = absl::AnyInvocable<absl::Status() &&>;
  using ErrorCallback = absl::AnyInvocable<void(absl::Status) &&>;
  using BlockUntilDone = absl::AnyInvocable<absl::Status(void*) &&>;

  FakeTailScheduler() : block_state_(std::make_shared<BlockState>()) {}

  absl::Status WaitFor(void* main_stream, void* other_stream) {
    main_streams_.push_back(main_stream);
    waited_streams_.push_back(other_stream);
    if (other_stream == failing_stream_) return wait_status_;
    return absl::OkStatus();
  }

  absl::Status Enqueue(void* main_stream, HostCallback callback,
                       ErrorCallback error_callback) {
    ++enqueue_count_;
    enqueue_stream_ = main_stream;
    if (!enqueue_status_.ok()) {
      if (invoke_error_callback_on_enqueue_failure_) {
        std::move(error_callback)(enqueue_status_);
        synchronous_error_callback_returned_.Notify();
        allow_failed_enqueue_return_.WaitForNotification();
      }
      return enqueue_status_;
    }
    callback_ = std::move(callback);
    error_callback_ = std::move(error_callback);
    return absl::OkStatus();
  }

  void FailWaitFor(void* stream, absl::Status status) {
    failing_stream_ = stream;
    wait_status_ = std::move(status);
  }

  void FailEnqueue(absl::Status status) { enqueue_status_ = std::move(status); }

  void FailEnqueueWithSynchronousErrorCallback(absl::Status status) {
    enqueue_status_ = std::move(status);
    invoke_error_callback_on_enqueue_failure_ = true;
  }

  bool WaitForSynchronousErrorCallback(absl::Duration timeout) {
    return synchronous_error_callback_returned_.WaitForNotificationWithTimeout(
        timeout);
  }

  void AllowFailedEnqueueReturn() { allow_failed_enqueue_return_.Notify(); }

  void FailBlockUntilDone(absl::Status status) {
    block_state_->status = std::move(status);
  }

  void PauseCallbackExit() { block_state_->pause_callback_exit = true; }

  void AllowCallbackExit() { block_state_->allow_callback_exit.Notify(); }

  void PauseBlockUntilDone() { block_state_->pause_block = true; }

  void AllowBlockUntilDone() { block_state_->allow_block.Notify(); }

  BlockUntilDone MakeBlockUntilDone() {
    return [state = block_state_](void* main_stream) mutable {
      state->stream.store(main_stream);
      ++state->calls;
      state->started.Notify();
      state->callback_exited.WaitForNotification();
      if (state->pause_block) {
        state->allow_block.WaitForNotification();
      }
      absl::Status status = state->status;
      state->finished.Notify();
      return status;
    };
  }

  absl::Status RunSuccessCallback() {
    if (!callback_) {
      return absl::FailedPreconditionError("success callback is not pending");
    }
    HostCallback callback = std::move(callback_);
    callback_ = nullptr;
    absl::Status status = std::move(callback)();
    block_state_->callback_body_returned.Notify();
    if (block_state_->pause_callback_exit) {
      block_state_->allow_callback_exit.WaitForNotification();
    }
    block_state_->callback_exited.Notify();
    return status;
  }

  absl::Status RunErrorCallback(absl::Status status) {
    if (!error_callback_) {
      return absl::FailedPreconditionError("error callback is not pending");
    }
    ErrorCallback error_callback = std::move(error_callback_);
    error_callback_ = nullptr;
    std::move(error_callback)(std::move(status));
    return absl::OkStatus();
  }

  const std::vector<void*>& main_streams() const { return main_streams_; }
  const std::vector<void*>& waited_streams() const { return waited_streams_; }
  int enqueue_count() const { return enqueue_count_; }
  void* enqueue_stream() const { return enqueue_stream_; }
  int block_count() const { return block_state_->calls.load(); }
  void* block_stream() const { return block_state_->stream.load(); }

  bool WaitForCallbackBodyReturned(absl::Duration timeout) {
    return block_state_->callback_body_returned.WaitForNotificationWithTimeout(
        timeout);
  }

  bool WaitForBlockStarted(absl::Duration timeout) {
    return block_state_->started.WaitForNotificationWithTimeout(timeout);
  }

  bool WaitForBlockFinished(absl::Duration timeout) {
    return block_state_->finished.WaitForNotificationWithTimeout(timeout);
  }

 private:
  struct BlockState {
    std::atomic<int> calls = 0;
    std::atomic<void*> stream = nullptr;
    absl::Status status = absl::OkStatus();
    bool pause_callback_exit = false;
    bool pause_block = false;
    absl::Notification callback_body_returned;
    absl::Notification allow_callback_exit;
    absl::Notification callback_exited;
    absl::Notification started;
    absl::Notification allow_block;
    absl::Notification finished;
  };

  std::vector<void*> main_streams_;
  std::vector<void*> waited_streams_;
  void* failing_stream_ = nullptr;
  absl::Status wait_status_ = absl::OkStatus();
  absl::Status enqueue_status_ = absl::OkStatus();
  int enqueue_count_ = 0;
  void* enqueue_stream_ = nullptr;
  HostCallback callback_;
  ErrorCallback error_callback_;
  std::shared_ptr<BlockState> block_state_;
  bool invoke_error_callback_on_enqueue_failure_ = false;
  absl::Notification synchronous_error_callback_returned_;
  absl::Notification allow_failed_enqueue_return_;
};

class DeferredGpuExecutionCompletionTestPeer {
 public:
  static absl::Status Schedule(
      DeferredGpuExecutionCompletion& completion, void* main_stream,
      absl::Span<void* const> execution_streams, FakeTailScheduler& scheduler,
      DeferredGpuExecutionCompletion::BeforeTail before_tail = nullptr) {
    auto wait_for = [&scheduler](void* main, void* other) {
      return scheduler.WaitFor(main, other);
    };
    auto enqueue = [&scheduler](
                       void* main,
                       DeferredGpuExecutionCompletion::HostCallback cb,
                       DeferredGpuExecutionCompletion::ErrorCallback error_cb) {
      return scheduler.Enqueue(main, std::move(cb), std::move(error_cb));
    };
    return completion.ScheduleImpl(main_stream, execution_streams,
                                   std::move(before_tail), wait_for,
                                   scheduler.MakeBlockUntilDone(), enqueue);
  }

  static absl::Status Schedule(
      DeferredGpuExecution& execution, FakeTailScheduler& scheduler,
      DeferredGpuExecution::BeforeTail before_tail = nullptr) {
    if (!execution.active()) {
      return absl::FailedPreconditionError(
          "deferred GPU execution is not active");
    }
    if (execution.scheduled_) {
      return absl::FailedPreconditionError(
          "deferred GPU execution is already scheduled");
    }

    execution.scheduled_ = true;
    std::vector<void*> streams;
    streams.reserve(execution.execution_streams_.size());
    for (stream_executor::Stream* stream : execution.execution_streams_) {
      streams.push_back(stream);
    }
    return Schedule(*execution.completion_, execution.main_stream_, streams,
                    scheduler, std::move(before_tail));
  }
};

namespace {

using absl_testing::StatusIs;

struct CleanupProbe {
  std::atomic<int> calls = 0;
  std::thread::id cleanup_thread;
  absl::Notification called;
};

DeferredGpuExecutionCompletion::Cleanup MakeCleanup(
    std::shared_ptr<CleanupProbe> probe) {
  return [probe = std::move(probe)]() mutable {
    ++probe->calls;
    probe->cleanup_thread = std::this_thread::get_id();
    probe->called.Notify();
  };
}

void ExpectCleanupCalled(const std::shared_ptr<CleanupProbe>& probe) {
  ASSERT_TRUE(probe->called.WaitForNotificationWithTimeout(absl::Seconds(10)));
  EXPECT_EQ(probe->calls.load(), 1);
}

TEST(DeferredGpuExecutionCompletionTest, CompletionBeforeArmRunsCleanup) {
  int main_stream;
  FakeTailScheduler scheduler;
  DeferredGpuExecutionCompletion completion;
  ASSERT_OK(DeferredGpuExecutionCompletionTestPeer::Schedule(
      completion, &main_stream, {}, scheduler));
  ASSERT_OK(scheduler.RunSuccessCallback());

  auto probe = std::make_shared<CleanupProbe>();
  ASSERT_OK(completion.Arm(MakeCleanup(probe)));
  ExpectCleanupCalled(probe);

  // A late error notification cannot dispatch the already-consumed cleanup.
  ASSERT_OK(scheduler.RunErrorCallback(absl::InternalError("late failure")));
  EXPECT_EQ(probe->calls.load(), 1);
}

TEST(DeferredGpuExecutionCompletionTest, ArmBeforeCompletionRunsCleanup) {
  int main_stream;
  FakeTailScheduler scheduler;
  DeferredGpuExecutionCompletion completion;
  auto probe = std::make_shared<CleanupProbe>();
  ASSERT_OK(completion.Arm(MakeCleanup(probe)));
  ASSERT_OK(DeferredGpuExecutionCompletionTestPeer::Schedule(
      completion, &main_stream, {}, scheduler));

  std::thread::id callback_thread = std::this_thread::get_id();
  ASSERT_OK(scheduler.RunSuccessCallback());
  ExpectCleanupCalled(probe);
  EXPECT_NE(probe->cleanup_thread, callback_thread);
}

TEST(DeferredGpuExecutionCompletionTest,
     CleanupWaitsForCallbackExitAndStreamQuiescence) {
  int main_stream;
  FakeTailScheduler scheduler;
  scheduler.PauseCallbackExit();
  scheduler.PauseBlockUntilDone();
  DeferredGpuExecutionCompletion completion;
  auto probe = std::make_shared<CleanupProbe>();
  ASSERT_OK(completion.Arm(MakeCleanup(probe)));
  ASSERT_OK(DeferredGpuExecutionCompletionTestPeer::Schedule(
      completion, &main_stream, {}, scheduler));

  absl::Status callback_status;
  std::thread callback_thread(
      [&] { callback_status = scheduler.RunSuccessCallback(); });

  EXPECT_TRUE(scheduler.WaitForCallbackBodyReturned(absl::Seconds(10)));
  EXPECT_TRUE(scheduler.WaitForBlockStarted(absl::Seconds(10)));
  EXPECT_EQ(scheduler.block_count(), 1);
  EXPECT_EQ(scheduler.block_stream(), &main_stream);
  EXPECT_FALSE(
      probe->called.WaitForNotificationWithTimeout(absl::Milliseconds(50)));

  // Let the host callback registry finish returning from the tail callback.
  // Cleanup must still wait for the worker-side stream synchronization.
  scheduler.AllowCallbackExit();
  callback_thread.join();
  ASSERT_OK(callback_status);
  EXPECT_FALSE(
      probe->called.WaitForNotificationWithTimeout(absl::Milliseconds(50)));

  scheduler.AllowBlockUntilDone();
  ExpectCleanupCalled(probe);
}

TEST(DeferredGpuExecutionCompletionTest,
     StreamSynchronizationFailureQuarantinesCleanup) {
  int main_stream;
  FakeTailScheduler scheduler;
  scheduler.FailBlockUntilDone(absl::InternalError("stream sync failed"));
  DeferredGpuExecutionCompletion completion;
  auto probe = std::make_shared<CleanupProbe>();
  std::weak_ptr<CleanupProbe> weak_probe = probe;
  ASSERT_OK(completion.Arm(MakeCleanup(probe)));
  probe.reset();
  ASSERT_OK(DeferredGpuExecutionCompletionTestPeer::Schedule(
      completion, &main_stream, {}, scheduler));

  ASSERT_OK(scheduler.RunSuccessCallback());
  ASSERT_TRUE(scheduler.WaitForBlockFinished(absl::Seconds(10)));
  ASSERT_FALSE(weak_probe.expired());
  EXPECT_FALSE(weak_probe.lock()->called.WaitForNotificationWithTimeout(
      absl::Milliseconds(50)));
  EXPECT_EQ(weak_probe.lock()->calls.load(), 0);
}

TEST(DeferredGpuExecutionCompletionTest,
     JoinsEachDistinctNonNullStreamBeforeEnqueue) {
  int main_stream;
  int stream_a;
  int stream_b;
  std::vector<void*> streams = {nullptr,      &stream_a, &stream_a,
                                &main_stream, &stream_b, nullptr};
  FakeTailScheduler scheduler;
  DeferredGpuExecutionCompletion completion;

  ASSERT_OK(DeferredGpuExecutionCompletionTestPeer::Schedule(
      completion, &main_stream, streams, scheduler));

  EXPECT_EQ(scheduler.main_streams(),
            (std::vector<void*>{&main_stream, &main_stream}));
  EXPECT_EQ(scheduler.waited_streams(),
            (std::vector<void*>{&stream_a, &stream_b}));
  EXPECT_EQ(scheduler.enqueue_count(), 1);
  EXPECT_EQ(scheduler.enqueue_stream(), &main_stream);

  auto probe = std::make_shared<CleanupProbe>();
  ASSERT_OK(completion.Arm(MakeCleanup(probe)));
  ASSERT_OK(scheduler.RunSuccessCallback());
  ExpectCleanupCalled(probe);
}

TEST(DeferredGpuExecutionCompletionTest,
     BeforeTailRunsAfterJoinsAndBeforeEnqueue) {
  int main_stream;
  int stream_a;
  int stream_b;
  std::vector<void*> streams = {&stream_a, &stream_b};
  FakeTailScheduler scheduler;
  DeferredGpuExecutionCompletion completion;
  bool before_tail_called = false;
  auto move_only_token = std::make_unique<int>(7);

  ASSERT_OK(DeferredGpuExecutionCompletionTestPeer::Schedule(
      completion, &main_stream, streams, scheduler,
      [&, token = std::move(move_only_token)]() -> absl::Status {
        EXPECT_EQ(scheduler.waited_streams(), streams);
        EXPECT_EQ(scheduler.enqueue_count(), 0);
        EXPECT_EQ(*token, 7);
        before_tail_called = true;
        return absl::OkStatus();
      }));

  EXPECT_TRUE(before_tail_called);
  EXPECT_EQ(scheduler.enqueue_count(), 1);
  auto probe = std::make_shared<CleanupProbe>();
  ASSERT_OK(completion.Arm(MakeCleanup(probe)));
  ASSERT_OK(scheduler.RunSuccessCallback());
  ExpectCleanupCalled(probe);
}

TEST(DeferredGpuExecutionCompletionTest,
     BeforeTailFailureQuarantinesCleanupWithoutEnqueue) {
  int main_stream;
  int stream;
  FakeTailScheduler scheduler;
  DeferredGpuExecutionCompletion completion;
  auto probe = std::make_shared<CleanupProbe>();
  std::weak_ptr<CleanupProbe> weak_probe = probe;
  ASSERT_OK(completion.Arm(MakeCleanup(probe)));
  probe.reset();

  EXPECT_THAT(
      DeferredGpuExecutionCompletionTestPeer::Schedule(
          completion, &main_stream, std::vector<void*>{&stream}, scheduler,
          [&]() -> absl::Status {
            EXPECT_EQ(scheduler.waited_streams(),
                      (std::vector<void*>{&stream}));
            return absl::InternalError("before-tail failed");
          }),
      StatusIs(absl::StatusCode::kInternal));
  EXPECT_EQ(scheduler.enqueue_count(), 0);
  ASSERT_FALSE(weak_probe.expired());
  EXPECT_EQ(weak_probe.lock()->calls.load(), 0);
}

TEST(DeferredGpuExecutionCompletionTest, WaitFailureQuarantinesCleanup) {
  int main_stream;
  int stream;
  FakeTailScheduler scheduler;
  scheduler.FailWaitFor(&stream, absl::InternalError("wait failed"));
  DeferredGpuExecutionCompletion completion;
  auto probe = std::make_shared<CleanupProbe>();
  std::weak_ptr<CleanupProbe> weak_probe = probe;
  ASSERT_OK(completion.Arm(MakeCleanup(probe)));
  probe.reset();

  EXPECT_THAT(
      DeferredGpuExecutionCompletionTestPeer::Schedule(
          completion, &main_stream, std::vector<void*>{&stream}, scheduler),
      StatusIs(absl::StatusCode::kInternal));
  EXPECT_EQ(scheduler.enqueue_count(), 0);
  ASSERT_FALSE(weak_probe.expired());
  EXPECT_EQ(weak_probe.lock()->calls.load(), 0);
}

TEST(DeferredGpuExecutionCompletionTest, EnqueueFailureQuarantinesCleanup) {
  int main_stream;
  FakeTailScheduler scheduler;
  scheduler.FailEnqueue(absl::InternalError("enqueue failed"));
  DeferredGpuExecutionCompletion completion;
  auto probe = std::make_shared<CleanupProbe>();
  std::weak_ptr<CleanupProbe> weak_probe = probe;
  ASSERT_OK(completion.Arm(MakeCleanup(probe)));
  probe.reset();

  EXPECT_THAT(DeferredGpuExecutionCompletionTestPeer::Schedule(
                  completion, &main_stream, {}, scheduler),
              StatusIs(absl::StatusCode::kInternal));
  ASSERT_FALSE(weak_probe.expired());
  EXPECT_EQ(weak_probe.lock()->calls.load(), 0);
}

TEST(DeferredGpuExecutionCompletionTest,
     SynchronousEnqueueErrorCallbackDoesNotReleaseOwners) {
  int main_stream;
  FakeTailScheduler scheduler;
  scheduler.FailEnqueueWithSynchronousErrorCallback(
      absl::InternalError("enqueue failed"));
  DeferredGpuExecutionCompletion completion;
  auto probe = std::make_shared<CleanupProbe>();
  std::weak_ptr<CleanupProbe> weak_probe = probe;
  ASSERT_OK(completion.Arm(MakeCleanup(probe)));
  probe.reset();

  absl::Status schedule_status;
  std::thread schedule_thread([&] {
    schedule_status = DeferredGpuExecutionCompletionTestPeer::Schedule(
        completion, &main_stream, {}, scheduler);
  });

  bool callback_returned =
      scheduler.WaitForSynchronousErrorCallback(absl::Seconds(10));
  EXPECT_TRUE(callback_returned);
  if (callback_returned) {
    std::shared_ptr<CleanupProbe> retained_probe = weak_probe.lock();
    EXPECT_NE(retained_probe, nullptr);
    if (retained_probe != nullptr) {
      EXPECT_EQ(retained_probe->calls.load(), 0);
    }
  }

  scheduler.AllowFailedEnqueueReturn();
  schedule_thread.join();
  EXPECT_TRUE(absl::IsInternal(schedule_status));
  ASSERT_FALSE(weak_probe.expired());
  EXPECT_EQ(weak_probe.lock()->calls.load(), 0);
}

TEST(DeferredGpuExecutionCompletionTest,
     RuntimeCallbackFailureBeforeArmQuarantinesCleanup) {
  int main_stream;
  FakeTailScheduler scheduler;
  DeferredGpuExecutionCompletion completion;
  ASSERT_OK(DeferredGpuExecutionCompletionTestPeer::Schedule(
      completion, &main_stream, {}, scheduler));
  ASSERT_OK(scheduler.RunErrorCallback(absl::InternalError("stream failed")));

  auto probe = std::make_shared<CleanupProbe>();
  std::weak_ptr<CleanupProbe> weak_probe = probe;
  ASSERT_OK(completion.Arm(MakeCleanup(probe)));
  probe.reset();

  ASSERT_FALSE(weak_probe.expired());
  EXPECT_EQ(weak_probe.lock()->calls.load(), 0);

  // Failure is terminal even if the success callback is spuriously delivered.
  ASSERT_OK(scheduler.RunSuccessCallback());
  ASSERT_TRUE(scheduler.WaitForBlockFinished(absl::Seconds(10)));
  EXPECT_EQ(weak_probe.lock()->calls.load(), 0);
}

TEST(DeferredGpuExecutionCompletionTest,
     ExplicitQuarantineBeforeArmRetainsCleanup) {
  DeferredGpuExecutionCompletion completion;
  ASSERT_OK(completion.Quarantine(
      absl::FailedPreconditionError("remote quiescence is unknown")));

  auto probe = std::make_shared<CleanupProbe>();
  std::weak_ptr<CleanupProbe> weak_probe = probe;
  ASSERT_OK(completion.Arm(MakeCleanup(probe)));
  probe.reset();

  ASSERT_FALSE(weak_probe.expired());
  EXPECT_EQ(weak_probe.lock()->calls.load(), 0);

  int main_stream;
  FakeTailScheduler scheduler;
  EXPECT_THAT(DeferredGpuExecutionCompletionTestPeer::Schedule(
                  completion, &main_stream, {}, scheduler),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(DeferredGpuExecutionCompletionTest, RejectsInvalidOrDuplicateCalls) {
  int main_stream;
  FakeTailScheduler scheduler;
  DeferredGpuExecutionCompletion completion;

  EXPECT_THAT(completion.Arm(nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));
  auto probe = std::make_shared<CleanupProbe>();
  ASSERT_OK(completion.Arm(MakeCleanup(probe)));
  EXPECT_THAT(completion.Arm([] {}),
              StatusIs(absl::StatusCode::kFailedPrecondition));

  ASSERT_OK(DeferredGpuExecutionCompletionTestPeer::Schedule(
      completion, &main_stream, {}, scheduler));
  EXPECT_THAT(DeferredGpuExecutionCompletionTestPeer::Schedule(
                  completion, &main_stream, {}, scheduler),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  ASSERT_OK(scheduler.RunSuccessCallback());
  ExpectCleanupCalled(probe);
}

TEST(DeferredGpuExecutionCompletionTest, NullMainStreamFailsAndQuarantines) {
  FakeTailScheduler scheduler;
  DeferredGpuExecutionCompletion completion;
  auto probe = std::make_shared<CleanupProbe>();
  std::weak_ptr<CleanupProbe> weak_probe = probe;
  ASSERT_OK(completion.Arm(MakeCleanup(probe)));
  probe.reset();

  EXPECT_THAT(DeferredGpuExecutionCompletionTestPeer::Schedule(
                  completion, nullptr, {}, scheduler),
              StatusIs(absl::StatusCode::kInvalidArgument));
  ASSERT_FALSE(weak_probe.expired());
  EXPECT_EQ(weak_probe.lock()->calls.load(), 0);
}

TEST(DeferredGpuExecutionTest, RunsCleanupsInInsertionOrder) {
  int main_stream_storage;
  auto* main_stream =
      reinterpret_cast<stream_executor::Stream*>(&main_stream_storage);
  std::vector<int> cleanup_order;
  absl::Notification cleanup_done;
  FakeTailScheduler scheduler;
  DeferredGpuExecution execution;
  bool before_tail_called = false;

  EXPECT_FALSE(execution.active());
  ASSERT_OK(
      execution.Defer(main_stream, {}, [&] { cleanup_order.push_back(1); }));
  ASSERT_TRUE(execution.active());
  ASSERT_OK(execution.AddCleanup([&] { cleanup_order.push_back(2); }));
  ASSERT_OK(execution.AddCleanup([&] {
    cleanup_order.push_back(3);
    cleanup_done.Notify();
  }));

  ASSERT_OK(DeferredGpuExecutionCompletionTestPeer::Schedule(
      execution, scheduler, [&]() -> absl::Status {
        before_tail_called = true;
        return absl::OkStatus();
      }));
  EXPECT_TRUE(before_tail_called);
  ASSERT_OK(execution.Arm());
  ASSERT_OK(scheduler.RunSuccessCallback());
  ASSERT_TRUE(cleanup_done.WaitForNotificationWithTimeout(absl::Seconds(10)));
  EXPECT_EQ(cleanup_order, (std::vector<int>{1, 2, 3}));
}

TEST(DeferredGpuExecutionTest, DestructorSafelyArmsUnscheduledCleanup) {
  int main_stream_storage;
  auto* main_stream =
      reinterpret_cast<stream_executor::Stream*>(&main_stream_storage);
  auto probe = std::make_shared<CleanupProbe>();
  std::weak_ptr<CleanupProbe> weak_probe = probe;
  {
    DeferredGpuExecution execution;
    ASSERT_OK(execution.Defer(main_stream, {}, MakeCleanup(probe)));
    probe.reset();
  }

  ASSERT_FALSE(weak_probe.expired());
  EXPECT_EQ(weak_probe.lock()->calls.load(), 0);
}

TEST(DeferredGpuExecutionTest, ExplicitQuarantineRetainsOrderedCleanup) {
  int main_stream_storage;
  auto* main_stream =
      reinterpret_cast<stream_executor::Stream*>(&main_stream_storage);
  auto probe = std::make_shared<CleanupProbe>();
  std::weak_ptr<CleanupProbe> weak_probe = probe;
  DeferredGpuExecution execution;
  ASSERT_OK(execution.Defer(main_stream, {}, MakeCleanup(probe)));
  ASSERT_OK(execution.Quarantine(
      absl::FailedPreconditionError("remote completion is unknown")));
  EXPECT_TRUE(execution.quarantined());
  probe.reset();
  ASSERT_OK(execution.Arm());

  ASSERT_FALSE(weak_probe.expired());
  EXPECT_EQ(weak_probe.lock()->calls.load(), 0);
}

TEST(DeferredGpuExecutionTest,
     QuarantineRetainsOwnersAttachedByCallerBeforeArm) {
  int main_stream_storage;
  auto* main_stream =
      reinterpret_cast<stream_executor::Stream*>(&main_stream_storage);
  auto internal_resource = std::make_shared<CleanupProbe>();
  auto outer_owner = std::make_shared<CleanupProbe>();
  std::weak_ptr<CleanupProbe> weak_internal = internal_resource;
  std::weak_ptr<CleanupProbe> weak_outer = outer_owner;

  DeferredGpuExecution execution;
  ASSERT_OK(execution.Defer(main_stream, {}, MakeCleanup(internal_resource)));
  ASSERT_OK(execution.Quarantine(absl::FailedPreconditionError(
      "remote resource initialization did not quiesce")));
  ASSERT_OK(execution.AddCleanup(MakeCleanup(outer_owner)));
  internal_resource.reset();
  outer_owner.reset();
  ASSERT_OK(execution.Arm());

  // Models ExecuteThunks retaining internal provider resources first and PJRT
  // attaching buffers, executable, and main-stream owners after it observes
  // an active quarantined execution.
  ASSERT_FALSE(weak_internal.expired());
  ASSERT_FALSE(weak_outer.expired());
  EXPECT_EQ(weak_internal.lock()->calls.load(), 0);
  EXPECT_EQ(weak_outer.lock()->calls.load(), 0);
}

TEST(DeferredGpuExecutionTest, RejectsInvalidLifecycleCalls) {
  int main_stream_storage;
  auto* main_stream =
      reinterpret_cast<stream_executor::Stream*>(&main_stream_storage);
  FakeTailScheduler scheduler;
  DeferredGpuExecution execution;

  EXPECT_THAT(execution.AddCleanup([] {}),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(execution.Schedule(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(execution.Arm(), StatusIs(absl::StatusCode::kFailedPrecondition));
  ASSERT_OK(execution.Defer(main_stream, {}, [] {}));
  EXPECT_THAT(execution.Defer(main_stream, {}, [] {}),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  ASSERT_OK(execution.Arm());
  EXPECT_THAT(execution.AddCleanup([] {}),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(execution.Arm(), StatusIs(absl::StatusCode::kFailedPrecondition));
  ASSERT_OK(
      DeferredGpuExecutionCompletionTestPeer::Schedule(execution, scheduler));
  ASSERT_OK(scheduler.RunSuccessCallback());
  ASSERT_TRUE(scheduler.WaitForBlockFinished(absl::Seconds(10)));
}

}  // namespace
}  // namespace xla::gpu
