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

#include <cstddef>
#include <cstring>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/barrier.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/collectives_test_util.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/collectives/nccl_symmetric_memory.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/core/collectives/registered_memory.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/future.h"
#include "xla/runtime/device_id.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

static constexpr GlobalDeviceId kD0(0);
static constexpr GlobalDeviceId kD1(1);

// Test checks that there are no deadlocks when NCCL API is used concurrently
// from multiple threads.
// Test launches two parallel threads which should use the same NCCL
// communicator asynchronously. The first thread should perform memory
// registration and the second thread should perform AllReduce.
// Memory registration operations should wait for the AllReduce to complete
// using a barrier.
TEST(NcclCommunicatorTest, AsyncApiCalls) {
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("CUDA"));

  constexpr int kNumDevices = 2;
  ASSERT_OK_AND_ASSIGN(std::vector<se::StreamExecutor*> executors,
                       CreateExecutors(platform, kNumDevices));

  if (!executors[0]->CanEnablePeerAccessTo(executors[1])) {
    GTEST_SKIP() << "Test requires peer access between devices";
  }

  if (!executors[0]
           ->GetDeviceDescription()
           .cuda_compute_capability()
           .IsAtLeastHopper()) {
    GTEST_SKIP() << "Test requires at least Hopper architecture";
  }

  ASSERT_OK_AND_ASSIGN(auto comms, CreateCommunicators(executors, {kD0, kD1},
                                                       /*blocking=*/true));

  ASSERT_OK_AND_ASSIGN(auto allocators, CreateMemoryAllocators(executors));
  ASSERT_OK_AND_ASSIGN(auto sym_allocs, Allocate(allocators, 1024));
  ASSERT_OK_AND_ASSIGN(auto send_allocs, Allocate(allocators, 1024));
  ASSERT_OK_AND_ASSIGN(auto recv_allocs, Allocate(allocators, 1024));

  ASSERT_OK_AND_ASSIGN(auto stream0, executors[0]->CreateStream());
  ASSERT_OK_AND_ASSIGN(auto stream1, executors[1]->CreateStream());

  constexpr int kNumThreads = 4;
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "nccl_test", kNumThreads);

  constexpr int kNumIterations = 1000;
  std::vector<std::unique_ptr<absl::Barrier>> registration_barriers;
  registration_barriers.reserve(kNumIterations);
  std::vector<std::unique_ptr<absl::Barrier>> all_reduce_barriers;
  all_reduce_barriers.reserve(kNumIterations);
  for (int i = 0; i < kNumIterations; ++i) {
    registration_barriers.push_back(
        std::make_unique<absl::Barrier>(kNumThreads));
    all_reduce_barriers.push_back(std::make_unique<absl::Barrier>(kNumThreads));
  }

  // Register memory, synchronize with barrier, unregister memory
  // asynchronously.
  std::function<absl::Status(int)> memory_registration_fn =
      [&](int rank) -> absl::Status {
    for (int i = 0; i < kNumIterations; ++i) {
      if (i > 0) {
        all_reduce_barriers[i - 1]->Block();
      }
      ASSIGN_OR_RETURN(auto sym_mem, comms[rank]->CreateSymmetricMemory(
                                         sym_allocs[rank]->address()));
      registration_barriers[i]->Block();
    }
    all_reduce_barriers[kNumIterations - 1]->Block();
    return absl::OkStatus();
  };

  // Perform AllReduce.
  std::function<absl::Status(int)> all_reduce_fn =
      [&](int rank) -> absl::Status {
    se::Stream* stream = rank == 0 ? stream0.get() : stream1.get();
    GpuCollectives::Executor gpu_exec(stream);
    for (int i = 0; i < kNumIterations; ++i) {
      registration_barriers[i]->Block();
      RETURN_IF_ERROR(comms[rank]
                          ->AllReduce(send_allocs[rank]->address(),
                                      recv_allocs[rank]->address(), F32, 256,
                                      ReductionKind::SUM, gpu_exec)
                          .Await());
      all_reduce_barriers[i]->Block();
    }
    return absl::OkStatus();
  };

  tsl::Executor& exec = *pool.AsExecutor();
  std::vector<Future<>> futures;
  futures.reserve(kNumThreads);
  futures.push_back(
      MakeFutureOn(exec, [&]() { return memory_registration_fn(0); }));
  futures.push_back(
      MakeFutureOn(exec, [&]() { return memory_registration_fn(1); }));
  futures.push_back(MakeFutureOn(exec, [&]() { return all_reduce_fn(0); }));
  futures.push_back(MakeFutureOn(exec, [&]() { return all_reduce_fn(1); }));

  for (auto& f : futures) {
    EXPECT_OK(f.Await());
  }
}

// Non-blocking NCCL communicators require every NCCL API call to execute on
// the communicator's owning thread and require host API output handles to be
// consumed only after ncclCommGetAsyncError reports completion.
TEST(NcclCommunicatorTest, NonBlockingDeviceResourceCreation) {
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("CUDA"));

  if (platform->VisibleDeviceCount() < 2) {
    GTEST_SKIP() << "Test requires at least 2 GPUs";
  }

  ASSERT_OK_AND_ASSIGN(std::vector<se::StreamExecutor*> executors,
                       CreateExecutors(platform, 2));

  if (!executors[0]->CanEnablePeerAccessTo(executors[1])) {
    GTEST_SKIP() << "Test requires peer access between devices";
  }

  if (!executors[0]
           ->GetDeviceDescription()
           .cuda_compute_capability()
           .IsAtLeastHopper()) {
    GTEST_SKIP() << "Test requires at least Hopper architecture";
  }

  ASSERT_OK_AND_ASSIGN(auto comms, CreateCommunicators(executors, {kD0, kD1},
                                                       /*blocking=*/false));
  ASSERT_OK_AND_ASSIGN(std::size_t num_ranks0, comms[0]->NumRanks());
  ASSERT_OK_AND_ASSIGN(std::size_t num_ranks1, comms[1]->NumRanks());
  EXPECT_EQ(num_ranks0, 2);
  EXPECT_EQ(num_ranks1, 2);

  if (!comms[0]->SupportsDeviceComm() || !comms[1]->SupportsDeviceComm()) {
    GTEST_SKIP() << "GPU communicators do not support device-initiated comms";
  }

  ASSERT_OK_AND_ASSIGN(auto allocators, CreateMemoryAllocators(executors));
  ASSERT_OK_AND_ASSIGN(auto sym_allocs, Allocate(allocators, 1024));
  ASSERT_OK_AND_ASSIGN(auto registered_allocs, Allocate(allocators, 1024));

  tsl::thread::ThreadPool pool(tsl::Env::Default(), "nccl_test", 2);
  tsl::Executor& exec = *pool.AsExecutor();

  auto symmetric_memory_futures =
      CreateSymmetricMemory(exec, comms, sym_allocs);
  ASSERT_OK_AND_ASSIGN(
      auto symmetric_memory,
      AwaitSymmetricMemory(std::move(symmetric_memory_futures)));
  EXPECT_EQ(symmetric_memory.size(), 2);
  for (const std::unique_ptr<SymmetricMemory>& memory : symmetric_memory) {
    auto* nccl_memory = dynamic_cast<NcclSymmetricMemory*>(memory.get());
    ASSERT_NE(nccl_memory, nullptr);

    auto packed = memory->PackKernelArg();
    ASSERT_EQ(packed.size_bytes(), sizeof(ncclWindow_t));
    ncclWindow_t packed_win;
    std::memcpy(&packed_win, packed.data(), sizeof(packed_win));
    EXPECT_EQ(packed_win, nccl_memory->win());
  }
  ASSERT_OK_AND_ASSIGN(se::DeviceAddressBase peer0,
                       symmetric_memory[0]->peer_addr(RankId(1)));
  ASSERT_OK_AND_ASSIGN(se::DeviceAddressBase peer1,
                       symmetric_memory[1]->peer_addr(RankId(0)));
  EXPECT_NE(peer0.opaque(), nullptr);
  EXPECT_NE(peer1.opaque(), nullptr);

  auto registered0 = MakeFutureOn(exec, [&] {
    return comms[0]->CreateRegisteredMemory(registered_allocs[0]->address());
  });
  auto registered1 = MakeFutureOn(exec, [&] {
    return comms[1]->CreateRegisteredMemory(registered_allocs[1]->address());
  });
  ASSERT_OK_AND_ASSIGN(auto registered_memory0, std::move(registered0).Await());
  ASSERT_OK_AND_ASSIGN(auto registered_memory1, std::move(registered1).Await());
  EXPECT_EQ(registered_memory0->addr().opaque(),
            registered_allocs[0]->address().opaque());
  EXPECT_EQ(registered_memory1->addr().opaque(),
            registered_allocs[1]->address().opaque());

  GpuDeviceCommunicator::Requirements requirements;
  requirements.local_barrier_count = 1;
  auto device_comm0 = MakeFutureOn(
      exec, [&] { return comms[0]->CreateDeviceComm(requirements); });
  auto device_comm1 = MakeFutureOn(
      exec, [&] { return comms[1]->CreateDeviceComm(requirements); });

  ASSERT_OK_AND_ASSIGN(auto dev_comm0, std::move(device_comm0).Await());
  ASSERT_OK_AND_ASSIGN(auto dev_comm1, std::move(device_comm1).Await());
  EXPECT_TRUE(dev_comm0->platform_comm().handle);
  EXPECT_TRUE(dev_comm1->platform_comm().handle);
}

}  // namespace
}  // namespace xla::gpu
