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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_RCCL_TYPES_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_RCCL_TYPES_H_

#include <memory>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "rocm/rocm_config.h"  // IWYU pragma: keep
#include "xla/backends/gpu/collectives/cancellation_token.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"

#if (TF_ROCM_VERSION >= 50200)
#include "rocm/include/rccl/rccl.h"
#else
#include "rocm/include/rccl.h"
#endif  // TF_ROCM_VERSION >= 50200

namespace xla::gpu {

// State shared by an RCCL communicator and provider resources derived from it.
// The abort bit remains valid after the parent object is destroyed and
// prevents child destructors from calling RCCL with an aborted communicator.
struct RcclCommState {
  explicit RcclCommState(ncclComm_t comm,
                         std::shared_ptr<CancellationToken> cancel = nullptr)
      : comm(comm),
        cancel(cancel != nullptr ? std::move(cancel)
                                 : std::make_shared<CancellationToken>()) {}

  ncclComm_t comm;
  bool aborted ABSL_GUARDED_BY(mutex) = false;
  bool destroyed ABSL_GUARDED_BY(mutex) = false;
  std::shared_ptr<CancellationToken> cancel;
  GpuProviderHostStorage host_storage;
  absl::Mutex mutex;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_RCCL_TYPES_H_
