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

#include <cstddef>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"

namespace xla::gpu {

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

}  // namespace xla::gpu
