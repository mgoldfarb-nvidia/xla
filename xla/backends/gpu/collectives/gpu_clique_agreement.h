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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_GPU_CLIQUE_AGREEMENT_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_GPU_CLIQUE_AGREEMENT_H_

#include <cstdint>
#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/executable_run_options.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/runtime/device_id.h"
#include "xla/runtime/process_id.h"

namespace xla::gpu {

// Configuration for a hierarchical GPU clique agreement.
struct GpuCliqueAgreementOptions {
  // Bounds process-local rendezvous and distributed key-value-store reads.
  // KeyValueStoreInterface::Set does not expose a timeout and therefore cannot
  // be bounded by this option.
  absl::Duration timeout = absl::Minutes(2);
};

// A request contributed by one local GPU rank to a clique-wide agreement.
//
// `run_id` is process-local and is used only to isolate local rendezvous. It is
// deliberately never serialized or used in a distributed key. `launch_id` is
// compared when a globally coordinated value is available; zero is valid for
// resource-initialization rounds because the clique session and monotonic
// generation already provide a globally shared namespace.
//
// `clique_session_id` must be a canonical, globally identical identifier for
// the lifetime of the underlying communicator session. It is the distributed
// key namespace, so a new communicator incarnation must use a new session id.
//
// `phase`, `logical_slot`, and `canonical_payload` are agreement data, not key
// components. This makes differently ordered operations collide at the same
// monotonically increasing session generation and fail with a useful error.
struct GpuCliqueAgreementRequest {
  GpuCliqueAgreementRequest(RunId run_id, int32_t launch_id,
                            std::string clique_session_id,
                            GpuCliqueKey clique_key, std::string phase,
                            int64_t logical_slot,
                            GlobalDeviceId global_device_id,
                            absl::Status local_status,
                            std::string canonical_payload);

  RunId run_id;
  int32_t launch_id;
  std::string clique_session_id;
  GpuCliqueKey clique_key;
  std::string phase;
  int64_t logical_slot;
  GlobalDeviceId global_device_id;
  absl::Status local_status;
  std::string canonical_payload;
};

// Implements hierarchical agreement for an operation associated with a GPU
// clique. All GPU ranks owned by this process first rendezvous in memory. One
// local leader then participates in a two-phase proposal/vote protocol through
// the shared key-value store. The exact process membership is derived from the
// clique devices and `device_to_process`.
//
// This class can represent one process owning many GPUs or one GPU. Different
// processes can own different numbers of participating GPUs. `Agree` must be
// called once by every local GPU rank in the request's clique, in the same
// per-session order. The return status is broadcast to all local callers.
class GpuCliqueAgreement {
 public:
  GpuCliqueAgreement(
      ProcessId process_id,
      absl::flat_hash_map<GlobalDeviceId, ProcessId> device_to_process,
      std::shared_ptr<KeyValueStoreInterface> kv_store,
      GpuCliqueAgreementOptions options = {});

  // Runs a single agreement round. A successful result means every rank
  // contributed an OK local status and byte-identical canonical payload, and
  // every participating process voted for the same proposal transcript.
  // Transport errors and timeouts have an uncertain distributed outcome and
  // must be treated as terminal for the clique session; callers must not retry
  // the same session.
  absl::Status Agree(const GpuCliqueAgreementRequest& request) const;

 private:
  struct Impl;
  std::shared_ptr<const Impl> impl_;
};

// Runs clique agreement through `agreement`. A null agreement is accepted only
// for a clique that declares every rank local; in that case this helper uses an
// in-process-only agreement. This fallback keeps direct local runtime users
// working without silently skipping required distributed coordination.
absl::Status AgreeGpuClique(const GpuCliqueAgreement* agreement,
                            const GpuCliqueAgreementRequest& request);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_GPU_CLIQUE_AGREEMENT_H_
