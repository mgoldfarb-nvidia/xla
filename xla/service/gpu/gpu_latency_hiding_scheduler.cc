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

#include "xla/service/gpu/gpu_latency_hiding_scheduler.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/numbers.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/side_effect_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/latency_hiding_scheduler.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

namespace {

// A threshold for which we consider AR to be costly perf-wise.
static constexpr int64_t kCostlyAllReduceThreshold = 30 * 1024 * 1024;

// Multiplier which we apply to expand the base cost for the costly AR.
static constexpr int64_t kCostlyAllReduceMultiplier = 4;

// Multipliers for p2p collectives.
static constexpr int64_t kCostlyP2PSendMultiplier = 1024;
static constexpr int64_t kCostlyP2PCollectivePermuteMultiplier = 14;
static constexpr int64_t kCostlyP2PRecvMultiplier = 6;

// Number of P2P collectives that can be in flight at the same time. Note that
// this does not mean that these collectives run in parallel but synchronisation
// may not happen after each one of them.
static constexpr int64_t kNumAsyncCollectivesP2P = 4;

// Number of async memcpy operations that can be in flight at the same time.
static constexpr int64_t kNumAsyncMemcpy = std::numeric_limits<int>::max();
;

// Classifies `hlo` instruction as noop or not.
bool IsNopInstruction(const HloInstruction& hlo) {
  switch (hlo.opcode()) {
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kBitcast:
    case HloOpcode::kConstant:
    case HloOpcode::kParameter:
    case HloOpcode::kTuple:
    case HloOpcode::kPartitionId:
    case HloOpcode::kReplicaId:
    case HloOpcode::kOptimizationBarrier:
      return true;
    default:
      return hlo.IsEffectiveBitcast();
  }
}

bool IsAsyncComputeStartOp(const HloInstruction& hlo) {
  return hlo.opcode() == HloOpcode::kAsyncStart &&
         !hlo_query::IsCollectiveCommunicationOp(hlo.async_wrapped_opcode()) &&
         hlo.async_execution_thread() != hlo.parent()->execution_thread();
}

bool IsAsyncComputeDoneOp(const HloInstruction& hlo) {
  return hlo.opcode() == HloOpcode::kAsyncDone &&
         !hlo_query::IsCollectiveCommunicationOp(hlo.async_wrapped_opcode()) &&
         hlo.async_execution_thread() != hlo.parent()->execution_thread();
}

// Returns the pipeline stream for a P2P instruction recorded in a frontend
// attribute.
int64_t GetPipelineStream(const HloInstruction& start) {
  auto it = start.frontend_attributes().map().find(kSendRecvPipelineAttr);
  if (it != start.frontend_attributes().map().end() && it->second == "1") {
    return 1;
  }
  return 0;
}

// Returns the resource type and resource usage for a P2P instruction.
std::pair<GpuResourceType, ResourceUsageType> GetP2PResourceAndUsage(
    const HloInstruction& instr, const CanonicalAsyncOp& op) {
  ResourceUsageType usage = op.outer == HloOpcode::kAsyncStart
                                ? ResourceUsageType::kResourceRelease
                                : ResourceUsageType::kResourceOccupy;
  int64_t pipeline = GetPipelineStream(instr);
  HloOpcode opcode = op.inner;
  GpuResourceType resource;
  if (pipeline == 0) {
    resource = opcode == HloOpcode::kSend
                   ? GpuResourceType::kGpuAsyncStreamSend0
                   : GpuResourceType::kGpuAsyncStreamRecv0;
  } else {
    resource = opcode == HloOpcode::kSend
                   ? GpuResourceType::kGpuAsyncStreamSend1
                   : GpuResourceType::kGpuAsyncStreamRecv1;
  }

  return {resource, usage};
}

bool ShapeHasHostMemorySpace(const Shape& shape) {
  return shape.IsArray() && shape.has_layout() &&
         shape.layout().memory_space() ==
             static_cast<int64_t>(stream_executor::MemorySpace::kHost);
}

bool IsSlicingMemcpy(const HloInstruction& hlo) {
  if (hlo.opcode() == HloOpcode::kFusion) {
    return ShapeHasHostMemorySpace(hlo.shape()) ||
           ShapeHasHostMemorySpace(hlo.operand(0)->shape());
  }
  return false;
}

bool IsMemcpyAsyncStartOp(const HloInstruction& hlo) {
  if (hlo.opcode() == HloOpcode::kCopyStart) {
    return true;
  }
  if (hlo.opcode() != HloOpcode::kAsyncStart) {
    return false;
  }
  return IsSlicingMemcpy(*hlo.async_wrapped_instruction());
}

bool IsMemcpyAsyncDoneOp(const HloInstruction& hlo) {
  if (hlo.opcode() == HloOpcode::kCopyDone) {
    return true;
  }
  if (hlo.opcode() != HloOpcode::kAsyncDone) {
    return false;
  }
  return IsSlicingMemcpy(*hlo.async_wrapped_instruction());
}

// Marks async start operations to be scheduled as early as possible.
// It allows maximum overlap of operations while respecting dependencies.
// Besides async collectives, copy-start is async memcpy D2H/H2D, the beginning
// of a host offloading segment.
bool IsGpuAsyncStart(const HloInstruction& hlo) {
  return (hlo_query::IsAsyncCollectiveStartOp(&hlo,
                                              /*include_send_recv=*/true) &&
          !IsGPUSyncCollective(hlo)) ||
         IsAsyncComputeStartOp(hlo) || IsMemcpyAsyncStartOp(hlo);
}

// Marks async done operations to be scheduled as late as possible.
bool IsGpuAsyncDone(const HloInstruction& hlo) {
  return (hlo_query::IsAsyncCollectiveDoneOp(&hlo,
                                             /*include_send_recv=*/true) &&
          !IsGPUSyncCollective(*hlo.operand(0))) ||
         IsAsyncComputeDoneOp(hlo) || IsMemcpyAsyncDoneOp(hlo);
}

bool IsAsyncPair(const HloInstruction& from, const HloInstruction& target) {
  return IsGpuAsyncStart(from) && IsGpuAsyncDone(target);
}

// Count the maximum overlapping count in subgroups of group and other
size_t CountOverlappingRanks(const std::vector<std::vector<int64_t>>& group,
                             const std::vector<std::vector<int64_t>>& other) {
  size_t overlapping_count = 0;
  for (const std::vector<int64_t>& curr_replica_group : group) {
    absl::flat_hash_set<int64_t> curr_replica_ids;
    for (const int64_t curr_replica_id : curr_replica_group) {
      curr_replica_ids.insert(curr_replica_id);
    }

    for (const std::vector<int64_t>& replica_group : other) {
      size_t subgroup_count = 0;
      for (const int64_t replica_id : replica_group) {
        if (curr_replica_ids.contains(replica_id)) ++subgroup_count;
      }
      overlapping_count = std::max(overlapping_count, subgroup_count);
    }
  }
  return overlapping_count;
}

}  // namespace

HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction(
    int64_t pointer_size, std::optional<int64_t> memory_space) {
  return [pointer_size, memory_space](const Shape& shape) -> int64_t {
    // Filter by memory space if specified
    if (memory_space.has_value() && shape.has_layout() &&
        shape.layout().memory_space() != memory_space.value()) {
      return 0;
    }
    int64_t size = ShapeUtil::ByteSizeOf(shape, pointer_size);
    if (shape.IsTuple() || shape.is_static()) {
      return size;
    }
    // Each dynamic dimension size is represented as a S32.
    int64_t metadata_size = sizeof(int32_t) * shape.dimensions().size();
    return size + metadata_size;
  };
}

namespace {

// Proposal A2 (docs/lhs_pgle_baseline_improvements.md):
// When PGLE has no entry for a collective, estimate its latency from
// shape + bytes + replica-group size instead of falling back to flat
// kHighLatency=5000 cycles.
//
// Returns latency in microseconds (matches ApproximateLatencyEstimator's
// abstract unit, where CyclesPerMicrosecond() == 1).
double EstimateCollectiveLatencyFromShapeUs(
    const HloInstruction& instr, int64_t bytes,
    HloOpcode async_inner_opcode) {
  // Mirror of ApproximateLatencyEstimator::{kLowLatency, kHighLatency}
  // (protected members; can't be accessed directly from a free function).
  // Keep these in sync with the parent's values.
  constexpr double kFallbackLowLatencyUs = 1.0;
  constexpr double kFallbackHighLatencyUs = 5000.0;

  if (bytes <= 0) {
    return kFallbackHighLatencyUs;
  }

  // Conservative effective bandwidth (per-direction, GB/s). Tuned for
  // current targets where collectives may cross node boundaries via IB —
  // intra-node NVLink is faster but for an unprofiled fallback we err on
  // the side of "this is not free, schedule overlap." A future per-
  // architecture / per-replica-locality table is tracked under the same
  // proposal.
  constexpr double kEffectiveBandwidthGBps = 12.0;

  // Replica-group size determines the ring-step factor. We assume the
  // instruction's replica_groups are populated; if not (e.g. bare
  // collective without explicit groups), assume a moderate group size.
  int64_t rg_size = 8;
  const auto* coll = DynCast<HloCollectiveInstruction>(&instr);
  if (coll != nullptr && !coll->replica_groups().empty()) {
    rg_size = std::max<int64_t>(2, coll->replica_groups()[0].replica_ids_size());
  }
  double ring_factor = static_cast<double>(rg_size - 1) / rg_size;

  // All-reduce performs reduce-scatter + all-gather, so its bandwidth-time
  // factor is roughly 2x a single-direction collective.
  if (async_inner_opcode == HloOpcode::kAllReduce) {
    ring_factor *= 2.0;
  }

  // bytes / (GB/s * 1e9 bytes-per-GB) = seconds. * 1e6 = microseconds.
  double us =
      (static_cast<double>(bytes) * ring_factor) /
      (kEffectiveBandwidthGBps * 1.0e9) * 1.0e6;

  // Floor at kLowLatency so very small collectives don't return zero.
  // Cap at 10x kHighLatency so a runaway shape doesn't produce absurd
  // values.
  return std::clamp(us, kFallbackLowLatencyUs, 10.0 * kFallbackHighLatencyUs);
}

}  // namespace

namespace internal {
double EstimateCollectiveLatencyFromShapeUsForTesting(
    const HloInstruction& instr, int64_t bytes,
    HloOpcode async_inner_opcode) {
  return EstimateCollectiveLatencyFromShapeUs(instr, bytes,
                                               async_inner_opcode);
}
}  // namespace internal

CanonicalAsyncOp GpuGetCanonicalAsyncOp(const HloInstruction& hlo) {
  switch (hlo.opcode()) {
    case HloOpcode::kSend:
      return {HloOpcode::kAsyncStart, HloOpcode::kSend};
    case HloOpcode::kSendDone:
      return {HloOpcode::kAsyncDone, HloOpcode::kSend};
    case HloOpcode::kRecv:
      return {HloOpcode::kAsyncStart, HloOpcode::kRecv};
    case HloOpcode::kRecvDone:
      return {HloOpcode::kAsyncDone, HloOpcode::kRecv};
    default:
      return DefaultGetCanonicalAsyncOp(hlo);
  }
}

bool GpuScheduleCrossesOverlapLimit(
    const DefaultSchedulerCore::SchedulingState& sched_state,
    const HloGraphNode* node) {
  for (const auto& [resource, limit] : sched_state.max_concurrent_resource) {
    // No resources in flight of this kind. Continue.
    auto it = sched_state.resource_occupiers_in_flight.find(resource);
    if (it == sched_state.resource_occupiers_in_flight.end() ||
        it->second.empty()) {
      continue;
    }
    // Number of instances of 'resource' needed if this instruction was
    // to be scheduled.
    const int64_t num_resources_needed =
        sched_state.async_tracker->GetNumResourcesPerInstruction(
            resource, node->GetInstr());

    if (limit < num_resources_needed) {
      return true;
    }
  }

  if (node->GetResources().size() == 0) {
    return false;
  }
  auto resource_type = node->GetResources().at(0).first;
  // If the candidate collective has more than 1 overlapping ranks with
  // in-flight collectives, they can form cyclic dependency and cannot be
  // overlapped
  if (resource_type == xla::ResourceTypeToIndex(
                           GpuResourceType::kGpuAsyncStreamCollectives) &&
      sched_state.resource_occupiers_in_flight.contains(resource_type) &&
      !sched_state.resource_occupiers_in_flight.at(resource_type).empty()) {
    const HloInstruction& curr_hlo_inst = node->GetInstr();
    if (sched_state.async_tracker->IsSupportedAsyncDone(curr_hlo_inst)) {
      CHECK(
          hlo_query::IsAsyncCollectiveStartOp(curr_hlo_inst.operand(0), true));
      const HloInstruction* curr_start_inst =
          curr_hlo_inst.IsAsynchronous()
              ? curr_hlo_inst.operand(0)->async_wrapped_instruction()
              : curr_hlo_inst.operand(0);

      auto get_replica_groups = [](const HloInstruction* inst) {
        if (inst->opcode() != HloOpcode::kCollectivePermuteStart)
          return inst->device_list()->flattened_replica_groups();
        std::vector<std::vector<int64_t>> g;
        absl::c_transform(inst->source_target_pairs(), std::back_inserter(g),
                          [](auto& p) -> std::vector<int64_t> {
                            return {p.first, p.second};
                          });
        return g;
      };

      // If candidate can be overlapped with in-flight collectives
      bool can_overlap = true;
      for (const auto async_occupier :
           sched_state.resource_occupiers_in_flight.at(resource_type)) {
        if (sched_state.async_tracker->IsSupportedAsyncStart(*async_occupier)) {
          const HloInstruction* occupier =
              async_occupier->opcode() == HloOpcode::kAsyncStart
                  ? async_occupier->async_wrapped_instruction()
                  : async_occupier;

          size_t overlapping_count =
              CountOverlappingRanks(get_replica_groups(curr_start_inst),
                                    get_replica_groups(occupier));
          if (overlapping_count > 1) {
            can_overlap = false;
            VLOG(3) << "Collectives have " << overlapping_count
                    << "overlapping ranks and cannot be overlapped. Candidate "
                       "collective: "
                    << curr_start_inst->ToString()
                    << ", in flight collective: " << occupier->ToString();
            break;
          }
        }
      }
      if (!can_overlap) return true;
    }
  }

  return false;
}

//===----------------------------------------------------------------------===//
// GpuAsyncTrackerBase
//===----------------------------------------------------------------------===//
GpuAsyncTrackerBase::GpuAsyncTrackerBase(const SchedulerConfig& config,
                                         GetCanonicalAsyncOpFunc func)
    : AsyncTracker(config, func) {}

bool GpuAsyncTrackerBase::IsSupportedAsyncDone(
    const HloInstruction& hlo) const {
  return IsGpuAsyncDone(hlo);
}

bool GpuAsyncTrackerBase::IsSupportedAsyncStart(
    const HloInstruction& hlo) const {
  return IsGpuAsyncStart(hlo);
}

static bool IsPartiallyPipelinedSendRecvDone(const HloInstruction* instr) {
  // Is send-done/recv-done but does not have send/recv operand.
  return HloPredicateIsOp<HloOpcode::kSendDone, HloOpcode::kRecvDone>(instr) &&
         HloPredicateIsNotOp<HloOpcode::kSend, HloOpcode::kRecv>(
             instr->operand(0));
}

static bool IsPartiallyPipelinedSendRecv(const HloInstruction* instr) {
  // Is send/recv but does not feed into send-done/recv-done.
  return HloPredicateIsOp<HloOpcode::kSend, HloOpcode::kRecv>(instr) &&
         instr->user_count() == 1 &&
         HloPredicateIsNotOp<HloOpcode::kSendDone, HloOpcode::kRecvDone>(
             instr->users().front());
}

void GpuAsyncTrackerBase::PostProcessScheduleGraph(
    HloScheduleGraph* schedule_graph,
    const LatencyEstimator* latency_estimator) const {
  if (schedule_graph->GetOriginalInstrList().empty()) return;
  auto debug_options = schedule_graph->GetOriginalInstrList()
                           .front()
                           ->GetModule()
                           ->config()
                           .debug_options();

  for (const HloInstruction* inst : schedule_graph->GetOriginalInstrList()) {
    // Schedule partially pipelined send/recv instructions late so that they can
    // overlap with compute. Schedule send/recv late and, when unblocked,
    // schedule send-done/recv-done early.
    bool enable_pipeline_parallelism_opt =
        debug_options.xla_gpu_experimental_pipeline_parallelism_opt_level() !=
        DebugOptions::PIPELINE_PARALLELISM_OPT_LEVEL_DISABLE;
    if (enable_pipeline_parallelism_opt && IsPartiallyPipelinedSendRecv(inst)) {
      HloGraphNode& node = schedule_graph->GetNode(inst);
      node.SetForceDelay(true);
      VLOG(5) << "Setting force delay for instruction: " << inst->ToString();
    }
    if (enable_pipeline_parallelism_opt &&
        IsPartiallyPipelinedSendRecvDone(inst)) {
      HloGraphNode& node = schedule_graph->GetNode(inst);
      node.SetForceEarly(true);
      VLOG(5) << "Setting force early for instruction: " << inst->ToString();
    }

    // Force pipelined Recv to be closed to Recvdone so that copies inserted
    // for RecvDone can be eliminated.
    if (debug_options.xla_gpu_enable_pipelined_p2p() &&
        inst->opcode() == HloOpcode::kRecv &&
        inst->frontend_attributes().map().count(kSendRecvPipelineAttr) > 0) {
      HloGraphNode& node = schedule_graph->GetNode(inst);
      node.SetForceEarly(true);
      VLOG(5) << "Setting force early for instruction: " << inst->ToString();
    }

    if (inst->has_backend_config()) {
      auto gpu_config = inst->backend_config<GpuBackendConfig>();
      if (gpu_config.ok()) {
        HloGraphNode& node = schedule_graph->GetNode(inst);
        node.SetForceDelay(gpu_config->force_earliest_schedule());
        VLOG(5) << "Setting force delay for instruction: " << inst->ToString();
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// GpuAsyncTracker
//===----------------------------------------------------------------------===//
GpuAsyncTracker::GpuAsyncTracker(const SchedulerConfig& config)
    : GpuAsyncTrackerBase(config) {}

// Returns the per-stream resource for a collective with _xla_stream_annotation
// pointing to one of the collective multi-streams, or nullopt if not annotated
// or the annotation is out of range.
static std::optional<GpuResourceType> GetAnnotatedCollectiveStreamResource(
    const HloInstruction& instr) {
  const HloInstruction& start_instr =
      GpuGetCanonicalAsyncOp(instr).outer == HloOpcode::kAsyncDone
          ? *instr.operand(0)
          : instr;
  auto& attrs = start_instr.frontend_attributes().map();
  auto it = attrs.find(kXlaStreamAnnotationAttr);
  if (it == attrs.end()) return std::nullopt;
  int32_t stream_id;
  if (!absl::SimpleAtoi(it->second, &stream_id)) return std::nullopt;
  // Collective streams begin at kNumComputeStreams+1 (e.g. 5 and 6 when
  // kNumComputeStreams=4).
  int idx = stream_id - (static_cast<int>(kNumComputeStreams) + 1);
  if (idx == 0) return GpuResourceType::kGpuAsyncStreamCollectives0;
  if (idx == 1) return GpuResourceType::kGpuAsyncStreamCollectives1;
  return std::nullopt;
}

static bool IsAnnotatedForGpuAsyncStreamCollectivesP2P(
    const HloInstruction& instr) {
  const HloInstruction& start_instr =
      GpuGetCanonicalAsyncOp(instr).outer == HloOpcode::kAsyncDone
          ? *instr.operand(0)
          : instr;
  auto it =
      start_instr.frontend_attributes().map().find(kCollectiveStreamAttrName);
  if (it == start_instr.frontend_attributes().map().end()) return false;
  return it->second == kCollectiveStreamP2P;
}

// Hardcoded list of sync custom-call targets that internally use NCCL.
// LHS doesn't see their internal collective machinery, so it can pipeline
// XLA-emitted async collectives across them on different streams, leading
// to NCCL state / shared-buffer contention (Xid faults at high
// overlap_limit + multi_streaming=true). Listing the target name here
// causes GpuAsyncTracker to emit a kGpuAsyncStreamCollectives resource
// pair (occupy + release at the same scheduling step) for the custom
// call, which makes LHS's overlap-limit accounting count it against the
// global collective concurrency limit.
//
// Sync emission only — this is a temporary hypothesis-test mechanism
// pending the proper async-wrap pass + frontend-attribute label
// described in docs/lhs_pgle_baseline_improvements.md (follow-up work).
// If the test confirms the hypothesis, generalize to a frontend-attr
// driven mechanism so users can opt in their own custom calls without
// editing this list.
bool IsKnownCollectiveCustomCall(const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kCustomCall) return false;
  const absl::string_view target =
      Cast<HloCustomCallInstruction>(&instr)->custom_call_target();
  return target == "hybrid_ep_dispatch" || target == "hybrid_ep_combine";
}

ResourcesVector GpuAsyncTracker::GetResourcesFromInstructionImpl(
    const HloInstruction& instr) const {
  // Treat known sync collective custom calls as occupying the GPU
  // collective stream resource for one scheduling step. This blocks LHS
  // from co-scheduling them with another collective at the same dispatch
  // tick and forces them to serialize against in-flight async collectives
  // through the overlap-limit counter.
  if (IsKnownCollectiveCustomCall(instr)) {
    ResourcesVector r;
    r.push_back(std::make_pair(
        ResourceTypeToIndex(GpuResourceType::kGpuAsyncStreamCollectives),
        ResourceUsageType::kResourceOccupy));
    r.push_back(std::make_pair(
        ResourceTypeToIndex(GpuResourceType::kGpuAsyncStreamCollectives),
        ResourceUsageType::kResourceRelease));
    return r;
  }
  CanonicalAsyncOp op = GetCanonicalAsyncOp(instr);
  if (op.outer == HloOpcode::kAsyncStart || op.outer == HloOpcode::kAsyncDone) {
    ResourceUsageType usage;
    GpuResourceType resource;

    if (IsAnnotatedForGpuAsyncStreamCollectivesP2P(instr)) {
      resource = GpuResourceType::kGpuAsyncStreamCollectivesP2P;
      usage = op.outer == HloOpcode::kAsyncStart
                  ? ResourceUsageType::kResourceRelease
                  : ResourceUsageType::kResourceOccupy;
    } else if (op.inner == HloOpcode::kSend || op.inner == HloOpcode::kRecv) {
      std::tie(resource, usage) = GetP2PResourceAndUsage(instr, op);
    } else {
      usage = op.outer == HloOpcode::kAsyncStart
                  ? ResourceUsageType::kResourceRelease
                  : ResourceUsageType::kResourceOccupy;
      if (hlo_query::IsCollectiveCommunicationOp(op.inner)) {
        // If pinned to a specific collective multi-stream via
        // _xla_stream_annotation, use its dedicated resource so the scheduler
        // knows it can overlap with collectives on the other stream.
        if (auto per_stream = GetAnnotatedCollectiveStreamResource(instr)) {
          resource = *per_stream;
        } else {
          resource = GpuResourceType::kGpuAsyncStreamCollectives;
        }
      } else {
        resource = GpuResourceType::kGpuAsyncStreamComputes;
      }
    }
    return {std::make_pair(ResourceTypeToIndex(resource), usage)};
  }
  return GpuAsyncTrackerBase::GetResourcesFromInstructionImpl(instr);
}

int64_t GpuAsyncTracker::GetNumTargetDefinedResources() const {
  return ResourceTypeToIndex(GpuResourceType::kGpuResourceTypeEnd) -
         ResourceTypeToIndex(ResourceType::kTargetDefinedResourceTypeBegin);
};

// Returns how many instructions using the given resource_type we can overlap
int64_t GpuAsyncTracker::GetNumAvailableResources(int64_t resource_type) const {
  CHECK_LT(resource_type,
           ResourceTypeToIndex(GpuResourceType::kGpuResourceTypeEnd));
  if (resource_type < GetTargetDefinedResourceTypeBegin()) {
    return GpuAsyncTrackerBase::GetNumAvailableResources(resource_type);
  }

  // We will allow upto 1 outstanding collective on the async stream. This
  // controls the number of collectives in flight in the schedule (a
  // collective is in flight if the start is issued but not done). As an
  // example, with 1, LHS will generate the schedule: s0,e0,s1,e1, i.e., s1
  // is not scheduled until e0 is scheduled. With 2, the scheduler can
  // schedule s0,s1,e0,e1, because it assumes that the 2 instances of the
  // resources do not interfere with each other. If we do want to support > 1
  // async stream, we can increase this number and then do a post-pass on the
  // scheduled code to assign async stream-id to collectives (and actually
  // support > 1 async stream in the runtime).
  // The only case we'd allow 2 for now is when the current resource is
  // for an async computation operation which will be allocated with
  // a dedicated compute stream. It can run concurrently with
  // another collective.
  if (resource_type ==
      ResourceTypeToIndex(GpuResourceType::kGpuAsyncStreamComputes)) {
    return config_.parallel_async_compute_limit;
  }

  if (resource_type ==
      ResourceTypeToIndex(GpuResourceType::kGpuAsyncStreamCollectives)) {
    return config_.parallel_collective_overlap_limit;
  }

  if (resource_type ==
          ResourceTypeToIndex(GpuResourceType::kGpuAsyncStreamCollectives0) ||
      resource_type ==
          ResourceTypeToIndex(GpuResourceType::kGpuAsyncStreamCollectives1)) {
    return 1;
  }

  if (resource_type ==
          ResourceTypeToIndex(GpuResourceType::kGpuAsyncStreamCollectivesP2P) ||
      resource_type ==
          ResourceTypeToIndex(GpuResourceType::kGpuAsyncStreamSend0) ||
      resource_type ==
          ResourceTypeToIndex(GpuResourceType::kGpuAsyncStreamSend1) ||
      resource_type ==
          ResourceTypeToIndex(GpuResourceType::kGpuAsyncStreamRecv0) ||
      resource_type ==
          ResourceTypeToIndex(GpuResourceType::kGpuAsyncStreamRecv1)) {
    return kNumAsyncCollectivesP2P;
  }

  if (resource_type ==
      ResourceTypeToIndex(GpuResourceType::kGpuAsyncStreamMemcpy)) {
    constexpr int64_t kNumAsyncMemcpy = std::numeric_limits<int>::max();
    return kNumAsyncMemcpy;
  }

  return 1;
}

absl::string_view GpuAsyncTracker::GetResourceName(
    int64_t resource_type) const {
  CHECK_LT(resource_type,
           ResourceTypeToIndex(GpuResourceType::kGpuResourceTypeEnd));
  if (resource_type < GetTargetDefinedResourceTypeBegin()) {
    return GpuAsyncTrackerBase::GetResourceName(resource_type);
  }
  switch (static_cast<GpuResourceType>(resource_type)) {
    case GpuResourceType::kGpuAsyncStreamCollectivesP2P:
      return "kGpuAsyncStreamCollectivesP2P";
    case GpuResourceType::kGpuAsyncStreamSend0:
      return "kGpuAsyncStreamSend0";
    case GpuResourceType::kGpuAsyncStreamSend1:
      return "kGpuAsyncStreamSend1";
    case GpuResourceType::kGpuAsyncStreamRecv0:
      return "kGpuAsyncStreamRecv0";
    case GpuResourceType::kGpuAsyncStreamRecv1:
      return "kGpuAsyncStreamRecv1";
    case GpuResourceType::kGpuAsyncStreamCollectives:
      return "kGpuAsyncStreamCollectives";
    case GpuResourceType::kGpuAsyncStreamCollectives0:
      return "kGpuAsyncStreamCollectives0";
    case GpuResourceType::kGpuAsyncStreamCollectives1:
      return "kGpuAsyncStreamCollectives1";
    case GpuResourceType::kGpuAsyncStreamComputes:
      return "kGpuAsyncStreamComputes";
    default:
      return "kUnsupportedResource";
  }
}

ResourceHazardType GpuAsyncTracker::GetResourceHazardType(
    int64_t resource_type) const {
  CHECK_LT(resource_type,
           ResourceTypeToIndex(GpuResourceType::kGpuResourceTypeEnd));
  if (resource_type < GetTargetDefinedResourceTypeBegin()) {
    return GpuAsyncTrackerBase::GetResourceHazardType(resource_type);
  }
  return ResourceHazardType::kUnshareable;
}

int64_t GpuAsyncTracker::GetNumResourcesPerInstruction(
    int64_t resource_type, const HloInstruction& instr) const {
  CHECK_LT(resource_type,
           ResourceTypeToIndex(GpuResourceType::kGpuResourceTypeEnd));
  int64_t num_resources =
      GpuAsyncTrackerBase::GetNumResourcesPerInstruction(resource_type, instr);

  if (num_resources <= 0 || instr.opcode() != HloOpcode::kWhile) {
    return num_resources;
  }

  // For while-loop with pipelined Send/Recv, the while-body first releases
  // the Send/Recv resource and then uses the resource. Therefore, subtract 1
  // from num_resources for the relevant resource type.
  int64_t first_p2p_resource =
      ResourceTypeToIndex(GpuResourceType::kGpuAsyncStreamSend0);
  if (resource_type < first_p2p_resource ||
      resource_type > first_p2p_resource + kP2pResourceCount) {
    return num_resources;
  }
  auto find_instruction_for_pipeline = [&](HloOpcode opcode, int64_t pipeline) {
    for (auto user1 : instr.users()) {
      if (user1->opcode() == HloOpcode::kGetTupleElement) {
        for (auto user2 : user1->users()) {
          if (user2->opcode() == opcode) {
            if (GetPipelineStream(*user2) == pipeline) {
              return true;
            }
          }
        }
      }
    }
    return false;
  };
  bool found;
  // Look into the users of the while-result to find pipelined Send-done or
  // Recv-done.
  if (resource_type == first_p2p_resource) {
    found = find_instruction_for_pipeline(HloOpcode::kSendDone, 0);
  } else if (resource_type == first_p2p_resource + 1) {
    found = find_instruction_for_pipeline(HloOpcode::kSendDone, 1);
  } else if (resource_type == first_p2p_resource + 2) {
    found = find_instruction_for_pipeline(HloOpcode::kRecvDone, 0);
  } else {
    found = find_instruction_for_pipeline(HloOpcode::kRecvDone, 1);
  }
  return num_resources - (found ? 1 : 0);
}

//===----------------------------------------------------------------------===//
// GpuLatencyEstimator
//===----------------------------------------------------------------------===//
GpuLatencyEstimator::GpuLatencyEstimator(int64_t pointer_size,
                                         GetCanonicalAsyncOpFunc func)
    : ApproximateLatencyEstimator(func), pointer_size_(pointer_size) {}

ApproximateLatencyEstimator::TimeCost GpuLatencyEstimator::NodeCost(
    const HloInstruction* instr) const {
  if (IsNopInstruction(*instr)) {
    return 0.0;
  }
  // Consider cublas/cuddn/softmax custom calls as medium cost. Since the
  // latency between async-start and async-done is 5000 and cost of each
  // custom call is 1000, the LHS will try to schedule approximately 5 of
  // these in between each start/end pair.
  if (instr->opcode() == HloOpcode::kCustomCall) {
    if (const std::optional<TimeCost> latency =
            GetLatencyFromMetadata(*instr)) {
      return *latency;
    }
    if (IsCublasGemm(*instr) || IsCustomCallToDnnConvolution(*instr)) {
      return ApproximateLatencyEstimator::kMediumCost;
    }
    // consider other custom calls as medium cost for now. Keeping the case
    // explicitly separate for further tuning.
    return ApproximateLatencyEstimator::kMediumCost;
  }
  return ApproximateLatencyEstimator::NodeCost(instr);
}

ApproximateLatencyEstimator::TimeCost GpuLatencyEstimator::GetLatencyBetween(
    const HloGraphNode& from, const HloGraphNode& to) const {
  if (IsAsyncPair(from, to)) {
    if (const std::optional<TimeCost> latency =
            GetLatencyFromMetadata(from.GetInstr())) {
      return *latency;
    }
    if (IsAnnotatedForGpuAsyncStreamCollectivesP2P(from.GetInstr())) {
      HloOpcode inner_opcode = GpuGetCanonicalAsyncOp(from.GetInstr()).inner;
      if (inner_opcode == HloOpcode::kSend) {
        return kCostlyP2PSendMultiplier * kHighLatency;
      } else if (inner_opcode == HloOpcode::kCollectivePermute) {
        // The collective permutes in p2p communication force-synchronize all
        // devices and destroy the desired staggering. The latency we assign
        // them here must compensate for that. We give them the time they take
        // plus the maximum time any of them will have to wait for their
        // furthest peer.
        int64_t num_partitions =
            from.GetInstr().GetModule()->config().num_partitions();
        int64_t cycle_length = num_partitions / 8;
        int64_t staggering_factor = std::max<int64_t>(cycle_length - 1, 1);
        return staggering_factor * kCostlyP2PRecvMultiplier *
                   ApproximateLatencyEstimator::kHighLatency +
               kCostlyP2PCollectivePermuteMultiplier *
                   ApproximateLatencyEstimator::kHighLatency;
      } else {
        return kCostlyP2PRecvMultiplier *
               ApproximateLatencyEstimator::kHighLatency;
      }
    }

    bool enable_approx_collectives =
        from.GetInstr()
            .GetModule()
            ->config()
            .debug_options()
            .xla_gpu_enable_approx_costly_collectives();
    bool is_all_reduce = from.GetInstr().opcode() == HloOpcode::kAllReduceStart;
    bool collective_size_exceeds_threshold =
        ShapeSizeBytesFunction(pointer_size_)(from.GetInstr().shape()) >
        kCostlyAllReduceThreshold;
    if (enable_approx_collectives && is_all_reduce &&
        collective_size_exceeds_threshold) {
      return ApproximateLatencyEstimator::kHighLatency *
             kCostlyAllReduceMultiplier;
    }

    // Proposal A2: when enabled, estimate latency from shape rather than
    // returning the flat kHighLatency fallback. Helps when PGLE has no
    // entry for the collective (e.g. recently-added or renamed
    // instructions).
    if (from.GetInstr()
            .GetModule()
            ->config()
            .debug_options()
            .xla_gpu_lhs_unprofiled_latency_model()) {
      const HloInstruction& start = from.GetInstr();
      const HloInstruction* shape_carrier = &start;
      // For async-start wrappers, dig into the wrapped op for replica-
      // group + shape attribution (the wrapper's shape is a tuple; the
      // wrapped op has the proper collective shape).
      if (start.opcode() == HloOpcode::kAsyncStart) {
        shape_carrier = start.async_wrapped_instruction();
      }
      int64_t bytes =
          ShapeSizeBytesFunction(pointer_size_)(shape_carrier->shape());
      HloOpcode inner = GpuGetCanonicalAsyncOp(start).inner;
      return EstimateCollectiveLatencyFromShapeUs(*shape_carrier, bytes,
                                                  inner);
    }

    return ApproximateLatencyEstimator::kHighLatency;
  }
  // Every other instruction we consider synchronous, which means the
  // latency between each of them is always one unit.
  return ApproximateLatencyEstimator::kLowLatency;
}

//===----------------------------------------------------------------------===//
// GPUProfileStatisticsAggregator
//===----------------------------------------------------------------------===//

void GPUProfileStatisticsAggregator::HandleMissingInstructionCost(
    const HloInstruction& instruction) {
  if (!IsNopInstruction(instruction) &&
      HloPredicateIsNotOp<HloOpcode::kWhile>(&instruction) &&
      HloPredicateIsNotOp<HloOpcode::kCopy>(&instruction)) {
    missing_instructions_.insert(&instruction);
  }
}

void GPUProfileStatisticsAggregator::HandleFoundInstructionCost(
    const HloInstruction& instruction) {
  found_instructions_count_++;
}

void GPUProfileStatisticsAggregator::HandleMissingInstructionLatency(
    const HloInstruction& from, const HloInstruction& to) {
  if (IsAsyncPair(from, to)) {
    missing_instructions_.insert(&from);
  }
}

void GPUProfileStatisticsAggregator::HandleFoundInstructionLatency(
    const HloInstruction& from, const HloInstruction& to) {
  found_instructions_count_++;
}

}  // namespace gpu
}  // namespace xla
