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

#include "xla/backends/gpu/collectives/gpu_clique_agreement.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "tsl/platform/fingerprint.h"

namespace xla::gpu {

GpuCliqueAgreementRequest::GpuCliqueAgreementRequest(
    RunId run_id, int32_t launch_id, std::string clique_session_id,
    GpuCliqueKey clique_key, std::string phase, int64_t logical_slot,
    GlobalDeviceId global_device_id, absl::Status local_status,
    std::string canonical_payload)
    : run_id(run_id),
      launch_id(launch_id),
      clique_session_id(std::move(clique_session_id)),
      clique_key(std::move(clique_key)),
      phase(std::move(phase)),
      logical_slot(logical_slot),
      global_device_id(global_device_id),
      local_status(std::move(local_status)),
      canonical_payload(std::move(canonical_payload)) {}

struct GpuCliqueAgreement::Impl {
  Impl(ProcessId process_id,
       absl::flat_hash_map<GlobalDeviceId, ProcessId> device_to_process,
       std::shared_ptr<KeyValueStoreInterface> kv_store,
       GpuCliqueAgreementOptions options)
      : process_id(process_id),
        device_to_process(std::move(device_to_process)),
        kv_store(std::move(kv_store)),
        options(options) {}

  ProcessId process_id;
  absl::flat_hash_map<GlobalDeviceId, ProcessId> device_to_process;
  std::shared_ptr<KeyValueStoreInterface> kv_store;
  GpuCliqueAgreementOptions options;
};

namespace {

constexpr uint32_t kProtocolVersion = 1;
constexpr absl::string_view kKeyPrefix = "xla_gpu_clique_agreement/v1";

class BinaryWriter {
 public:
  void AppendUint32(uint32_t value) {
    for (size_t i = 0; i < sizeof(value); ++i) {
      data_.push_back(static_cast<char>(value & 0xff));
      value >>= 8;
    }
  }

  void AppendInt32(int32_t value) {
    AppendUint32(static_cast<uint32_t>(value));
  }

  void AppendUint64(uint64_t value) {
    for (size_t i = 0; i < sizeof(value); ++i) {
      data_.push_back(static_cast<char>(value & 0xff));
      value >>= 8;
    }
  }

  void AppendInt64(int64_t value) {
    AppendUint64(static_cast<uint64_t>(value));
  }

  void AppendString(absl::string_view value) {
    AppendUint64(value.size());
    data_.append(value.data(), value.size());
  }

  std::string Take() { return std::move(data_); }

 private:
  std::string data_;
};

class BinaryReader {
 public:
  explicit BinaryReader(absl::string_view data) : data_(data) {}

  absl::StatusOr<uint32_t> ReadUint32() {
    if (data_.size() < sizeof(uint32_t)) {
      return absl::DataLossError("Truncated uint32 in agreement record");
    }
    uint32_t value = 0;
    for (size_t i = sizeof(value); i-- > 0;) {
      value <<= 8;
      value |= static_cast<unsigned char>(data_[i]);
    }
    data_.remove_prefix(sizeof(value));
    return value;
  }

  absl::StatusOr<int32_t> ReadInt32() {
    ASSIGN_OR_RETURN(uint32_t value, ReadUint32());
    return static_cast<int32_t>(value);
  }

  absl::StatusOr<uint64_t> ReadUint64() {
    if (data_.size() < sizeof(uint64_t)) {
      return absl::DataLossError("Truncated uint64 in agreement record");
    }
    uint64_t value = 0;
    for (size_t i = sizeof(value); i-- > 0;) {
      value <<= 8;
      value |= static_cast<unsigned char>(data_[i]);
    }
    data_.remove_prefix(sizeof(value));
    return value;
  }

  absl::StatusOr<int64_t> ReadInt64() {
    ASSIGN_OR_RETURN(uint64_t value, ReadUint64());
    return static_cast<int64_t>(value);
  }

  absl::StatusOr<std::string> ReadString() {
    ASSIGN_OR_RETURN(uint64_t size, ReadUint64());
    if (size > data_.size()) {
      return absl::DataLossError("Truncated string in agreement record");
    }
    std::string value(data_.substr(0, size));
    data_.remove_prefix(size);
    return value;
  }

  bool empty() const { return data_.empty(); }

 private:
  absl::string_view data_;
};

void AppendStatus(BinaryWriter& writer, const absl::Status& status) {
  writer.AppendInt32(static_cast<int32_t>(status.code()));
  writer.AppendString(status.message());
}

absl::Status ReadStatus(BinaryReader& reader, absl::Status* status) {
  ASSIGN_OR_RETURN(int32_t code, reader.ReadInt32());
  ASSIGN_OR_RETURN(std::string message, reader.ReadString());
  if (code < static_cast<int32_t>(absl::StatusCode::kOk) ||
      code > static_cast<int32_t>(absl::StatusCode::kUnauthenticated)) {
    return absl::DataLossError(
        absl::StrCat("Invalid status code in agreement record: ", code));
  }
  *status = absl::Status(static_cast<absl::StatusCode>(code), message);
  return absl::OkStatus();
}

bool StatusEquals(const absl::Status& lhs, const absl::Status& rhs) {
  return lhs.code() == rhs.code() && lhs.message() == rhs.message();
}

std::string CanonicalClique(const GpuCliqueKey& clique_key) {
  // num_local_participants is deliberately excluded: it is process-local and
  // can differ for uneven device ownership within the same global clique.
  BinaryWriter writer;
  writer.AppendUint64(clique_key.devices().size());
  for (GlobalDeviceId device : clique_key.devices()) {
    writer.AppendInt32(device.value());
  }
  writer.AppendUint64(clique_key.communication_id().value());
  writer.AppendUint64(clique_key.incarnations().size());
  for (IncarnationId incarnation : clique_key.incarnations()) {
    writer.AppendUint64(incarnation.value());
  }
  return writer.Take();
}

struct Issue {
  std::string sort_key;
  absl::Status status;
  std::string context;
};

absl::Status AggregateIssues(std::vector<Issue> issues) {
  if (issues.empty()) {
    return absl::OkStatus();
  }

  absl::c_sort(issues, [](const Issue& lhs, const Issue& rhs) {
    return lhs.sort_key < rhs.sort_key;
  });
  return absl::Status(
      issues.front().status.code(),
      absl::StrCat(
          "GPU clique agreement failed: ",
          absl::StrJoin(issues, "; ", [](std::string* out, const Issue& issue) {
            absl::StrAppend(out, issue.context, ": ", issue.status.message());
          })));
}

struct RankProposal {
  GlobalDeviceId device_id;
  absl::Status status;
  std::string canonical_payload;
};

struct ProcessProposal {
  ProcessId process_id;
  uint64_t generation;
  int32_t launch_id;
  std::string clique_session_id;
  std::string canonical_clique;
  std::string phase;
  int64_t logical_slot;
  absl::Status validation_status;
  std::vector<RankProposal> ranks;
};

std::string SerializeProposal(const ProcessProposal& proposal) {
  BinaryWriter writer;
  writer.AppendUint32(kProtocolVersion);
  writer.AppendInt32(proposal.process_id.value());
  writer.AppendUint64(proposal.generation);
  writer.AppendInt32(proposal.launch_id);
  writer.AppendString(proposal.clique_session_id);
  writer.AppendString(proposal.canonical_clique);
  writer.AppendString(proposal.phase);
  writer.AppendInt64(proposal.logical_slot);
  AppendStatus(writer, proposal.validation_status);
  writer.AppendUint64(proposal.ranks.size());
  for (const RankProposal& rank : proposal.ranks) {
    writer.AppendInt32(rank.device_id.value());
    AppendStatus(writer, rank.status);
    writer.AppendString(rank.canonical_payload);
  }
  return writer.Take();
}

absl::StatusOr<ProcessProposal> DeserializeProposal(absl::string_view data) {
  BinaryReader reader(data);
  ASSIGN_OR_RETURN(uint32_t version, reader.ReadUint32());
  if (version != kProtocolVersion) {
    return absl::DataLossError(absl::StrFormat(
        "Unsupported GPU clique agreement proposal version %d", version));
  }

  ASSIGN_OR_RETURN(int32_t process_id, reader.ReadInt32());
  ASSIGN_OR_RETURN(uint64_t generation, reader.ReadUint64());
  ASSIGN_OR_RETURN(int32_t launch_id, reader.ReadInt32());
  ASSIGN_OR_RETURN(std::string clique_session_id, reader.ReadString());
  ASSIGN_OR_RETURN(std::string canonical_clique, reader.ReadString());
  ASSIGN_OR_RETURN(std::string phase, reader.ReadString());
  ASSIGN_OR_RETURN(int64_t logical_slot, reader.ReadInt64());
  absl::Status validation_status;
  RETURN_IF_ERROR(ReadStatus(reader, &validation_status));
  ASSIGN_OR_RETURN(uint64_t num_ranks, reader.ReadUint64());
  if (num_ranks > std::numeric_limits<int32_t>::max()) {
    return absl::DataLossError("Too many ranks in agreement proposal");
  }

  std::vector<RankProposal> ranks;
  ranks.reserve(num_ranks);
  for (uint64_t i = 0; i < num_ranks; ++i) {
    ASSIGN_OR_RETURN(int32_t device_id, reader.ReadInt32());
    absl::Status status;
    RETURN_IF_ERROR(ReadStatus(reader, &status));
    ASSIGN_OR_RETURN(std::string canonical_payload, reader.ReadString());
    ranks.push_back({GlobalDeviceId(device_id), std::move(status),
                     std::move(canonical_payload)});
  }
  if (!reader.empty()) {
    return absl::DataLossError("Trailing bytes in agreement proposal");
  }

  return ProcessProposal{ProcessId(process_id),
                         generation,
                         launch_id,
                         std::move(clique_session_id),
                         std::move(canonical_clique),
                         std::move(phase),
                         logical_slot,
                         std::move(validation_status),
                         std::move(ranks)};
}

struct ProcessVote {
  ProcessId process_id;
  uint64_t generation;
  tsl::Fprint128 transcript_fingerprint;
  absl::Status decision;
};

std::string SerializeVote(const ProcessVote& vote) {
  BinaryWriter writer;
  writer.AppendUint32(kProtocolVersion);
  writer.AppendInt32(vote.process_id.value());
  writer.AppendUint64(vote.generation);
  writer.AppendUint64(vote.transcript_fingerprint.low64);
  writer.AppendUint64(vote.transcript_fingerprint.high64);
  AppendStatus(writer, vote.decision);
  return writer.Take();
}

absl::StatusOr<ProcessVote> DeserializeVote(absl::string_view data) {
  BinaryReader reader(data);
  ASSIGN_OR_RETURN(uint32_t version, reader.ReadUint32());
  if (version != kProtocolVersion) {
    return absl::DataLossError(absl::StrFormat(
        "Unsupported GPU clique agreement vote version %d", version));
  }

  ASSIGN_OR_RETURN(int32_t process_id, reader.ReadInt32());
  ASSIGN_OR_RETURN(uint64_t generation, reader.ReadUint64());
  ASSIGN_OR_RETURN(uint64_t transcript_fingerprint_low, reader.ReadUint64());
  ASSIGN_OR_RETURN(uint64_t transcript_fingerprint_high, reader.ReadUint64());
  absl::Status decision;
  RETURN_IF_ERROR(ReadStatus(reader, &decision));
  if (!reader.empty()) {
    return absl::DataLossError("Trailing bytes in agreement vote");
  }
  return ProcessVote{ProcessId(process_id),
                     generation,
                     {transcript_fingerprint_low, transcript_fingerprint_high},
                     std::move(decision)};
}

struct Membership {
  std::vector<ProcessId> processes;
  absl::btree_map<ProcessId, std::vector<GlobalDeviceId>> devices;
};

absl::StatusOr<Membership> DeriveMembership(
    const GpuCliqueKey& clique_key,
    const absl::flat_hash_map<GlobalDeviceId, ProcessId>& device_to_process) {
  Membership membership;
  absl::btree_set<GlobalDeviceId> seen_devices;
  for (GlobalDeviceId device : clique_key.devices()) {
    if (!seen_devices.insert(device).second) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Clique contains duplicate global device %d", device.value()));
    }
    auto process = device_to_process.find(device);
    if (process == device_to_process.end()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "No process mapping for global device %d", device.value()));
    }
    membership.devices[process->second].push_back(device);
  }
  if (membership.devices.empty()) {
    return absl::InvalidArgumentError("GPU clique must not be empty");
  }

  membership.processes.reserve(membership.devices.size());
  for (auto& [process, devices] : membership.devices) {
    absl::c_sort(devices, [](GlobalDeviceId lhs, GlobalDeviceId rhs) {
      return lhs.value() < rhs.value();
    });
    membership.processes.push_back(process);
  }
  return membership;
}

struct LocalValue {
  GpuCliqueAgreementRequest request;
  absl::Time deadline;
};

struct LocalRoundKey {
  ProcessId process_id;
  int64_t run_id;
  std::string clique_session_id;

  bool operator==(const LocalRoundKey& other) const {
    return process_id == other.process_id && run_id == other.run_id &&
           clique_session_id == other.clique_session_id;
  }

  template <typename H>
  friend H AbslHashValue(H hash, const LocalRoundKey& key) {
    return H::combine(std::move(hash), key.process_id, key.run_id,
                      key.clique_session_id);
  }
};

struct GenerationKey {
  ProcessId process_id;
  std::string clique_session_id;

  bool operator==(const GenerationKey& other) const {
    return process_id == other.process_id &&
           clique_session_id == other.clique_session_id;
  }

  template <typename H>
  friend H AbslHashValue(H hash, const GenerationKey& key) {
    return H::combine(std::move(hash), key.process_id, key.clique_session_id);
  }
};

struct LocalRound {
  explicit LocalRound(size_t expected_participants)
      : expected_participants(expected_participants) {}

  size_t expected_participants;
  absl::btree_map<GlobalDeviceId, LocalValue> values;
  bool leader_running = false;
  bool ready = false;
  absl::Status result;
  absl::CondVar cv;
};

// The rendezvous and generation state intentionally has process lifetime. A
// timed-out local round remains as a tombstone so a late rank observes the same
// failure instead of joining a new round with the same sequence number.
struct ProcessAgreementState {
  absl::Mutex mutex;
  absl::flat_hash_map<GenerationKey, uint64_t> next_generation
      ABSL_GUARDED_BY(mutex);
  absl::flat_hash_map<LocalRoundKey, std::shared_ptr<LocalRound>> local_rounds
      ABSL_GUARDED_BY(mutex);
};

ProcessAgreementState& GetProcessAgreementState() {
  static ProcessAgreementState& state = *new ProcessAgreementState;
  return state;
}

absl::StatusOr<uint64_t> NextGeneration(ProcessId process_id,
                                        absl::string_view session_id) {
  ProcessAgreementState& state = GetProcessAgreementState();
  absl::MutexLock lock(state.mutex);
  uint64_t& next =
      state.next_generation[GenerationKey{process_id, std::string(session_id)}];
  if (next == std::numeric_limits<uint64_t>::max()) {
    return absl::ResourceExhaustedError(
        "GPU clique agreement generation overflow");
  }
  return next++;
}

struct LocalProposal {
  ProcessProposal proposal;
  Membership membership;
};

absl::StatusOr<LocalProposal> BuildLocalProposal(
    ProcessId process_id,
    const absl::flat_hash_map<GlobalDeviceId, ProcessId>& device_to_process,
    uint64_t generation, absl::Span<const LocalValue> values) {
  if (values.empty()) {
    return absl::InternalError("Local agreement has no participants");
  }

  const GpuCliqueAgreementRequest& baseline = values.front().request;
  ASSIGN_OR_RETURN(Membership membership,
                   DeriveMembership(baseline.clique_key, device_to_process));
  auto local_devices = membership.devices.find(process_id);
  if (local_devices == membership.devices.end()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Process %d does not own a device in the GPU clique",
                        process_id.value()));
  }

  std::vector<Issue> issues;
  if (values.size() != local_devices->second.size()) {
    issues.push_back(
        {"00/count",
         absl::InvalidArgumentError(
             absl::StrFormat("expected %d local ranks but received %d",
                             local_devices->second.size(), values.size())),
         absl::StrFormat("process %d local rank count", process_id.value())});
  }

  std::vector<GlobalDeviceId> actual_devices;
  actual_devices.reserve(values.size());
  std::string baseline_clique = CanonicalClique(baseline.clique_key);
  for (const LocalValue& value : values) {
    const GpuCliqueAgreementRequest& request = value.request;
    actual_devices.push_back(request.global_device_id);
    std::string device_key =
        absl::StrFormat("%010d", request.global_device_id.value());

    auto owner = device_to_process.find(request.global_device_id);
    if (owner == device_to_process.end() || owner->second != process_id) {
      issues.push_back(
          {absl::StrCat("01/owner/", device_key),
           absl::InvalidArgumentError("rank is not owned by this process"),
           absl::StrFormat("global device %d",
                           request.global_device_id.value())});
    }
    if (!request.clique_key.rank(request.global_device_id).has_value()) {
      issues.push_back(
          {absl::StrCat("01/clique/", device_key),
           absl::InvalidArgumentError("rank is not a member of the clique"),
           absl::StrFormat("global device %d",
                           request.global_device_id.value())});
    }

    absl::StatusOr<Membership> rank_membership =
        DeriveMembership(request.clique_key, device_to_process);
    if (!rank_membership.ok()) {
      issues.push_back({absl::StrCat("02/topology/", device_key),
                        rank_membership.status(),
                        absl::StrFormat("global device %d topology",
                                        request.global_device_id.value())});
    } else {
      auto rank_local_devices = rank_membership->devices.find(process_id);
      size_t expected = rank_local_devices == rank_membership->devices.end()
                            ? 0
                            : rank_local_devices->second.size();
      if (request.clique_key.num_local_participants() !=
          static_cast<int64_t>(expected)) {
        issues.push_back(
            {absl::StrCat("02/local_count/", device_key),
             absl::InvalidArgumentError(absl::StrFormat(
                 "clique key declares %d local participants; topology has %d",
                 request.clique_key.num_local_participants(), expected)),
             absl::StrFormat("global device %d",
                             request.global_device_id.value())});
      }
    }

    if (request.run_id.ToInt() != baseline.run_id.ToInt()) {
      issues.push_back(
          {absl::StrCat("03/run/", device_key),
           absl::InvalidArgumentError("local RunId differs across ranks"),
           absl::StrFormat("global device %d",
                           request.global_device_id.value())});
    }
    if (request.clique_session_id != baseline.clique_session_id) {
      issues.push_back(
          {absl::StrCat("03/session/", device_key),
           absl::InvalidArgumentError("clique session id differs across ranks"),
           absl::StrFormat("global device %d",
                           request.global_device_id.value())});
    }
    if (request.launch_id != baseline.launch_id) {
      issues.push_back(
          {absl::StrCat("03/launch/", device_key),
           absl::InvalidArgumentError("launch id differs across local ranks"),
           absl::StrFormat("global device %d",
                           request.global_device_id.value())});
    }
    if (CanonicalClique(request.clique_key) != baseline_clique) {
      issues.push_back(
          {absl::StrCat("03/clique/", device_key),
           absl::InvalidArgumentError("clique differs across local ranks"),
           absl::StrFormat("global device %d",
                           request.global_device_id.value())});
    }
    if (request.phase != baseline.phase) {
      issues.push_back(
          {absl::StrCat("03/phase/", device_key),
           absl::InvalidArgumentError("phase differs across local ranks"),
           absl::StrFormat("global device %d",
                           request.global_device_id.value())});
    }
    if (request.logical_slot != baseline.logical_slot) {
      issues.push_back({absl::StrCat("03/slot/", device_key),
                        absl::InvalidArgumentError(
                            "logical slot differs across local ranks"),
                        absl::StrFormat("global device %d",
                                        request.global_device_id.value())});
    }
  }

  absl::c_sort(actual_devices, [](GlobalDeviceId lhs, GlobalDeviceId rhs) {
    return lhs.value() < rhs.value();
  });
  if (actual_devices != local_devices->second) {
    issues.push_back(
        {"04/devices",
         absl::InvalidArgumentError(
             "local callers do not exactly match topology-owned clique ranks"),
         absl::StrFormat("process %d local rank set", process_id.value())});
  }

  std::vector<RankProposal> ranks;
  ranks.reserve(values.size());
  for (const LocalValue& value : values) {
    ranks.push_back({value.request.global_device_id, value.request.local_status,
                     value.request.canonical_payload});
  }

  return LocalProposal{
      ProcessProposal{process_id, generation, baseline.launch_id,
                      baseline.clique_session_id, std::move(baseline_clique),
                      baseline.phase, baseline.logical_slot,
                      AggregateIssues(std::move(issues)), std::move(ranks)},
      std::move(membership)};
}

std::string ProcessKey(absl::string_view base, absl::string_view kind,
                       ProcessId process_id) {
  return absl::StrFormat("%s/%s/%d", base, kind, process_id.value());
}

std::string HexEncode(absl::string_view value) {
  constexpr char kHexDigits[] = "0123456789abcdef";
  std::string encoded;
  encoded.reserve(value.size() * 2);
  for (unsigned char byte : value) {
    encoded.push_back(kHexDigits[byte >> 4]);
    encoded.push_back(kHexDigits[byte & 0x0f]);
  }
  return encoded;
}

std::string GenerationKeyPrefix(absl::string_view session_id,
                                uint64_t generation) {
  return absl::StrFormat("%s/session/%s/generation/%d", kKeyPrefix,
                         HexEncode(session_id), generation);
}

absl::Status CheckDeadline(absl::Time deadline, absl::string_view operation) {
  if (absl::Now() >= deadline) {
    return absl::DeadlineExceededError(
        absl::StrCat("GPU clique agreement timed out during ", operation));
  }
  return absl::OkStatus();
}

absl::Status SetWithDeadline(KeyValueStoreInterface& kv_store,
                             absl::string_view key, absl::string_view value,
                             absl::Time deadline) {
  RETURN_IF_ERROR(CheckDeadline(deadline, "key-value-store Set"));
  absl::Status status = kv_store.Set(key, value);
  if (!status.ok()) {
    return absl::Status(
        status.code(),
        absl::StrCat("Failed to publish GPU clique agreement key ", key, ": ",
                     status.message()));
  }
  return CheckDeadline(deadline, "key-value-store Set");
}

absl::StatusOr<std::string> GetWithDeadline(KeyValueStoreInterface& kv_store,
                                            absl::string_view key,
                                            absl::Time deadline) {
  absl::Duration remaining = deadline - absl::Now();
  if (remaining <= absl::ZeroDuration()) {
    return absl::DeadlineExceededError(
        absl::StrCat("Timed out waiting for GPU clique agreement key ", key));
  }
  absl::StatusOr<std::string> value = kv_store.Get(key, remaining);
  if (!value.ok()) {
    if (absl::IsNotFound(value.status()) ||
        absl::IsDeadlineExceeded(value.status())) {
      return absl::DeadlineExceededError(
          absl::StrCat("Timed out waiting for GPU clique agreement key ", key,
                       ": ", value.status().message()));
    }
    return absl::Status(value.status().code(),
                        absl::StrCat("Failed to read GPU clique agreement key ",
                                     key, ": ", value.status().message()));
  }
  return value;
}

absl::Status ComputeDecision(uint64_t generation, const Membership& membership,
                             absl::Span<const ProcessProposal> proposals) {
  std::vector<Issue> issues;
  if (proposals.size() != membership.processes.size()) {
    return absl::InternalError(
        "Agreement proposal count does not match membership");
  }

  const ProcessProposal& baseline = proposals.front();
  std::vector<const RankProposal*> all_ranks;
  for (size_t i = 0; i < proposals.size(); ++i) {
    const ProcessProposal& proposal = proposals[i];
    ProcessId expected_process = membership.processes[i];
    std::string process_key =
        absl::StrFormat("%010d", expected_process.value());
    if (proposal.process_id != expected_process) {
      issues.push_back(
          {absl::StrCat("00/process/", process_key),
           absl::DataLossError(absl::StrFormat("proposal claims process %d",
                                               proposal.process_id.value())),
           absl::StrFormat("proposal for process %d",
                           expected_process.value())});
    }
    if (proposal.generation != generation) {
      issues.push_back(
          {absl::StrCat("00/generation/", process_key),
           absl::DataLossError(
               absl::StrFormat("proposal generation is %d; expected %d",
                               proposal.generation, generation)),
           absl::StrFormat("process %d generation", expected_process.value())});
    }
    if (!proposal.validation_status.ok()) {
      issues.push_back({absl::StrCat("20/local/", process_key),
                        proposal.validation_status,
                        absl::StrFormat("process %d local validation",
                                        expected_process.value())});
    }

    auto expected_devices = membership.devices.find(expected_process);
    std::vector<GlobalDeviceId> proposal_devices;
    proposal_devices.reserve(proposal.ranks.size());
    for (const RankProposal& rank : proposal.ranks) {
      proposal_devices.push_back(rank.device_id);
      all_ranks.push_back(&rank);
    }
    if (expected_devices == membership.devices.end() ||
        proposal_devices != expected_devices->second) {
      issues.push_back(
          {absl::StrCat("00/ranks/", process_key),
           absl::DataLossError(
               "proposal ranks do not exactly match topology ownership"),
           absl::StrFormat("process %d rank set", expected_process.value())});
    }

    if (proposal.clique_session_id != baseline.clique_session_id) {
      issues.push_back(
          {absl::StrCat("10/session/", process_key),
           absl::FailedPreconditionError(
               "clique session id differs across processes"),
           absl::StrFormat("process %d proposal", expected_process.value())});
    }
    if (proposal.launch_id != baseline.launch_id) {
      issues.push_back(
          {absl::StrCat("10/launch/", process_key),
           absl::FailedPreconditionError(
               absl::StrFormat("launch id %d differs from %d",
                               proposal.launch_id, baseline.launch_id)),
           absl::StrFormat("process %d proposal", expected_process.value())});
    }
    if (proposal.canonical_clique != baseline.canonical_clique) {
      issues.push_back(
          {absl::StrCat("10/clique/", process_key),
           absl::FailedPreconditionError("clique differs across processes"),
           absl::StrFormat("process %d proposal", expected_process.value())});
    }
    if (proposal.phase != baseline.phase) {
      issues.push_back(
          {absl::StrCat("10/phase/", process_key),
           absl::FailedPreconditionError(absl::StrFormat(
               "phase '%s' differs from '%s'", proposal.phase, baseline.phase)),
           absl::StrFormat("process %d proposal", expected_process.value())});
    }
    if (proposal.logical_slot != baseline.logical_slot) {
      issues.push_back(
          {absl::StrCat("10/slot/", process_key),
           absl::FailedPreconditionError(
               absl::StrFormat("logical slot %d differs from %d",
                               proposal.logical_slot, baseline.logical_slot)),
           absl::StrFormat("process %d proposal", expected_process.value())});
    }
  }

  absl::c_sort(all_ranks, [](const RankProposal* lhs, const RankProposal* rhs) {
    return lhs->device_id.value() < rhs->device_id.value();
  });
  if (!all_ranks.empty()) {
    const std::string& baseline_payload = all_ranks.front()->canonical_payload;
    for (const RankProposal* rank : all_ranks) {
      std::string device_key =
          absl::StrFormat("%010d", rank->device_id.value());
      if (!rank->status.ok()) {
        issues.push_back({absl::StrCat("30/status/", device_key), rank->status,
                          absl::StrFormat("global device %d local status",
                                          rank->device_id.value())});
      }
      if (rank->canonical_payload != baseline_payload) {
        issues.push_back({absl::StrCat("40/payload/", device_key),
                          absl::FailedPreconditionError(
                              "canonical payload differs across ranks"),
                          absl::StrFormat("global device %d payload",
                                          rank->device_id.value())});
      }
    }
  }

  return AggregateIssues(std::move(issues));
}

tsl::Fprint128 TranscriptFingerprint(
    absl::Span<const std::string> serialized_proposals) {
  BinaryWriter writer;
  writer.AppendUint64(serialized_proposals.size());
  for (const std::string& proposal : serialized_proposals) {
    writer.AppendString(proposal);
  }
  return tsl::Fingerprint128(writer.Take());
}

absl::Status ValidateVotes(uint64_t generation,
                           tsl::Fprint128 transcript_fingerprint,
                           const Membership& membership,
                           absl::Span<const ProcessVote> votes) {
  if (votes.size() != membership.processes.size()) {
    return absl::InternalError(
        "Agreement vote count does not match membership");
  }

  std::vector<Issue> issues;
  const absl::Status& baseline_decision = votes.front().decision;
  for (size_t i = 0; i < votes.size(); ++i) {
    const ProcessVote& vote = votes[i];
    ProcessId expected_process = membership.processes[i];
    std::string process_key =
        absl::StrFormat("%010d", expected_process.value());
    if (vote.process_id != expected_process) {
      issues.push_back(
          {absl::StrCat("00/process/", process_key),
           absl::DataLossError(absl::StrFormat("vote claims process %d",
                                               vote.process_id.value())),
           absl::StrFormat("vote for process %d", expected_process.value())});
    }
    if (vote.generation != generation) {
      issues.push_back(
          {absl::StrCat("00/generation/", process_key),
           absl::DataLossError("vote generation differs"),
           absl::StrFormat("process %d vote", expected_process.value())});
    }
    if (!(vote.transcript_fingerprint == transcript_fingerprint)) {
      issues.push_back(
          {absl::StrCat("00/transcript/", process_key),
           absl::FailedPreconditionError(
               "processes observed different proposal transcripts"),
           absl::StrFormat("process %d vote", expected_process.value())});
    }
    if (!StatusEquals(vote.decision, baseline_decision)) {
      issues.push_back(
          {absl::StrCat("00/decision/", process_key),
           absl::FailedPreconditionError(
               "processes computed different agreement decisions"),
           absl::StrFormat("process %d vote", expected_process.value())});
    }
  }
  return AggregateIssues(std::move(issues));
}

absl::Status RunAgreementLeader(
    ProcessId process_id,
    const absl::flat_hash_map<GlobalDeviceId, ProcessId>& device_to_process,
    const std::shared_ptr<KeyValueStoreInterface>& kv_store,
    absl::Span<const LocalValue> values, absl::Time deadline) {
  const std::string& session_id = values.front().request.clique_session_id;
  ASSIGN_OR_RETURN(uint64_t generation, NextGeneration(process_id, session_id));
  ASSIGN_OR_RETURN(
      LocalProposal local,
      BuildLocalProposal(process_id, device_to_process, generation, values));

  if (local.membership.processes.size() == 1) {
    return ComputeDecision(generation, local.membership,
                           absl::MakeConstSpan(&local.proposal, 1));
  }
  if (kv_store == nullptr) {
    return absl::FailedPreconditionError(
        "A non-local GPU clique agreement requires a key-value store");
  }

  std::string key_prefix = GenerationKeyPrefix(session_id, generation);
  std::string proposal_value = SerializeProposal(local.proposal);
  RETURN_IF_ERROR(
      SetWithDeadline(*kv_store, ProcessKey(key_prefix, "proposal", process_id),
                      proposal_value, deadline));

  std::vector<std::string> serialized_proposals;
  std::vector<ProcessProposal> proposals;
  serialized_proposals.reserve(local.membership.processes.size());
  proposals.reserve(local.membership.processes.size());
  for (ProcessId process : local.membership.processes) {
    ASSIGN_OR_RETURN(
        std::string serialized,
        GetWithDeadline(*kv_store, ProcessKey(key_prefix, "proposal", process),
                        deadline));
    ASSIGN_OR_RETURN(ProcessProposal proposal, DeserializeProposal(serialized));
    serialized_proposals.push_back(std::move(serialized));
    proposals.push_back(std::move(proposal));
  }

  absl::Status decision =
      ComputeDecision(generation, local.membership, proposals);
  tsl::Fprint128 transcript_fingerprint =
      TranscriptFingerprint(serialized_proposals);
  ProcessVote local_vote{process_id, generation, transcript_fingerprint,
                         decision};
  RETURN_IF_ERROR(SetWithDeadline(*kv_store,
                                  ProcessKey(key_prefix, "vote", process_id),
                                  SerializeVote(local_vote), deadline));

  std::vector<ProcessVote> votes;
  votes.reserve(local.membership.processes.size());
  for (ProcessId process : local.membership.processes) {
    ASSIGN_OR_RETURN(
        std::string serialized,
        GetWithDeadline(*kv_store, ProcessKey(key_prefix, "vote", process),
                        deadline));
    ASSIGN_OR_RETURN(ProcessVote vote, DeserializeVote(serialized));
    votes.push_back(std::move(vote));
  }
  RETURN_IF_ERROR(ValidateVotes(generation, transcript_fingerprint,
                                local.membership, votes));
  return decision;
}

absl::Status LocalRendezvous(
    ProcessId process_id,
    const absl::flat_hash_map<GlobalDeviceId, ProcessId>& device_to_process,
    const std::shared_ptr<KeyValueStoreInterface>& kv_store,
    const GpuCliqueAgreementRequest& request, size_t expected_participants,
    absl::Time deadline) {
  ProcessAgreementState& state = GetProcessAgreementState();
  std::shared_ptr<LocalRound> round;
  LocalRoundKey round_key;
  bool is_leader = false;
  std::vector<LocalValue> leader_values;

  {
    absl::MutexLock lock(state.mutex);
    round_key = LocalRoundKey{process_id, request.run_id.ToInt(),
                              request.clique_session_id};

    auto [it, inserted] = state.local_rounds.try_emplace(
        round_key, std::make_shared<LocalRound>(expected_participants));
    round = it->second;
    if (!inserted && round->expected_participants != expected_participants &&
        !round->ready) {
      round->result = absl::InvalidArgumentError(absl::StrFormat(
          "Local GPU ranks derived different participant counts: %d and %d",
          round->expected_participants, expected_participants));
      round->ready = true;
      round->cv.SignalAll();
    }

    auto [value_it, value_inserted] = round->values.try_emplace(
        request.global_device_id, LocalValue{request, deadline});
    if (!value_inserted && !round->ready) {
      round->result = absl::InvalidArgumentError(absl::StrFormat(
          "Global device %d joined a local agreement round more than once",
          request.global_device_id.value()));
      round->ready = true;
      round->cv.SignalAll();
    }

    if (!round->ready && !round->leader_running &&
        round->values.size() == round->expected_participants) {
      round->leader_running = true;
      is_leader = true;
      leader_values.reserve(round->values.size());
      for (const auto& [device, value] : round->values) {
        leader_values.push_back(value);
      }
      // All local ranks have arrived, so this round no longer needs to be
      // discoverable. Removing it before the leader runs lets the same key be
      // reused by a later sequential agreement without retaining per-RunId
      // sequence counters for the lifetime of the process.
      state.local_rounds.erase(round_key);
    }

    if (!is_leader) {
      while (!round->ready) {
        if (round->cv.WaitWithDeadline(&state.mutex, deadline) &&
            !round->ready) {
          round->result = absl::DeadlineExceededError(absl::StrFormat(
              "Timed out waiting for %d local GPU ranks in clique session %s",
              round->expected_participants, request.clique_session_id));
          round->ready = true;
          round->cv.SignalAll();
        }
      }
      absl::Status result = round->result;
      return result;
    }
  }

  absl::Time leader_deadline = deadline;
  for (const LocalValue& value : leader_values) {
    leader_deadline = std::min(leader_deadline, value.deadline);
  }
  absl::Status leader_result = RunAgreementLeader(
      process_id, device_to_process, kv_store, leader_values, leader_deadline);

  absl::MutexLock lock(state.mutex);
  if (!round->ready) {
    round->result = std::move(leader_result);
    round->ready = true;
    round->cv.SignalAll();
  }
  absl::Status result = round->result;
  return result;
}

}  // namespace

GpuCliqueAgreement::GpuCliqueAgreement(
    ProcessId process_id,
    absl::flat_hash_map<GlobalDeviceId, ProcessId> device_to_process,
    std::shared_ptr<KeyValueStoreInterface> kv_store,
    GpuCliqueAgreementOptions options)
    : impl_(std::make_shared<Impl>(process_id, std::move(device_to_process),
                                   std::move(kv_store), options)) {}

absl::Status GpuCliqueAgreement::Agree(
    const GpuCliqueAgreementRequest& request) const {
  if (impl_->options.timeout <= absl::ZeroDuration() ||
      impl_->options.timeout >= absl::InfiniteDuration()) {
    return absl::InvalidArgumentError(
        "GPU clique agreement timeout must be positive and finite");
  }
  if (request.clique_session_id.empty()) {
    return absl::InvalidArgumentError(
        "GPU clique agreement requires a non-empty clique session id");
  }

  ASSIGN_OR_RETURN(
      Membership membership,
      DeriveMembership(request.clique_key, impl_->device_to_process));
  auto local_devices = membership.devices.find(impl_->process_id);
  if (local_devices == membership.devices.end()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Process %d does not participate in the GPU clique",
                        impl_->process_id.value()));
  }

  return LocalRendezvous(impl_->process_id, impl_->device_to_process,
                         impl_->kv_store, request, local_devices->second.size(),
                         absl::Now() + impl_->options.timeout);
}

absl::Status AgreeGpuClique(const GpuCliqueAgreement* agreement,
                            const GpuCliqueAgreementRequest& request) {
  if (agreement != nullptr) {
    return agreement->Agree(request);
  }
  if (!request.clique_key.is_local()) {
    return absl::FailedPreconditionError(
        "A non-local GPU clique requires a configured clique agreement");
  }

  absl::flat_hash_map<GlobalDeviceId, ProcessId> local_topology;
  local_topology.reserve(request.clique_key.devices().size());
  for (GlobalDeviceId device : request.clique_key.devices()) {
    local_topology.emplace(device, ProcessId(0));
  }
  GpuCliqueAgreement local_agreement(ProcessId(0), std::move(local_topology),
                                     /*kv_store=*/nullptr);
  return local_agreement.Agree(request);
}

}  // namespace xla::gpu
