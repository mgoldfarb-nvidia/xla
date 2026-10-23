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

#ifndef XLA_FFI_API_COLLECTIVES_API_H_
#define XLA_FFI_API_COLLECTIVES_API_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/collectives_c_api.h"

// C++ wrapper for the XLA FFI Collectives API.
namespace xla::ffi {

// Mirrors `XLA_FFI_CollectiveGroupMode` / `xla::CollectiveOpGroupMode`.
enum class GroupMode {
  kCrossReplica = XLA_FFI_GROUP_CROSS_REPLICA,
  kCrossPartition = XLA_FFI_GROUP_CROSS_PARTITION,
  kCrossReplicaAndPartition = XLA_FFI_GROUP_CROSS_REPLICA_AND_PARTITION,
  kFlattenedId = XLA_FFI_GROUP_FLATTENED_ID,
};

struct CollectiveMemoryRegion {
  const void* buffer;
  size_t byte_size;
  uint64_t flags = 0;
};

struct WindowLookup {
  XLA_FFI_Window* window;
  size_t offset;
};

struct CommunicatorVersion {
  std::string backend;
  int64_t build_version;
  int64_t runtime_version;
};

struct DeviceCommunicatorRequirements {
  int32_t barrier_count = 0;
  int32_t lsa_barrier_count = 0;
  int32_t rail_gin_barrier_count = 0;
  int32_t gin_signal_count = 0;
  bool gin_connection_full = false;
};

namespace internal {

// C++ wrapper for the XLA FFI Collectives extension API.
// Unified implementation for internal and external FFI modules.
template <typename ErrorPolicy>
class CommunicatorContextBase {
 public:
  using Status = typename ErrorPolicy::Status;
  template <typename T>
  using StatusOr = typename ErrorPolicy::template StatusOr<T>;

  CommunicatorContextBase(const XLA_FFI_Api* api,
                          const XLA_FFI_Collectives_Extension* ext)
      : api_(api), ext_(ext) {}

  // Requests the clique for `groups` so it is acquired before execution.
  // Prepare stage only.
  Status RequestCommunicator(GroupMode group_mode,
                             const std::vector<std::vector<int64_t>>& groups,
                             int64_t communication_id) {
    std::vector<XLA_FFI_ReplicaGroup> raw_groups = ToRawGroups(groups);
    XLA_FFI_Communicator_Request_Args args;
    args.struct_size = XLA_FFI_Communicator_Request_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.group_mode = static_cast<XLA_FFI_CollectiveGroupMode>(group_mode);
    args.groups = raw_groups.data();
    args.num_groups = raw_groups.size();
    args.communication_id = communication_id;
    if (XLA_FFI_Error* err = ext_->request_communicator(ext_, &args)) {
      return ErrorPolicy::TakeError(api_, err);
    }
    return ErrorPolicy::Ok();
  }

  // Returns the non-owning communicator handle for `groups`. The handle is
  // backend-defined; the caller reinterprets it (e.g. as `ncclComm_t`).
  StatusOr<XLA_FFI_Communicator*> GetCommunicator(
      GroupMode group_mode, const std::vector<std::vector<int64_t>>& groups,
      int64_t communication_id) {
    std::vector<XLA_FFI_ReplicaGroup> raw_groups = ToRawGroups(groups);
    XLA_FFI_Communicator_Get_Args args;
    args.struct_size = XLA_FFI_Communicator_Get_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.group_mode = static_cast<XLA_FFI_CollectiveGroupMode>(group_mode);
    args.groups = raw_groups.data();
    args.num_groups = raw_groups.size();
    args.communication_id = communication_id;
    args.communicator = nullptr;
    if (XLA_FFI_Error* err = ext_->get_communicator(ext_, &args)) {
      return StatusOr<XLA_FFI_Communicator*>(ErrorPolicy::TakeError(api_, err));
    }
    return args.communicator;
  }

  //===--------------------------------------------------------------------===//
  // Collective memory window
  //===--------------------------------------------------------------------===//
  //
  // Request window registration for a batch of already-allocated buffers in
  // Prepare; look up an opaque backend-defined window handle per buffer in
  // Init/Execute. The handler reinterprets the window and calls the backend's
  // collective device APIs directly to obtain local, peer, and multicast
  // pointers. Allocation stays on the JAX-side.
  //
  // RequestWindow requires the corresponding communicator/clique to have been
  // requested first via RequestCommunicator or RequestDeviceCommunicator with
  // the same (group_mode, groups, communication_id); backends may return
  // FailedPrecondition otherwise.

  Status RequestWindow(GroupMode group_mode,
                       const std::vector<std::vector<int64_t>>& groups,
                       int64_t communication_id,
                       const std::vector<CollectiveMemoryRegion>& regions) {
    std::vector<XLA_FFI_ReplicaGroup> raw_groups = ToRawGroups(groups);
    std::vector<XLA_FFI_CollectiveMemoryRegion> raw_regions;
    raw_regions.reserve(regions.size());
    for (const CollectiveMemoryRegion& r : regions) {
      raw_regions.push_back(
          XLA_FFI_CollectiveMemoryRegion{r.buffer, r.byte_size, r.flags});
    }
    XLA_FFI_Window_Request_Args args;
    args.struct_size = XLA_FFI_Window_Request_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.group_mode = static_cast<XLA_FFI_CollectiveGroupMode>(group_mode);
    args.groups = raw_groups.data();
    args.num_groups = raw_groups.size();
    args.communication_id = communication_id;
    args.regions = raw_regions.data();
    args.num_regions = raw_regions.size();
    if (XLA_FFI_Error* err = ext_->request_window(ext_, &args)) {
      return ErrorPolicy::TakeError(api_, err);
    }
    return ErrorPolicy::Ok();
  }

  StatusOr<WindowLookup> GetWindow(
      GroupMode group_mode, const std::vector<std::vector<int64_t>>& groups,
      int64_t communication_id, const void* buffer) {
    std::vector<XLA_FFI_ReplicaGroup> raw_groups = ToRawGroups(groups);
    XLA_FFI_Window_Get_Args args;
    args.struct_size = XLA_FFI_Window_Get_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.group_mode = static_cast<XLA_FFI_CollectiveGroupMode>(group_mode);
    args.groups = raw_groups.data();
    args.num_groups = raw_groups.size();
    args.communication_id = communication_id;
    args.buffer = buffer;
    args.window = nullptr;
    args.window_offset = 0;
    if (XLA_FFI_Error* err = ext_->get_window(ext_, &args)) {
      return StatusOr<WindowLookup>(ErrorPolicy::TakeError(api_, err));
    }
    return WindowLookup{args.window, args.window_offset};
  }

  // Queries the active backend locally, including during Prepare.
  StatusOr<CommunicatorVersion> GetCommunicatorVersion() {
    if (ext_ == nullptr ||
        ext_->extension_base.id.major_version !=
            XLA_FFI_Extension_Collectives_MajorVersion ||
        ext_->extension_base.id.minor_version < 3 ||
        ext_->extension_base.struct_size <
            XLA_FFI_Collectives_Extension_STRUCT_SIZE ||
        ext_->get_communicator_version == nullptr) {
      return ErrorPolicy::FromErrorCode(
          XLA_FFI_Error_Code_UNIMPLEMENTED,
          "Communicator versions require collectives extension version 0.3");
    }
    XLA_FFI_Collectives_GetCommunicatorVersion_Args args = {};
    args.struct_size =
        XLA_FFI_Collectives_GetCommunicatorVersion_Args_STRUCT_SIZE;
    if (XLA_FFI_Error* err = ext_->get_communicator_version(ext_, &args)) {
      return ErrorPolicy::TakeError(api_, err);
    }
    if (args.backend == nullptr) {
      return ErrorPolicy::FromErrorCode(XLA_FFI_Error_Code_INTERNAL,
                                        "Collectives backend name is null");
    }
    return CommunicatorVersion{args.backend, args.build_version,
                               args.runtime_version};
  }

  // Requests the clique and device communicator in Prepare. The expected
  // version is the caller's device-header version, not a queried XLA version.
  Status RequestDeviceCommunicator(
      GroupMode group_mode, const std::vector<std::vector<int64_t>>& groups,
      int64_t communication_id,
      const DeviceCommunicatorRequirements& requirements,
      int64_t expected_build_version) {
    if (ext_ == nullptr ||
        ext_->extension_base.id.major_version !=
            XLA_FFI_Extension_Collectives_MajorVersion ||
        ext_->extension_base.id.minor_version < 3 ||
        ext_->extension_base.struct_size <
            XLA_FFI_Collectives_Extension_STRUCT_SIZE ||
        ext_->request_device_communicator == nullptr) {
      return ErrorPolicy::FromErrorCode(
          XLA_FFI_Error_Code_UNIMPLEMENTED,
          "Device communicators require collectives extension version 0.3");
    }
    std::vector<XLA_FFI_ReplicaGroup> raw_groups = ToRawGroups(groups);
    XLA_FFI_DeviceCommunicator_Requirements raw_requirements =
        ToRawRequirements(requirements);
    XLA_FFI_DeviceCommunicator_Request_Args args = {};
    args.struct_size = XLA_FFI_DeviceCommunicator_Request_Args_STRUCT_SIZE;
    args.group_mode = static_cast<XLA_FFI_CollectiveGroupMode>(group_mode);
    args.groups = raw_groups.data();
    args.num_groups = raw_groups.size();
    args.communication_id = communication_id;
    args.requirements = &raw_requirements;
    args.expected_build_version = expected_build_version;
    if (XLA_FFI_Error* err = ext_->request_device_communicator(ext_, &args)) {
      return ErrorPolicy::TakeError(api_, err);
    }
    return ErrorPolicy::Ok();
  }

  // Returns a borrowed host descriptor in Initialize/Execute. Copy its value
  // into kernel arguments; XLA owns its resources. Identical requests share
  // synchronization resources and require ordered kernel execution.
  StatusOr<const XLA_FFI_DeviceCommunicator*> GetDeviceCommunicator(
      GroupMode group_mode, const std::vector<std::vector<int64_t>>& groups,
      int64_t communication_id,
      const DeviceCommunicatorRequirements& requirements,
      int64_t expected_build_version) {
    if (ext_ == nullptr ||
        ext_->extension_base.id.major_version !=
            XLA_FFI_Extension_Collectives_MajorVersion ||
        ext_->extension_base.id.minor_version < 3 ||
        ext_->extension_base.struct_size <
            XLA_FFI_Collectives_Extension_STRUCT_SIZE ||
        ext_->get_device_communicator == nullptr) {
      return ErrorPolicy::FromErrorCode(
          XLA_FFI_Error_Code_UNIMPLEMENTED,
          "Device communicators require collectives extension version 0.3");
    }
    std::vector<XLA_FFI_ReplicaGroup> raw_groups = ToRawGroups(groups);
    XLA_FFI_DeviceCommunicator_Requirements raw_requirements =
        ToRawRequirements(requirements);
    XLA_FFI_DeviceCommunicator_Get_Args args = {};
    args.struct_size = XLA_FFI_DeviceCommunicator_Get_Args_STRUCT_SIZE;
    args.group_mode = static_cast<XLA_FFI_CollectiveGroupMode>(group_mode);
    args.groups = raw_groups.data();
    args.num_groups = raw_groups.size();
    args.communication_id = communication_id;
    args.requirements = &raw_requirements;
    args.expected_build_version = expected_build_version;
    if (XLA_FFI_Error* err = ext_->get_device_communicator(ext_, &args)) {
      return ErrorPolicy::TakeError(api_, err);
    }
    if (args.communicator == nullptr) {
      return ErrorPolicy::FromErrorCode(XLA_FFI_Error_Code_INTERNAL,
                                        "Device communicator is null");
    }
    return args.communicator;
  }

 private:
  static XLA_FFI_DeviceCommunicator_Requirements ToRawRequirements(
      const DeviceCommunicatorRequirements& requirements) {
    return XLA_FFI_DeviceCommunicator_Requirements{
        XLA_FFI_DeviceCommunicator_Requirements_STRUCT_SIZE,
        nullptr,
        requirements.barrier_count,
        requirements.lsa_barrier_count,
        requirements.rail_gin_barrier_count,
        requirements.gin_signal_count,
        requirements.gin_connection_full};
  }

  // Converts a vector of replica groups to a vector of `XLA_FFI_ReplicaGroup`.
  // The results reference the id storage in `groups`, which must outlive them.
  static std::vector<XLA_FFI_ReplicaGroup> ToRawGroups(
      const std::vector<std::vector<int64_t>>& groups) {
    std::vector<XLA_FFI_ReplicaGroup> raw_groups;
    raw_groups.reserve(groups.size());
    for (const std::vector<int64_t>& group : groups) {
      raw_groups.push_back(XLA_FFI_ReplicaGroup{group.data(), group.size()});
    }
    return raw_groups;
  }

  const XLA_FFI_Api* api_;
  const XLA_FFI_Collectives_Extension* ext_;
};

// Common base struct for internal and external Collectives extensions.
// Defines traits for CtxDecoding<Extension<Collectives>>.
template <typename CommunicatorContextT>
struct CollectivesExtensionBase {
  using Type = CommunicatorContextT;
  using CExtension = XLA_FFI_Collectives_Extension;

  static constexpr auto kName = "CollectivesExtension";
  static constexpr int32_t kExtensionType = XLA_FFI_Extension_Collectives;
  static constexpr int32_t kMajorVersion =
      XLA_FFI_Extension_Collectives_MajorVersion;
  static constexpr int32_t kMinorVersion =
      XLA_FFI_Extension_Collectives_MinorVersion;

  // Builds a context from the extension.
  static CommunicatorContextT Create(const XLA_FFI_Api* api,
                                     const CExtension* ext) {
    return CommunicatorContextT(api, ext);
  }
};

}  // namespace internal

}  // namespace xla::ffi

#endif  // XLA_FFI_API_COLLECTIVES_API_H_
