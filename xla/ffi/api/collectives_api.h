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

struct DeviceCommunicatorRequirements {
  int32_t lsa_barrier_count = 0;
};

struct DeviceCommunicatorLookup {
  const XLA_FFI_DeviceCommunicator* device_communicator;
  size_t byte_size;
  int32_t version;
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

  // Requests both the clique and its device communicator in Prepare. Identical
  // requests share synchronization resources and must execute in order on the
  // device. All ranks must request identical requirements.
  Status RequestDeviceCommunicator(
      GroupMode group_mode, const std::vector<std::vector<int64_t>>& groups,
      int64_t communication_id, DeviceCommunicatorRequirements requirements) {
    if (ext_->extension_base.id.minor_version <
            kDeviceCommunicatorMinorVersion ||
        ext_->extension_base.struct_size <
            XLA_FFI_STRUCT_SIZE(XLA_FFI_Collectives_Extension,
                                request_device_communicator) ||
        ext_->request_device_communicator == nullptr) {
      return ErrorPolicy::FromErrorCode(
          XLA_FFI_Error_Code_UNIMPLEMENTED,
          "Device communicators are not supported by this collectives "
          "extension");
    }
    std::vector<XLA_FFI_ReplicaGroup> raw_groups = ToRawGroups(groups);
    XLA_FFI_DeviceCommunicator_Requirements raw_requirements = {
        XLA_FFI_DeviceCommunicator_Requirements_STRUCT_SIZE, nullptr,
        requirements.lsa_barrier_count};
    XLA_FFI_DeviceCommunicator_Request_Args args;
    args.struct_size = XLA_FFI_DeviceCommunicator_Request_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.group_mode = static_cast<XLA_FFI_CollectiveGroupMode>(group_mode);
    args.groups = raw_groups.data();
    args.num_groups = raw_groups.size();
    args.communication_id = communication_id;
    args.requirements = &raw_requirements;
    if (XLA_FFI_Error* err = ext_->request_device_communicator(ext_, &args)) {
      return ErrorPolicy::TakeError(api_, err);
    }
    return ErrorPolicy::Ok();
  }

  // Returns a borrowed host-resident descriptor for the exact requirements
  // requested in Prepare. Read or copy it during this invocation, and verify
  // byte_size and the backend-defined version before using it with device code.
  StatusOr<DeviceCommunicatorLookup> GetDeviceCommunicator(
      GroupMode group_mode, const std::vector<std::vector<int64_t>>& groups,
      int64_t communication_id, DeviceCommunicatorRequirements requirements) {
    if (ext_->extension_base.id.minor_version <
            kDeviceCommunicatorMinorVersion ||
        ext_->extension_base.struct_size <
            XLA_FFI_STRUCT_SIZE(XLA_FFI_Collectives_Extension,
                                get_device_communicator) ||
        ext_->get_device_communicator == nullptr) {
      return StatusOr<DeviceCommunicatorLookup>(
          ErrorPolicy::FromErrorCode(XLA_FFI_Error_Code_UNIMPLEMENTED,
                                     "Device communicators are not supported "
                                     "by this collectives extension"));
    }
    std::vector<XLA_FFI_ReplicaGroup> raw_groups = ToRawGroups(groups);
    XLA_FFI_DeviceCommunicator_Requirements raw_requirements = {
        XLA_FFI_DeviceCommunicator_Requirements_STRUCT_SIZE, nullptr,
        requirements.lsa_barrier_count};
    XLA_FFI_DeviceCommunicator_Get_Args args;
    args.struct_size = XLA_FFI_DeviceCommunicator_Get_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.group_mode = static_cast<XLA_FFI_CollectiveGroupMode>(group_mode);
    args.groups = raw_groups.data();
    args.num_groups = raw_groups.size();
    args.communication_id = communication_id;
    args.requirements = &raw_requirements;
    args.device_communicator = nullptr;
    args.byte_size = 0;
    args.version = 0;
    if (XLA_FFI_Error* err = ext_->get_device_communicator(ext_, &args)) {
      return StatusOr<DeviceCommunicatorLookup>(
          ErrorPolicy::TakeError(api_, err));
    }
    return DeviceCommunicatorLookup{args.device_communicator, args.byte_size,
                                    args.version};
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

 private:
  static constexpr int32_t kDeviceCommunicatorMinorVersion = 3;

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

  // Device communicator methods check availability independently so existing
  // communicator and window operations remain usable with older runtimes.
  static bool Support(int32_t major_version, int32_t minor_version) {
    return major_version == kMajorVersion && minor_version >= 2;
  }

  // Builds a context from the extension.
  static CommunicatorContextT Create(const XLA_FFI_Api* api,
                                     const CExtension* ext) {
    return CommunicatorContextT(api, ext);
  }
};

}  // namespace internal

}  // namespace xla::ffi

#endif  // XLA_FFI_API_COLLECTIVES_API_H_
