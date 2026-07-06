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

#ifndef XLA_FFI_API_GPU_COLLECTIVES_H_
#define XLA_FFI_API_GPU_COLLECTIVES_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/c_api_gpu_collectives.h"
#include "xla/ffi/api/ffi.h"

namespace xla::ffi {

enum class PeerAccess : uint8_t {
  kLocalDomain = XLA_FFI_GPU_PEER_ACCESS_LOCAL_DOMAIN,
  kHierarchical = XLA_FFI_GPU_PEER_ACCESS_HIERARCHICAL,
  kDirectAnyPeer = XLA_FFI_GPU_PEER_ACCESS_DIRECT_ANY_PEER,
};

enum class DeviceCommunicationFeature : uint64_t {
  kLocalMulticast = XLA_FFI_GPU_DEVICE_COMM_FEATURE_LOCAL_MULTICAST,
  kNetworkDeviceOperations =
      XLA_FFI_GPU_DEVICE_COMM_FEATURE_NETWORK_DEVICE_OPERATIONS,
};

// Type-safe set of optional device-communication features.
class DeviceCommunicationFeatures {
 public:
  constexpr DeviceCommunicationFeatures() = default;
  constexpr DeviceCommunicationFeatures(  // NOLINT
      DeviceCommunicationFeature feature)
      : bits_(static_cast<uint64_t>(feature)) {}
  explicit constexpr DeviceCommunicationFeatures(uint64_t bits) : bits_(bits) {}

  constexpr uint64_t bits() const { return bits_; }
  constexpr bool empty() const { return bits_ == 0; }
  constexpr bool contains(DeviceCommunicationFeature feature) const {
    return (bits_ & static_cast<uint64_t>(feature)) != 0;
  }

  constexpr DeviceCommunicationFeatures& operator|=(
      DeviceCommunicationFeatures other) {
    bits_ |= other.bits_;
    return *this;
  }

  friend constexpr DeviceCommunicationFeatures operator|(
      DeviceCommunicationFeatures lhs, DeviceCommunicationFeatures rhs) {
    return DeviceCommunicationFeatures(lhs.bits_ | rhs.bits_);
  }

  friend constexpr bool operator==(DeviceCommunicationFeatures lhs,
                                   DeviceCommunicationFeatures rhs) {
    return lhs.bits_ == rhs.bits_;
  }

  friend constexpr bool operator!=(DeviceCommunicationFeatures lhs,
                                   DeviceCommunicationFeatures rhs) {
    return !(lhs == rhs);
  }

 private:
  uint64_t bits_ = 0;
};

constexpr DeviceCommunicationFeatures operator|(
    DeviceCommunicationFeature lhs, DeviceCommunicationFeature rhs) {
  return DeviceCommunicationFeatures(lhs) | DeviceCommunicationFeatures(rhs);
}

// An all-zero value requests local-domain access with no auxiliary resources.
// It is distinct from the trait-only legacy default, which adds one team
// barrier when the handler makes no explicit request.
struct DeviceCommunicationRequirements {
  PeerAccess peer_access = PeerAccess::kLocalDomain;
  DeviceCommunicationFeatures required_features;
  DeviceCommunicationFeatures preferred_features;
  uint32_t local_barriers = 0;
  uint32_t team_barriers = 0;
  uint32_t notification_slots = 0;
  uint32_t completion_slots = 0;
};

enum class CommunicationTopology : uint8_t {
  kLocalDomain = XLA_FFI_GPU_TOPOLOGY_LOCAL_DOMAIN,
  kHierarchical = XLA_FFI_GPU_TOPOLOGY_HIERARCHICAL,
  kAllPeers = XLA_FFI_GPU_TOPOLOGY_ALL_PEERS,
};

struct DeviceCommunicationInfo {
  int64_t rank = 0;
  int64_t team_size = 0;
  int64_t local_rank = 0;
  int64_t local_domain_size = 0;
  int64_t local_domain_count = 0;
  CommunicationTopology topology = CommunicationTopology::kLocalDomain;
  DeviceCommunicationFeatures enabled_features;
  uint32_t team_barrier_count = 0;
  uint32_t local_barrier_count = 0;
  uint32_t notification_slot_count = 0;
  uint32_t completion_slot_count = 0;
};

// Context tag for a provider-defined device communicator passed by value to an
// FFI handler. The schema identifies the provider ABI, while the version must
// exactly match the provider headers used to compile the handler.
//
// Example:
//   using DeviceComm = ProviderDeviceComm<MyDeviceComm, kSchema, kVersion>;
//   Ffi::Bind().Ctx<DeviceComm>().To([](MyDeviceComm comm) { ... });
template <typename T, uint64_t abi_schema, uint64_t abi_version>
struct ProviderDeviceComm {};

// Header-only wrapper for the GPU device-communication C extension. XLA owns
// resource selection, registration, and lifetime; handlers can retrieve the
// resulting device communicator and the provider's registered-memory handle
// for each of any number of tagged buffers independently. Buffer registration
// does not change normal FFI argument/result access rules.
class GpuCollectives {
 public:
  GpuCollectives(const XLA_FFI_Api* api, XLA_FFI_ExecutionContext* ctx)
      : api_(api), ctx_(ctx), extension_(FindExtension(api)) {}

  // Returns true when the complete version 1.0 operation prefix is available.
  // Individual version 1.1 operations perform their own compatibility checks.
  bool available() const {
    return CheckExtension(XLA_FFI_STRUCT_SIZE(XLA_FFI_GpuCollectives_Extension,
                                              get_registered_memory_handle),
                          /*required_minor_version=*/0)
        .has_value();
  }

  // Declares the communication semantics and logical resources required by the
  // handler. This operation is valid only during Prepare.
  Error Request(const DeviceCommunicationRequirements& requirements) const {
    ErrorOr<const XLA_FFI_GpuCollectives_Extension*> extension =
        CheckExtension(XLA_FFI_STRUCT_SIZE(XLA_FFI_GpuCollectives_Extension,
                                           request_device_communication),
                       /*required_minor_version=*/1);
    if (!extension.has_value()) return extension.error();
    if ((*extension)->request_device_communication == nullptr) {
      return Unimplemented("RequestDeviceCommunication is unavailable");
    }

    XLA_FFI_GpuDeviceCommunication_Requirements c_requirements = {};
    c_requirements.struct_size =
        XLA_FFI_GpuDeviceCommunication_Requirements_STRUCT_SIZE;
    c_requirements.peer_access =
        static_cast<XLA_FFI_GpuPeerAccess>(requirements.peer_access);
    c_requirements.required_features = requirements.required_features.bits();
    c_requirements.preferred_features = requirements.preferred_features.bits();
    c_requirements.local_barrier_count = requirements.local_barriers;
    c_requirements.team_barrier_count = requirements.team_barriers;
    c_requirements.notification_slot_count = requirements.notification_slots;
    c_requirements.completion_slot_count = requirements.completion_slots;

    XLA_FFI_GpuCollectives_RequestDeviceCommunication_Args args = {};
    args.struct_size =
        XLA_FFI_GpuCollectives_RequestDeviceCommunication_Args_STRUCT_SIZE;
    args.ctx = ctx_;
    args.requirements = &c_requirements;

    XLA_FFI_Error* error = (*extension)->request_device_communication(&args);
    return error ? TakeError(error) : Error::Success();
  }

  // Returns immutable information about resources acquired after Prepare. This
  // operation is valid during Initialize and Execute.
  Error GetInfo(DeviceCommunicationInfo* info) const {
    ErrorOr<const XLA_FFI_GpuCollectives_Extension*> extension =
        CheckExtension(XLA_FFI_STRUCT_SIZE(XLA_FFI_GpuCollectives_Extension,
                                           get_device_communication_info),
                       /*required_minor_version=*/1);
    if (!extension.has_value()) return extension.error();
    if ((*extension)->get_device_communication_info == nullptr) {
      return Unimplemented("GetDeviceCommunicationInfo is unavailable");
    }
    if (info == nullptr) {
      return Error::InvalidArgument("info must not be null");
    }

    XLA_FFI_GpuDeviceCommunication_Info c_info = {};
    c_info.struct_size = XLA_FFI_GpuDeviceCommunication_Info_STRUCT_SIZE;

    XLA_FFI_GpuCollectives_GetDeviceCommunicationInfo_Args args = {};
    args.struct_size =
        XLA_FFI_GpuCollectives_GetDeviceCommunicationInfo_Args_STRUCT_SIZE;
    args.ctx = ctx_;
    args.info = &c_info;

    XLA_FFI_Error* error = (*extension)->get_device_communication_info(&args);
    if (error != nullptr) return TakeError(error);

    CommunicationTopology topology;
    switch (c_info.topology) {
      case XLA_FFI_GPU_TOPOLOGY_LOCAL_DOMAIN:
        topology = CommunicationTopology::kLocalDomain;
        break;
      case XLA_FFI_GPU_TOPOLOGY_HIERARCHICAL:
        topology = CommunicationTopology::kHierarchical;
        break;
      case XLA_FFI_GPU_TOPOLOGY_ALL_PEERS:
        topology = CommunicationTopology::kAllPeers;
        break;
      default:
        return Unimplemented("Unknown device-communication topology");
    }

    info->rank = c_info.rank;
    info->team_size = c_info.team_size;
    info->local_rank = c_info.local_rank;
    info->local_domain_size = c_info.local_domain_size;
    info->local_domain_count = c_info.local_domain_count;
    info->topology = topology;
    info->enabled_features =
        DeviceCommunicationFeatures(c_info.enabled_features);
    info->team_barrier_count = c_info.team_barrier_count;
    info->local_barrier_count = c_info.local_barrier_count;
    info->notification_slot_count = c_info.notification_slot_count;
    info->completion_slot_count = c_info.completion_slot_count;
    return Error::Success();
  }

  Error GetDeviceComm(uint64_t expected_abi_schema,
                      uint64_t expected_abi_version, void* destination,
                      size_t destination_size) const {
    ErrorOr<const XLA_FFI_GpuCollectives_Extension*> extension = CheckExtension(
        XLA_FFI_STRUCT_SIZE(XLA_FFI_GpuCollectives_Extension, get_device_comm),
        /*required_minor_version=*/0);
    if (!extension.has_value()) return extension.error();
    if ((*extension)->get_device_comm == nullptr) {
      return Unimplemented("GetDeviceComm is unavailable");
    }
    if (destination == nullptr) {
      return Error::InvalidArgument("destination must not be null");
    }
    if (destination_size == 0) {
      return Error::InvalidArgument("destination size must not be zero");
    }

    XLA_FFI_GpuCollectives_GetDeviceComm_Args args = {};
    args.struct_size = XLA_FFI_GpuCollectives_GetDeviceComm_Args_STRUCT_SIZE;
    args.ctx = ctx_;
    args.expected_abi_schema = expected_abi_schema;
    args.expected_abi_version = expected_abi_version;
    args.destination = destination;
    args.destination_size = destination_size;

    XLA_FFI_Error* error = (*extension)->get_device_comm(&args);
    return error ? TakeError(error) : Error::Success();
  }

  Error GetRegisteredMemoryHandle(AnyBuffer buffer,
                                  uint64_t expected_abi_schema,
                                  uint64_t expected_abi_version,
                                  void* destination, size_t destination_size,
                                  uint64_t* offset) const {
    AnyBuffer::Dimensions dimensions = buffer.dimensions();
    XLA_FFI_Buffer c_buffer = {
        XLA_FFI_Buffer_STRUCT_SIZE,
        /*extension_start=*/nullptr,
        static_cast<XLA_FFI_DataType>(buffer.element_type()),
        buffer.untyped_data(),
        static_cast<int64_t>(dimensions.size()),
        const_cast<int64_t*>(dimensions.begin()),
    };
    return GetRegisteredMemoryHandle(&c_buffer, expected_abi_schema,
                                     expected_abi_version, destination,
                                     destination_size, offset);
  }

  Error GetRegisteredMemoryHandle(const XLA_FFI_Buffer* buffer,
                                  uint64_t expected_abi_schema,
                                  uint64_t expected_abi_version,
                                  void* destination, size_t destination_size,
                                  uint64_t* offset) const {
    ErrorOr<const XLA_FFI_GpuCollectives_Extension*> extension =
        CheckExtension(XLA_FFI_STRUCT_SIZE(XLA_FFI_GpuCollectives_Extension,
                                           get_registered_memory_handle),
                       /*required_minor_version=*/0);
    if (!extension.has_value()) return extension.error();
    if ((*extension)->get_registered_memory_handle == nullptr) {
      return Unimplemented("GetRegisteredMemoryHandle is unavailable");
    }
    if (buffer == nullptr) {
      return Error::InvalidArgument("buffer must not be null");
    }
    if (destination == nullptr) {
      return Error::InvalidArgument("destination must not be null");
    }
    if (destination_size == 0) {
      return Error::InvalidArgument("destination size must not be zero");
    }
    if (offset == nullptr) {
      return Error::InvalidArgument("offset must not be null");
    }

    XLA_FFI_GpuCollectives_GetRegisteredMemoryHandle_Args args = {};
    args.struct_size =
        XLA_FFI_GpuCollectives_GetRegisteredMemoryHandle_Args_STRUCT_SIZE;
    args.ctx = ctx_;
    args.buffer = buffer;
    args.expected_abi_schema = expected_abi_schema;
    args.expected_abi_version = expected_abi_version;
    args.destination = destination;
    args.destination_size = destination_size;

    XLA_FFI_Error* error = (*extension)->get_registered_memory_handle(&args);
    if (error != nullptr) return TakeError(error);
    *offset = args.offset;
    return Error::Success();
  }

 private:
  static const XLA_FFI_GpuCollectives_Extension* FindExtension(
      const XLA_FFI_Api* api) {
    if (api == nullptr) return nullptr;
    XLA_FFI_Extension_Base* extension = api->extension_start;
    while (extension != nullptr) {
      if (extension->struct_size < XLA_FFI_Extension_Base_STRUCT_SIZE) {
        return nullptr;
      }
      if (extension->type == XLA_FFI_Extension_GpuCollectives) {
        return reinterpret_cast<const XLA_FFI_GpuCollectives_Extension*>(
            extension);
      }
      extension = extension->next;
    }
    return nullptr;
  }

  ErrorOr<const XLA_FFI_GpuCollectives_Extension*> CheckExtension(
      size_t required_struct_size, int32_t required_minor_version) const {
    if (extension_ == nullptr) {
      return Unexpected(
          Unimplemented("GPU device-communication extension not found"));
    }
    if (extension_->extension_base.struct_size < required_struct_size) {
      return Unexpected(Unimplemented(
          "GPU device-communication operation is unavailable in this extension "
          "table"));
    }
    if (extension_->api_major_version != XLA_FFI_GPU_COLLECTIVES_API_MAJOR) {
      return Unexpected(Unimplemented(
          "Incompatible GPU device-communication extension major version"));
    }
    if (extension_->api_minor_version < required_minor_version) {
      return Unexpected(Unimplemented(
          "GPU device-communication operation requires a newer extension minor "
          "version"));
    }
    return extension_;
  }

  Error TakeError(XLA_FFI_Error* error) const {
    XLA_FFI_Error_Code errc = XLA_FFI_Error_Code_UNKNOWN;
    if (api_->struct_size >=
            XLA_FFI_STRUCT_SIZE(XLA_FFI_Api, XLA_FFI_Error_GetCode) &&
        api_->XLA_FFI_Error_GetCode != nullptr) {
      XLA_FFI_Error_GetCode_Args args = {};
      args.struct_size = XLA_FFI_Error_GetCode_Args_STRUCT_SIZE;
      args.error = error;
      api_->XLA_FFI_Error_GetCode(&args);
      errc = args.errc;
    }

    const char* message = internal::GetErrorMessage(api_, error);
    std::string owned_message = message == nullptr ? std::string() : message;
    internal::DestroyError(api_, error);
    return Error(errc, std::move(owned_message));
  }

  static Error Unimplemented(std::string message) {
    return Error(ErrorCode::kUnimplemented, std::move(message));
  }

  const XLA_FFI_Api* api_;
  XLA_FFI_ExecutionContext* ctx_;
  const XLA_FFI_GpuCollectives_Extension* extension_;
};

// Preferred semantic name for new handlers. GpuCollectives remains the class
// name to preserve source compatibility with version 1.0 clients.
using GpuDeviceCommunication = GpuCollectives;

template <>
struct CtxDecoding<GpuCollectives> {
  using Type = GpuCollectives;

  static std::optional<Type> Decode(const XLA_FFI_Api* api,
                                    XLA_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine&) {
    return GpuCollectives(api, ctx);
  }
};

// Decodes immutable, provider-neutral information about the resources XLA
// acquired for this handler.
template <>
struct CtxDecoding<DeviceCommunicationInfo> {
  using Type = DeviceCommunicationInfo;

  static std::optional<Type> Decode(const XLA_FFI_Api* api,
                                    XLA_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine& diagnostic) {
    Type info;
    Error error = GpuCollectives(api, ctx).GetInfo(&info);
    if (!error.success()) {
      return diagnostic.Emit("Failed to get device communication info: ")
             << error.message();
    }
    return info;
  }
};

// Decodes a provider-specific device communicator while keeping provider ABI
// details out of the generic FFI C extension.
template <typename T, uint64_t abi_schema, uint64_t abi_version>
struct CtxDecoding<ProviderDeviceComm<T, abi_schema, abi_version>> {
  using Type = T;

  static_assert(std::is_trivially_copyable_v<Type>,
                "provider device communicator must be trivially copyable");
  static_assert(!std::is_array_v<Type>,
                "provider device communicator must not be an array");

  static std::optional<Type> Decode(const XLA_FFI_Api* api,
                                    XLA_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine& diagnostic) {
    Type device_comm{};
    Error error = GpuCollectives(api, ctx).GetDeviceComm(
        abi_schema, abi_version, &device_comm, sizeof(Type));
    if (!error.success()) {
      return diagnostic.Emit("Failed to get provider device communicator: ")
             << error.message();
    }
    return device_comm;
  }
};

}  // namespace xla::ffi

#endif  // XLA_FFI_API_GPU_COLLECTIVES_H_
