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

#ifndef XLA_FFI_API_C_API_GPU_COLLECTIVES_H_
#define XLA_FFI_API_C_API_GPU_COLLECTIVES_H_

#include "xla/ffi/api/c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// The GPU collectives extension is versioned independently from the root XLA
// FFI API. Minor versions may add extension-table entries or new argument
// structs, but do not add required fields to existing argument structs.
#define XLA_FFI_GPU_COLLECTIVES_API_MAJOR 1
#define XLA_FFI_GPU_COLLECTIVES_API_MINOR 1

// Minimum peer reachability required by a device-communication algorithm.
// The concrete operations available to the kernel are defined by the provider
// device ABI returned by GetDeviceComm.
typedef enum {
  XLA_FFI_GPU_PEER_ACCESS_LOCAL_DOMAIN = 0,
  XLA_FFI_GPU_PEER_ACCESS_HIERARCHICAL = 1,
  XLA_FFI_GPU_PEER_ACCESS_DIRECT_ANY_PEER = 2,
} XLA_FFI_GpuPeerAccess;

typedef uint64_t XLA_FFI_GpuDeviceCommunication_Features;

#define XLA_FFI_GPU_DEVICE_COMM_FEATURE_LOCAL_MULTICAST (UINT64_C(1) << 0)

// Requires the provider's device-network operation path even when local
// load/store access could satisfy peer reachability. For NCCL this requires a
// usable GIN connection.
#define XLA_FFI_GPU_DEVICE_COMM_FEATURE_NETWORK_DEVICE_OPERATIONS \
  (UINT64_C(1) << 1)

// Semantic requirements for one call-scoped device-communication resource
// set. All resource counts are logical namespace widths starting at slot zero.
// Notification/completion operations and any context qualification remain part
// of the provider device ABI; this generic API only reserves their index range.
// Unknown required feature bits return UNIMPLEMENTED; unknown preferred bits
// are ignored. If a bit appears in both masks, required wins. An explicit
// all-zero request asks for local-domain access with no auxiliary resources;
// unlike the trait-only default, it does not request a team barrier.
struct XLA_FFI_GpuDeviceCommunication_Requirements {
  size_t struct_size;
  XLA_FFI_Extension_Base* extension_start;

  XLA_FFI_GpuPeerAccess peer_access;
  XLA_FFI_GpuDeviceCommunication_Features required_features;
  XLA_FFI_GpuDeviceCommunication_Features preferred_features;
  uint32_t local_barrier_count;
  uint32_t team_barrier_count;
  uint32_t notification_slot_count;
  uint32_t completion_slot_count;
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_GpuDeviceCommunication_Requirements,
                             completion_slot_count);
// Stable minimum prefix accepted by version 1.1 implementations. The generic
// STRUCT_SIZE may grow when later minor versions append fields.
enum {
  XLA_FFI_GpuDeviceCommunication_Requirements_STRUCT_SIZE_V1_1 =
      XLA_FFI_STRUCT_SIZE(XLA_FFI_GpuDeviceCommunication_Requirements,
                          completion_slot_count)
};

// Requests semantic device-communication resources. Available only during
// FFI Prepare for a handler declared with USES_DEVICE_COMMUNICATION. Repeating
// an identical request is idempotent; a conflicting request is an error. If a
// handler makes no request, XLA installs the legacy default of one team
// barrier and otherwise zero-valued requirements.
struct XLA_FFI_GpuCollectives_RequestDeviceCommunication_Args {
  size_t struct_size;
  XLA_FFI_Extension_Base* extension_start;

  XLA_FFI_ExecutionContext* ctx;
  const XLA_FFI_GpuDeviceCommunication_Requirements* requirements;
};

XLA_FFI_DEFINE_STRUCT_TRAITS(
    XLA_FFI_GpuCollectives_RequestDeviceCommunication_Args, requirements);
enum {
  XLA_FFI_GpuCollectives_RequestDeviceCommunication_Args_STRUCT_SIZE_V1_1 =
      XLA_FFI_STRUCT_SIZE(
          XLA_FFI_GpuCollectives_RequestDeviceCommunication_Args, requirements)
};

typedef XLA_FFI_Error* XLA_FFI_GpuCollectives_RequestDeviceCommunication(
    XLA_FFI_GpuCollectives_RequestDeviceCommunication_Args* args);

// Handler-visible topology realized for the requested communication data
// plane. Networking used solely to implement a team barrier does not grant a
// stronger topology to the handler.
typedef enum {
  XLA_FFI_GPU_TOPOLOGY_LOCAL_DOMAIN = 0,
  XLA_FFI_GPU_TOPOLOGY_HIERARCHICAL = 1,
  XLA_FFI_GPU_TOPOLOGY_ALL_PEERS = 2,
} XLA_FFI_GpuCommunicationTopology;

// Immutable information for a realized device-communication resource set.
// Callers must zero-initialize the complete struct and set struct_size before
// calling GetDeviceCommunicationInfo.
struct XLA_FFI_GpuDeviceCommunication_Info {
  size_t struct_size;
  XLA_FFI_Extension_Base* extension_start;

  int64_t rank;
  int64_t team_size;
  int64_t local_rank;
  int64_t local_domain_size;
  int64_t local_domain_count;
  XLA_FFI_GpuCommunicationTopology topology;
  XLA_FFI_GpuDeviceCommunication_Features enabled_features;
  uint32_t team_barrier_count;
  uint32_t local_barrier_count;
  uint32_t notification_slot_count;
  uint32_t completion_slot_count;
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_GpuDeviceCommunication_Info,
                             completion_slot_count);
enum {
  XLA_FFI_GpuDeviceCommunication_Info_STRUCT_SIZE_V1_1 = XLA_FFI_STRUCT_SIZE(
      XLA_FFI_GpuDeviceCommunication_Info, completion_slot_count)
};

struct XLA_FFI_GpuCollectives_GetDeviceCommunicationInfo_Args {
  size_t struct_size;
  XLA_FFI_Extension_Base* extension_start;

  XLA_FFI_ExecutionContext* ctx;
  XLA_FFI_GpuDeviceCommunication_Info* info;
};

XLA_FFI_DEFINE_STRUCT_TRAITS(
    XLA_FFI_GpuCollectives_GetDeviceCommunicationInfo_Args, info);
enum {
  XLA_FFI_GpuCollectives_GetDeviceCommunicationInfo_Args_STRUCT_SIZE_V1_1 =
      XLA_FFI_STRUCT_SIZE(
          XLA_FFI_GpuCollectives_GetDeviceCommunicationInfo_Args, info)
};

typedef XLA_FFI_Error* XLA_FFI_GpuCollectives_GetDeviceCommunicationInfo(
    XLA_FFI_GpuCollectives_GetDeviceCommunicationInfo_Args* args);

// Returns the execution-scoped device communicator selected and configured by
// XLA for an FFI handler declared with USES_DEVICE_COMMUNICATION. It spans the
// full execution group (all replicas and partitions, ordered by flattened ID).
// The communicator satisfies the semantic requirements requested during
// Prepare, or the legacy default of one team barrier when no request was made.
// On NCCL, XLA realizes those requirements with LSA, Multimem, and GIN as
// needed. This operation copies the provider kernel argument into caller-owned
// storage after validating the expected provider schema, ABI version, and
// exact destination size. It is available during Initialize and Execute.
struct XLA_FFI_GpuCollectives_GetDeviceComm_Args {
  size_t struct_size;
  XLA_FFI_Extension_Base* extension_start;

  XLA_FFI_ExecutionContext* ctx;
  uint64_t expected_abi_schema;
  uint64_t expected_abi_version;
  void* destination;
  size_t destination_size;
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_GpuCollectives_GetDeviceComm_Args,
                             destination_size);
enum {
  XLA_FFI_GpuCollectives_GetDeviceComm_Args_STRUCT_SIZE_V1_0 =
      XLA_FFI_STRUCT_SIZE(XLA_FFI_GpuCollectives_GetDeviceComm_Args,
                          destination_size)
};

typedef XLA_FFI_Error* XLA_FFI_GpuCollectives_GetDeviceComm(
    XLA_FFI_GpuCollectives_GetDeviceComm_Args* args);

// Returns the provider's registered-memory handle for a tagged FFI argument or
// result buffer.
// A handler can retrieve any number of tagged buffers independently; no buffer
// count or registration list is part of the API. For this callsite's
// communication team, XLA registers each complete backing allocation once, and
// offset locates the requested buffer view within that allocation. Registration
// does not grant write access: normal FFI argument/result and output-aliasing
// rules continue to apply. A tagged buffer is a logical memory-space-1
// custom-call operand/result (set by frontends with operands_memory_spaces/
// results_memory_spaces). The provider kernel argument is copied into
// caller-owned storage after validating the expected provider schema, ABI
// version, and exact destination size. It is a provider-defined registration
// handle, not buffer->data or an ordinary device pointer. This operation is
// available during the Initialize and Execute stages.
struct XLA_FFI_GpuCollectives_GetRegisteredMemoryHandle_Args {
  size_t struct_size;
  XLA_FFI_Extension_Base* extension_start;

  XLA_FFI_ExecutionContext* ctx;
  const XLA_FFI_Buffer* buffer;
  uint64_t expected_abi_schema;
  uint64_t expected_abi_version;
  void* destination;
  size_t destination_size;
  uint64_t offset;  // out: requested view offset in the registered allocation
};

XLA_FFI_DEFINE_STRUCT_TRAITS(
    XLA_FFI_GpuCollectives_GetRegisteredMemoryHandle_Args, offset);
enum {
  XLA_FFI_GpuCollectives_GetRegisteredMemoryHandle_Args_STRUCT_SIZE_V1_0 =
      XLA_FFI_STRUCT_SIZE(
          XLA_FFI_GpuCollectives_GetRegisteredMemoryHandle_Args, offset)
};

typedef XLA_FFI_Error* XLA_FFI_GpuCollectives_GetRegisteredMemoryHandle(
    XLA_FFI_GpuCollectives_GetRegisteredMemoryHandle_Args* args);

// GPU device-communication extension table.
struct XLA_FFI_GpuCollectives_Extension {
  XLA_FFI_Extension_Base extension_base;
  int32_t api_major_version;
  int32_t api_minor_version;

  XLA_FFI_GpuCollectives_GetDeviceComm* get_device_comm;
  XLA_FFI_GpuCollectives_GetRegisteredMemoryHandle*
      get_registered_memory_handle;
  XLA_FFI_GpuCollectives_RequestDeviceCommunication*
      request_device_communication;
  XLA_FFI_GpuCollectives_GetDeviceCommunicationInfo*
      get_device_communication_info;
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_GpuCollectives_Extension,
                             get_device_communication_info);

#ifdef __cplusplus
}
#endif

#endif  // XLA_FFI_API_C_API_GPU_COLLECTIVES_H_
