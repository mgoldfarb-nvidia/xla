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
// FFI API. Minor versions are append-only and backwards compatible.
#define XLA_FFI_GPU_COLLECTIVES_API_MAJOR 1
#define XLA_FFI_GPU_COLLECTIVES_API_MINOR 0

// Caller-owned storage and metadata for one opaque device kernel argument. A
// null destination with zero capacity is a size/alignment query and must
// succeed. The implementation validates that the provider compile-time and
// runtime ABIs match before returning their common ABI version.
struct XLA_FFI_GpuCollective_PackedKernelArg {
  size_t struct_size;
  XLA_FFI_Extension_Base* extension_start;

  void* destination;
  size_t capacity;

  size_t size;           // out: required size, including on RESOURCE_EXHAUSTED
  size_t alignment;      // out: required destination alignment
  uint64_t schema;       // out: stable packed-argument schema identifier
  uint64_t abi_version;  // out: validated compile-time/runtime ABI version
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_GpuCollective_PackedKernelArg,
                             abi_version);

// Returns the execution-scoped device communicator selected and configured by
// XLA for an FFI handler declared with USES_DEVICE_COMMUNICATION. It spans the
// full execution group (all replicas and partitions, ordered by flattened ID).
// The initial contract provides one load/store-accessible synchronization slot
// (slot 0) per callsite; handlers requiring more slots are not supported. This
// query is available during the Initialize and Execute stages.
struct XLA_FFI_GpuCollectives_GetDeviceComm_Args {
  size_t struct_size;
  XLA_FFI_Extension_Base* extension_start;

  XLA_FFI_ExecutionContext* ctx;
  XLA_FFI_GpuCollective_PackedKernelArg* kernel_arg;
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_GpuCollectives_GetDeviceComm_Args,
                             kernel_arg);

typedef XLA_FFI_Error* XLA_FFI_GpuCollectives_GetDeviceComm(
    XLA_FFI_GpuCollectives_GetDeviceComm_Args* args);

// Returns the device-memory handle for a tagged FFI argument or result buffer.
// A handler can query any number of tagged buffers independently; no buffer
// count or registration list is part of the API. For this callsite's
// communication team, XLA registers each complete backing allocation once, and
// offset locates the requested buffer view within that allocation. Registration
// does not grant write access: normal FFI argument/result and output-aliasing
// rules continue to apply. A tagged buffer is a logical memory-space-1
// custom-call operand/result (set by frontends with operands_memory_spaces/
// results_memory_spaces). This query is available during the Initialize and
// Execute stages.
struct XLA_FFI_GpuCollectives_GetDeviceMemory_Args {
  size_t struct_size;
  XLA_FFI_Extension_Base* extension_start;

  XLA_FFI_ExecutionContext* ctx;
  const XLA_FFI_Buffer* buffer;
  XLA_FFI_GpuCollective_PackedKernelArg* kernel_arg;
  uint64_t offset;  // out: requested view offset in the registered allocation
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_GpuCollectives_GetDeviceMemory_Args,
                             offset);

typedef XLA_FFI_Error* XLA_FFI_GpuCollectives_GetDeviceMemory(
    XLA_FFI_GpuCollectives_GetDeviceMemory_Args* args);

// GPU device-communication extension table.
struct XLA_FFI_GpuCollectives_Extension {
  XLA_FFI_Extension_Base extension_base;
  int32_t api_major_version;
  int32_t api_minor_version;

  XLA_FFI_GpuCollectives_GetDeviceComm* get_device_comm;
  XLA_FFI_GpuCollectives_GetDeviceMemory* get_device_memory;
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_GpuCollectives_Extension,
                             get_device_memory);

#ifdef __cplusplus
}
#endif

#endif  // XLA_FFI_API_C_API_GPU_COLLECTIVES_H_
