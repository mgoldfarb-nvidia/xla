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
#include <utility>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/c_api_gpu_collectives.h"
#include "xla/ffi/api/ffi.h"

namespace xla::ffi {

// Header-only wrapper for the GPU device-communication C extension. XLA owns
// resource selection, registration, and lifetime; handlers can retrieve the
// resulting device communicator and each of any number of tagged device-memory
// arguments independently. Buffer registration does not change normal FFI
// argument/result access rules.
class GpuCollectives {
 public:
  GpuCollectives(const XLA_FFI_Api* api, XLA_FFI_ExecutionContext* ctx)
      : api_(api), ctx_(ctx), extension_(FindExtension(api)) {}

  bool available() const { return CheckExtension().has_value(); }

  Error GetDeviceComm(uint64_t expected_abi_schema,
                      uint64_t expected_abi_version, void* destination,
                      size_t destination_size) const {
    ErrorOr<const XLA_FFI_GpuCollectives_Extension*> extension =
        CheckExtension();
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

  Error GetDeviceMemory(AnyBuffer buffer, uint64_t expected_abi_schema,
                        uint64_t expected_abi_version, void* destination,
                        size_t destination_size, uint64_t* offset) const {
    AnyBuffer::Dimensions dimensions = buffer.dimensions();
    XLA_FFI_Buffer c_buffer = {
        XLA_FFI_Buffer_STRUCT_SIZE,
        /*extension_start=*/nullptr,
        static_cast<XLA_FFI_DataType>(buffer.element_type()),
        buffer.untyped_data(),
        static_cast<int64_t>(dimensions.size()),
        const_cast<int64_t*>(dimensions.begin()),
    };
    return GetDeviceMemory(&c_buffer, expected_abi_schema,
                           expected_abi_version, destination, destination_size,
                           offset);
  }

  Error GetDeviceMemory(const XLA_FFI_Buffer* buffer,
                        uint64_t expected_abi_schema,
                        uint64_t expected_abi_version, void* destination,
                        size_t destination_size, uint64_t* offset) const {
    ErrorOr<const XLA_FFI_GpuCollectives_Extension*> extension =
        CheckExtension();
    if (!extension.has_value()) return extension.error();
    if ((*extension)->get_device_memory == nullptr) {
      return Unimplemented("GetDeviceMemory is unavailable");
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

    XLA_FFI_GpuCollectives_GetDeviceMemory_Args args = {};
    args.struct_size = XLA_FFI_GpuCollectives_GetDeviceMemory_Args_STRUCT_SIZE;
    args.ctx = ctx_;
    args.buffer = buffer;
    args.expected_abi_schema = expected_abi_schema;
    args.expected_abi_version = expected_abi_version;
    args.destination = destination;
    args.destination_size = destination_size;

    XLA_FFI_Error* error = (*extension)->get_device_memory(&args);
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

  ErrorOr<const XLA_FFI_GpuCollectives_Extension*> CheckExtension() const {
    if (extension_ == nullptr) {
      return Unexpected(
          Unimplemented("GPU device-communication extension not found"));
    }
    if (extension_->extension_base.struct_size <
        XLA_FFI_GpuCollectives_Extension_STRUCT_SIZE) {
      return Unexpected(Unimplemented(
          "GPU device-communication extension table is too small"));
    }
    if (extension_->api_major_version != XLA_FFI_GPU_COLLECTIVES_API_MAJOR) {
      return Unexpected(Unimplemented(
          "Incompatible GPU device-communication extension major version"));
    }
    if (extension_->api_minor_version < XLA_FFI_GPU_COLLECTIVES_API_MINOR) {
      return Unexpected(Unimplemented(
          "GPU device-communication extension minor version is too old"));
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

template <>
struct CtxDecoding<GpuCollectives> {
  using Type = GpuCollectives;

  static std::optional<Type> Decode(const XLA_FFI_Api* api,
                                    XLA_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine&) {
    return GpuCollectives(api, ctx);
  }
};

}  // namespace xla::ffi

#endif  // XLA_FFI_API_GPU_COLLECTIVES_H_
