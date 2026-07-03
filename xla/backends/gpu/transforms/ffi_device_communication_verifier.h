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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_FFI_DEVICE_COMMUNICATION_VERIFIER_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_FFI_DEVICE_COMMUNICATION_VERIFIER_H_

#include <string>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla::gpu {

// Verifies the side-effect contract for a single typed FFI custom call. An
// unregistered target is accepted because handlers may be unavailable during
// deviceless compilation and resolved later during thunk emission.
absl::Status VerifyFfiDeviceCommunicationCustomCall(
    const HloCustomCallInstruction& custom_call,
    absl::string_view platform_name);

// Device communication changes state outside of HLO-visible buffers. Require
// handlers declaring USES_DEVICE_COMMUNICATION to be represented by an
// explicitly side-effecting custom call so HLO transformations preserve them.
class FfiDeviceCommunicationVerifier : public HloModulePass {
 public:
  explicit FfiDeviceCommunicationVerifier(std::string platform_name)
      : platform_name_(std::move(platform_name)) {}

  absl::string_view name() const override {
    return "ffi-device-communication-verifier";
  }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  std::string platform_name_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_FFI_DEVICE_COMMUNICATION_VERIFIER_H_
