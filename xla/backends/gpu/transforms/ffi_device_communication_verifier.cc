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

#include "xla/backends/gpu/transforms/ffi_device_communication_verifier.h"

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/ffi/ffi_registry.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

absl::Status VerifyFfiDeviceCommunicationCustomCall(
    const HloCustomCallInstruction& custom_call,
    absl::string_view platform_name) {
  if (custom_call.api_version() !=
          CustomCallApiVersion::API_VERSION_TYPED_FFI ||
      custom_call.custom_call_has_side_effect()) {
    return absl::OkStatus();
  }

  absl::StatusOr<ffi::HandlerRegistration> registration =
      ffi::FindHandler(custom_call.custom_call_target(), platform_name);
  if (!registration.ok()) {
    // An FFI handler need not be linked into a deviceless compiler process.
    // Preserve existing unknown-handler behavior and validate it after handler
    // resolution in ThunkEmitter instead.
    if (absl::IsNotFound(registration.status())) {
      return absl::OkStatus();
    }
    return registration.status();
  }

  if (!ffi::UsesDeviceCommunication(registration->metadata)) {
    return absl::OkStatus();
  }

  return absl::InvalidArgumentError(absl::StrFormat(
      "Typed FFI custom call '%s' (target '%s') is registered with "
      "USES_DEVICE_COMMUNICATION and must set "
      "custom_call_has_side_effect=true",
      custom_call.name(), custom_call.custom_call_target()));
}

absl::StatusOr<bool> FfiDeviceCommunicationVerifier::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  for (const HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (const HloInstruction* instruction : computation->instructions()) {
      const auto* custom_call = DynCast<HloCustomCallInstruction>(instruction);
      if (custom_call != nullptr) {
        RETURN_IF_ERROR(VerifyFfiDeviceCommunicationCustomCall(*custom_call,
                                                               platform_name_));
      }
    }
  }
  return false;
}

}  // namespace xla::gpu
