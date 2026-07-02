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

#include <memory>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/ffi.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

using absl_testing::IsOkAndHolds;
using absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::HasSubstr;

constexpr absl::string_view kMetadataTraitTarget =
    "__xla_test$$ffi_device_communication_metadata_trait";
constexpr absl::string_view kRegistrationTraitTarget =
    "__xla_test$$ffi_device_communication_registration_trait";
constexpr absl::string_view kNoDeviceCommunicationTarget =
    "__xla_test$$ffi_without_device_communication";

absl::Status NoOp() { return absl::OkStatus(); }

XLA_FFI_DEFINE_HANDLER(kMetadataTraitHandler, NoOp, ffi::Ffi::Bind(),
                       {ffi::Traits::kUsesDeviceCommunication});
XLA_FFI_DEFINE_HANDLER(kRegistrationTraitHandler, NoOp, ffi::Ffi::Bind());
XLA_FFI_DEFINE_HANDLER(kNoDeviceCommunicationHandler, NoOp, ffi::Ffi::Bind());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), kMetadataTraitTarget, "Host",
                         kMetadataTraitHandler);
XLA_FFI_REGISTER_HANDLER(
    ffi::GetXlaFfiApi(), kRegistrationTraitTarget, "Host",
    kRegistrationTraitHandler,
    static_cast<XLA_FFI_Handler_Traits>(ffi::Traits::kUsesDeviceCommunication));
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), kNoDeviceCommunicationTarget,
                         "Host", kNoDeviceCommunicationHandler);

class FfiDeviceCommunicationVerifierTest
    : public HloHardwareIndependentTestBase {};

absl::StatusOr<bool> RunVerifier(HloModule* module) {
  return FfiDeviceCommunicationVerifier("Host").Run(module);
}

TEST_F(FfiDeviceCommunicationVerifierTest, RejectsHandlerMetadataTrait) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(R"(
    HloModule test

    ENTRY main {
      ROOT call = f32[] custom-call(),
        custom_call_target="__xla_test$$ffi_device_communication_metadata_trait",
        api_version=API_VERSION_TYPED_FFI
    }
  )"));

  EXPECT_THAT(RunVerifier(module.get()),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr(kMetadataTraitTarget),
                             HasSubstr("USES_DEVICE_COMMUNICATION"),
                             HasSubstr("custom_call_has_side_effect=true"))));
}

TEST_F(FfiDeviceCommunicationVerifierTest, RejectsMergedRegistrationTrait) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(R"(
    HloModule test

    ENTRY main {
      ROOT call = f32[] custom-call(),
        custom_call_target="__xla_test$$ffi_device_communication_registration_trait",
        api_version=API_VERSION_TYPED_FFI
    }
  )"));

  EXPECT_THAT(RunVerifier(module.get()),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr(kRegistrationTraitTarget)));
}

TEST_F(FfiDeviceCommunicationVerifierTest, AcceptsExplicitSideEffect) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(R"(
    HloModule test

    ENTRY main {
      ROOT call = f32[] custom-call(),
        custom_call_target="__xla_test$$ffi_device_communication_metadata_trait",
        custom_call_has_side_effect=true,
        api_version=API_VERSION_TYPED_FFI
    }
  )"));

  EXPECT_THAT(RunVerifier(module.get()), IsOkAndHolds(false));
}

TEST_F(FfiDeviceCommunicationVerifierTest,
       AcceptsHandlerWithoutDeviceCommunication) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(R"(
    HloModule test

    ENTRY main {
      ROOT call = f32[] custom-call(),
        custom_call_target="__xla_test$$ffi_without_device_communication",
        api_version=API_VERSION_TYPED_FFI
    }
  )"));

  EXPECT_THAT(RunVerifier(module.get()), IsOkAndHolds(false));
}

TEST_F(FfiDeviceCommunicationVerifierTest,
       AcceptsUnknownHandlerForDevicelessCompilation) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(R"(
    HloModule test

    ENTRY main {
      ROOT call = f32[] custom-call(),
        custom_call_target="__xla_test$$unregistered_ffi_handler",
        api_version=API_VERSION_TYPED_FFI
    }
  )"));

  EXPECT_THAT(RunVerifier(module.get()), IsOkAndHolds(false));
}

TEST_F(FfiDeviceCommunicationVerifierTest, IgnoresLegacyCustomCall) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(R"(
    HloModule test

    ENTRY main {
      ROOT call = f32[] custom-call(),
        custom_call_target="__xla_test$$ffi_device_communication_metadata_trait",
        api_version=API_VERSION_STATUS_RETURNING
    }
  )"));

  EXPECT_THAT(RunVerifier(module.get()), IsOkAndHolds(false));
}

}  // namespace
}  // namespace xla::gpu
