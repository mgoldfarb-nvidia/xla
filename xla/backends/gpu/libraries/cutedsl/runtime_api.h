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

#ifndef XLA_BACKENDS_GPU_LIBRARIES_CUTEDSL_RUNTIME_API_H_
#define XLA_BACKENDS_GPU_LIBRARIES_CUTEDSL_RUNTIME_API_H_

#include "CuteDSLRuntime.h"

#ifdef __cplusplus
namespace xla::gpu::cutedsl {

// XLA's private dispatch table for the CuTeDSL runtime ABI.
struct RuntimeApi {
  decltype(&CuteDSLRT_Module_Create_From_Bytes) module_create_from_bytes;
  decltype(&CuteDSLRT_Module_Get_Function) module_get_function;
  decltype(&CuteDSLRT_Function_Run) function_run;
  decltype(&CuteDSLRT_Module_Destroy) module_destroy;
  decltype(&CuteDSLRT_GetErrorName) get_error_name;
  decltype(&CuteDSLRT_GetErrorString) get_error_string;
};

}  // namespace xla::gpu::cutedsl
#endif

#endif  // XLA_BACKENDS_GPU_LIBRARIES_CUTEDSL_RUNTIME_API_H_
