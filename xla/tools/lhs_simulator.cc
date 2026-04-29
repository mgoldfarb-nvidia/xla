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

// LHS Simulator: drives `ScheduleGpuModule` on a previously-dumped HLO so
// the post-LHS schedule can be inspected without doing codegen or running
// on a GPU. Useful for fast A/B iteration on collective-hint textprotos:
//
//   1. Take a `*.after_spmd_partitioner.hlo.pb` from any prior dump (it
//      contains the HLO right before the scheduling pipeline runs).
//   2. Pair it with the matching `*.gpu_target_config.pbtxt`.
//   3. Re-run scheduling with whatever `--xla_gpu_collective_hints_file`
//      we want to test.
//   4. Inspect the resulting scheduled HLO via existing analyzers
//      (`analyze_schedule_groups.py`, etc.).
//
// What this driver runs:
//   * The same `xla::gpu::ScheduleGpuModule` entry point that
//     `GpuCompiler::CompileToBackendResult` invokes.
//   * Which in turn runs `RunLatencyHidingSchedulerPasses` ->
//     `CollectiveHintsAnnotatorPass` -> `LegalizeSchedulingAnnotations` ->
//     `LatencyHidingScheduler`.
//
// What it skips: pre-scheduling passes (assumed already applied on the
// after_spmd_partitioner dump), all post-scheduling passes (Remat, Stream
// Attribute Annotator, FusionWrapper, etc.), and codegen. So the simulator
// does NOT exercise the cuDNN graph cache or any backend-config rewrites
// that happen after scheduling.

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/alias_info.h"
#include "xla/service/gpu/gpu_hlo_schedule.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tools/hlo_module_loader.h"
#include "xla/tsl/util/command_line_flags.h"
#include "xla/xla.pb.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/status.h"

namespace {
const char* const kUsage = R"(
LHS Simulator — drive ScheduleGpuModule on an HLO + hints file without
GPU compilation, so collective-hint textprotos can be A/B-tested locally.

Required:
  --hlo_input=PATH         Path to .hlo.pb / .hlo.pbtxt (after_spmd_partitioner
                           recommended; after_optimizations also works but the
                           schedule will be re-derived).
  --target_config=PATH     Path to gpu_target_config.pbtxt (XLA dumps these
                           alongside HLO when --xla_dump_to is set).

Optional:
  --hints_file=PATH        Path to CollectiveHintsConfig textproto. Sets
                           --xla_gpu_collective_hints_file in DebugOptions.
  --pgle_profile=PATH      Path to a ProfiledInstructionsProto file (.pb /
                           .pbtxt) or a directory containing one named by
                           module fingerprint. Wires LHS's PGLE estimator
                           with measured per-instruction cycle counts.
                           Without this, LHS uses a pessimistic default
                           (kHighLatency=5000) for every collective and
                           the resulting schedule order is "valid but
                           pessimistic" — usable for structural validation
                           but not as a predictor of runtime overlap.
  --pgle_latency_scaling_factor=F  Multiplier on PGLE-sourced latencies
                                   (default 1.0). Useful for sweeping how
                                   schedule shape changes as the
                                   estimator weighs collective time more
                                   or less heavily.
  --output=PATH            Where to write the scheduled HLO (default: stdout).
  --output_format=FMT      "hlo" (text, default) or "pb" (proto).
  --pointer_size=N         Pointer size in bytes (default: 8).

Example:
  bazel run -c opt //xla/tools:lhs_simulator -- \
      --hlo_input=$DUMPS/module_43775.jit_train_step.after_spmd_partitioner.hlo.pb \
      --target_config=$DUMPS/module_43775.jit_train_step.gpu_target_config.pbtxt \
      --hints_file=$EXTRAS/hybridep_zero_sum_steal_serial_v5.textproto \
      --output=/tmp/v5_scheduled.hlo
)";
}  // namespace

namespace xla::gpu {

absl::Status RunSimulator(const std::string& hlo_input,
                          const std::string& target_config_path,
                          const std::string& hints_file,
                          const std::string& pgle_profile,
                          float pgle_latency_scaling_factor,
                          int64_t parallel_collective_overlap_limit,
                          int64_t parallel_async_compute_limit,
                          const std::string& output_path,
                          const std::string& output_format,
                          int64_t pointer_size,
                          bool strip_existing_annotations) {
  // 1. Load HLO
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                      LoadModuleFromFile(hlo_input));
  std::cerr << "Loaded HLO module: " << module->name()
            << " (computations=" << module->computation_count() << ")\n";

  // 2. Patch DebugOptions on the module config to include the hints file.
  HloModuleConfig& config = module->mutable_config();
  DebugOptions debug_opts = config.debug_options();
  if (!hints_file.empty()) {
    debug_opts.set_xla_gpu_collective_hints_file(hints_file);
    std::cerr << "Set xla_gpu_collective_hints_file=" << hints_file << "\n";
  }
  if (!pgle_profile.empty()) {
    // Wires LHS's ProfileGuidedLatencyEstimator: replaces the pessimistic
    // kHighLatency=5000 default with measured per-instruction cycle counts.
    // Format expected by ReadProfileFromSources: a `.pb` (binary) or
    // `.pbtxt` (text) file containing tensorflow::profiler::ProfiledInstructionsProto,
    // OR a directory of files named `<module-fingerprint>.{pb,pbtxt}`.
    debug_opts.set_xla_gpu_pgle_profile_file_or_directory_path(pgle_profile);
    std::cerr << "Set xla_gpu_pgle_profile_file_or_directory_path="
              << pgle_profile << "\n";
  }
  if (pgle_latency_scaling_factor != 1.0f) {
    debug_opts.set_xla_gpu_pgle_latency_scaling_factor(
        pgle_latency_scaling_factor);
    std::cerr << "Set xla_gpu_pgle_latency_scaling_factor="
              << pgle_latency_scaling_factor << "\n";
  }
  if (parallel_collective_overlap_limit > 0) {
    debug_opts.set_xla_gpu_experimental_parallel_collective_overlap_limit(
        parallel_collective_overlap_limit);
    std::cerr << "Set xla_gpu_experimental_parallel_collective_overlap_limit="
              << parallel_collective_overlap_limit << "\n";
  }
  if (parallel_async_compute_limit > 0) {
    debug_opts.set_xla_gpu_experimental_parallel_async_compute_limit(
        parallel_async_compute_limit);
    std::cerr << "Set xla_gpu_experimental_parallel_async_compute_limit="
              << parallel_async_compute_limit << "\n";
  }
  // Force LHS on; some serialized HLO modules have it disabled.
  debug_opts.set_xla_gpu_enable_latency_hiding_scheduler(true);
  config.set_debug_options(debug_opts);

  // 3. Resolve device description.
  TF_ASSIGN_OR_RETURN(GpuTargetConfig target_config,
                      GetTargetConfigFromFile(target_config_path));
  const stream_executor::DeviceDescription& gpu_device_info =
      target_config.device_description;
  std::cerr << "Target device: "
            << gpu_device_info.gpu_compute_capability().ToString()
            << " (" << gpu_device_info.name() << ")\n";

  // 4. Construct supporting state.
  mlir::MLIRContext mlir_context;
  auto alias_info = std::make_unique<GpuAliasInfo>(gpu_device_info);

  // 5. Drop any existing schedule so LHS re-derives from scratch.
  if (module->has_schedule()) {
    module->clear_schedule();
    std::cerr << "Cleared pre-existing schedule on input HLO.\n";
  }

  // 5b. Strip pre-existing _scheduling_group_id (and other LHS-side scheduling
  //     metadata) annotations so the new hints file's effect is the only
  //     source of grouping. Without this, an input HLO that was previously
  //     compiled with a different hints file carries annotations forward and
  //     contaminates the simulation's output.
  if (strip_existing_annotations) {
    int n_stripped = 0;
    static constexpr absl::string_view kStripKeys[] = {
        "_scheduling_group_id",
        "_xla_force_earliest_schedule",
        "_xla_stream_id",
        "latency_metadata",
    };
    for (HloComputation* comp : module->MakeNonfusionComputations()) {
      for (HloInstruction* instr : comp->instructions()) {
        for (absl::string_view key : kStripKeys) {
          if (instr->erase_frontend_attribute(key) > 0) ++n_stripped;
        }
      }
    }
    if (n_stripped > 0) {
      std::cerr << "Stripped " << n_stripped
                << " pre-existing scheduling annotation(s) from input HLO.\n";
    }
  }

  // 6. Run scheduling.
  std::cerr << "Running ScheduleGpuModule…\n";
  TF_ASSIGN_OR_RETURN(
      ScheduleMetadata metadata,
      ScheduleGpuModule(module.get(), pointer_size, gpu_device_info,
                        &mlir_context, alias_info.get()));
  std::cerr << "  scheduler_mem_limit = " << metadata.scheduler_mem_limit
            << " bytes\n"
            << "  peak_memory_usage   = " << metadata.peak_memory_usage
            << " bytes\n";

  // 7. Emit scheduled HLO.
  if (output_format == "pb") {
    HloProto proto;
    *proto.mutable_hlo_module() = module->ToProto();
    std::string serialized;
    if (!proto.SerializeToString(&serialized)) {
      return absl::InternalError("failed to serialize HloProto");
    }
    if (output_path.empty()) {
      std::cout.write(serialized.data(), serialized.size());
    } else {
      std::ofstream os(output_path, std::ios::binary);
      os.write(serialized.data(), serialized.size());
      std::cerr << "Wrote " << serialized.size() << " bytes -> "
                << output_path << "\n";
    }
  } else {
    // text HLO
    std::string text = module->ToString();
    if (output_path.empty()) {
      std::cout << text;
    } else {
      std::ofstream os(output_path);
      os << text;
      std::cerr << "Wrote " << text.size() << " chars -> " << output_path
                << "\n";
    }
  }
  return absl::OkStatus();
}

}  // namespace xla::gpu

int main(int argc, char** argv) {
  std::string hlo_input;
  std::string target_config;
  std::string hints_file;
  std::string pgle_profile;
  float pgle_latency_scaling_factor = 1.0f;
  int64_t parallel_collective_overlap_limit = 0;  // 0 = leave unchanged
  int64_t parallel_async_compute_limit = 0;       // 0 = leave unchanged
  std::string output_path;
  std::string output_format = "hlo";
  int64_t pointer_size = 8;
  bool strip_existing_annotations = true;

  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("hlo_input", &hlo_input,
                "Path to input HLO (.hlo.pb or .hlo.pbtxt)"),
      tsl::Flag("target_config", &target_config,
                "Path to gpu_target_config.pbtxt"),
      tsl::Flag("hints_file", &hints_file,
                "Path to CollectiveHintsConfig textproto (optional)"),
      tsl::Flag("pgle_profile", &pgle_profile,
                "Path to a ProfiledInstructionsProto file (.pb binary or "
                ".pbtxt text), or a directory containing one named by "
                "module fingerprint. Wires LHS's "
                "ProfileGuidedLatencyEstimator with measured per-instruction "
                "cycle counts. Without this, LHS falls back to a flat "
                "kHighLatency=5000 default for every collective."),
      tsl::Flag("pgle_latency_scaling_factor", &pgle_latency_scaling_factor,
                "Multiplier applied to PGLE-sourced latencies (default 1.0). "
                "Useful for sweeping how schedule shape changes as the "
                "estimator weighs collective time more or less heavily."),
      tsl::Flag("parallel_collective_overlap_limit",
                &parallel_collective_overlap_limit,
                "Override xla_gpu_experimental_parallel_collective_overlap_limit "
                "(default 1). Bumping >1 lets LHS keep more collectives in "
                "flight concurrently (subject to memory pressure). Also "
                "switches LHS into prioritize_compute_over_async_start mode."),
      tsl::Flag("parallel_async_compute_limit",
                &parallel_async_compute_limit,
                "Override xla_gpu_experimental_parallel_async_compute_limit "
                "(default 2). Caps concurrent async-compute slots."),
      tsl::Flag("output", &output_path,
                "Output path for scheduled HLO (default: stdout)"),
      tsl::Flag("output_format", &output_format,
                "Output format: 'hlo' (text) or 'pb' (proto)"),
      tsl::Flag("pointer_size", &pointer_size, "Pointer size in bytes"),
      tsl::Flag("strip_existing_annotations", &strip_existing_annotations,
                "If true (default), remove _scheduling_group_id and related "
                "scheduling annotations from the input HLO before re-running "
                "the annotator. Prevents leftover annotations from a prior "
                "compile from polluting the simulation output."),
  };

  std::string usage(kUsage);
  bool parse_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  tsl::port::InitMain(argv[0], &argc, &argv);

  if (!parse_ok || hlo_input.empty() || target_config.empty()) {
    std::cerr << usage;
    std::cerr << "\nFlags:\n" << tsl::Flags::Usage(argv[0], flag_list);
    return 1;
  }

  absl::Status status = xla::gpu::RunSimulator(
      hlo_input, target_config, hints_file, pgle_profile,
      pgle_latency_scaling_factor, parallel_collective_overlap_limit,
      parallel_async_compute_limit, output_path, output_format,
      pointer_size, strip_existing_annotations);
  if (!status.ok()) {
    std::cerr << "ERROR: " << status << "\n";
    return 2;
  }
  return 0;
}
